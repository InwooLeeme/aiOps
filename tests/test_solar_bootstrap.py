"""샘플/최신 업로드 선택과 초기 등록의 게이트·재시작 검증."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "project"))
import mlflow
import numpy as np
from daily_helpers import daily_rows as hourly_rows
from data import storage
from data.daily_features import SolarScaler, load_rows
from fastapi.testclient import TestClient
from serving_app import model_loader
from serving_app import train_and_register as training
from serving_app.config import MODEL_NAME
from serving_app.main import app
from test_day1 import csv_bytes
from test_day2 import constant_model


class DataFallbackTests(unittest.TestCase):
    def test_no_upload_uses_real_solar_sample_and_preview_labels_source(self):
        with (
            tempfile.TemporaryDirectory() as root,
            patch.object(storage, "UPLOAD_DIR", root),
        ):
            path = storage.latest_upload()
            rows = load_rows(path)
            self.assertEqual(len(rows), 2192)
            self.assertEqual(rows[0]["timestamp"], "2019-01-01T00:00:00")
            with patch.dict(os.environ, LOADING_MODE="lazy"), TestClient(app) as client:
                response = client.get("/data/preview")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["source"], "sample")
            self.assertEqual(len(response.json()["example"]["sequence"]), 14)

    def test_upload_preferred_and_bad_upload_not_silently_replaced(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "solar_new.csv"
            path.write_text("bad")
            self.assertEqual(storage.latest_upload(root), str(path))
            with self.assertRaises(ValueError):
                load_rows(storage.latest_upload(root))

    def test_initial_training_without_csv_resolves_latest_dataset(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "solar.csv"
            rows = hourly_rows(80)
            path.write_bytes(csv_bytes(rows))
            with (
                patch.object(storage, "latest_upload", return_value=str(path)),
                patch.object(
                    training, "_train", side_effect=RuntimeError("reached training")
                ) as train,
            ):
                with self.assertRaisesRegex(RuntimeError, "reached training"):
                    training.train_and_register()
            self.assertEqual(train.call_args.args[0], rows)


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.old_uri = mlflow.get_tracking_uri()
        self.addCleanup(mlflow.set_tracking_uri, self.old_uri)
        uri = f"sqlite:///{self.root / 'mlflow.db'}"
        self.enterContext(
            patch.dict(os.environ, MODEL_SOURCE="mlflow", MLFLOW_TRACKING_URI=uri)
        )
        mlflow.set_tracking_uri(uri)
        mlflow.create_experiment(
            "JejuSolar", artifact_location=(self.root / "artifacts").as_uri()
        )
        self.client = mlflow.MlflowClient()
        rows = hourly_rows(80)
        self.path = self.root / "solar.csv"
        self.path.write_bytes(csv_bytes(rows))
        self.enterContext(
            patch.object(storage, "latest_upload", return_value=str(self.path))
        )
        self.sequence = rows[:14]
        scaler = SolarScaler().fit(rows)
        metadata = {
            "mode": "scratch",
            "epochs": 1,
            "seed": 42,
            "seq_len": 14,
            "gate_passed": True,
            "error_threshold_mwh": 10.0,
            "metrics": {"validation": {"model": {"rmse": 1.0}}},
        }
        self.train = self.enterContext(
            patch.object(
                training,
                "_train",
                return_value=(
                    constant_model(),
                    {
                        "scaler": scaler,
                        "train": {"X": np.zeros((1, 14, 8), dtype="float32")},
                    },
                    metadata,
                    True,
                ),
            )
        )

    def test_empty_registry_prepares_once_and_restart_reuses_production(self):
        from serving_app.bootstrap import prepare_service

        first = prepare_service()
        second = prepare_service()
        self.assertEqual(first["status"], "prepared")
        self.assertEqual(second["status"], "existing")
        self.assertEqual(self.train.call_count, 1)
        self.assertEqual(
            len(self.client.search_model_versions(f"name='{MODEL_NAME}'")), 1
        )
        self.assertEqual(model_loader._load_from_mlflow().registry_version, "1")

    def test_empty_readonly_model_mount_prepares_runtime_bundle(self):
        from serving_app import bootstrap

        mount = self.root / "empty-mount"
        mount.mkdir()

        def train(csv_path, *, directory):
            model, prepared, metadata, _ = self.train.return_value
            training.save_bundle(model, prepared["scaler"], metadata, directory)

        with (
            patch.dict(os.environ, MODEL_SOURCE="local"),
            patch.object(model_loader, "LOCAL_BUNDLE_DIR", mount),
            patch.object(bootstrap, "RUNTIME_DIR", self.root / "runtime"),
            patch.object(bootstrap, "train_local", side_effect=train),
        ):
            result = bootstrap.prepare_service()
            self.assertEqual(result["status"], "prepared")
            self.assertEqual(
                model_loader.LOCAL_BUNDLE_DIR,
                self.root / "runtime/bootstrap-solar-daily",
            )
            self.assertEqual(list(mount.iterdir()), [])
            self.assertGreaterEqual(
                model_loader._load_from_local().predict_one(self.sequence), 0
            )

    def test_failed_gate_does_not_create_production(self):
        from serving_app.bootstrap import prepare_service

        model, prepared, metadata, _ = self.train.return_value
        self.train.return_value = (
            model,
            prepared,
            {**metadata, "gate_passed": False},
            False,
        )
        with self.assertRaisesRegex(ValueError, "게이트"):
            prepare_service()
        self.assertEqual(self.client.search_model_versions(f"name='{MODEL_NAME}'"), [])

    def test_candidate_load_failure_leaves_version_unpromoted(self):
        from serving_app.bootstrap import prepare_service

        with patch.object(
            model_loader, "_load_from_mlflow", side_effect=ValueError("bad bundle")
        ):
            with self.assertRaisesRegex(ValueError, "bad bundle"):
                prepare_service()
        self.assertEqual(
            self.client.get_model_version(MODEL_NAME, "1").current_stage, "None"
        )
