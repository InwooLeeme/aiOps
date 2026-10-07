"""운영 배치 누적, 자동 재학습 및 안전한 모델 교체 회귀 검증."""

import os
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "project"))
from data.features import SolarScaler
from fastapi.testclient import TestClient
from serving_app import model_loader
from serving_app.main import app
from serving_app.routers import predict
from test_day1 import csv_bytes
from test_solar_data import hourly_rows


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.stack = self.enterContext(ExitStack())
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.rows = hourly_rows(60 * 24, "2024-01-01T00:00:00")
        self.path = self.root / "solar.csv"
        self.path.write_bytes(csv_bytes(self.rows))
        self.model = model_loader.LoadedModel(
            lambda x, training=False: [[5.0]],
            SolarScaler().fit(self.rows[:72]),
            "production",
            "1",
            {"validation_end": "2023-11-30T23:00:00", "drift_threshold_mwh": 10.0},
        )
        for context in (
            patch.dict(os.environ, {"MODEL_SOURCE": "mlflow", "LOADING_MODE": "lazy"}),
            patch.object(model_loader, "_model_cache", self.model),
            patch.object(predict, "latest_upload", return_value=str(self.path)),
            patch.object(predict, "REPLAY_LOG", self.root / "replay.jsonl"),
            patch.object(predict, "_last_replay", {}),
            patch.object(predict, "_batch_contexts", {}),
            patch.object(predict, "recent_predictions", []),
            patch.object(
                app.state, "request_log_path", self.root / "requests.jsonl", create=True
            ),
        ):
            self.stack.enter_context(context)
        self.training = self.stack.enter_context(
            patch(
                "serving_app.train_and_register.fine_tune",
                return_value={
                    "promoted": False,
                    "gate_passed": False,
                    "version": None,
                    "rmse": 20.0,
                },
            )
        )
        self.client = self.stack.enter_context(TestClient(app))

    def replay(self, start="2024-02-01T00:00:00", limit=168):
        response = self.client.post(
            "/predict/batch-test", json={"start_timestamp": start, "limit": limit}
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_partial_batches_accumulate_and_trigger_once_when_ready(self):
        first = self.replay(limit=84)
        self.assertEqual(first["drift_check"]["status"], "insufficient_data")
        self.training.assert_not_called()
        second = self.replay("2024-02-04T12:00:00", 84)
        self.assertEqual(second["drift_check"]["count"], 168)
        self.assertEqual(second["drift_check"]["retraining"]["status"], "gate_rejected")
        self.assertEqual(self.training.call_count, 1)
        history = self.training.call_args.args[0]
        self.assertEqual(history[-1]["timestamp"], "2024-02-07T23:00:00")
        self.assertEqual(len(predict.recent_predictions), 168)
        self.assertIs(model_loader.get_model(), self.model)

    def test_repeat_batch_and_manual_retry_do_not_retrain_same_observations(self):
        self.replay()
        result = self.replay()
        self.assertEqual(self.training.call_count, 1)
        self.assertEqual(result["drift_check"]["new_count"], 0)
        manual = self.client.post(
            "/retrain", json={"cutoff_timestamp": "2024-02-07T23:00:00"}
        )
        self.assertEqual(manual.json()["status"], "blocked")
        self.assertEqual(self.training.call_count, 1)
        self.assertEqual(
            len((self.root / "replay.jsonl").read_text().splitlines()), 168
        )

    def test_overlap_only_adds_new_hours_and_reverse_replay_is_rejected(self):
        self.replay(limit=100)
        result = self.replay("2024-02-03T00:00:00", 120)
        self.assertEqual(result["drift_check"]["count"], 168)
        self.assertEqual(result["drift_check"]["new_count"], 68)
        self.assertEqual(self.training.call_count, 1)
        response = self.client.post(
            "/predict/batch-test",
            json={"start_timestamp": "2024-01-20T00:00:00", "limit": 168},
        )
        self.assertEqual(response.status_code, 409)
        self.assertEqual(predict._last_replay["cutoff"], "2024-02-07T23:00:00")

    def test_dataset_or_model_change_starts_fresh_monitoring_window(self):
        for change in ("dataset", "model"):
            with self.subTest(change=change):
                predict._last_replay.clear()
                predict._batch_contexts.clear()
                predict.recent_predictions.clear()
                self.replay(limit=84)
                if change == "dataset":
                    self.rows[0]["temperature"] = 12.0
                    self.path.write_bytes(csv_bytes(self.rows))
                else:
                    self.model.registry_version = "2"
                result = self.replay("2024-02-04T12:00:00", 84)
                self.assertEqual(result["drift_check"]["count"], 84)
                self.training.assert_not_called()

    def test_gap_does_not_count_as_continuous_week(self):
        self.replay(limit=84)
        result = self.replay("2024-02-04T13:00:00", 84)
        self.assertFalse(result["drift_check"]["ready"])
        self.training.assert_not_called()

    def test_normal_or_local_batches_never_train(self):
        self.model.metadata["drift_threshold_mwh"] = 200.0
        self.assertEqual(self.replay()["drift_check"]["status"], "ok")
        predict._last_replay.clear()
        predict._batch_contexts.clear()
        predict.recent_predictions.clear()
        self.model.metadata["drift_threshold_mwh"] = 10.0
        with patch.dict(os.environ, {"MODEL_SOURCE": "local"}):
            result = self.replay()
        self.assertEqual(result["drift_check"]["retraining"]["status"], "blocked")
        self.training.assert_not_called()

    def test_training_failure_is_reported_without_losing_predictions_or_retrying(self):
        self.training.side_effect = RuntimeError("training unavailable")
        result = self.replay()
        self.assertEqual(len(result["records"]), 168)
        self.assertEqual(result["drift_check"]["retraining"]["status"], "failed")
        self.assertIs(model_loader.get_model(), self.model)
        self.replay()
        self.assertEqual(self.training.call_count, 1)

    def test_future_observations_rejected_without_poisoning_monitoring(self):
        self.path.write_bytes(csv_bytes(hourly_rows(240, "2099-01-01T00:00:00")))
        response = self.client.post(
            "/predict/batch-test",
            json={"start_timestamp": "2099-01-04T00:00:00", "limit": 168},
        )
        self.assertEqual(response.status_code, 422)
        self.assertEqual(predict.recent_predictions, [])
        self.assertEqual(predict._last_replay, {})
        self.assertFalse((self.root / "replay.jsonl").exists())

    def test_replay_response_identifies_evaluated_source_bytes(self):
        import hashlib

        result = self.replay(limit=1)
        self.assertEqual(
            result["dataset_sha256"], hashlib.sha256(self.path.read_bytes()).hexdigest()
        )

    def test_training_selection_end_is_checked_even_if_validation_end_is_older(self):
        self.model.metadata["training_end"] = "2024-02-02T00:00:00"
        response = self.client.post(
            "/predict/batch-test", json={"start_timestamp": "2024-02-01T00:00:00"}
        )
        self.assertEqual(response.status_code, 422)
        self.training.assert_not_called()


class PromotionTests(unittest.TestCase):
    """실제 임시 MLflow 저장소와 Keras 모델로 승격·캐시 결과를 확인한다."""

    def setUp(self):
        import mlflow
        from mlflow.tracking import MlflowClient
        from serving_app.config import MODEL_NAME
        from test_day2 import constant_model

        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.old_uri = mlflow.get_tracking_uri()
        self.addCleanup(mlflow.set_tracking_uri, self.old_uri)
        self.uri = f"sqlite:///{self.root / 'mlflow.db'}"
        self.enterContext(
            patch.dict(
                os.environ, {"MODEL_SOURCE": "mlflow", "MLFLOW_TRACKING_URI": self.uri}
            )
        )
        mlflow.set_tracking_uri(self.uri)
        exp = mlflow.create_experiment(
            "JejuSolar", artifact_location=(self.root / "artifacts").as_uri()
        )
        self.client = MlflowClient()
        self.name = MODEL_NAME
        self.sequence = hourly_rows(72, "2024-01-01T00:00:00")
        self.scaler = SolarScaler().fit(self.sequence)
        self.models = []
        for value in (0.25, 0.5):
            model = constant_model(value)
            with mlflow.start_run(experiment_id=exp) as run:
                mlflow.tensorflow.log_model(model, name="model", pip_requirements=[])
                bundle = self.root / "bundle"
                from serving_app.train_and_register import save_bundle

                save_bundle(
                    model,
                    self.scaler,
                    {"gate_passed": True, "parent_version": "1"},
                    bundle,
                )
                mlflow.log_artifacts(str(bundle), "bundle")
                version = mlflow.register_model(
                    f"runs:/{run.info.run_id}/model", self.name
                )
                self.models.append(version)
        self.client.transition_model_version_stage(self.name, "1", "Production")
        self.incumbent = model_loader._load_from_mlflow()
        self.enterContext(patch.object(model_loader, "_model_cache", self.incumbent))

    def test_candidate_is_loaded_smoked_promoted_and_served_as_same_version(self):
        self.assertTrue(
            hasattr(model_loader, "promote_candidate"),
            "검증한 후보만 승격·교체하는 경로 필요",
        )
        result = model_loader.promote_candidate("2", self.incumbent, self.sequence)
        self.assertEqual(result.registry_version, "2")
        self.assertIs(model_loader.get_model(), result)
        self.assertAlmostEqual(result.predict_one(self.sequence), 11.5)
        self.assertEqual(
            self.client.get_model_version(self.name, "2").current_stage, "Production"
        )
        self.assertEqual(
            self.client.get_model_version(self.name, "1").current_stage, "Archived"
        )

    def test_corrupt_candidate_keeps_production_and_serving_cache(self):
        self.assertTrue(
            hasattr(model_loader, "promote_candidate"),
            "후보 로딩 실패 시 운영 모델 보존 필요",
        )
        damaged = self.root / "scaler.json"
        damaged.write_text("{}")
        self.client.log_artifact(self.models[1].run_id, str(damaged), "bundle")
        with self.assertRaisesRegex(ValueError, "artifact"):
            model_loader.promote_candidate("2", self.incumbent, self.sequence)
        self.assertIs(model_loader.get_model(), self.incumbent)
        self.assertEqual(
            self.client.get_model_version(self.name, "1").current_stage, "Production"
        )
        self.assertEqual(
            self.client.get_model_version(self.name, "2").current_stage, "None"
        )

    def test_candidate_smoke_failure_keeps_production(self):
        self.assertTrue(
            hasattr(model_loader, "promote_candidate"), "후보 예측 검증 필요"
        )
        with self.assertRaises(ValueError):
            model_loader.promote_candidate("2", self.incumbent, self.sequence[:1])
        self.assertIs(model_loader.get_model(), self.incumbent)
        self.assertEqual(
            self.client.get_model_version(self.name, "1").current_stage, "Production"
        )

    def test_external_production_change_blocks_stale_training_result(self):
        self.assertTrue(
            hasattr(model_loader, "promote_candidate"), "기존 운영 버전 확인 필요"
        )
        self.client.transition_model_version_stage(self.name, "1", "Archived")
        with self.assertRaisesRegex(ValueError, "변경"):
            model_loader.promote_candidate("2", self.incumbent, self.sequence)
        self.assertIs(model_loader.get_model(), self.incumbent)
        self.assertEqual(
            self.client.get_model_version(self.name, "2").current_stage, "None"
        )

    def test_batch_promotion_changes_next_prediction_and_resets_monitoring(self):
        rows = hourly_rows(60 * 24, "2024-01-01T00:00:00")
        path = self.root / "solar.csv"
        path.write_bytes(csv_bytes(rows))
        self.incumbent.metadata.update(
            validation_end="2023-11-30T23:00:00", drift_threshold_mwh=1.0
        )
        with (
            patch.dict(os.environ, {"LOADING_MODE": "lazy"}),
            patch.object(predict, "latest_upload", return_value=str(path)),
            patch.object(predict, "REPLAY_LOG", self.root / "replay.jsonl"),
            patch.object(predict, "_last_replay", {}),
            patch.object(predict, "_batch_contexts", {}),
            patch.object(predict, "recent_predictions", []),
            patch.object(
                app.state, "request_log_path", self.root / "requests.jsonl", create=True
            ),
            patch(
                "serving_app.train_and_register.fine_tune",
                return_value={
                    "promoted": False,
                    "gate_passed": True,
                    "version": "2",
                    "rmse": 1.0,
                },
            ),
            TestClient(app) as client,
        ):
            response = client.post(
                "/predict/batch-test",
                json={"start_timestamp": "2024-02-01T00:00:00", "limit": 168},
            )
            self.assertEqual(response.status_code, 200, response.text)
            result = response.json()["drift_check"]["retraining"]
            self.assertTrue(result["promoted"])
            self.assertEqual(result["served_model_version"], "2")
            self.assertEqual(predict.recent_predictions, [])
            prediction = client.post("/predict", json={"sequence": rows[:72]}).json()
            self.assertEqual(prediction["model_version"], "2")
            self.assertEqual(prediction["predicted_generation_mwh"], 11.5)

    def test_deferred_registration_never_promotes_until_activation(self):
        import numpy as np
        from serving_app.train_and_register import _log_and_register, passes_gate
        from test_day2 import constant_model

        for error in (3.0, 1.0):
            scores = {
                name: {"rmse": rmse, "monthly": {"2024-02": {"rmse": rmse}}}
                for name, rmse in (
                    ("model", error),
                    ("persistence", 2.0),
                    ("previous_day", 4.0),
                    ("incumbent", 2.5),
                )
            }
            metadata = {
                "epochs": 1,
                "seed": 42,
                "seq_len": 72,
                "mode": "fine-tune",
                "metrics": {"validation": scores},
                "error_threshold_mwh": 1.5 * error,
                "gate_passed": error == 1.0,
                "parent_version": "1",
            }
            passed = passes_gate(error, {"persistence": 2.0, "previous_day": 4.0}, 2.5)
            result = _log_and_register(
                constant_model(),
                self.scaler,
                metadata,
                passed,
                np.zeros((1, 72, 10), dtype="float32"),
                promote=False,
            )
            self.assertFalse(result["promoted"])
            self.assertEqual(result["gate_passed"], passed)
            self.assertEqual(
                self.client.get_model_version(self.name, "1").current_stage,
                "Production",
            )
            if passed:
                self.assertEqual(
                    self.client.get_model_version(
                        self.name, result["version"]
                    ).current_stage,
                    "None",
                )
            else:
                self.assertIsNone(result["version"])
