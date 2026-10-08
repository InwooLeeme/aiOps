"""Solar training chronology, deployment gate, and artifact identity regressions."""

import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "project"))

import numpy as np
from data.daily_features import FEATURE_COLUMNS, SEQ_LEN, SolarScaler
from serving_app import model_loader
from serving_app.lstm_model import build_model
from serving_app.train_and_register import (
    _predictions,
    evaluate_predictions,
    fine_tune,
    passes_gate,
    prepare_training_data,
    save_bundle,
)
from tensorflow import keras


def rows(start, count, generation=2.0):
    begin = datetime.fromisoformat(start)
    return [
        {
            "timestamp": (begin + timedelta(days=i)).isoformat(),
            "region": "제주",
            "generation_mwh": generation + i % 3,
            "capacity_mw": 10.0,
            "temperature": 20.0,
            "humidity": 50.0,
            "wind_speed": 3.0,
            "cloud_cover": 4.0,
        }
        for i in range(count)
    ]


def constant_model(value=0.5):
    return keras.Sequential(
        [
            keras.layers.Input((SEQ_LEN, len(FEATURE_COLUMNS))),
            keras.layers.GlobalAveragePooling1D(),
            keras.layers.Dense(
                1,
                kernel_initializer="zeros",
                bias_initializer=keras.initializers.Constant(value),
            ),
        ]
    )


class Day2Tests(unittest.TestCase):
    def test_untrained_residual_model_preserves_last_generation_through_save_load(self):
        keras.utils.set_random_seed(42)
        model = build_model()
        inputs = np.full((2, SEQ_LEN, len(FEATURE_COLUMNS)), 0.7, dtype="float32")
        inputs[0, -1, 0] = 0.25
        inputs[1, -1, 0] = 1.25
        np.testing.assert_allclose(
            model(inputs, training=False), [[0.25], [1.25]], atol=1e-7
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.keras"
            model.save(path)
            restored = keras.models.load_model(path, safe_mode=True)
            np.testing.assert_allclose(
                restored(inputs, training=False), [[0.25], [1.25]], atol=1e-7
            )

    def test_local_model_version_changes_with_model_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "solar"
            scaler = SolarScaler().fit(rows("2022-01-01", 80))
            with patch.object(model_loader, "LOCAL_BUNDLE_DIR", target):
                save_bundle(constant_model(0.25), scaler, {}, target)
                first = model_loader._load_from_local()
                save_bundle(constant_model(0.75), scaler, {}, target)
                second = model_loader._load_from_local()
            self.assertNotEqual(first.version, second.version)
            self.assertAlmostEqual(first.predict_one(rows("2023-01-01", 14)), 2.5)
            self.assertAlmostEqual(second.predict_one(rows("2023-01-01", 14)), 3.5)

    def test_scaler_fit_excludes_validation_and_test_and_splits_target_year(self):
        source = rows("2022-12-01", 32) + rows("2023-12-01", 32, generation=100.0)
        prepared = prepare_training_data(source)
        self.assertEqual(prepared["train"]["timestamps"][-1], "2022-12-31T00:00:00")
        self.assertEqual(prepared["validation"]["timestamps"][0], "2023-01-01T00:00:00")
        self.assertEqual(prepared["test"]["timestamps"][0], "2024-01-01T00:00:00")
        self.assertAlmostEqual(prepared["scaler"].inverse_target(1.0), 4.0)
        self.assertEqual(prepared["validation"]["persistence"][0], 2.0)
        self.assertEqual(prepared["validation"]["weekly_mean"][0], 20 / 7)

    def test_gate_requires_finite_candidate_strictly_better_than_both_baselines(self):
        self.assertTrue(passes_gate(0.9, {"persistence": 1.0, "weekly_mean": 1.1}))
        self.assertFalse(passes_gate(1.0, {"persistence": 1.0, "weekly_mean": 1.1}))
        self.assertFalse(passes_gate(0.9, {"persistence": 1.0, "weekly_mean": 0.8}))
        self.assertFalse(passes_gate(float("nan"), {"persistence": 1.0}))
        self.assertFalse(passes_gate(0.1, {"persistence": float("nan")}))
        self.assertFalse(passes_gate(0.9, {"persistence": 1.0}, incumbent=0.8))

    def test_metrics_report_daytime_proxy_without_hiding_night_errors(self):
        result = evaluate_predictions(
            [0, 4], [2, 3], ["2023-01-01T02:00:00", "2023-01-01T12:00:00"]
        )
        self.assertAlmostEqual(result["mae"], 1.5)
        self.assertAlmostEqual(result["rmse"], (2.5) ** 0.5)
        self.assertNotIn("daytime_mae", result)

    def test_nonfinite_model_predictions_are_rejected_before_zero_clipping(self):
        class NonfiniteModel:
            def predict(self, inputs, **kwargs):
                return np.asarray([[float("nan")]])

        scaler = SolarScaler().fit(rows("2022-01-01", 80))
        with self.assertRaisesRegex(ValueError, "유한"):
            _predictions(NonfiniteModel(), {"X": np.zeros((1, 14, 8))}, scaler)

    def test_fine_tune_rejects_short_or_future_observations_before_loading_model(self):
        with self.assertRaisesRegex(ValueError, "90일"):
            fine_tune(rows("2024-01-01", 89))
        with self.assertRaisesRegex(ValueError, "미래"):
            fine_tune(rows("2099-01-01", 90))

    def test_fine_tune_validation_must_follow_incumbent_model_selection(self):
        source = rows("2023-01-01", 120)
        incumbent = SimpleNamespace(
            scaler=SolarScaler().fit(source),
            metadata={
                "training_end": "2022-12-31T00:00:00",
                "validation_end": "2023-12-31T00:00:00",
            },
        )
        with patch.object(model_loader, "get_model", return_value=incumbent):
            with self.assertRaisesRegex(ValueError, "검증 구간"):
                fine_tune(source)

    def test_local_bundle_uses_its_scaler_and_detects_swapped_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "solar"
            scaler = SolarScaler().fit(rows("2022-01-01", 80))
            metadata = {
                "training_end": "2022-12-31T00:00:00",
                "error_threshold_mwh": 1.0,
            }
            save_bundle(constant_model(), scaler, metadata, target)
            with patch.object(model_loader, "LOCAL_BUNDLE_DIR", target):
                model = model_loader._load_from_local()
                self.assertAlmostEqual(model.predict_one(rows("2023-01-01", 14)), 3.0)
                replacement = SolarScaler().fit(rows("2022-01-01", 80, generation=100))
                replacement.save(target / "scaler.json")
                with self.assertRaisesRegex(ValueError, "artifact|아티팩트"):
                    model_loader._load_from_local()

    def test_missing_bundle_is_explicit_and_failed_reload_preserves_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            incumbent = object()
            with (
                patch.object(model_loader, "LOCAL_BUNDLE_DIR", Path(directory)),
                patch.object(model_loader, "_model_cache", incumbent),
                patch.dict(os.environ, {"MODEL_SOURCE": "local"}),
            ):
                with self.assertRaises(FileNotFoundError):
                    model_loader.reload_model()
                self.assertIs(model_loader.get_model(), incumbent)

    def test_registry_loads_scaler_and_metadata_from_same_pinned_model_run(self):
        import mlflow
        import mlflow.tensorflow
        from mlflow.tracking import MlflowClient
        from serving_app.config import MODEL_NAME

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            uri = f"sqlite:///{root / 'tracking.db'}"
            old_uri = mlflow.get_tracking_uri()
            try:
                mlflow.set_tracking_uri(uri)
                experiment = mlflow.create_experiment(
                    "solar-pair", artifact_location=(root / "artifacts").as_uri()
                )
                bundle = root / "bundle"
                save_bundle(
                    constant_model(),
                    SolarScaler().fit(rows("2022-01-01", 80)),
                    {"training_end": "2022-12-31T00:00:00", "error_threshold_mwh": 1.0},
                    bundle,
                )
                with mlflow.start_run(experiment_id=experiment) as run:
                    mlflow.tensorflow.log_model(
                        constant_model(), name="model", pip_requirements=[]
                    )
                    mlflow.log_artifacts(str(bundle), "bundle")
                    version = mlflow.register_model(
                        f"runs:/{run.info.run_id}/model", MODEL_NAME
                    )
                MlflowClient().transition_model_version_stage(
                    MODEL_NAME, version.version, "Production"
                )
                with (
                    patch.dict(os.environ, {"MLFLOW_TRACKING_URI": uri}),
                    patch.object(model_loader, "LOCAL_BUNDLE_DIR", root / "missing"),
                ):
                    loaded = model_loader._load_from_mlflow()
                self.assertEqual(loaded.registry_version, str(version.version))
                self.assertEqual(loaded.metadata["training_end"], "2022-12-31T00:00:00")
                self.assertAlmostEqual(loaded.predict_one(rows("2023-01-01", 14)), 3.0)
            finally:
                mlflow.set_tracking_uri(old_uri)


if __name__ == "__main__":
    unittest.main()
