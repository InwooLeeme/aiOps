"""계절 지연의 시각 정렬, 모델 저장 호환성, 평가 구간 분리 검증."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "project"))
import numpy as np
from serving_app import train_and_register as training
from serving_app.lstm_model import build_model
from tensorflow import keras


class SeasonalTests(unittest.TestCase):
    def test_daily_lags_match_target_hour_and_survive_safe_reload(self):
        model = build_model()
        self.assertIn("previous_day_generation", [layer.name for layer in model.layers])
        # Window is Jan 1 00:00 .. Jan 3 23:00; target is Jan 4 00:00.
        # Previous day is index 48, two days ago index 24 (not 47 and 23).
        x = np.zeros((1, 14, 8), dtype="float32")
        x[0, :, 0] = np.arange(14) ** 2
        probe = keras.Model(
            model.input, model.get_layer("previous_day_generation").output
        )
        np.testing.assert_allclose(probe(x), [[169]])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.keras"
            model.save(path)
            restored = keras.models.load_model(path, safe_mode=True)
            np.testing.assert_allclose(restored(x), model(x), atol=1e-6)
            clone = keras.models.clone_model(restored)
            clone.set_weights(restored.get_weights())
            np.testing.assert_allclose(clone(x), model(x), atol=1e-6)

    def test_selection_and_gate_windows_are_disjoint_in_time(self):
        self.assertTrue(hasattr(training, "split_selection_validation"))
        partition = {
            "timestamps": [
                "2023-01-01T00:00:00",
                "2023-06-30T00:00:00",
                "2023-07-01T00:00:00",
                "2023-12-31T00:00:00",
            ],
            "X": np.zeros((4, 14, 8)),
            "y": np.array([1, 2, 100, 200]),
            "persistence": [1, 2, 3, 4],
            "weekly_mean": [4, 3, 2, 1],
        }
        selection, gate = training.split_selection_validation(partition)
        np.testing.assert_array_equal(selection["y"], [1, 2])
        np.testing.assert_array_equal(gate["y"], [100, 200])
        self.assertLess(max(selection["timestamps"]), min(gate["timestamps"]))
        with self.assertRaises(ValueError):
            training.split_selection_validation(
                {k: v[:2] for k, v in partition.items()}
            )

    def test_monthly_metrics_include_night_errors_and_missing_months(self):
        result = training.evaluate_predictions(
            [0, 10, 20],
            [4, 8, 23],
            ["2024-01-01T00:00:00", "2024-01-01T12:00:00", "2024-03-01T12:00:00"],
        )
        self.assertIn("monthly", result)
        self.assertEqual(set(result["monthly"]), {"2024-01", "2024-03"})
        self.assertAlmostEqual(result["monthly"]["2024-01"]["rmse"], 10**0.5)
        self.assertEqual(result["monthly"]["2024-01"]["mae"], 3)
        self.assertEqual(result["monthly"]["2024-03"]["n_samples"], 1)

    def test_retraining_allows_missing_days_in_selection_but_not_gate(self):
        from types import SimpleNamespace
        from unittest.mock import patch

        from data.daily_features import SolarScaler
        from test_day2 import constant_model, rows

        source = rows("2024-01-01", 120)
        incumbent = SimpleNamespace(
            _keras_model=constant_model(),
            scaler=SolarScaler().fit(source),
            registry_version="1",
            version="1",
            metadata={"validation_end": "2023-12-31T00:00:00"},
        )

        def register(model, scaler, metadata, *args, **kwargs):
            return metadata

        # Feb 16 is in the selection week. Valid windows on later days remain.
        selection_gap = [r for r in source if r["timestamp"] != "2024-03-20T00:00:00"]
        with (
            patch.object(training, "_fit"),
            patch.object(training, "_log_and_register", side_effect=register),
        ):
            result = training.fine_tune(
                selection_gap, incumbent=incumbent, epochs=1, promote=False
            )
            self.assertEqual(result["metrics"]["validation"]["model"]["n_samples"], 14)
            gate_gap = [r for r in source if r["timestamp"] != "2024-04-25T00:00:00"]
            with self.assertRaises(ValueError):
                training.fine_tune(
                    gate_gap, incumbent=incumbent, epochs=1, promote=False
                )

    def test_retraining_early_stop_never_uses_final_gate_targets(self):
        from types import SimpleNamespace
        from unittest.mock import patch

        from data.daily_features import SolarScaler
        from test_day2 import constant_model, rows

        source = rows("2024-01-01", 120)
        incumbent = SimpleNamespace(
            _keras_model=constant_model(),
            scaler=SolarScaler().fit(source),
            registry_version="1",
            version="1",
            metadata={"validation_end": "2023-12-31T00:00:00"},
        )
        captured = {}

        def fit(model, scaler, train, selection, epochs):
            captured["training_end"] = max(train["timestamps"])
            captured["selection_end"] = max(selection["timestamps"])

        def register(model, scaler, metadata, *args, **kwargs):
            return metadata

        with (
            patch.object(training, "_fit", side_effect=fit),
            patch.object(training, "_log_and_register", side_effect=register),
        ):
            result = training.fine_tune(
                source, incumbent=incumbent, epochs=1, promote=False
            )
        self.assertLess(captured["selection_end"], result["validation_start"])
        self.assertLess(captured["training_end"], result["selection_start"])
        self.assertEqual(result["validation_end"], "2024-04-29T00:00:00")
