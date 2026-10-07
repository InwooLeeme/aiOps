"""전날 기준 보정의 시간 정렬, 실제 학습 및 안전한 아티팩트 재로딩."""

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "project"))
import numpy as np
from data.features import SolarScaler
from serving_app import lstm_model
from tensorflow import keras


class DailyResidualTests(unittest.TestCase):
    def test_gate_targets_cannot_change_candidate_selection_or_weights(self):
        from serving_app import train_and_register as training
        from test_day2 import constant_model, rows

        source = rows("2024-01-01", 60 * 24)
        for i, row in enumerate(source):
            row["generation_mwh"] = 10 + 0.01 * i + 0.4 * (i % 24)
        incumbent = SimpleNamespace(
            _keras_model=constant_model(),
            scaler=SolarScaler().fit(source),
            registry_version="1",
            version="1",
            metadata={"validation_end": "2023-12-31T23:00:00"},
        )

        def register(model, scaler, metadata, *args, **kwargs):
            return metadata, model.get_weights()

        with (
            patch.object(training, "_fit"),
            patch.object(training, "_log_and_register", side_effect=register),
        ):
            original, weights = training.fine_tune(source, incumbent=incumbent)
            self.assertIn("selection_candidates", original)
            changed = [dict(r) for r in source]
            for row in changed[-168:]:
                row["generation_mwh"] += 100
            altered, other_weights = training.fine_tune(changed, incumbent=incumbent)
        self.assertEqual(
            original["selection_candidates"], altered["selection_candidates"]
        )
        self.assertEqual(original["selected_candidate"], altered["selected_candidate"])
        for a, b in zip(weights, other_weights, strict=True):
            np.testing.assert_array_equal(a, b)
        self.assertNotEqual(original["gate_passed"], altered["gate_passed"])

    def test_untrained_model_uses_previous_day_target_hour(self):
        self.assertTrue(hasattr(lstm_model, "build_daily_residual"))
        model = lstm_model.build_daily_residual()
        x = np.zeros((2, 72, 10), dtype="float32")
        x[:, -24, 0] = [0.3, 0.8]
        x[:, -1, 0] = [0.9, 0.1]
        np.testing.assert_allclose(model(x), [[0.3], [0.8]], atol=1e-6)

    def test_learns_hour_dependent_correction_and_survives_reload(self):
        self.assertTrue(hasattr(lstm_model, "build_daily_residual"))
        from serving_app.daily_residual import fit_daily_residual

        rng = np.random.default_rng(7)

        def samples(count):
            x = np.zeros((count, 72, 10), dtype="float32")
            hours = np.arange(count) % 24
            phase = 2 * np.pi * hours / 24
            x[:, -1, 6] = (np.sin(phase) + 1) / 2
            x[:, -1, 7] = (np.cos(phase) + 1) / 2
            x[:, -24, 0] = 0.5
            x[:, -25, 0] = 0.4
            x[:, -1, 0] = rng.uniform(0.2, 0.6, count)
            # Opposite corrections at adjacent hours; no scenario flag as input.
            y = 0.5 + np.where(hours % 2, 1, -1) * (x[:, -1, 0] - 0.4)
            return {"X": x, "y": y * 100}

        scaler = SolarScaler()
        scaler.minimum = [0] * 10
        scaler.maximum = [100] + [1] * 9
        train, heldout = samples(24 * 25), samples(24 * 5)
        model = lstm_model.build_daily_residual()
        fit_daily_residual(model, scaler, train, alpha=1e-4)
        predictions = np.asarray(model(heldout["X"])).reshape(-1) * 100
        self.assertLess(np.sqrt(np.mean((predictions - heldout["y"]) ** 2)), 0.1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.keras"
            model.save(path)
            restored = keras.models.load_model(path, safe_mode=True)
            np.testing.assert_allclose(
                restored(heldout["X"]), model(heldout["X"]), atol=1e-6
            )
            clone = keras.models.clone_model(restored)
            clone.set_weights(restored.get_weights())
            np.testing.assert_allclose(
                clone(heldout["X"]), model(heldout["X"]), atol=1e-6
            )
