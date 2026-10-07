import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "project"))

import mlflow
import mlflow.tensorflow
from data.features import HAICScaler
from mlflow.tracking import MlflowClient
from serving_app import model_loader
from serving_app.train_and_register import _register_if_gate_passed
from tensorflow import keras


class Day2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        cls.uri = f"sqlite:///{cls.root / 'tracking.db'}"
        cls.previous_uri = mlflow.get_tracking_uri()
        mlflow.set_tracking_uri(cls.uri)
        experiment = mlflow.create_experiment(
            "test-day2",
            artifact_location=(cls.root / "artifacts").as_uri(),
        )
        model = keras.Sequential(
            [
                keras.layers.Input((20, 2)),
                keras.layers.GlobalAveragePooling1D(),
                keras.layers.Dense(
                    1,
                    kernel_initializer="zeros",
                    bias_initializer=keras.initializers.Constant(0.5),
                ),
            ]
        )
        with mlflow.start_run(experiment_id=experiment) as run:
            cls.run_id = run.info.run_id
            mlflow.tensorflow.log_model(
                model,
                name="model",
                pip_requirements=["tensorflow==2.21.0"],
            )
        scaler = HAICScaler().fit(
            [
                {"Close": 100, "Volume": 1000},
                {"Close": 200, "Volume": 2000},
            ]
        )
        cls.scaler_path = cls.root / "scaler.pkl"
        scaler.save(cls.scaler_path)

    @classmethod
    def tearDownClass(cls):
        mlflow.set_tracking_uri(cls.previous_uri)
        cls.directory.cleanup()

    def test_gate_accepts_boundary_and_rejects_worse_model(self):
        mlflow.set_tracking_uri(self.uri)
        passed = _register_if_gate_passed(None, self.run_id, 4.0)
        self.assertTrue(passed["promoted"])
        previous = MlflowClient().get_latest_versions("HAIC_Predictor", ["Production"])[
            0
        ]
        failed = _register_if_gate_passed(None, self.run_id, 4.0001)
        self.assertFalse(failed["promoted"])
        current = MlflowClient().get_latest_versions("HAIC_Predictor", ["Production"])[
            0
        ]
        self.assertEqual(current.version, previous.version)

    def test_production_loader_uses_configured_store_and_local_scaler(self):
        mlflow.set_tracking_uri(self.uri)
        _register_if_gate_passed(None, self.run_id, 4.0)
        mlflow.set_tracking_uri(f"sqlite:///{self.root / 'wrong.db'}")
        with (
            patch.dict(os.environ, {"MLFLOW_TRACKING_URI": self.uri}),
            patch.object(model_loader, "SCALER_PATH", self.scaler_path),
        ):
            try:
                model = model_loader._load_from_mlflow()
                prediction = model.predict_one([{"close": 125, "volume": 1500}] * 20)
            except Exception as exc:
                self.fail(
                    f"Production loading is not ready: {type(exc).__name__}: {exc}"
                )
        self.assertEqual(model.version, "production")
        self.assertAlmostEqual(prediction, 150.0)


if __name__ == "__main__":
    unittest.main()
