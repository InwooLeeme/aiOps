import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "project"))

from data.features import HAICScaler
from data.storage import UPLOAD_DIR, latest_upload
from fastapi.testclient import TestClient
from serving_app import model_loader
from serving_app.main import app


class MeanPriceModel:
    """학습 비용 없이 서빙 전처리와 출력 복원을 검사하는 예측기."""

    def predict(self, x, verbose=0):
        if x.shape != (1, 20, 2):
            raise ValueError("Expected a single 20-day, two-feature sequence")
        return x[:, :, 0].mean(axis=1, keepdims=True)


class Day1Tests(unittest.TestCase):
    def setUp(self):
        model_loader._model_cache = None
        metrics_directory = tempfile.TemporaryDirectory()
        self.addCleanup(metrics_directory.cleanup)
        metrics_path = patch.object(
            app.state,
            "request_log_path",
            Path(metrics_directory.name) / "requests.log",
            create=True,
        )
        metrics_path.start()
        self.addCleanup(metrics_path.stop)
        self.env = patch.dict(
            os.environ, {"LOADING_MODE": "lazy", "MODEL_SOURCE": "local"}
        )
        self.env.start()
        self.addCleanup(self.env.stop)
        self.addCleanup(setattr, model_loader, "_model_cache", None)

    def test_prediction_restores_price_units(self):
        scaler = HAICScaler().fit(
            [
                {"Close": 100, "Volume": 1000},
                {"Close": 200, "Volume": 2000},
            ]
        )
        model = model_loader.LoadedModel(MeanPriceModel(), scaler, "test")
        try:
            result = model.predict_one([{"close": 125, "volume": 1500}] * 20)
        except (NameError, AttributeError) as exc:
            self.fail(f"Day1 prediction remains incomplete: {exc}")
        self.assertAlmostEqual(result, 125.0)

    def test_lazy_loading_reuses_model(self):
        loads = []

        def load():
            loads.append(object())
            return loads[-1]

        with patch.object(model_loader, "_load_model", side_effect=load):
            try:
                first = model_loader.get_model()
                second = model_loader.get_model()
            except NameError as exc:
                self.fail(f"Lazy loading remains incomplete: {exc}")
        self.assertIs(first, second)
        self.assertEqual(len(loads), 1)

    def test_uploaded_data_is_found_from_other_working_directory(self):
        before = Path.cwd()
        Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
        with (
            tempfile.NamedTemporaryFile(dir=UPLOAD_DIR, suffix=".csv") as uploaded,
            tempfile.TemporaryDirectory() as directory,
        ):
            try:
                os.chdir(directory)
                try:
                    result = latest_upload()
                except FileNotFoundError as exc:
                    self.fail(
                        f"Data path still depends on the working directory: {exc}"
                    )
                self.assertTrue(Path(result).is_file())
                self.assertEqual(Path(result), Path(uploaded.name))
            finally:
                os.chdir(before)

    def test_eager_loading_prepares_real_model_at_startup(self):
        with patch.dict(os.environ, {"LOADING_MODE": "eager"}):
            with TestClient(app) as client:
                self.assertTrue(client.get("/health").json()["model_loaded"])

    def test_invalid_sequences_are_rejected_before_loading(self):
        sequence = [{"close": 160, "volume": 1200000}] * 20
        cases = [
            sequence[:19],
            sequence + [sequence[0]],
            [{"close": 0, "volume": 1200000}] + sequence[1:],
            [{"close": 160, "volume": -1}] + sequence[1:],
        ]
        with TestClient(app) as client:
            for points in cases:
                with self.subTest(points=points):
                    self.assertEqual(
                        client.post("/predict", json={"sequence": points}).status_code,
                        422,
                    )
            self.assertFalse(client.get("/health").json()["model_loaded"])

    def test_real_local_model_prediction_and_health(self):
        with TestClient(app, raise_server_exceptions=False) as client:
            self.assertFalse(client.get("/health").json()["model_loaded"])
            response = client.post(
                "/predict",
                json={
                    "sequence": [
                        {"close": 160 + i * 0.1, "volume": 1200000} for i in range(20)
                    ],
                },
            )
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()["model_version"], "v1-local")
            self.assertGreater(response.json()["predicted_close"], 0)
            self.assertTrue(client.get("/health").json()["model_loaded"])


if __name__ == "__main__":
    unittest.main()
