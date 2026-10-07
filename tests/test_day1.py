import csv
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "project"))
from data.features import COLUMN_MAP, SolarScaler
from fastapi.testclient import TestClient
from serving_app import model_loader
from serving_app.main import app
from test_solar_data import hourly_rows


class MeanGenerationModel:
    def __call__(self, x, training=False):
        return x[:, :, 0].mean(axis=1, keepdims=True)


def csv_bytes(rows):
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(COLUMN_MAP)
    writer.writerows([[r[v] for v in COLUMN_MAP.values()] for r in rows])
    return stream.getvalue().encode("cp949")


class Day1Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.env = patch.dict(
            os.environ, {"LOADING_MODE": "lazy", "MODEL_SOURCE": "local"}
        )
        self.env.start()
        self.addCleanup(self.env.stop)
        self.rows = hourly_rows(260, "2024-01-01T00:00:00")
        self.scaler = SolarScaler().fit(self.rows[:72])
        self.model = model_loader.LoadedModel(
            MeanGenerationModel(),
            self.scaler,
            "test",
            metadata={
                "validation_end": "2023-12-31T23:00:00",
                "training_end": "2022-12-31T23:00:00",
                "drift_threshold_mwh": 10.0,
            },
        )
        self.cache = patch.object(model_loader, "_model_cache", self.model)
        self.cache.start()
        self.addCleanup(self.cache.stop)
        self.log_patch = patch.object(
            app.state, "request_log_path", self.root / "requests.log", create=True
        )
        self.log_patch.start()
        self.addCleanup(self.log_patch.stop)

    def test_zero_generation_negative_temperature_and_decimal_wind_are_valid(self):
        with TestClient(app) as client:
            response = client.post("/predict", json={"sequence": self.rows[:72]})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertAlmostEqual(response.json()["predicted_generation_mwh"], 11.5)
        self.assertEqual(response.json()["target_timestamp"], "2024-01-04T00:00:00")

    def test_invalid_sequence_gap_duplicates_or_future_order_rejected(self):
        good = self.rows[:72]
        cases = [
            good[:-1],
            good[:30] + good[31:] + [self.rows[72]],
            good[:71] + [good[70]],
            list(reversed(good)),
        ]
        with TestClient(app) as client:
            for rows in cases:
                with self.subTest(rows=rows[-1]):
                    self.assertEqual(
                        client.post("/predict", json={"sequence": rows}).status_code,
                        422,
                    )

    def test_upload_cp949_preview_keeps_time_and_generation_units(self):
        with (
            patch("serving_app.routers.data.UPLOAD_DIR", str(self.root)),
            patch(
                "serving_app.routers.data.latest_upload",
                lambda: str(next(self.root.glob("solar_*.csv"))),
            ),
        ):
            with TestClient(app) as client:
                uploaded = client.post(
                    "/data/upload",
                    files={"file": ("jeju.csv", csv_bytes(self.rows), "text/csv")},
                )
                self.assertEqual(uploaded.status_code, 200, uploaded.text)
                preview = client.get("/data/preview").json()
                self.assertEqual(preview["rows"], 260)
                self.assertEqual(preview["unit"], "MWh")
                self.assertEqual(
                    preview["example"]["sequence"][-1]["timestamp"],
                    "2024-01-11T19:00:00",
                )

    def test_replay_does_not_mix_repeated_batches_or_evaluate_training_dates(self):
        self.assertTrue(
            hasattr(
                __import__("serving_app.routers.predict", fromlist=["replay"]), "replay"
            ),
            "CSV 이력 재생 필요",
        )
        csv_path = self.root / "solar_test.csv"
        csv_path.write_bytes(csv_bytes(self.rows))
        with (
            patch(
                "serving_app.routers.predict.latest_upload", return_value=str(csv_path)
            ),
            patch("serving_app.routers.predict.REPLAY_LOG", self.root / "replay.jsonl"),
        ):
            with TestClient(app) as client:
                response = client.post(
                    "/predict/batch-test",
                    json={"start_timestamp": "2024-01-04T00:00:00", "limit": 168},
                )
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(len(response.json()["records"]), 168)
                again = client.post(
                    "/predict/batch-test",
                    json={"start_timestamp": "2024-01-04T00:00:00", "limit": 168},
                )
                self.assertEqual(again.json()["drift_check"]["count"], 168)
                # Later replay errors must not authorize retraining in its past.
                self.assertEqual(
                    client.post(
                        "/retrain", json={"cutoff_timestamp": "2024-01-05T00:00:00"}
                    ).status_code,
                    422,
                )
                self.assertEqual(
                    client.post(
                        "/predict/batch-test",
                        json={"start_timestamp": "2023-01-01T00:00:00"},
                    ).status_code,
                    422,
                )
                self.assertEqual(
                    client.post(
                        "/retrain", json={"cutoff_timestamp": "2024-12-31T00:00:00"}
                    ).status_code,
                    422,
                )

    def test_missing_model_returns_actionable_service_error(self):
        with patch.object(
            model_loader,
            "get_model",
            side_effect=FileNotFoundError("태양광 모델을 먼저 학습하세요"),
        ):
            with TestClient(app) as client:
                response = client.post("/predict", json={"sequence": self.rows[:72]})
        self.assertEqual(response.status_code, 503)
        self.assertIn("학습", response.json()["detail"])
