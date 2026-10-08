import csv
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "project"))
from daily_helpers import daily_rows as hourly_rows
from data.daily_features import COLUMN_MAP, SolarScaler
from fastapi.testclient import TestClient
from serving_app import model_loader
from serving_app.main import app


class MeanGenerationModel:
    def __call__(self, x, training=False):
        return x[:, :, 0].mean(axis=1, keepdims=True)


def csv_bytes(rows):
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow([*COLUMN_MAP.values(), "granularity"])
    writer.writerows([[*[r[v] for v in COLUMN_MAP.values()], "daily"] for r in rows])
    return stream.getvalue().encode("cp949")


class Day1Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.enterContext(
            patch.object(
                app.state, "forecast_db_path", self.root / "forecasts.db", create=True
            )
        )
        self.env = patch.dict(
            os.environ, {"LOADING_MODE": "lazy", "MODEL_SOURCE": "local"}
        )
        self.env.start()
        self.addCleanup(self.env.stop)
        self.rows = hourly_rows(260, "2024-01-01T00:00:00")
        self.scaler = SolarScaler().fit(self.rows[:14])
        self.model = model_loader.LoadedModel(
            MeanGenerationModel(),
            self.scaler,
            "test",
            metadata={
                "validation_end": "2023-12-31T00:00:00",
                "training_end": "2022-12-31T00:00:00",
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
            response = client.post("/predict", json={"sequence": self.rows[:14]})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertAlmostEqual(response.json()["predicted_generation_mwh"], 6.5)
        self.assertEqual(response.json()["target_timestamp"], "2024-01-15T00:00:00")

    def test_invalid_sequence_gap_duplicates_or_future_order_rejected(self):
        good = self.rows[:14]
        cases = [
            good[:-1],
            good[:7] + good[8:] + [self.rows[14]],
            good[:13] + [good[12]],
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
                    "2024-09-16T00:00:00",
                )

    def test_removed_manual_features_are_not_exposed(self):
        with TestClient(app) as client:
            for path, payload in (
                ("/predict/batch-test", {"start_timestamp": "2024-01-15T00:00:00"}),
                ("/retrain", {"cutoff_timestamp": "2024-01-10T00:00:00"}),
            ):
                with self.subTest(path=path):
                    response = client.post(path, json=payload)
                    self.assertIn(response.status_code, (404, 405), response.text)
                    self.assertNotIn(path, client.get("/openapi.json").json()["paths"])

    def test_missing_model_returns_actionable_service_error(self):
        with patch.object(
            model_loader,
            "get_model",
            side_effect=FileNotFoundError("태양광 모델을 먼저 학습하세요"),
        ):
            with TestClient(app) as client:
                response = client.post("/predict", json={"sequence": self.rows[:14]})
        self.assertEqual(response.status_code, 503)
        self.assertIn("학습", response.json()["detail"])
