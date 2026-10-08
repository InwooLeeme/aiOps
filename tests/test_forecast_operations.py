import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "project"))
from daily_helpers import daily_rows as hourly_rows
from data.daily_features import SolarScaler
from fastapi.testclient import TestClient
from serving_app import model_loader
from serving_app.main import app
from test_day1 import MeanGenerationModel, csv_bytes


class ForecastOperationsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.rows = hourly_rows(73, "2024-01-01T00:00:00")
        self.model = model_loader.LoadedModel(
            MeanGenerationModel(),
            SolarScaler().fit(self.rows[:14]),
            "test",
            metadata={
                "training_end": "2022-12-31T00:00:00",
                "validation_end": "2023-12-31T00:00:00",
                "drift_threshold_mwh": 10,
            },
        )
        for p in (
            patch.dict(os.environ, {"MODEL_SOURCE": "local", "LOADING_MODE": "lazy"}),
            patch.object(model_loader, "_model_cache", self.model),
            patch.object(
                app.state, "forecast_db_path", self.root / "forecasts.db", create=True
            ),
            patch.object(
                app.state, "request_log_path", self.root / "requests.log", create=True
            ),
            patch("serving_app.routers.data.UPLOAD_DIR", str(self.root)),
        ):
            p.start()
            self.addCleanup(p.stop)

    def test_prediction_reports_its_data_time_and_excludes_historical_requests(self):
        with TestClient(app) as client:
            result = client.post("/predict", json={"sequence": self.rows[:14]}).json()
            self.assertEqual(result.get("input_end_timestamp"), "2024-01-14T00:00:00")
            self.assertEqual(result.get("forecast_context"), "historical")
            self.assertFalse(result["monitoring_eligible"])
            self.assertTrue(result["prediction_id"])

    def test_uploaded_actuals_match_saved_prediction_and_survive_new_client(self):
        with TestClient(app) as client:
            client.post("/predict", json={"sequence": self.rows[:14]})
            response = client.get("/predictions/recent")
            self.assertEqual(response.status_code, 200)
            self.assertIsNone(response.json()["records"][0]["actual"])
            result = client.post(
                "/data/upload",
                files={"file": ("data.csv", csv_bytes(self.rows), "text/csv")},
            )
            self.assertEqual(result.status_code, 200)
        with TestClient(app) as client:
            data = client.get("/predictions/recent").json()
            record = data["records"][0]
            self.assertEqual(record["actual"], 14)
            self.assertEqual(record["absolute_error_mwh"], 7.5)
            self.assertEqual(data["monitoring"]["count"], 0)

    def test_fresh_prediction_is_counted_once_after_actual_arrives(self):
        from serving_app import forecasts

        with (
            TestClient(app) as client,
            patch.object(
                forecasts, "now_kst", return_value=datetime(2024, 1, 15, 0, 30)
            ),
        ):
            first = client.post("/predict", json={"sequence": self.rows[:14]}).json()
            second = client.post("/predict", json={"sequence": self.rows[:14]}).json()
            self.assertTrue(first["monitoring_eligible"])
            self.assertEqual(first["prediction_id"], second["prediction_id"])
            with patch.object(
                forecasts, "now_kst", return_value=datetime(2024, 1, 16, 1)
            ):
                for _ in range(2):
                    client.post(
                        "/data/upload",
                        files={"file": ("data.csv", csv_bytes(self.rows), "text/csv")},
                    )
            data = client.get("/predictions/recent").json()
            self.assertEqual(len(data["records"]), 1)
            self.assertEqual(data["monitoring"]["count"], 1)
            self.assertEqual(data["monitoring"]["rmse"], 7.5)
            self.assertFalse(data["monitoring"]["ready"])

    def test_future_input_rejected_and_training_overlap_not_monitored(self):
        from serving_app import forecasts

        with (
            TestClient(app) as client,
            patch.object(
                forecasts, "now_kst", return_value=datetime(2024, 1, 14, 23, 30)
            ),
        ):
            self.assertEqual(
                client.post("/predict", json={"sequence": self.rows[:14]}).status_code,
                422,
            )
        self.model.metadata["validation_end"] = "2024-01-15T00:00:00"
        with (
            TestClient(app) as client,
            patch.object(
                forecasts, "now_kst", return_value=datetime(2024, 1, 15, 0, 30)
            ),
        ):
            data = client.post("/predict", json={"sequence": self.rows[:14]}).json()
            self.assertFalse(data["monitoring_eligible"])
            self.assertEqual(data["exclusion_reason"], "training_overlap")

    def test_preview_reports_stale_source_and_exact_target(self):
        path = self.root / "data.csv"
        path.write_bytes(csv_bytes(self.rows))
        with (
            patch("serving_app.routers.data.latest_upload", return_value=str(path)),
            TestClient(app) as client,
        ):
            data = client.get("/data/preview").json()
            self.assertEqual(data.get("input_end_timestamp"), "2024-03-13T00:00:00")
            self.assertEqual(data.get("target_timestamp"), "2024-03-14T00:00:00")
            self.assertEqual(data.get("forecast_context"), "historical")

    def test_registry_comparison_uses_same_window_incumbent_and_artifact_period(self):
        from types import SimpleNamespace

        from serving_app.routers.dashboard import version_info

        class Client:
            def get_run(self, run_id):
                return SimpleNamespace(
                    data=SimpleNamespace(
                        metrics={"rmse": 16.31, "validation_incumbent_rmse": 441.65},
                        params={"mode": "fine-tune"},
                        tags={"simulation": "true"},
                    )
                )

            def download_artifacts(inner, run_id, path):
                metadata = self.root / "metadata.json"
                metadata.write_text(
                    '{"validation_start":"2024-05-31T18:00:00","validation_end":"2024-06-07T17:00:00","parent_version":"3"}'
                )
                return str(metadata)

        version = SimpleNamespace(
            version="4",
            run_id="test",
            creation_timestamp=1000,
            current_stage="Production",
        )
        result = version_info(Client(), version)
        self.assertEqual(result.get("incumbent_rmse"), 441.65)
        self.assertEqual(result.get("validation_start"), "2024-05-31T18:00:00")
        self.assertEqual(result.get("parent_version"), "3")
        self.assertTrue(result.get("simulation"))

    def seed_window(self):
        from serving_app import forecasts

        rows = hourly_rows(28, "2024-01-01T00:00:00")
        path = self.root / "forecasts.db"
        for row in rows[14:]:
            target = datetime.fromisoformat(row["timestamp"])
            context = forecasts.forecast_context(
                (target - timedelta(days=1)).isoformat(),
                now=target + timedelta(minutes=30),
            )
            forecasts.save_prediction(path, context, 100, self.model)
        return path, rows

    def test_complete_observation_window_triggers_once_and_is_persisted(self):
        from serving_app import forecasts

        path, rows = self.seed_window()
        with (
            patch.object(forecasts, "now_kst", return_value=datetime(2024, 1, 29)),
            patch.dict(os.environ, {"MODEL_SOURCE": "mlflow"}),
            patch(
                "serving_app.train_and_register.fine_tune",
                return_value={"gate_passed": False, "promoted": False},
            ),
        ):
            result = forecasts.reconcile(path, rows)
            self.assertEqual(result["status"], "evaluated")
            self.assertEqual(result["check"]["status"], "performance_degraded")
            self.assertEqual(result["check"]["retraining"]["status"], "gate_rejected")
            repeated = forecasts.reconcile(path, rows)
            self.assertEqual(repeated["status"], "already_evaluated")
            self.assertEqual(repeated["matched"], 0)
        data = forecasts.summary(path, self.model)
        self.assertEqual(data["monitoring"]["count"], 14)
        self.assertEqual(
            data["last_evaluation"]["cutoff_timestamp"], "2024-01-28T00:00:00"
        )
        self.model.registry_version = "new-model"
        self.assertEqual(forecasts.summary(path, self.model)["monitoring"]["count"], 0)

    def test_simulation_model_never_enters_operational_monitoring(self):
        from serving_app import forecasts

        self.model.metadata["simulation"] = True
        with (
            TestClient(app) as client,
            patch.object(
                forecasts, "now_kst", return_value=datetime(2024, 1, 15, 0, 30)
            ),
        ):
            result = client.post("/predict", json={"sequence": self.rows[:14]}).json()
            self.assertFalse(result["monitoring_eligible"])
            self.assertEqual(result["exclusion_reason"], "simulation_model")

    def test_blocked_evaluation_can_retry_with_full_history(self):
        from serving_app import forecasts

        path, rows = self.seed_window()
        with (
            patch.object(forecasts, "now_kst", return_value=datetime(2024, 1, 29)),
            patch.dict(os.environ, {"MODEL_SOURCE": "mlflow"}),
        ):
            with patch(
                "serving_app.train_and_register.fine_tune",
                side_effect=ValueError("history incomplete"),
            ):
                first = forecasts.reconcile(path, rows)
                self.assertEqual(first["check"]["retraining"]["status"], "blocked")
            with patch(
                "serving_app.train_and_register.fine_tune",
                return_value={"gate_passed": False, "promoted": False},
            ):
                second = forecasts.reconcile(path, rows)
                self.assertEqual(second["status"], "evaluated")
                self.assertEqual(
                    second["check"]["retraining"]["status"], "gate_rejected"
                )

    def test_changed_or_missing_monitoring_actuals_do_not_start_training(self):
        from serving_app import forecasts

        path, rows = self.seed_window()
        with patch.object(forecasts, "now_kst", return_value=datetime(2024, 1, 29)):
            forecasts.reconcile(path, rows)
            changed = [dict(r) for r in rows]
            changed[-1]["generation_mwh"] = 500
            result = forecasts.reconcile(path, changed)
            self.assertEqual(result["status"], "observation_mismatch")
            self.assertEqual(
                forecasts.reconcile(path, rows[-7:])["status"], "observation_mismatch"
            )

    def test_inference_finishing_after_target_is_not_operational_forecast(self):
        from serving_app import forecasts

        with (
            TestClient(app) as client,
            patch.object(
                forecasts,
                "now_kst",
                side_effect=[
                    datetime(2024, 1, 15, 0, 59, 59),
                    datetime(2024, 1, 15, 1, 0, 1),
                ],
            ),
        ):
            result = client.post("/predict", json={"sequence": self.rows[:14]}).json()
            self.assertFalse(result["monitoring_eligible"])
