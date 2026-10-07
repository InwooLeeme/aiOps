import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "project"))

import mlflow
from data.storage import latest_upload
from fastapi.testclient import TestClient
from mlflow.tracking import MlflowClient
from serving_app import model_loader
from serving_app.main import app


class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.log_path = self.root / "requests.log"
        self.env = patch.dict(
            os.environ, {"MODEL_SOURCE": "local", "LOADING_MODE": "lazy"}
        )
        self.env.start()
        self.addCleanup(self.env.stop)
        self.path_patch = patch.object(
            app.state, "request_log_path", self.log_path, create=True
        )
        self.path_patch.start()
        self.addCleanup(self.path_patch.stop)
        self.cache = patch.object(model_loader, "_model_cache", None)
        self.cache.start()
        self.addCleanup(self.cache.stop)

    def test_metrics_count_real_predictions_and_validation_errors_only(self):
        with TestClient(app) as client:
            response = client.get("/metrics/summary")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["request_count"], 0)
            client.get("/health")
            sequence = [{"close": 165, "volume": 1200000}] * 20
            self.assertEqual(
                client.post("/predict", json={"sequence": sequence}).status_code,
                200,
            )
            self.assertEqual(
                client.post("/predict", json={"sequence": sequence[:19]}).status_code,
                422,
            )
            summary = client.get("/metrics/summary").json()
            self.assertEqual(summary["request_count"], 2)
            self.assertEqual(summary["success_rate"], 50.0)
            self.assertEqual(summary["error_rate"], 0.5)
            self.assertGreater(summary["avg_latency_ms"], 0)
            self.assertEqual(sum(p["request_count"] for p in summary["series"]), 2)

    def test_metrics_filter_time_window_and_skip_damaged_log_lines(self):
        now = time.time()
        records = [
            {"timestamp": now - 400, "status": 500, "latency_ms": 900},
            {"timestamp": now - 10, "status": 200, "latency_ms": 10},
            {"timestamp": now - 5, "status": 422, "latency_ms": 30},
        ]
        self.log_path.write_text(
            "broken line\n" + "\n".join(json.dumps(r) for r in records) + "\n"
        )
        with TestClient(app) as client:
            response = client.get("/metrics/summary?window=5m")
            self.assertEqual(response.status_code, 200)
            summary = response.json()
            self.assertEqual(summary["request_count"], 2)
            self.assertEqual(summary["avg_latency_ms"], 20.0)
            self.assertEqual(summary["success_rate"], 50.0)
            self.assertEqual(
                client.get("/metrics/summary?window=1h").json()["request_count"], 3
            )
            self.assertEqual(
                client.get("/metrics/summary?window=invalid").status_code, 422
            )

    def test_dataset_preview_uses_latest_upload_and_matches_prediction_input(self):
        csv_path = self.root / "latest.csv"
        csv_path.write_text(
            "Date,Close,Volume\n"
            + "\n".join(f"2026-01-{i + 1:02d},{100 + i},{1000 + i}" for i in range(25))
        )
        with patch(
            "serving_app.routers.data.latest_upload", return_value=str(csv_path)
        ):
            with TestClient(app) as client:
                response = client.get("/data/preview")
                self.assertEqual(response.status_code, 200)
                data = response.json()
                self.assertEqual(data["rows"], 25)
                self.assertEqual(data["avg_volume"], 1012)
                self.assertEqual((data["min_close"], data["max_close"]), (100, 124))
                self.assertEqual(len(data["example"]["sequence"]), 20)
                self.assertEqual(data["example"]["sequence"][0]["close"], 105)
                self.assertEqual(data["example"]["sequence"][-1]["volume"], 1024)

    def test_system_info_reports_runtime_environment(self):
        with TestClient(app) as client:
            with patch.dict(
                os.environ, {"MODEL_SOURCE": "mlflow", "LOADING_MODE": "eager"}
            ):
                response = client.get("/system/info")
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["model_source"], "mlflow")
                self.assertEqual(response.json()["loading_mode"], "eager")

    def test_registry_history_keeps_archived_stage_and_flags_stale_serving_cache(self):
        previous_uri = mlflow.get_tracking_uri()
        self.addCleanup(mlflow.set_tracking_uri, previous_uri)
        uri = f"sqlite:///{self.root / 'tracking.db'}"
        mlflow.set_tracking_uri(uri)
        registry = MlflowClient(tracking_uri=uri)
        experiment = registry.create_experiment("dashboard-test")
        registry.create_registered_model("HAIC_Predictor")
        for mode, score in [("scratch", 2.03), ("fine-tune", 1.69)]:
            run = registry.create_run(experiment)
            registry.log_param(run.info.run_id, "mode", mode)
            registry.log_metric(run.info.run_id, "rmse", score)
            version = registry.create_model_version(
                "HAIC_Predictor", (self.root / "model").as_uri(), run.info.run_id
            )
            registry.transition_model_version_stage(
                "HAIC_Predictor",
                version.version,
                "Production",
                archive_existing_versions=True,
            )
        cached = SimpleNamespace(version="production", registry_version="1")
        with (
            patch.dict(
                os.environ, {"MLFLOW_TRACKING_URI": uri, "MODEL_SOURCE": "mlflow"}
            ),
            patch.object(model_loader, "_model_cache", cached),
            TestClient(app) as client,
        ):
            response = client.get("/models/overview")
            self.assertEqual(response.status_code, 200)
            result = response.json()
            self.assertEqual(result["production"]["version"], "2")
            self.assertEqual(result["served_model"]["version"], "1")
            self.assertTrue(result["reload_required"])
            self.assertEqual(result["versions"][0]["rmse"], 1.69)
            self.assertEqual(result["versions"][1]["stage"], "Archived")

    def test_missing_registry_does_not_create_database_or_fake_model(self):
        missing = self.root / "missing.db"
        with (
            patch.dict(os.environ, {"MLFLOW_TRACKING_URI": f"sqlite:///{missing}"}),
            TestClient(app) as client,
        ):
            response = client.get("/models/overview")
            self.assertEqual(response.status_code, 200)
            self.assertIsNone(response.json()["production"])
            self.assertEqual(response.json()["versions"], [])
            self.assertFalse(missing.exists())

    def test_recent_events_ignore_bad_dates_and_keep_newest_first(self):
        (self.root / "aiops.log").write_text(
            "2026-01-02 09:00:00,000 [WARNING] [WARN] drift detected\n"
            "2026-01-02 09:00:01,000 [INFO] [OK] production promoted\n"
            "2026-99-02 09:00:02,000 [INFO] damaged timestamp\n"
        )
        with (
            patch("serving_app.config.LOG_DIR", self.root),
            TestClient(app, raise_server_exceptions=False) as client,
        ):
            response = client.get("/events/recent")
            self.assertEqual(response.status_code, 200)
            events = response.json()
            self.assertEqual(len(events), 2)
            self.assertEqual(events[0]["message"], "[OK] production promoted")
            self.assertEqual(events[1]["level"], "WARNING")

    def test_uploaded_csv_is_used_by_preview(self):
        csv_bytes = (
            Path(__file__).resolve().parents[1] / "project/data/sample_haic_prices.csv"
        ).read_bytes()
        with (
            patch("serving_app.routers.data.UPLOAD_DIR", str(self.root)),
            patch(
                "serving_app.routers.data.latest_upload",
                lambda: latest_upload(str(self.root)),
            ),
            TestClient(app) as client,
        ):
            response = client.post(
                "/data/upload", files={"file": ("sample.csv", csv_bytes, "text/csv")}
            )
            self.assertEqual(response.status_code, 200)
            preview = client.get("/data/preview").json()
            self.assertEqual(preview["filename"], response.json()["filename"])
            self.assertEqual(preview["source"], "upload")
            self.assertEqual(preview["rows"], 756)


if __name__ == "__main__":
    unittest.main()
