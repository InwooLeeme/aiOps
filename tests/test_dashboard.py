import json
import os
import sys
import tempfile
import time
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "project"))

import mlflow
from data.daily_features import SolarScaler
from data.storage import latest_upload
from fastapi.testclient import TestClient
from mlflow.tracking import MlflowClient
from serving_app import config, model_loader
from serving_app.main import app


class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.enterContext(
            patch.object(
                app.state, "forecast_db_path", self.root / "forecasts.db", create=True
            )
        )
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
        sequence = self.solar_rows(14)
        cached = model_loader.LoadedModel(
            keras_model=lambda x, training=False: [[0.5]],
            scaler=SolarScaler().fit(sequence),
            version="test-solar",
        )
        with (
            patch.object(model_loader, "_model_cache", cached),
            TestClient(app) as client,
        ):
            response = client.get("/metrics/summary")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["request_count"], 0)
            client.get("/health")
            self.assertEqual(
                client.post("/predict", json={"sequence": sequence}).status_code,
                200,
            )
            self.assertEqual(
                client.post("/predict", json={"sequence": sequence[:13]}).status_code,
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

    @staticmethod
    def solar_rows(count):
        start = datetime(2024, 1, 1)
        return [
            {
                "timestamp": (start + timedelta(days=i)).isoformat(),
                "region": "제주",
                "generation_mwh": float(i % 10),
                "capacity_mw": 100.0,
                "temperature": 15.0,
                "humidity": 60.0,
                "wind_speed": 3.0,
                "cloud_cover": 4.0,
            }
            for i in range(count)
        ]

    def write_solar_csv(self, path, count=240):
        import csv

        rows = self.solar_rows(count)
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=[*rows[0], "granularity"])
            writer.writeheader()
            writer.writerows([{**r, "granularity": "daily"} for r in rows])

    def test_dataset_preview_uses_latest_upload_and_matches_prediction_input(self):
        csv_path = self.root / "latest.csv"
        self.write_solar_csv(csv_path)
        with patch(
            "serving_app.routers.data.latest_upload", return_value=str(csv_path)
        ):
            with TestClient(app) as client:
                response = client.get("/data/preview")
                self.assertEqual(response.status_code, 200)
                data = response.json()
                self.assertEqual(data["rows"], 240)
                self.assertEqual(data["region"], "제주")
                self.assertEqual(data["missing_days"], 0)
                self.assertEqual(
                    (data["min_generation_mwh"], data["max_generation_mwh"]), (0, 9)
                )
                self.assertEqual(len(data["example"]["sequence"]), 14)
                self.assertIn("timestamp", data["example"]["sequence"][0])
                self.assertEqual(data["example"]["sequence"][-1]["capacity_mw"], 100)

    def test_metrics_uses_loaded_model_threshold_and_never_invents_one(self):
        with TestClient(app) as client:
            self.assertIsNone(
                client.get("/metrics/summary").json()["drift"]["threshold"]
            )
            cached = SimpleNamespace(metadata={"drift_threshold_mwh": 1.25})
            with patch.object(model_loader, "_model_cache", cached):
                self.assertEqual(
                    client.get("/metrics/summary").json()["drift"]["threshold"], 1.25
                )

    def test_system_info_reports_runtime_environment(self):
        with TestClient(app) as client:
            with patch.dict(
                os.environ, {"MODEL_SOURCE": "mlflow", "LOADING_MODE": "eager"}
            ):
                response = client.get("/system/info")
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["model_source"], "mlflow")
                self.assertEqual(response.json()["loading_mode"], "eager")
                self.assertEqual(response.json()["unit"], "MWh")
                self.assertEqual(response.json()["seq_len"], 14)
                self.assertEqual(response.json()["n_features"], 8)
                self.assertIsNone(response.json()["rmse_threshold"])

    def test_registry_history_keeps_archived_stage_and_flags_stale_serving_cache(self):
        previous_uri = mlflow.get_tracking_uri()
        self.addCleanup(mlflow.set_tracking_uri, previous_uri)
        uri = f"sqlite:///{self.root / 'tracking.db'}"
        mlflow.set_tracking_uri(uri)
        registry = MlflowClient(tracking_uri=uri)
        experiment = registry.create_experiment("dashboard-test")
        registry.create_registered_model(config.MODEL_NAME)
        for mode, score in [("scratch", 2.03), ("fine-tune", 1.69)]:
            run = registry.create_run(experiment)
            registry.log_param(run.info.run_id, "mode", mode)
            registry.log_metric(run.info.run_id, "rmse", score)
            version = registry.create_model_version(
                config.MODEL_NAME, (self.root / "model").as_uri(), run.info.run_id
            )
            registry.transition_model_version_stage(
                config.MODEL_NAME,
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

    def test_local_model_reports_validation_metrics_without_registry(self):
        cached = SimpleNamespace(
            version="solar-local",
            registry_version=None,
            metadata={
                "validation_metrics": {"rmse": 2.75},
                "mode": "scratch",
                "gate_baseline_rmse": 3.1,
                "gate_passed": False,
                "drift_threshold_mwh": 4.125,
            },
        )
        with (
            patch.dict(
                os.environ,
                {"MLFLOW_TRACKING_URI": f"sqlite:///{self.root / 'absent.db'}"},
            ),
            patch.object(model_loader, "_model_cache", cached),
            TestClient(app) as client,
        ):
            result = client.get("/models/overview").json()
            self.assertEqual(result["served_model"]["rmse"], 2.75)
            self.assertEqual(result["served_model"]["stage"], "Local")
            self.assertIs(result["served_model"]["gate_passed"], False)
            self.assertEqual(client.get("/system/info").json()["rmse_gate"], 3.1)

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

    def test_log_api_preserves_develop_file_access_inside_log_directory(self):
        (self.root / "aiops.log").write_text("legacy training result", encoding="utf-8")
        (self.root / "custom.log").write_text("custom event", encoding="utf-8")
        outside = self.root.parent / (self.root.name + "-outside")
        outside.write_text("outside")
        self.addCleanup(outside.unlink)
        (self.root / "linked.log").symlink_to(outside)
        with (
            patch("serving_app.routers.logs.LOG_DIR", str(self.root)),
            TestClient(app) as client,
        ):
            names = [item["name"] for item in client.get("/logs").json()]
            self.assertIn("aiops.log", names)
            self.assertIn("custom.log", names)
            self.assertNotIn("linked.log", names)
            self.assertEqual(
                client.get("/logs/aiops.log").json()["content"],
                "legacy training result",
            )
            self.assertEqual(client.get("/logs/linked.log").status_code, 400)

    def test_recent_events_ignore_bad_dates_and_keep_newest_first(self):
        (self.root / "solar_aiops.log").write_text(
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
        path = self.root / "input.csv"
        self.write_solar_csv(path)
        csv_bytes = path.read_bytes()
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
            self.assertEqual(preview["rows"], 240)


if __name__ == "__main__":
    unittest.main()
