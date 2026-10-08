"""버튼 배치의 운영 재학습, 데이터 보존, 중복 및 지표 검증."""

import copy
import hashlib
import os
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "project"))
from daily_helpers import daily_rows as hourly_rows
from fastapi.testclient import TestClient
from serving_app import model_loader
from serving_app.main import app
from serving_app.monitoring.simulation import inject_curtailment
from serving_app.routers import predict
from test_day1 import csv_bytes


class SimulationTests(unittest.TestCase):
    def setUp(self):
        self.stack = self.enterContext(ExitStack())
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.rows = hourly_rows(120, "2024-02-01T00:00:00")
        for r in self.rows:
            r["generation_mwh"] = 100.0
        self.path = self.root / "solar.csv"
        self.path.write_bytes(csv_bytes(self.rows))
        self.model = SimpleNamespace(
            registry_version="1",
            version="production",
            metadata={
                "validation_end": "2023-11-30T00:00:00",
                "drift_threshold_mwh": 10.0,
            },
            predict_one=lambda w: w[-1]["generation_mwh"],
        )
        for ctx in (
            patch.dict(os.environ, MODEL_SOURCE="mlflow", LOADING_MODE="lazy"),
            patch.object(model_loader, "_model_cache", self.model),
            patch.object(predict, "latest_upload", return_value=str(self.path)),
            patch.object(predict, "REPLAY_LOG", self.root / "replay.jsonl"),
            patch.object(predict, "SIMULATION_LOG", self.root / "simulation.jsonl"),
            patch.object(predict, "_last_replay", {}),
            patch.object(predict, "_batch_contexts", {}),
            patch.object(predict, "recent_predictions", []),
            patch.object(
                app.state, "request_log_path", self.root / "requests.jsonl", create=True
            ),
        ):
            self.stack.enter_context(ctx)
        self.train = self.stack.enter_context(
            patch(
                "serving_app.train_and_register.fine_tune",
                return_value={
                    "gate_passed": False,
                    "promoted": False,
                    "version": None,
                    "rmse": 20.0,
                },
            )
        )
        self.client = self.stack.enter_context(TestClient(app))

    def run_batch(self, scenario):
        response = self.client.post(
            "/simulation/run",
            json={"scenario": scenario, "start_timestamp": "2024-05-01"},
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_normal_and_drift_share_operating_pipeline_and_preserve_source(self):
        before = self.path.read_bytes()
        normal = self.run_batch("normal")
        self.assertEqual(normal["drift_check"]["status"], "ok")
        self.assertIsNone(normal["retraining"])
        result = self.run_batch("drift")
        self.assertEqual(result["model_name"], "JejuSolarDailyPredictor")
        self.assertEqual(result["retraining"]["status"], "gate_rejected")
        self.assertEqual(len(predict.recent_predictions), 14)
        self.assertEqual(self.path.read_bytes(), before)
        history = self.train.call_args.args[0]
        self.assertEqual(history[-1]["timestamp"], "2024-05-14T00:00:00")
        by_time = {r["timestamp"]: r for r in history}
        self.assertEqual(by_time["2024-05-04T00:00:00"]["generation_mwh"], 100)
        self.assertEqual(by_time["2024-05-05T00:00:00"]["generation_mwh"], 10)
        self.assertTrue(self.train.call_args.kwargs["metadata_extra"]["simulation"])
        self.assertEqual(self.client.get("/simulation/status").json(), result)
        self.assertEqual(self.client.get("/metrics/summary").json()["request_count"], 2)

    def test_repeat_batch_does_not_retrain_or_duplicate_observations(self):
        self.run_batch("drift")
        again = self.run_batch("drift")
        self.assertEqual(again["drift_check"]["new_count"], 0)
        self.assertEqual(self.train.call_count, 1)
        self.assertEqual(len((self.root / "replay.jsonl").read_text().splitlines()), 14)

    def test_new_curtailment_is_not_suppressed_by_legacy_batch_history(self):
        source_hash = hashlib.sha256(self.path.read_bytes()).hexdigest()
        legacy_hash = hashlib.sha256(
            (source_hash + ":daily-curtailment-v1").encode()
        ).hexdigest()
        predict._batch_contexts[(legacy_hash, "1")] = {
            "cutoff": "2024-05-14T00:00:00",
            "records": [],
            "check": {"rmse": 0.0, "status": "ok", "retraining": None},
        }
        result = self.run_batch("drift")
        self.assertEqual(result["drift_check"]["new_count"], 14)
        self.assertEqual(result["drift_check"]["status"], "performance_degraded")
        self.assertEqual(result["retraining"]["status"], "gate_rejected")

    def test_switching_scenarios_does_not_repeat_same_training(self):
        self.run_batch("drift")
        self.run_batch("normal")
        result = self.run_batch("drift")
        self.assertEqual(self.train.call_count, 1)
        self.assertEqual(result["drift_check"]["new_count"], 0)
        self.assertEqual(len(predict.recent_predictions), 14)

    def test_gate_pass_activates_operating_model_and_clears_old_errors(self):
        self.train.return_value = {
            "gate_passed": True,
            "promoted": False,
            "version": "2",
            "rmse": 5.0,
        }
        loaded = SimpleNamespace(registry_version="2")
        with patch.object(
            model_loader, "promote_candidate", return_value=loaded
        ) as promote:
            result = self.run_batch("drift")
        self.assertEqual(result["served_model_version"], "2")
        self.assertTrue(result["retraining"]["promoted"])
        self.assertIs(promote.call_args.args[1], self.model)
        self.assertEqual(predict.recent_predictions, [])

    def test_load_failure_does_not_claim_promotion(self):
        self.train.return_value = {
            "gate_passed": True,
            "promoted": False,
            "version": "2",
            "rmse": 5.0,
        }
        with patch.object(
            model_loader, "promote_candidate", side_effect=ValueError("bad bundle")
        ):
            result = self.run_batch("drift")
        self.assertEqual(result["retraining"]["status"], "activation_failed")
        self.assertFalse(result["retraining"]["promoted"])
        self.assertEqual(result["served_model_version"], "1")

    def test_missing_hour_rejected_before_any_monitoring_or_training(self):
        self.path.write_bytes(
            csv_bytes([r for r in self.rows if r["timestamp"] != "2024-05-06T00:00:00"])
        )
        response = self.client.post(
            "/simulation/run",
            json={"scenario": "drift", "start_timestamp": "2024-05-01"},
        )
        self.assertEqual(response.status_code, 422)
        self.assertEqual(predict.recent_predictions, [])
        self.train.assert_not_called()

    def test_selection_overlap_local_and_concurrent_requests_rejected(self):
        self.model.metadata["training_end"] = "2024-06-01T00:00:00"
        self.assertEqual(
            self.client.post(
                "/simulation/run",
                json={"scenario": "drift", "start_timestamp": "2024-05-01"},
            ).status_code,
            422,
        )
        with patch.dict(os.environ, MODEL_SOURCE="local"):
            self.assertEqual(
                self.client.post(
                    "/simulation/run",
                    json={"scenario": "normal", "start_timestamp": "2024-05-01"},
                ).status_code,
                422,
            )
        with predict._replay_lock:
            self.assertEqual(
                self.client.post(
                    "/simulation/run",
                    json={"scenario": "normal", "start_timestamp": "2024-05-01"},
                ).status_code,
                409,
            )

    def test_legacy_isolated_simulator_result_is_not_shown_as_operating_result(self):
        import json

        old = {
            "exists": True,
            "model_name": "JejuSolarSimulator",
            "simulation_model_version": "3",
        }
        (self.root / "simulation.jsonl").write_text(json.dumps(old))
        result = self.client.get("/simulation/status").json()
        self.assertFalse(result["exists"])
        self.assertEqual(json.loads((self.root / "simulation.jsonl").read_text()), old)

    def test_curtailment_preserves_missing_and_night_and_source(self):
        self.rows[0]["generation_mwh"] = None
        before = copy.deepcopy(self.rows)
        changed = inject_curtailment(self.rows)
        self.assertIsNone(changed[0]["generation_mwh"])
        self.assertEqual(changed[1]["generation_mwh"], 10)
        self.assertEqual(changed[2]["generation_mwh"], 100)
        self.assertEqual(
            [r["generation_mwh"] for r in changed[1:7]],
            [10, 100, 100, 10, 100, 100],
        )
        self.assertEqual(self.rows, before)
