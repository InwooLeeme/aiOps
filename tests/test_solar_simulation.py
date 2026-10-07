"""시뮬레이션의 데이터 일관성, 자동 재학습 및 운영 모델 격리."""

import copy
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "project"))
from data.features import validate_sequence
from fastapi.testclient import TestClient
from serving_app.main import app
from serving_app.monitoring import simulation
from serving_app.routers import predict
from serving_app.schemas import RetrainRequest
from test_day1 import csv_bytes
from test_solar_data import hourly_rows


class SimulationTests(unittest.TestCase):
    def setUp(self):
        self.rows = hourly_rows(120 * 24, "2024-02-01T00:00:00")
        for row in self.rows:
            row["generation_mwh"] = 100.0
        self.model = SimpleNamespace(
            registry_version="1",
            version="production",
            metadata={
                "validation_end": "2023-11-30T23:00:00",
                "drift_threshold_mwh": 10.0,
            },
            predict_one=lambda window: window[-1]["generation_mwh"],
        )

    def test_normal_does_not_retrain_or_mutate_observations(self):
        original = copy.deepcopy(self.rows)
        with patch.object(simulation, "fine_tune") as train:
            result = simulation.run_simulation(self.rows, self.model, "normal")
        self.assertEqual(result["drift_check"]["status"], "ok")
        self.assertIsNone(result["retraining"])
        train.assert_not_called()
        self.assertEqual(self.rows, original)

    def test_drift_retrains_same_transformed_history_without_future_or_live_reload(
        self,
    ):
        original = copy.deepcopy(self.rows)
        loaded = SimpleNamespace(registry_version="2", predict_one=lambda window: 5.0)
        with (
            patch.object(
                simulation,
                "fine_tune",
                return_value={"promoted": True, "version": "2", "rmse": 5.0},
            ) as train,
            patch.object(simulation, "_load_from_mlflow", return_value=loaded) as load,
            patch("serving_app.model_loader.reload_model") as live_reload,
        ):
            result = simulation.run_simulation(self.rows, self.model, "drift")
        self.assertEqual(result["drift_check"]["status"], "performance_degraded")
        history = train.call_args.args[0]
        self.assertEqual(history[-1]["timestamp"], result["cutoff_timestamp"])
        self.assertEqual(
            train.call_args.kwargs["model_name"], simulation.SIMULATION_MODEL_NAME
        )
        self.assertIs(train.call_args.kwargs["incumbent"], self.model)
        self.assertTrue(train.call_args.kwargs["metadata_extra"]["simulation"])
        by_time = {r["timestamp"]: r for r in history}
        self.assertTrue(
            all(
                r["actual"] == by_time[r["timestamp"]]["generation_mwh"]
                for r in result["records"]
            )
        )
        self.assertEqual(by_time["2024-05-24T12:00:00"]["generation_mwh"], 10.0)
        self.assertEqual(by_time["2024-05-24T13:00:00"]["generation_mwh"], 100.0)
        self.assertEqual(self.rows, original)
        load.assert_called_once_with(simulation.SIMULATION_MODEL_NAME)
        live_reload.assert_not_called()
        self.assertEqual(result["simulation_model_version"], "2")

    def test_failed_gate_never_loads_or_claims_new_simulation_model(self):
        with (
            patch.object(
                simulation,
                "fine_tune",
                return_value={"promoted": False, "version": None},
            ) as train,
            patch.object(simulation, "_load_from_mlflow") as load,
        ):
            result = simulation.run_simulation(self.rows, self.model, "drift")
        train.assert_called_once()
        load.assert_not_called()
        self.assertIsNone(result["simulation_model_version"])

    def test_reload_smoke_uses_valid_input_not_target_weather(self):
        cutoff = "2024-05-30T16:00:00"
        next(r for r in self.rows if r["timestamp"] == cutoff)["temperature"] = None

        def validate(window):
            validate_sequence(window)
            return 5.0

        loaded = SimpleNamespace(registry_version="2", predict_one=validate)
        with (
            patch.object(
                simulation, "fine_tune", return_value={"promoted": True, "version": "2"}
            ),
            patch.object(simulation, "_load_from_mlflow", return_value=loaded),
        ):
            result = simulation.run_simulation(self.rows, self.model, "drift")
        self.assertEqual(result["simulation_model_version"], "2")

    def test_reload_failure_preserves_promoted_result_for_ui_and_history(self):
        with (
            patch.object(
                simulation, "fine_tune", return_value={"promoted": True, "version": "2"}
            ),
            patch.object(
                simulation, "_load_from_mlflow", side_effect=ValueError("bad bundle")
            ),
        ):
            result = simulation.run_simulation(self.rows, self.model, "drift")
        self.assertTrue(result["retraining"]["promoted"])
        self.assertFalse(result["reload_verified"])
        self.assertEqual(result["reload_error"], "bad bundle")

    def test_missing_evaluation_hour_rejects_before_training(self):
        self.rows = [r for r in self.rows if r["timestamp"] != "2024-05-26T12:00:00"]
        with patch.object(simulation, "fine_tune") as train:
            with self.assertRaisesRegex(ValueError, "168"):
                simulation.run_simulation(self.rows, self.model, "drift")
        train.assert_not_called()

    def test_rejects_overlap_with_model_selection_period(self):
        self.model.metadata["validation_end"] = "2024-06-01T00:00:00"
        with self.assertRaisesRegex(ValueError, "학습|검증"):
            simulation.run_simulation(self.rows, self.model, "drift")

    def test_missing_values_and_night_generation_are_preserved(self):
        self.rows[0]["generation_mwh"] = None
        changed = simulation.inject_curtailment(self.rows)
        self.assertIsNone(changed[0]["generation_mwh"])
        self.assertEqual(changed[2]["generation_mwh"], 100.0)
        self.assertEqual(changed[12]["generation_mwh"], 10.0)

    def test_browser_minute_precision_cutoff_is_normalized(self):
        self.assertEqual(
            RetrainRequest(cutoff_timestamp="2024-01-10T00:00").cutoff_timestamp,
            "2024-01-10T00:00:00",
        )

    def test_simulation_api_persists_result_without_authorizing_real_retraining(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv = root / "solar.csv"
            csv.write_bytes(csv_bytes(self.rows))
            before = dict(predict._last_replay)
            with (
                patch.dict(
                    os.environ, {"MODEL_SOURCE": "mlflow", "LOADING_MODE": "lazy"}
                ),
                patch.object(predict, "current_model", return_value=self.model),
                patch.object(predict, "latest_upload", return_value=str(csv)),
                patch.object(predict, "SIMULATION_LOG", root / "simulation.jsonl"),
                TestClient(app) as client,
            ):
                response = client.post("/simulation/run", json={"scenario": "normal"})
                self.assertEqual(response.status_code, 200, response.text)
                result = response.json()
                self.assertTrue(result["simulation"])
                self.assertEqual(client.get("/simulation/status").json(), result)
                self.assertEqual(
                    json.loads((root / "simulation.jsonl").read_text()), result
                )
                self.assertEqual(predict._last_replay, before)
                self.assertEqual(
                    client.post(
                        "/simulation/run", json={"scenario": "unknown"}
                    ).status_code,
                    422,
                )
                with patch.dict(os.environ, {"MODEL_SOURCE": "local"}):
                    self.assertEqual(
                        client.post(
                            "/simulation/run", json={"scenario": "normal"}
                        ).status_code,
                        422,
                    )
                with predict._replay_lock:
                    self.assertEqual(
                        client.post(
                            "/simulation/run", json={"scenario": "normal"}
                        ).status_code,
                        409,
                    )


if __name__ == "__main__":
    unittest.main()
