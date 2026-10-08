import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "project"))

from serving_app.monitoring.drift_detector import assess_drift
from serving_app.routers.dashboard import version_info
from serving_app.train_and_register import _log_and_register, evaluate_predictions


class WapeTests(unittest.TestCase):
    def test_mlflow_records_percentage_and_skips_undefined_month(self):
        scores = evaluate_predictions([0, 100], [10, 110], ["2024-01-01", "2024-02-01"])
        metadata = {
            "epochs": 1,
            "seed": 42,
            "seq_len": 14,
            "mode": "scratch",
            "metrics": {"validation": {"model": scores}},
            "error_threshold_mwh": 15,
        }
        with (
            patch("serving_app.tracking.configure_experiment"),
            patch("mlflow.start_run", return_value=MagicMock()),
            patch("mlflow.log_params"),
            patch("mlflow.set_tags") as tags,
            patch("mlflow.log_metric") as log,
            patch("mlflow.tensorflow.log_model"),
            patch("mlflow.log_artifacts"),
            patch("serving_app.train_and_register.save_bundle"),
        ):
            _log_and_register(None, None, metadata, False, None)
        logged = {call.args[0]: call.args[1] for call in log.call_args_list}
        self.assertEqual(logged["validation_model_wape_pct"], 20)
        self.assertEqual(logged["validation_model_2024-02_wape_pct"], 10)
        self.assertNotIn("validation_model_2024-01_wape_pct", logged)
        self.assertTrue(all(value is not None for value in logged.values()))
        self.assertEqual(tags.call_args.args[0]["wape_evaluated"], "true")

    def test_total_is_ratio_of_sums_and_monthly_zero_is_undefined(self):
        scores = evaluate_predictions(
            [0, 100, 900],
            [10, 120, 810],
            ["2024-01-01", "2024-02-01", "2024-02-02"],
        )
        self.assertIn("wape_pct", scores)
        self.assertAlmostEqual(scores["wape_pct"], 12.0)
        self.assertIsNone(scores["monthly"]["2024-01"]["wape_pct"])
        self.assertAlmostEqual(scores["monthly"]["2024-02"]["wape_pct"], 11.0)

    def test_perfect_zero_and_over_100_percent_are_not_conflated(self):
        for actual, prediction, expected in [(10, 10, 0), (0, 5, None), (1, 5, 400)]:
            scores = evaluate_predictions([actual], [prediction], ["2024-01-01"])
            self.assertIn("wape_pct", scores)
            self.assertEqual(scores["wape_pct"], expected)

    def test_monitoring_uses_same_filtered_window_as_rmse(self):
        records = [
            {"timestamp": "2023-01-01", "actual": 10000, "predicted": 0},
            {"timestamp": "2024-01-01", "actual": 100, "predicted": 999},
            {"timestamp": "2024-01-01", "actual": 100, "predicted": 120},
            {"timestamp": "2024-01-02", "actual": 0, "predicted": 10},
            {"timestamp": "2024-01-03", "actual": None, "predicted": 900},
        ]
        result = assess_drift(records, 100)
        self.assertIn("wape_pct", result)
        self.assertEqual(result["count"], 2)
        self.assertEqual(result["wape_pct"], 30)
        self.assertEqual(result["status"], "insufficient_data")
        self.assertIsNone(assess_drift([], 100)["wape_pct"])
        self.assertIsNone(assess_drift(records[3:4], 100)["wape_pct"])

    def test_registry_exposes_wape_and_leaves_legacy_unknown(self):
        version = SimpleNamespace(
            run_id="run",
            version="1",
            creation_timestamp=1000,
            current_stage="Production",
        )
        client = Mock()
        for metrics, expected in [({}, None), ({"validation_model_wape_pct": 12}, 12)]:
            client.get_run.return_value.data = SimpleNamespace(
                metrics=metrics, params={"validation_start": "2024-01-01"}, tags={}
            )
            info = version_info(client, version)
            self.assertIn("wape_pct", info)
            self.assertEqual(info["wape_pct"], expected)
            self.assertEqual(info["wape_recorded"], expected is not None)
        client.get_run.return_value.data = SimpleNamespace(
            metrics={},
            params={"validation_start": "2024-01-01"},
            tags={"wape_evaluated": "true"},
        )
        info = version_info(client, version)
        self.assertTrue(info["wape_recorded"])
        self.assertIsNone(info["wape_pct"])
