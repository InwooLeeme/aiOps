import sys
import unittest
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "project"))
from serving_app.monitoring import drift_detector


class MonitoringTests(unittest.TestCase):
    def records(self, error=2.0):
        start = datetime(2024, 1, 1)
        return [
            {
                "timestamp": (start + timedelta(days=i)).isoformat(),
                "predicted": 10.0 + error,
                "actual": 10.0,
            }
            for i in range(168)
        ]

    def test_no_threshold_or_insufficient_observations_never_triggers_retraining(self):
        self.assertTrue(hasattr(drift_detector, "assess_drift"))
        self.assertEqual(
            drift_detector.assess_drift(self.records(), None)["status"],
            "threshold_unavailable",
        )
        self.assertFalse(drift_detector.assess_drift(self.records()[:10], 1.0)["ready"])

    def test_sustained_high_error_is_performance_degradation_not_proven_drift(self):
        self.assertTrue(hasattr(drift_detector, "assess_drift"))
        result = drift_detector.assess_drift(self.records(), 1.0)
        self.assertEqual(result["status"], "performance_degraded")
        self.assertAlmostEqual(result["rmse"], 2.0)
        self.assertEqual(
            drift_detector.assess_drift(self.records(), 3.0)["status"], "ok"
        )

    def test_duplicates_and_large_time_gaps_do_not_inflate_observation_count(self):
        self.assertTrue(hasattr(drift_detector, "assess_drift"))
        self.assertFalse(
            drift_detector.assess_drift(self.records()[:1] * 168, 1.0)["ready"]
        )
        records = self.records()
        records[-1]["timestamp"] = "2025-01-01T00:00:00"
        self.assertFalse(drift_detector.assess_drift(records, 1.0)["ready"])
