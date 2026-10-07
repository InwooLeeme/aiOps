import csv
import io
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "project"))
from data import features


def hourly_rows(count=80, start="2022-01-01T00:00:00"):
    first = datetime.fromisoformat(start)
    return [
        {
            "timestamp": (first + timedelta(hours=i)).isoformat(),
            "region": "제주",
            "generation_mwh": float(i % 24),
            "capacity_mw": 200.0,
            "temperature": -2.0,
            "humidity": 60.0,
            "wind_speed": 2.5,
            "cloud_cover": 3.0,
        }
        for i in range(count)
    ]


class SolarDataTests(unittest.TestCase):
    def test_cp949_actuals_are_decoded_without_turning_missing_target_into_zero(self):
        self.assertTrue(hasattr(features, "decode_csv"), "태양광 CSV 파서 필요")
        stream = io.StringIO()
        writer = csv.writer(stream)
        writer.writerow(
            [
                "시도명",
                "일자 및 시각",
                "태양광 발전량 합계(MWh)",
                "태양광설비용량(MW)",
                "기온",
                "습도",
                "풍속",
                "전운량(10분위)",
            ]
        )
        writer.writerow(["제주", "2022-01-01 00:00", "", 200, -2, 60, 2.5, 3])
        writer.writerow(["제주", "2022-01-01 01:00", 0, 200, -2, 60, 2.5, 3])
        rows = features.decode_csv(stream.getvalue().encode("cp949"))
        self.assertIsNone(rows[0]["generation_mwh"])
        self.assertEqual(rows[1]["generation_mwh"], 0)
        self.assertEqual(rows[0]["temperature"], -2)

    def test_gap_or_missing_actual_cannot_become_adjacent_training_steps(self):
        self.assertTrue(hasattr(features, "sample_windows"))
        rows = hourly_rows(80)
        self.assertEqual(len(list(features.sample_windows(rows))), 8)
        del rows[40]
        self.assertEqual(len(list(features.sample_windows(rows))), 0)
        rows = hourly_rows(80)
        rows[75]["generation_mwh"] = None
        targets = [r["timestamp"] for _, r in features.sample_windows(rows)]
        self.assertEqual(
            targets,
            ["2022-01-04T00:00:00", "2022-01-04T01:00:00", "2022-01-04T02:00:00"],
        )

    def test_labels_are_next_hour_not_last_input_and_target_weather_not_used(self):
        self.assertTrue(hasattr(features, "SolarScaler"))
        rows = hourly_rows(73)
        rows[-1]["generation_mwh"] = 99.0
        rows[-1]["temperature"] = None
        scaler = features.SolarScaler().fit(rows[:72])
        x, y = features.build_sequences(rows, scaler)
        self.assertEqual(y, [99.0])
        self.assertEqual(len(x[0]), 72)
        rows[-1]["temperature"] = 1000
        self.assertEqual(features.build_sequences(rows, scaler)[0], x)

    def test_scaler_round_trip_and_future_extremes_do_not_change_training_scale(self):
        self.assertTrue(hasattr(features, "SolarScaler"))
        rows = hourly_rows(80)
        scaler = features.SolarScaler().fit(rows[:72])
        self.assertAlmostEqual(scaler.inverse_target(scaler.scale_target(12.5)), 12.5)
        self.assertGreater(scaler.scale_target(1000), 1)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "scaler.json"
            scaler.save(p)
            restored = features.SolarScaler.load(p)
            self.assertEqual(
                restored.transform_point(rows[0]), scaler.transform_point(rows[0])
            )

    def test_duplicate_timestamps_rejected_and_gap_count_reported(self):
        self.assertTrue(hasattr(features, "validate_rows"))
        rows = hourly_rows(80)
        with self.assertRaises(ValueError):
            features.validate_rows(rows + [rows[0]])
        del rows[10:13]
        report = features.dataset_summary(rows)
        self.assertEqual(report["missing_hours"], 3)


if __name__ == "__main__":
    unittest.main()
