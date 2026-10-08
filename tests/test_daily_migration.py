"""일별 입력·일자 정렬·집계 품질에 대한 회귀 검증."""

import importlib.util
import sys
import unittest
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "project"))
from test_solar_data import hourly_rows


class DailyMigrationTests(unittest.TestCase):
    def daily(self):
        self.assertIsNotNone(
            importlib.util.find_spec("data.daily_features"), "일별 전처리가 필요합니다"
        )
        from data import daily_features

        return daily_features

    def test_aggregation_requires_all_hours_and_does_not_bridge_gap(self):
        daily = self.daily()
        raw = hourly_rows(24 * 17)
        rows = daily.aggregate_hourly(raw)
        self.assertEqual(rows[0]["generation_mwh"], sum(range(24)))
        self.assertEqual(rows[0]["capacity_mw"], 200)
        self.assertEqual(len(list(daily.sample_windows(rows))), 3)
        raw.pop(24 * 8)
        rows = daily.aggregate_hourly(raw)
        self.assertIsNone(rows[8]["generation_mwh"])
        self.assertEqual(list(daily.sample_windows(rows)), [])

    def test_saved_daily_file_and_hourly_aggregation_have_identical_model_inputs(self):
        daily = self.daily()
        root = Path(__file__).resolve().parents[1] / "project/data"
        saved = daily.load_rows(root / "aggregated/jeju_solar_daily_2019_2024.csv")
        source = daily.load_rows(root / "uploads/solar_jeju_2019_2024.csv")
        self.assertEqual(len(saved), 2192)
        self.assertEqual(saved, source)
        self.assertEqual(sum(daily.complete_point(r) for r in saved), 2098)

    def test_target_weather_never_enters_input(self):
        daily = self.daily()
        rows = daily.aggregate_hourly(hourly_rows(24 * 15))
        scaler = daily.SolarScaler().fit(rows[:14])
        x, y = daily.build_sequences(rows, scaler)
        rows[-1]["temperature"] = 999
        self.assertEqual(daily.build_sequences(rows, scaler), (x, y))
        self.assertEqual(len(x[0]), 14)
        self.assertEqual(len(x[0][0]), 8)

    def test_missing_hourly_capacity_only_invalidates_that_daily_feature(self):
        daily = self.daily()
        raw = hourly_rows(24 * 2)
        raw[3]["capacity_mw"] = None
        result = daily.aggregate_hourly(raw)
        self.assertIsNone(result[0]["capacity_mw"])
        self.assertEqual(result[0]["generation_mwh"], 276)
        self.assertTrue(daily.complete_point(result[1]))

    def test_old_hourly_bundle_cannot_be_served_as_daily(self):
        import json
        import tempfile

        from serving_app.model_loader import read_bundle

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model.keras").write_bytes(b"hourly")
            (root / "scaler.json").write_text("{}")
            (root / "metadata.json").write_text(
                json.dumps(
                    {
                        "feature_columns": self.daily().FEATURE_COLUMNS,
                        "seq_len": 14,
                        "unit": "MWh",
                        "granularity": "hourly",
                        "target": "next_hour_generation_mwh",
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "일별"):
                read_bundle(root)

    def test_incomplete_current_day_is_not_usable_as_observation(self):
        from serving_app import forecasts

        context = forecasts.forecast_context(
            "2024-01-14T00:00:00", now=datetime(2024, 1, 14, 23)
        )
        self.assertEqual(context["forecast_context"], "future_input")
        context = forecasts.forecast_context(
            "2024-01-14T00:00:00", now=datetime(2024, 1, 15, 0, 30)
        )
        self.assertEqual(context["target_timestamp"], "2024-01-15T00:00:00")
        self.assertEqual(context["forecast_context"], "current")


if __name__ == "__main__":
    unittest.main()
