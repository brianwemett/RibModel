import argparse
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rib_forecast import create_features, build_daily_dataset, forecast_days, save_weekly_prep, run


def options(path, output):
    return argparse.Namespace(input=str(path), output=str(output), inspect=False, date_column=None, item_column=None, quantity_column=None, stockout_column=None, rib_keyword=None, missing_days="unknown", test_fraction=0.2, prep_buffer=3, buffer_type="fixed", model="auto")


class ForecastTests(unittest.TestCase):
    def test_future_targets_do_not_change_past_features(self):
        index = pd.date_range("2025-01-01", periods=60)
        daily = pd.DataFrame({"rib_quantity": np.arange(60, dtype=float)}, index=index)
        before = create_features(daily)
        daily.loc[index[40]:, "rib_quantity"] = 10000
        after = create_features(daily)
        pd.testing.assert_frame_equal(before.loc[:index[40]], after.loc[:index[40]])
        self.assertEqual(before.loc[index[7], "lag_7"], 0)
        self.assertEqual(before.loc[index[7], "mean_7"], 3)

    def test_missing_day_is_unknown_and_observed_nonrib_day_is_zero(self):
        frame = pd.DataFrame({"date": ["2025-01-01", "2025-01-03", "2025-01-04"], "item": ["ribs", "salad", "ribs"], "quantity": [2, 1, 3]})
        args = options("unused", "unused")
        result = build_daily_dataset(frame, args)
        self.assertTrue(pd.isna(result.loc["2025-01-02", "rib_quantity"]))
        self.assertEqual(result.loc["2025-01-03", "rib_quantity"], 0)
        args.missing_days = "zero"
        self.assertEqual(build_daily_dataset(frame, args).loc["2025-01-02", "rib_quantity"], 0)

    def test_daily_partial_and_closed_days_never_become_lags(self):
        args = options("unused", "unused")
        frame = pd.DataFrame({"date": ["2026-10-28", "2026-10-29", "2026-10-30", "2026-10-31"],
                              "total_quarter_racks": [8, 0, 12, 3], "partial": [0, 0, 0, 1], "checks_day": [10, 0, 12, 2]})
        daily = build_daily_dataset(frame, args)
        self.assertTrue(pd.isna(daily.loc["2026-10-29", "rib_quantity"]))
        self.assertTrue(pd.isna(daily.loc["2026-10-31", "rib_quantity"]))
        self.assertEqual(daily.loc["2026-10-31", "reported_quantity"], 3)
        self.assertTrue(pd.isna(create_features(daily).loc["2026-10-30", "lag_1"]))
        self.assertEqual(daily.attrs["units"], "quarter racks")

    def test_daily_mapping_and_components_reject_inconsistent_counts(self):
        args = options("unused", "unused")
        frame = pd.DataFrame({"date": ["2026-10-30"], "total_quarter_racks": [8],
                              "full_rack_qty": [2], "half_rack_qty": [0], "kids_rib_qty": [0],
                              "ribs_from_combos_qty": [0], "ribs_from_plates_qty": [0], "ribs_from_other_qty": [0]})
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"mapping.csv"
            pd.DataFrame({"quarter_racks_per_qty": [4], "qty_sum": [2], "quarter_racks_total": [8]}).to_csv(path,index=False)
            args.item_mapping = str(path)
            self.assertEqual(build_daily_dataset(frame, args).attrs["mapping_total_quarter_racks"], 8)
            frame["total_quarter_racks"] = 9
            with self.assertRaisesRegex(ValueError, "components"):
                build_daily_dataset(frame, args)
            frame["total_quarter_racks"] = 8
            pd.DataFrame({"quarter_racks_per_qty": [4], "qty_sum": [1], "quarter_racks_total": [4]}).to_csv(path,index=False)
            with self.assertRaisesRegex(ValueError, "matching export periods"):
                build_daily_dataset(frame, args)
        with self.assertRaisesRegex(ValueError, "unique business dates"):
            build_daily_dataset(pd.concat([frame,frame]), args)

    def test_daily_end_to_end_preserves_partial_calendar_and_rack_units(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            dates = pd.date_range("2026-01-01", periods=70)
            frame = pd.DataFrame({"date": dates, "total_quarter_racks": 20, "checks_day": 10, "partial": 0})
            frame.loc[69, "partial"] = 1
            frame.loc[69, "total_quarter_racks"] = 1
            frame.to_csv(root/"sales.csv", index=False)
            run(options(root/"sales.csv", root/"outputs"))
            import json
            prep = json.loads((root/"outputs/prep_recommendation.json").read_text())
            self.assertEqual(prep["date"], str((dates[-1]+pd.Timedelta(days=1)).date()))
            self.assertEqual(prep["units"], "quarter racks")
            self.assertEqual(prep["missing_recent_sales_days"], 1)
            self.assertEqual(prep["recommended_whole_racks"], 6)
            heldout = pd.read_csv(root/"outputs/test_predictions.csv")
            self.assertNotIn(str(dates[-1].date()), heldout.date.tolist())

    def test_weekly_forecast_uses_predictions_in_future_lags(self):
        class LagModel:
            def predict(self, features):
                return features["lag_1"].to_numpy() + 4
        dates = pd.date_range("2026-01-01", periods=40)
        daily = pd.DataFrame({"rib_quantity": 20.0}, index=dates)
        result = forecast_days(daily, "poisson_regression", LagModel())
        self.assertEqual(result.predicted_quantity.tolist(), [24, 28, 32, 36, 40, 44, 48])
        self.assertEqual(result.date.iloc[0], "2026-02-10")
        self.assertEqual(result.date.iloc[-1], "2026-02-16")
        later = forecast_days(daily, "poisson_regression", LagModel(), "2026-02-12")
        self.assertEqual(later.predicted_quantity.iloc[0], 32)
        with self.assertRaisesRegex(ValueError, "follow the input period"):
            forecast_days(daily, "poisson_regression", LagModel(), "2026-01-01")

    def test_weekly_sheet_rounds_daily_and_skips_closed_day_buffer(self):
        dates = pd.date_range("2026-01-01", periods=40)
        daily = pd.DataFrame({"rib_quantity": 21.0}, index=dates)
        daily.attrs["units"] = "quarter racks"
        forecast = forecast_days(daily, "weekday_baseline", None, closed_dates=["2026-02-11"])
        self.assertEqual(forecast.predicted_quantity.iloc[1], 0)
        with tempfile.TemporaryDirectory() as folder:
            sheet = save_weekly_prep(daily, forecast, "weekday_baseline", 12, "fixed", Path(folder))
            self.assertEqual(sheet.recommended_whole_racks.sum(), 54)
            self.assertEqual(sheet.recommended_prep.iloc[1], 0)
            percent = save_weekly_prep(daily, forecast, "weekday_baseline", 10, "percent", Path(folder))
            self.assertEqual(percent.recommended_prep.iloc[0], 24)
            for name in ["weekly_prep_sheet.pdf", "weekly_prep_sheet.csv", "weekly_prep_sheet.png"]:
                self.assertGreater((Path(folder)/name).stat().st_size, 100)
        with self.assertRaisesRegex(ValueError, "Closed dates must fall"):
            forecast_days(daily, "weekday_baseline", None, closed_dates=["2026-01-01"])

    def test_end_to_end(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            index = pd.date_range("2025-01-01", periods=100)
            quantities = np.random.default_rng(42).poisson(12 + 5 * (index.day_of_week >= 5))
            pd.DataFrame({"date": index, "item": "baby back ribs", "quantity": quantities}).to_csv(root / "sales.csv", index=False)
            run(options(root / "sales.csv", root / "outputs"))
            metrics = pd.read_csv(root / "outputs/model_comparison.csv")
            self.assertEqual(len(metrics), 3)
            self.assertTrue(np.isfinite(metrics[["MAE", "RMSE"]]).all().all())
            predictions = pd.read_csv(root / "outputs/test_predictions.csv")
            self.assertEqual(len(predictions), 20)
            self.assertTrue((predictions.iloc[:, 1:] >= 0).all().all())
            import json
            prep = json.loads((root / "outputs/prep_recommendation.json").read_text())
            self.assertGreaterEqual(prep["recommended_prep"], prep["predicted_quantity"])
            for name in ["rib_sales_over_time.png", "actual_vs_predicted.png", "sales_by_weekday.png", "model_selection.txt"]:
                self.assertTrue((root / "outputs" / name).is_file())


if __name__ == "__main__":
    unittest.main()
