import argparse
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rib_forecast import create_features, build_daily_dataset, run


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

    def test_end_to_end(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            index = pd.date_range("2025-01-01", periods=100)
            quantities = np.random.default_rng(42).poisson(12 + 5 * (index.dayofweek >= 5))
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
