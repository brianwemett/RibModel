import argparse
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rib_forecast import build_daily_dataset, create_features, forecast_days, FEATURES
from weather_bias import WEATHER, validate_weather, read_future_weather, weather_adjustment, fit_weather_bias


class FakeWeatherModel:
    def predict(self, x):
        return np.exp(x["precip_in"].to_numpy()) * 20


class WeatherTests(unittest.TestCase):
    def test_validation_rejects_invalid_units_values_and_dates(self):
        for column, value in [("precip_in", -1), ("snowfall_in", float("inf")), ("temp_max_f", "rainy")]:
            with self.assertRaises(ValueError):
                validate_weather(pd.DataFrame({column: [value]}))
        with self.assertRaisesRegex(ValueError, "below low"):
            validate_weather(pd.DataFrame({"temp_max_f": [30], "temp_min_f": [40]}))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"forecast.csv"
            pd.DataFrame({"date": ["2026-11-01", "2026-11-01"], "precip_in": [0, 1]}).to_csv(path,index=False)
            with self.assertRaisesRegex(ValueError, "unique dates"):
                read_future_weather(argparse.Namespace(weather_forecast=str(path)), pd.Timestamp("2026-11-01"), pd.Timestamp("2026-11-07"))

    def test_no_forecast_is_neutral_and_large_adjustments_are_capped(self):
        index = pd.DatetimeIndex(["2026-11-01"])
        row = pd.DataFrame({"lag_1": [20]}, index=index)
        reference = pd.Series({"temp_max_f": 50., "temp_min_f": 35., "precip_in": 0., "snowfall_in": 0.})
        ctx = {"enabled": True, "model": FakeWeatherModel(), "reference": reference, "cap": .25}
        self.assertEqual(weather_adjustment(ctx, row, None)[0], 1)
        condition = pd.Series({"temp_max_f": np.nan, "temp_min_f": np.nan, "precip_in": 2., "snowfall_in": np.nan})
        ratio, status = weather_adjustment(ctx, row, condition)
        self.assertEqual(ratio, 1.25)
        self.assertIn("Partial", status)
        self.assertEqual(weather_adjustment({"enabled": False}, row, condition)[0], 1)

    def test_manual_next_day_forecast_and_recursive_adjustment(self):
        dates = pd.date_range("2026-01-01", periods=40)
        daily = pd.DataFrame({"rib_quantity": 20.}, index=dates)
        class LagModel:
            def predict(self, x):
                return x["lag_1"].to_numpy()
        next_date = dates[-1]+pd.Timedelta(days=1)
        args = argparse.Namespace(weather_rain=2.)
        future = read_future_weather(args, next_date, next_date+pd.Timedelta(days=6))
        reference = pd.Series({"temp_max_f": 50., "temp_min_f": 35., "precip_in": 0., "snowfall_in": 0.})
        ctx = {"enabled": True, "model": FakeWeatherModel(), "reference": reference, "cap": .25}
        result = forecast_days(daily,"poisson_regression",LagModel(),weather_context=ctx,future_weather=future)
        self.assertEqual(result.predicted_quantity.iloc[0], 25)
        self.assertEqual(result.predicted_quantity.iloc[1], 25)
        self.assertEqual(result.weather_multiplier.iloc[1], 1)
        closed = forecast_days(daily,"poisson_regression",LagModel(),closed_dates=[str(next_date.date())],weather_context=ctx,future_weather=future)
        self.assertEqual(closed.predicted_quantity.iloc[0], 0)
        self.assertEqual(closed.weather_multiplier.iloc[0], 1)

    def test_holdout_weather_does_not_change_training_reference_or_past_lags(self):
        dates = pd.date_range("2025-01-01", periods=120)
        rng = np.random.default_rng(71)
        daily = pd.DataFrame({"rib_quantity": rng.poisson(20,120), "temp_max_f": rng.uniform(35,80,120),
                              "temp_min_f": 20., "precip_in": rng.uniform(0,1,120), "snowfall_in": 0.},index=dates)
        features = create_features(daily)
        train, test = daily.iloc[:90],daily.iloc[90:]
        base = np.full(30,20.)
        with tempfile.TemporaryDirectory() as folder:
            context=fit_weather_bias(daily,features,train,test,base,"auto",Path(folder))
            self.assertIn("mae_improvement_fraction",context["report"])
            self.assertTrue((Path(folder)/"weather_comparison.csv").exists())
            train_reference = context["report"]["training_reference_weather"]
            self.assertAlmostEqual(train_reference["temp_max_f"], daily.loc[train.index, "temp_max_f"].median())
            changed = daily.copy()
            changed.loc[test.index, "temp_max_f"] = 100
            changed.loc[test.index, "rib_quantity"] = 1000
            changed_features = create_features(changed)
            pd.testing.assert_frame_equal(features.loc[train.index], changed_features.loc[train.index])
            changed_context = fit_weather_bias(changed,changed_features,changed.loc[train.index],changed.loc[test.index],base,"auto",Path(folder))
            self.assertEqual(train_reference, changed_context["report"]["training_reference_weather"])
            neutral = pd.Series(context["report"]["operational_reference_weather"])
            self.assertTrue(np.isfinite(neutral).all())
            no_weather=fit_weather_bias(daily[["rib_quantity"]],features,train,test,base,"auto",Path(folder))
            self.assertFalse(no_weather["enabled"])
            off=fit_weather_bias(daily,features,train,test,base,"off",Path(folder))
            self.assertFalse(off["enabled"])


if __name__ == "__main__":
    unittest.main()
