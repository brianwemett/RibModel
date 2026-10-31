"""Optional, bounded weather scenarios. Historical observations are not archived forecasts."""
import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

WEATHER = ["temp_max_f", "temp_min_f", "precip_in", "snowfall_in"]


def validate_weather(frame):
    """Missing values mean unknown, never automatically dry or zero degrees."""
    weather = pd.DataFrame(index=frame.index)
    for column in WEATHER:
        raw = frame[column] if column in frame else pd.Series(np.nan, index=frame.index)
        values = pd.to_numeric(raw, errors="coerce")
        if (raw.notna() & values.isna()).any() or np.isinf(values).any():
            raise ValueError(f"Invalid weather values in {column}.")
        if column in ["precip_in", "snowfall_in"] and (values < 0).any():
            raise ValueError(f"{column} cannot be negative; use inches, not rain probability.")
        if column in ["temp_max_f", "temp_min_f"] and ((values < -100) | (values > 150)).any():
            raise ValueError(f"{column} must be between -100 and 150 Fahrenheit.")
        weather[column] = values.astype(float)
    both = weather.temp_max_f.notna() & weather.temp_min_f.notna()
    if (weather.loc[both, "temp_max_f"] < weather.loc[both, "temp_min_f"]).any():
        raise ValueError("Weather high temperature cannot be below low temperature.")
    return weather


def weather_ratios(model, feature_rows, weather, reference, cap):
    actual = feature_rows.copy()
    neutral = feature_rows.copy()
    for column in WEATHER:
        actual[column] = weather[column].fillna(reference[column]).to_numpy()
        neutral[column] = reference[column]
    observed = np.maximum(model.predict(actual), 1e-9)
    typical = np.maximum(model.predict(neutral), 1e-9)
    return np.clip(observed / typical, 1 - cap, 1 + cap)


def fit_weather_bias(daily, features, train, test, base_prediction, mode, output, cap=0.25):
    """Learn weather effects with calendar/history controls and a chronological gate."""
    report = {"requested_mode": mode, "enabled": False, "max_adjustment_fraction": cap,
              "evaluation": "Exploratory: historical observed weather, not archived next-day forecasts.",
              "reference": "Training median weather; refitted full-history medians for operational scenarios.",
              "minimum_mae_improvement": 0.05}
    context = {"enabled": False, "report": report}
    weather = validate_weather(daily)
    complete_train = weather.loc[train.index].notna().all(axis=1)
    complete_test = weather.loc[test.index].notna().all(axis=1)
    report["complete_training_weather_days"] = int(complete_train.sum())
    report["complete_test_weather_days"] = int(complete_test.sum())
    if mode == "off":
        report["reason"] = "Weather adjustment disabled by --weather-bias off."
    elif complete_train.sum() < 60 or complete_test.sum() < 14 or weather.loc[train.index].notna().sum().min() < 30:
        report["reason"] = "Insufficient historical weather: need 60 complete training days and 14 complete test days."
        if mode == "on":
            raise ValueError(report["reason"])
    else:
        categorical = ["day_of_week", "month"]
        numeric = [c for c in features.columns if c not in categorical] + WEATHER
        def build_model():
            prep = ColumnTransformer([
                ("calendar", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
                ("numeric", make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler()), numeric),
            ])
            return make_pipeline(prep, PoissonRegressor(alpha=3.0, max_iter=1500))
        reference = weather.loc[train.index].median()
        report["training_reference_weather"] = reference.to_dict()
        x_train = features.loc[train.index].join(weather.loc[train.index])
        candidate = build_model().fit(x_train, train.rib_quantity)
        ratios = weather_ratios(candidate, features.loc[test.index], weather.loc[test.index], reference, cap)
        # Missing historical weather cannot establish an adjustment: retain base demand.
        ratios[~complete_test.to_numpy()] = 1.0
        adjusted = np.asarray(base_prediction) * ratios
        comparisons = []
        for name, prediction in [("without_weather", base_prediction), ("weather_adjusted", adjusted)]:
            comparisons.append({"model": name, "MAE": float(mean_absolute_error(test.rib_quantity, prediction)),
                                "RMSE": float(np.sqrt(mean_squared_error(test.rib_quantity, prediction)))})
        metrics = pd.DataFrame(comparisons)
        metrics.to_csv(output / "weather_comparison.csv", index=False)
        audit = pd.DataFrame({"date": test.index, "actual": test.rib_quantity.to_numpy(),
                              "without_weather": np.asarray(base_prediction), "weather_adjusted": adjusted,
                              "weather_multiplier": ratios, "complete_observed_weather": complete_test.to_numpy()})
        audit.to_csv(output / "weather_test_predictions.csv", index=False)
        before, after = comparisons[0]["MAE"], comparisons[1]["MAE"]
        improvement = (before - after) / before if before > 0 else 0.0
        report["metrics"] = comparisons
        report["mae_improvement_fraction"] = improvement
        enabled = mode == "on" or (improvement >= 0.05 and complete_test.mean() >= 0.8)
        report["enabled"] = bool(enabled)
        report["reason"] = ("Explicit exploratory override (--weather-bias on)." if mode == "on" else
                            "Passed 5% MAE improvement gate with at least 80% test-weather coverage." if enabled else
                            "No automatic adjustment: did not pass the improvement/coverage gate.")
        # Refit only after the evaluation; holdout medians never enter the comparison.
        known = daily.loc[daily.rib_quantity.notna()]
        full_reference = weather.loc[known.index].median()
        fitted = build_model().fit(features.loc[known.index].join(weather.loc[known.index]), known.rib_quantity)
        report["operational_reference_weather"] = full_reference.to_dict()
        context.update(enabled=bool(enabled), model=fitted, reference=full_reference, cap=cap)
        names = fitted.named_steps["columntransformer"].get_feature_names_out()
        coefficients = pd.DataFrame({"feature": names, "coefficient": fitted.named_steps["poissonregressor"].coef_})
        coefficients.loc[coefficients.feature.str.replace("numeric__", "", regex=False).isin(WEATHER)].to_csv(output / "weather_coefficients.csv", index=False)
    (output / "weather_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return context


def read_future_weather(args, next_date, end_date):
    """Manual forecast inputs; this module does not fetch current weather."""
    future = pd.DataFrame(columns=WEATHER, index=pd.DatetimeIndex([], name="date"), dtype=float)
    if getattr(args, "weather_forecast", None):
        frame = pd.read_csv(args.weather_forecast)
        if "date" not in frame:
            raise ValueError("Weather forecast CSV requires a date column.")
        dates = pd.to_datetime(frame.date, errors="coerce", format="mixed")
        if dates.isna().any() or dates.duplicated().any() or not dates.eq(dates.dt.normalize()).all() or dates.dt.tz is not None:
            raise ValueError("Forecast weather dates must be unique dates without times or timezones.")
        future = validate_weather(frame)
        future.index = pd.DatetimeIndex(dates, name="date")
        if (future.index < next_date).any() or (future.index > end_date).any():
            raise ValueError("Weather forecast dates must be within the upcoming forecast period.")
    supplied = {column: getattr(args, arg, None) for column, arg in zip(WEATHER, ["weather_high", "weather_low", "weather_rain", "weather_snow"])}
    if any(value is not None for value in supplied.values()):
        if next_date not in future.index:
            future.loc[next_date] = np.nan
        for column, value in supplied.items():
            if value is not None:
                future.loc[next_date, column] = value
    return validate_weather(future).sort_index()


def weather_adjustment(context, feature_row, conditions):
    if conditions is None or conditions.isna().all():
        return 1.0, "No forecast supplied"
    if not context or not context["enabled"]:
        return 1.0, "Weather adjustment inactive"
    weather = pd.DataFrame([conditions], index=feature_row.index)
    ratio = weather_ratios(context["model"], feature_row, weather, context["reference"], context["cap"])[0]
    status = "Forecast supplied" if conditions.notna().all() else "Partial forecast; unknown fields use reference"
    return float(ratio), status
