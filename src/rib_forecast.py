"""Small-data, one-day-ahead rib sales forecasting. Run with --help."""
import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from weather_bias import WEATHER, validate_weather, fit_weather_bias, read_future_weather, weather_adjustment
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CALENDAR = ["day_of_week", "month"]
NUMERIC = ["day_of_month", "week_of_year", "is_weekend", "lag_1", "lag_7", "mean_7", "mean_28", "same_weekday_4"]
FEATURES = CALENDAR + NUMERIC


def load_data(path):
    frame = pd.read_csv(path)
    print(f"Shape: {frame.shape}\nColumns: {list(frame.columns)}")
    print("Null counts:\n", frame.isna().sum().to_string())
    print("Example rows:\n", frame.head().to_string(index=False))
    return frame


def resolve_column(frame, explicit, aliases):
    if explicit:
        if explicit not in frame.columns:
            raise ValueError(f"Column {explicit!r} not found. Available: {list(frame.columns)}")
        return explicit
    names = {str(c).strip().lower().replace(" ", "_"): c for c in frame.columns}
    for alias in aliases:
        if alias in names:
            return names[alias]
    raise ValueError(f"Cannot identify {aliases[0]} column; set its CLI mapping. Available: {list(frame.columns)}")


def build_daily_dataset(frame, args):
    mode = getattr(args, "input_format", "auto")
    if mode == "daily" or (mode == "auto" and "total_quarter_racks" in frame):
        return load_daily_sales(frame, args)
    date_col = resolve_column(frame, args.date_column, ["date", "transaction_date", "datetime", "timestamp"])
    item_col = resolve_column(frame, args.item_column, ["item", "item_name", "product", "product_name"])
    qty_col = resolve_column(frame, args.quantity_column, ["quantity", "quantity_sold", "qty"])
    dates = pd.to_datetime(frame[date_col], errors="coerce", format="mixed")
    if dates.isna().any():
        raise ValueError("Missing/invalid dates found; clean them before forecasting.")
    try:
        dates = dates.dt.normalize()
    except AttributeError as exc:
        raise ValueError("Dates must use one consistent restaurant-local timezone.") from exc
    print(f"Date range: {dates.min()} to {dates.max()}")
    print("Most common items:\n", frame[item_col].value_counts().head(20).to_string())
    items = frame[item_col].astype("string").str.lower()
    keywords = args.rib_keyword or ["ribs", "half rack", "full rack", "baby back", "beef rib"]
    mask = pd.Series(False, index=frame.index)
    for word in keywords:
        mask |= items.str.contains(word.lower(), regex=False, na=False)
    if not mask.any():
        raise ValueError("No rib items matched; supply --rib-keyword (repeat for multiple keywords).")
    quantities = pd.to_numeric(frame.loc[mask, qty_col], errors="coerce")
    if quantities.isna().any() or not np.isfinite(quantities).all() or (quantities < 0).any():
        raise ValueError("Rib quantities must be finite and nonnegative; resolve returns/missing values first.")
    daily = quantities.groupby(dates.loc[mask]).sum().rename("rib_quantity").to_frame()
    # A date with other restaurant transactions but no ribs is an observed zero.
    # Entirely absent dates are unknown unless the user explicitly opts into zero filling.
    calendar = pd.date_range(dates.min(), dates.max(), freq="D")
    daily = daily.reindex(calendar)
    observed = pd.DatetimeIndex(dates.unique())
    daily.loc[daily.index.isin(observed), "rib_quantity"] = daily.loc[daily.index.isin(observed), "rib_quantity"].fillna(0)
    missing = int(daily.rib_quantity.isna().sum())
    if args.missing_days == "zero":
        daily["rib_quantity"] = daily.rib_quantity.fillna(0)
    if args.stockout_column:
        if args.stockout_column not in frame:
            raise ValueError("Stockout column not found.")
        values = frame[args.stockout_column].astype("string").str.strip().str.lower()
        if (~values.isin(["true", "false", "1", "0", "yes", "no"])).any():
            raise ValueError("Stockout flags must be true/false, yes/no, or 1/0 without missing values.")
        flags = values.isin(["true", "1", "yes"]).groupby(dates).any()
        daily["stockout"] = flags.reindex(calendar, fill_value=False)
    daily.index.name = "date"
    print(f"Entirely absent dates: {missing}; policy: {args.missing_days}")
    return daily


def load_daily_sales(frame, args):
    """Use already-normalized business dates and quarter racks without recounting items."""
    date_col = resolve_column(frame, args.date_column, ["date", "business_date"])
    qty_col = resolve_column(frame, args.quantity_column, ["total_quarter_racks", "rib_quantity"])
    dates = pd.to_datetime(frame[date_col], errors="coerce", format="mixed").dt.normalize()
    if dates.isna().any() or dates.duplicated().any():
        raise ValueError("Daily input requires valid, unique business dates.")
    qty = pd.to_numeric(frame[qty_col], errors="coerce")
    if qty.isna().any() or not np.isfinite(qty).all() or (qty < 0).any():
        raise ValueError("Daily quantities must be finite and nonnegative.")
    if qty_col == "total_quarter_racks":
        parts = {"full_rack_qty": 4, "half_rack_qty": 2, "kids_rib_qty": 1,
                 "ribs_from_combos_qty": 1, "ribs_from_plates_qty": 1, "ribs_from_other_qty": 1}
        if all(c in frame for c in parts):
            components = frame[list(parts)].apply(pd.to_numeric, errors="coerce")
            if components.isna().any().any() or (components < 0).any().any():
                raise ValueError("Rack components must be valid nonnegative quantities.")
            expected = sum(components[c] * factor for c, factor in parts.items())
            if not np.allclose(expected, qty):
                raise ValueError("Quarter-rack target does not match its daily components.")
        if "total_racks_equiv" in frame and not np.allclose(qty / 4, frame.total_racks_equiv):
            raise ValueError("Quarter-rack and rack-equivalent totals disagree.")
    daily = pd.DataFrame({"rib_quantity": qty.to_numpy()}, index=pd.DatetimeIndex(dates))
    daily["reported_quantity"] = daily.rib_quantity
    daily["exclusion_reason"] = ""
    if "partial" in frame:
        partial = pd.to_numeric(frame.partial, errors="coerce")
        if partial.isna().any() or not partial.isin([0, 1]).all():
            raise ValueError("partial must contain only 0 or 1.")
        daily.loc[partial.to_numpy() == 1, "exclusion_reason"] = "partial_day"
    if "checks_day" in frame and getattr(args, "closed_days", "exclude") == "exclude":
        checks = pd.to_numeric(frame.checks_day, errors="coerce")
        if checks.isna().any() or not np.isfinite(checks).all() or (checks < 0).any():
            raise ValueError("checks_day must contain finite nonnegative counts.")
        closed = (checks.to_numpy() == 0) & daily.exclusion_reason.eq("")
        if daily.loc[closed, "rib_quantity"].gt(0).any():
            raise ValueError("Positive rib sales on zero-check days; resolve closure flags.")
        daily.loc[closed, "exclusion_reason"] = "zero_checks_probable_closure"
    if args.stockout_column:
        col = resolve_column(frame, args.stockout_column, [])
        flags = frame[col].astype("string").str.strip().str.lower()
        if not flags.isin(["true", "false", "1", "0", "yes", "no"]).all():
            raise ValueError("Invalid stockout flags.")
        daily["stockout"] = flags.isin(["true", "1", "yes"]).to_numpy()
    for col in ["checks_day", "holiday_name", "local_event"]:
        if col in frame:
            daily[col] = frame[col].to_numpy()
    weather = validate_weather(frame)
    for column in WEATHER:
        daily[column] = weather[column].to_numpy()
    calendar = pd.date_range(dates.min(), dates.max(), freq="D")
    daily = daily.sort_index().reindex(calendar)
    absent = daily.reported_quantity.isna()
    daily.loc[absent, "exclusion_reason"] = "missing_date"
    if args.missing_days == "zero":
        daily.loc[absent, "rib_quantity"] = 0
        daily.loc[absent, "exclusion_reason"] = ""
    # Keep excluded dates on the calendar, but never use incomplete sales as lags/targets.
    daily.loc[daily.exclusion_reason.ne(""), "rib_quantity"] = np.nan
    daily.index.name = "date"
    daily.attrs["units"] = "quarter racks" if qty_col == "total_quarter_racks" else "input quantity units"
    daily.attrs["target_column"] = qty_col
    mapping_path = getattr(args, "item_mapping", None)
    if mapping_path:
        if qty_col != "total_quarter_racks":
            raise ValueError("Item-mapping reconciliation requires total_quarter_racks as target.")
        mapping = pd.read_csv(mapping_path)
        required = ["quarter_racks_per_qty", "qty_sum", "quarter_racks_total"]
        if not all(c in mapping for c in required):
            raise ValueError(f"Item mapping must contain {required}.")
        amounts = mapping[required].apply(pd.to_numeric, errors="coerce")
        if amounts.isna().any().any() or not np.isfinite(amounts).all().all() or (amounts < 0).any().any():
            raise ValueError("Item mapping contains invalid quantities.")
        if not np.allclose(amounts.quarter_racks_per_qty * amounts.qty_sum, amounts.quarter_racks_total):
            raise ValueError("Item mapping conversion arithmetic does not reconcile.")
        if not np.isclose(amounts.quarter_racks_total.sum(), qty.sum()):
            raise ValueError("Item mapping totals differ from daily sales; use matching export periods.")
        daily.attrs["mapping_total_quarter_racks"] = float(qty.sum())
    print(f"Daily input: {dates.min().date()} to {dates.max().date()}; units: {daily.attrs['units']}")
    print("Excluded dates:\n", daily.loc[daily.exclusion_reason.ne(""), ["reported_quantity", "exclusion_reason"]].to_string())
    return daily


def create_features(daily):
    features = pd.DataFrame(index=daily.index)
    features["day_of_week"] = daily.index.day_of_week
    features["month"] = daily.index.month
    features["day_of_month"] = daily.index.day
    features["week_of_year"] = daily.index.isocalendar().week.astype(int)
    features["is_weekend"] = (daily.index.day_of_week >= 5).astype(int)
    y = daily.rib_quantity
    features["lag_1"] = y.shift(1)
    features["lag_7"] = y.shift(7)
    features["mean_7"] = y.shift(1).rolling(7, min_periods=1).mean()
    features["mean_28"] = y.shift(1).rolling(28, min_periods=1).mean()
    features["same_weekday_4"] = pd.concat([y.shift(k) for k in (7, 14, 21, 28)], axis=1).mean(axis=1)
    return features


def build_models():
    numeric = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler())
    preprocess = ColumnTransformer([
        ("calendar", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CALENDAR),
        ("history", numeric, NUMERIC),
    ])
    # Poisson uses a log link for positive expected counts. The idealized model
    # assumes conditional variance equals its mean; overdispersion is common in sales.
    # Regularization stabilizes correlated lags; coefficients remain interpretable.
    poisson = make_pipeline(preprocess, PoissonRegressor(alpha=1.0, max_iter=1000))
    tree = HistGradientBoostingRegressor(max_iter=100, max_leaf_nodes=7, min_samples_leaf=10, l2_regularization=1.0, early_stopping=False, random_state=42)
    return {"poisson_regression": poisson, "gradient_boosting": tree}


def baseline_predict(train, index):
    averages = train.groupby(train.index.day_of_week).rib_quantity.mean()
    return np.array([averages.get(day, train.rib_quantity.mean()) for day in index.day_of_week])


def evaluate_model(actual, prediction):
    return {"MAE": float(mean_absolute_error(actual, prediction)), "RMSE": float(np.sqrt(mean_squared_error(actual, prediction)))}


def save_plot(path):
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def plot_results(daily, predictions, output):
    daily.rib_quantity.plot(figsize=(10, 4), title="Daily rib sales", ylabel="Quantity")
    save_plot(output / "rib_sales_over_time.png")
    for groups, name, title in [(daily.index.day_of_week, "weekday", "Average sales by weekday (0=Monday)"), (daily.index.month, "month", "Average sales by month")]:
        daily.rib_quantity.groupby(groups).mean().plot.bar(title=title, ylabel="Quantity")
        save_plot(output / f"sales_by_{name}.png")
    daily.rib_quantity.plot.hist(bins=20, title="Daily rib sales distribution", xlabel="Quantity")
    save_plot(output / "daily_sales_histogram.png")
    predictions.set_index("date").drop(columns=["stockout"], errors="ignore").plot(figsize=(12, 5), title="One-day-ahead held-out predictions", ylabel="Quantity")
    save_plot(output / "actual_vs_predicted.png")


def prep_quantity(demand, buffer, buffer_type):
    """Round up in source units; never feed the safety buffer back into forecasts."""
    amount = demand + buffer if buffer_type == "fixed" else demand * (1 + buffer / 100)
    return math.ceil(amount)


def forecast_days(daily, selected, model, start_date=None, closed_dates=(), weather_context=None, future_weather=None):
    """Seven days at one forecast origin, recursively using unbuffered predictions."""
    next_date = daily.index.max() + pd.Timedelta(days=1)
    start = pd.Timestamp(start_date) if start_date else next_date
    if pd.isna(start) or start != start.normalize() or start.tzinfo is not None:
        raise ValueError("Week start must be a calendar date without time or timezone.")
    if start < next_date or start > next_date + pd.Timedelta(days=90):
        raise ValueError("Week start must follow the input period and be within 90 days of it.")
    closed = set()
    for value in closed_dates:
        date = pd.Timestamp(value)
        if pd.isna(date) or date != date.normalize() or date.tzinfo is not None:
            raise ValueError("Closed dates must be calendar dates without time or timezone.")
        if date < next_date or date > start + pd.Timedelta(days=6):
            raise ValueError("Closed dates must fall between the next forecast date and the end of the prep week.")
        closed.add(date)
    history = daily[["rib_quantity"]].copy()
    known = daily.loc[daily.rib_quantity.notna()]
    rows = []
    # Bridge days before a requested later week also need forecasts, never future actuals.
    for date in pd.date_range(next_date, start + pd.Timedelta(days=6), freq="D"):
        history = history.reindex(history.index.append(pd.DatetimeIndex([date])))
        if date in closed:
            demand = 0.0
            # Historical probable closures are unknown in lag features, so match that policy.
        elif selected == "weekday_baseline":
            demand = float(baseline_predict(known, pd.DatetimeIndex([date]))[0])
            history.loc[date, "rib_quantity"] = demand
        else:
            demand = max(0.0, float(model.predict(create_features(history).loc[[date], FEATURES])[0]))
            history.loc[date, "rib_quantity"] = demand
        base_demand = demand
        conditions = future_weather.loc[date] if future_weather is not None and date in future_weather.index else None
        multiplier, weather_status = weather_adjustment(weather_context, create_features(history).loc[[date], FEATURES], conditions)
        if date in closed:
            multiplier, weather_status = 1.0, "Closed"
        else:
            demand *= multiplier
            history.loc[date, "rib_quantity"] = demand
        if date >= start:
            rows.append({"date": str(date.date()), "day": date.day_name(),
                         "status": "Closed" if date in closed else "Open (assumed)",
                         "predicted_quantity": demand, "without_weather_quantity": base_demand,
                         "weather_multiplier": multiplier, "weather_status": weather_status,
                         **{column: float(conditions[column]) if conditions is not None else np.nan for column in WEATHER}})
    return pd.DataFrame(rows)


def save_weekly_prep(daily, forecast, selected, buffer, buffer_type, output):
    """Write an editable CSV and a single-page, US-letter landscape kitchen sheet."""
    units = daily.attrs.get("units", "input quantity units")
    sheet = forecast.copy()
    sheet["units"] = units
    sheet["buffer_type"] = buffer_type
    sheet["buffer"] = buffer
    sheet["actual_units"] = "whole racks" if units == "quarter racks" else units
    sheet["recommended_prep"] = [0 if row.status == "Closed" else prep_quantity(row.predicted_quantity, buffer, buffer_type) for row in sheet.itertuples()]
    is_racks = units == "quarter racks"
    if is_racks:
        sheet["predicted_racks_equiv"] = sheet.predicted_quantity / 4
        sheet["recommended_whole_racks"] = np.ceil(sheet.recommended_prep / 4).astype(int)
    for name in ["actual_prepped", "actual_sold", "leftover", "notes"]:
        sheet[name] = ""
    sheet.to_csv(output / "weekly_prep_sheet.csv", index=False)
    date_range = f"{sheet.date.iloc[0]} through {sheet.date.iloc[-1]}"
    demand_col = "predicted_racks_equiv" if is_racks else "predicted_quantity"
    prep_col = "recommended_whole_racks" if is_racks else "recommended_prep"
    unit_label = "whole racks" if is_racks else units
    columns = ["Day / date", "Status", "Forecast\n(rack equiv.)" if is_racks else "Forecast", "Prep\n(whole racks)" if is_racks else "Prep", "Actually\nprepped", "Sold", "Leftover", "Notes / initials"]
    body = [[f"{row['day']}\n{row['date']}", "Closed" if row['status'] == "Closed" else "Open*", f"{row[demand_col]:.1f}", str(row[prep_col]), "", "", "", (f"Weather {100 * (row['weather_multiplier'] - 1):+.0f}%" if abs(row.get("weather_multiplier", 1) - 1) > 0.00001 else "Weather inactive" if row.get("weather_status") == "Weather adjustment inactive" else "")] for _, row in sheet.iterrows()]
    body.append(["WEEK TOTAL", "", f"{sheet[demand_col].sum():.1f}", str(sheet[prep_col].sum()), "", "", "", ""])
    fig = plt.figure(figsize=(11, 8.5), facecolor="white")
    fig.text(0.05, 0.94, "WEEKLY RIB PREP SHEET", fontsize=21, weight="bold")
    fig.text(0.05, 0.90, date_range, fontsize=13)
    buffer_label = f"{buffer / 4:g} racks/day ({buffer:g} quarters)" if is_racks and buffer_type == "fixed" else f"{buffer:g} {'%' if buffer_type == 'percent' else units}/day"
    fig.text(0.05, 0.862, f"Model: {selected.replace('_', ' ')}  |  Buffer: {buffer_label}  |  Prep rounded up daily", fontsize=9)
    ax = fig.add_axes([0.05, 0.30, 0.90, 0.53])
    ax.axis("off")
    table = ax.table(cellText=body, colLabels=columns, cellLoc="center", loc="center", bbox=[0, 0, 1, 1], colWidths=[.18, .08, .12, .11, .11, .09, .09, .22])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#9ca3af")
        if row == 0 or row == len(body):
            cell.set_facecolor("#e5e7eb")
            cell.set_text_props(weight="bold")
    latest = daily.loc[daily.rib_quantity.notna()].index.max()
    notes = [f"*All unmarked dates assumed open. Confirm closures, events and reservations before prep.",
             f"Write actuals in {unit_label}. Weekly prep total sums daily rounded quantities; no carryover assumed.",
             "Prep is the daily target before adjusting for usable inventory; record manager changes in Notes.",
             "Seven-day forecasts use earlier forecast demand in later lags. Refresh daily as sales arrive.",
             "Reported test errors measure one-day forecasts; seven-day accuracy has not been evaluated.",
             f"Latest complete sales: {latest.date()}. Source period ends: {daily.index.max().date()}."]
    if latest < daily.index.max():
        notes.append("PROVISIONAL: latest sales are incomplete; recent lag inputs are missing.")
    notes.append("Weather scenarios (if enabled): +/-25% cap; no supplied forecast means no adjustment.")
    fig.text(0.05, 0.26, "\n".join(notes), fontsize=9, va="top", linespacing=1.6)
    fig.text(0.05, 0.04, "Prepared by: ____________________     Reviewed by: ____________________     Date: ______________", fontsize=10)
    fig.savefig(output / "weekly_prep_sheet.pdf")
    fig.savefig(output / "weekly_prep_sheet.png", dpi=130)
    plt.close(fig)
    print(f"Weekly sheet: {date_range}; total prep {sheet[prep_col].sum()} {unit_label}")
    return sheet


def run(args):
    if not 0 < args.test_fraction < 0.5:
        raise ValueError("--test-fraction must be between 0 and 0.5.")
    if not np.isfinite(args.prep_buffer) or args.prep_buffer < 0:
        raise ValueError("--prep-buffer must be finite and nonnegative.")
    frame = load_data(args.input)
    if args.inspect:
        return
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    daily = build_daily_dataset(frame, args)
    daily.to_csv(output / "daily_rib_sales.csv")
    units = daily.attrs.get("units", "input quantity units")
    quality = {"calendar_days": len(daily), "usable_days": int(daily.rib_quantity.notna().sum()),
               "units": units, **daily.attrs}
    if "exclusion_reason" in daily:
        quality["excluded_dates"] = {str(date.date()): row.exclusion_reason for date, row in daily.iterrows() if row.exclusion_reason}
    (output / "data_quality.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
    # Preserve the full calendar before shifting so a lag means calendar days.
    features = create_features(daily)
    known = daily.loc[daily.rib_quantity.notna()]
    if len(known) < 35:
        raise ValueError("At least 35 observed calendar days are required; more history is recommended.")
    split = int(len(known) * (1 - args.test_fraction))
    train, test = known.iloc[:split], known.iloc[split:]
    print(f"Training: {train.index.min().date()} to {train.index.max().date()} ({len(train)} days)")
    print(f"Test: {test.index.min().date()} to {test.index.max().date()} ({len(test)} days)")
    print("Evaluation is one-day-ahead: previous test-day sales become available for subsequent lags.")
    if train.rib_quantity.sum() == 0:
        raise ValueError("Training period has no positive rib sales; collect history before fitting count models.")
    models = build_models()
    predictions = pd.DataFrame({"date": test.index, "actual": test.rib_quantity.to_numpy()})
    predictions["weekday_baseline"] = baseline_predict(train, test.index)
    for name, model in models.items():
        model.fit(features.loc[train.index, FEATURES], train.rib_quantity)
        predictions[name] = np.maximum(0, model.predict(features.loc[test.index, FEATURES]))
    names = ["weekday_baseline", *models]
    comparison = pd.DataFrame([{"model": name, **evaluate_model(predictions.actual, predictions[name])} for name in names])
    comparison.to_csv(output / "model_comparison.csv", index=False)
    if "stockout" in test:
        predictions["stockout"] = test.stockout.to_numpy()
    predictions.to_csv(output / "test_predictions.csv", index=False)
    for name in names:
        errors = predictions[["date", "actual"]].copy()
        errors["prediction"] = predictions[name]
        errors["absolute_error"] = abs(errors.actual - errors.prediction)
        errors["day_of_week"] = pd.DatetimeIndex(errors.date).day_name()
        if "stockout" in predictions:
            errors["stockout"] = predictions.stockout
        for col in ["checks_day", "holiday_name", "local_event"]:
            if col in test:
                errors[col] = test[col].to_numpy()
        errors = errors.sort_values("absolute_error", ascending=False)
        errors.to_csv(output / f"worst_days_{name}.csv", index=False)
        print(f"\nWorst days: {name}\n{errors.head(5).to_string(index=False)}")
    # Prefer a simple model unless added complexity reduces MAE by at least 5%.
    scores = comparison.set_index("model").MAE
    selected = "weekday_baseline"
    for name in models:
        if scores[name] < scores[selected] * 0.95:
            selected = name
    if args.model != "auto":
        selected = args.model
    next_date = daily.index.max() + pd.Timedelta(days=1)
    week_start = pd.Timestamp(getattr(args, "week_start", None) or next_date)
    future_weather = read_future_weather(args, next_date, week_start + pd.Timedelta(days=6))
    pd.DataFrame(index=pd.date_range(week_start, periods=7, freq="D"), columns=WEATHER).to_csv(output / "weather_forecast_template.csv", index_label="date")
    context = fit_weather_bias(daily, features, train, test, predictions[selected].to_numpy(),
                               getattr(args, "weather_bias", "auto"), output)
    (output / "weather_inputs.csv").write_text(future_weather.to_csv(index_label="date"), encoding="utf-8")
    model = None
    if selected != "weekday_baseline":
        model = build_models()[selected]
        model.fit(features.loc[known.index, FEATURES], known.rib_quantity)
    # Next-day JSON and the weekly sheet now share one forecasting implementation.
    closed = getattr(args, "closed_date", None) or []
    upcoming = forecast_days(daily, selected, model, None, [d for d in closed if pd.Timestamp(d) <= next_date + pd.Timedelta(days=6)], context, future_weather)
    first = upcoming.iloc[0]
    demand = float(first.predicted_quantity)
    prep = 0 if first.status == "Closed" else prep_quantity(demand, args.prep_buffer, args.buffer_type)
    recommendation = {"date": str(next_date.date()), "model": selected, "predicted_quantity": demand, "recommended_prep": prep, "buffer_type": args.buffer_type, "buffer": args.prep_buffer, "units": units}
    recommendation["weather_conditions"] = {column: (None if pd.isna(first[column]) else float(first[column])) for column in WEATHER}
    recommendation.update(without_weather_quantity=float(first.without_weather_quantity), weather_multiplier=float(first.weather_multiplier),
                          weather_status=first.weather_status, weather_adjustment_enabled=context["enabled"])
    if units == "quarter racks":
        recommendation["predicted_racks_equiv"] = demand / 4
        recommendation["recommended_racks_equiv"] = prep / 4
        recommendation["recommended_whole_racks"] = math.ceil(prep / 4)
    recommendation["latest_complete_sales_date"] = str(known.index.max().date())
    recommendation["missing_recent_sales_days"] = int((next_date - known.index.max()).days - 1)
    (output / "prep_recommendation.json").write_text(json.dumps(recommendation, indent=2), encoding="utf-8")
    weekly_forecast = forecast_days(daily, selected, model, getattr(args, "week_start", None), closed, context, future_weather)
    save_weekly_prep(daily, weekly_forecast, selected, args.prep_buffer, args.buffer_type, output)
    limitation = "Stockout flags are included in diagnostics; sales on flagged days remain censored lower bounds." if "stockout" in daily else "No stockout flags supplied: observed sales can underestimate demand."
    report = f"Selected model: {selected}\nPrefer baseline, then Poisson, then boosting; require 5% MAE improvement at each step.\nOverride with --model after reviewing errors, RMSE and operational cost.\nPoisson is interpretable; boosting captures interactions but is harder to explain.\nOne chronological holdout does not establish stability; verify across seasons before use.\n{limitation}\nMissing dates policy: {args.missing_days}.\nQuantities retain source units: normalize half/full racks before running.\nBuffer is a business rule, not a prediction interval; tune against waste and lost sales.\n"
    report += f"Target units: {units}. Prep buffer uses those same units.\n"
    if "exclusion_reason" in daily:
        report += f"Partial dates excluded; zero-check probable closure policy: {getattr(args, 'closed_days', 'exclude')}. Excluded dates are absent from targets and historical quantities.\n"
    report += "Near-zero days with checks are retained; confirm opening hours before operational use.\n"
    report += "Same-day checks, revenue and unverified events are not predictors. Weather is an optional bounded scenario adjustment.\n"
    report += context["report"]["reason"] + "\n" + context["report"]["evaluation"] + "\n"
    if recommendation["missing_recent_sales_days"]:
        report += "Recent sales are incomplete: next-day prediction has missing recent lags and is provisional.\n"
    (output / "model_selection.txt").write_text(report, encoding="utf-8")
    plot_results(daily, predictions, output)
    print("\n", comparison.to_string(index=False), "\n", report, "\nPrep recommendation:", recommendation)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--weather-bias", choices=["auto", "on", "off"], default="auto", help="Auto: require 5%% exploratory holdout MAE improvement; on: exploratory override")
    parser.add_argument("--weather-forecast", help="Future weather CSV: date,temp_max_f,temp_min_f,precip_in,snowfall_in")
    parser.add_argument("--weather-high", type=float, help="Next-day forecast high, Fahrenheit")
    parser.add_argument("--weather-low", type=float, help="Next-day forecast low, Fahrenheit")
    parser.add_argument("--weather-rain", type=float, help="Next-day forecast precipitation, inches (not probability)")
    parser.add_argument("--weather-snow", type=float, help="Next-day forecast snow, inches")
    parser.add_argument("--week-start", help="Seven-day sheet start (YYYY-MM-DD); default: day after last input date")
    parser.add_argument("--closed-date", action="append", help="Known future closure (YYYY-MM-DD); repeat for multiple dates")
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--input-format", choices=["auto", "daily", "transactions"], default="auto")
    parser.add_argument("--item-mapping", help="Audit mapping arithmetic and full-period quarter-rack totals")
    parser.add_argument("--closed-days", choices=["exclude", "keep"], default="exclude", help="Daily data: exclude zero-check probable closures by default")
    parser.add_argument("--inspect", action="store_true", help="Print CSV schema/sample and exit without training")
    for column in ("date", "item", "quantity", "stockout"):
        parser.add_argument(f"--{column}-column")
    parser.add_argument("--rib-keyword", action="append")
    parser.add_argument("--missing-days", choices=["unknown", "zero"], default="unknown")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--prep-buffer", type=float, default=3)
    parser.add_argument("--buffer-type", choices=["fixed", "percent"], default="fixed")
    parser.add_argument("--model", choices=["auto", "weekday_baseline", "poisson_regression", "gradient_boosting"], default="auto")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        run(parse_args())
    except (ValueError, OSError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
