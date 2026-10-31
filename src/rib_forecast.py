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
    extended = daily.reindex(daily.index.append(pd.DatetimeIndex([next_date])))
    if selected == "weekday_baseline":
        demand = float(baseline_predict(known, pd.DatetimeIndex([next_date]))[0])
    else:
        model = build_models()[selected]
        model.fit(features.loc[known.index, FEATURES], known.rib_quantity)
        demand = max(0.0, float(model.predict(create_features(extended).loc[[next_date], FEATURES])[0]))
    prep = math.ceil(demand + args.prep_buffer) if args.buffer_type == "fixed" else math.ceil(demand * (1 + args.prep_buffer / 100))
    recommendation = {"date": str(next_date.date()), "model": selected, "predicted_quantity": demand, "recommended_prep": prep, "buffer_type": args.buffer_type, "buffer": args.prep_buffer, "units": "input quantity units"}
    (output / "prep_recommendation.json").write_text(json.dumps(recommendation, indent=2), encoding="utf-8")
    limitation = "Stockout flags are included in diagnostics; sales on flagged days remain censored lower bounds." if "stockout" in daily else "No stockout flags supplied: observed sales can underestimate demand."
    report = f"Selected model: {selected}\nPrefer baseline, then Poisson, then boosting; require 5% MAE improvement at each step.\nOverride with --model after reviewing errors, RMSE and operational cost.\nPoisson is interpretable; boosting captures interactions but is harder to explain.\nOne chronological holdout does not establish stability; verify across seasons before use.\n{limitation}\nMissing dates policy: {args.missing_days}.\nQuantities retain source units: normalize half/full racks before running.\nBuffer is a business rule, not a prediction interval; tune against waste and lost sales.\n"
    (output / "model_selection.txt").write_text(report, encoding="utf-8")
    plot_results(daily, predictions, output)
    print("\n", comparison.to_string(index=False), "\n", report, "\nPrep recommendation:", recommendation)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="outputs")
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
