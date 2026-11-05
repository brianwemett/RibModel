# Restaurant Rib Demand Forecasting & Prep Optimization

An end-to-end machine-learning project that forecasts next-day restaurant rib demand and translates the forecast into a practical kitchen prep recommendation.

The project uses historical sales patterns, calendar information, lagged demand, and rolling demand features to compare a simple weekday baseline, regularized Poisson regression, and histogram gradient boosting. Model evaluation uses a chronological holdout so the test period represents future dates rather than a random sample of the past.

**Current status: operational pilot.** Forecasts are being provided to restaurant staff through a weekly prep sheet, with actual prep, sales, leftovers, and operational notes available for follow-up evaluation.

## Results

On the current prepared dataset of 350 usable daily observations, the models produced the following chronological holdout results:

| Model | MAE (full racks) | RMSE (full racks) |
| --- | ---: | ---: |
| Weekday baseline | 2.83 | 3.38 |
| **Poisson regression** | **2.55** | **3.06** |
| Gradient boosting | 2.78 | 3.43 |

Poisson regression reduced holdout MAE by approximately **10%** versus the simple weekday baseline while also outperforming the more flexible gradient-boosting model.

The result is intentionally treated as a model-selection result rather than a final estimate of future production performance. The same holdout helped choose the model, so additional chronological validation windows are needed to establish stability.

## Business Problem

Restaurant rib preparation has a direct operational tradeoff:

- **Over-preparation** creates food waste and unnecessary food cost.
- **Under-preparation** creates stockouts, lost sales, and a poorer guest experience.

The modeling goal is to estimate expected rib demand before service and convert that forecast into a useful prep recommendation.

The forecast and the prep decision are deliberately separated. The model predicts expected demand; an operational safety buffer is applied afterward. That buffer is a business rule, not a model confidence interval, and should ultimately be calibrated against the relative cost of waste and stockouts.

## Modeling Approach

The workflow is:

```text
historical sales
      ↓
daily demand table
      ↓
calendar + lagged demand features
      ↓
chronological train/test split
      ↓
weekday baseline
      ↓
Poisson regression
      ↓
histogram gradient boosting
      ↓
MAE / RMSE comparison
      ↓
selected model
      ↓
next-day forecast
      ↓
operational prep buffer
      ↓
daily / weekly kitchen prep recommendation
```

### Features

Calendar features:

- weekday
- month
- day of month
- ISO week
- weekend indicator

Historical demand features:

- previous-day sales
- previous-week sales
- trailing 7-day mean
- trailing 28-day mean
- previous four same weekdays

All lagged and rolling features are shifted before calculation so they use **past data only**. Preprocessing is fit on training dates only.

### Validation

The default evaluation uses the first 80% of known dates for training and the final 20% for testing.

This is a **chronological holdout**, not a random split. It better represents the real task: predicting future restaurant demand from information available in the past.

Evaluation simulates **one-day-ahead forecasting**. Each test day's observed sales are available before forecasting the following day. The fitted models remain fixed during the holdout evaluation.

### Models

#### Weekday baseline

The baseline predicts demand using the average historical sales for the same weekday from the training period.

It provides a simple operational benchmark: a more complex model should only be preferred if it meaningfully improves on a rule that is easy to understand and maintain.

#### Poisson regression

The Poisson model uses `sklearn.linear_model.PoissonRegressor` with:

- one-hot encoded weekday and month
- scaled numeric features
- median imputation for missing historical inputs
- regularization

Poisson regression is a useful comparison because daily demand is nonnegative and count-like, while the model remains relatively interpretable.

The strict Poisson equal-mean/variance assumption may not hold perfectly in restaurant demand, so performance is judged empirically on the chronological holdout rather than assumed from the model family.

#### Histogram gradient boosting

`HistGradientBoostingRegressor` provides a more flexible comparison that can capture nonlinear relationships and feature interactions automatically.

In the current data, that extra flexibility did not improve holdout performance enough to justify preferring it over Poisson regression.

## Model Selection

Automatic selection starts with the weekday baseline and only accepts a more complex candidate when it reduces current MAE by at least 5%.

That threshold is a transparent heuristic rather than proof that one model is universally superior. Final selection should also consider:

- MAE and RMSE
- worst prediction days
- interpretability
- stability
- complexity
- maintenance cost
- operational usefulness

The selected model can also be overridden explicitly from the command line.

## Key Findings

- Recent demand history plus broader calendar information improved on a weekday-only baseline.
- Poisson regression generalized better than the gradient-boosting model on the current holdout.
- Model complexity did not automatically produce better performance.
- Large misses are operationally useful diagnostic cases rather than just metric penalties.
- Observed sales may underestimate true demand on stockout days.
- The operational prep buffer should be tuned separately from the demand model.
- Weather appeared intuitively useful but did **not** improve holdout performance in the current sample.

## Weather Experiment

Historical high temperature, low temperature, precipitation, and snowfall can be evaluated as an optional Poisson-based adjustment.

The weather model compares expected demand at supplied conditions with expected demand at historical median weather while controlling for calendar and prior-sales features.

On the current prepared sample:

- core Poisson MAE: **2.55 full racks**
- weather-adjusted MAE: **2.93 full racks**

Because weather worsened holdout MAE by roughly 15%, automatic weather adjustment remains **off**.

This is an important design choice: plausible features are not retained simply because they sound useful. They must demonstrate out-of-sample value.

## Operational Pilot

The project generates a printable weekly prep sheet for restaurant staff.

The sheet includes:

- forecast demand
- buffered prep recommendation
- weekly prep total
- blanks for actual prep
- sold quantity
- leftovers
- operational notes

The pilot is intended to create a feedback loop between model performance and restaurant operations. Future evaluation can compare forecast error with the business outcomes that matter most: waste, stockouts, and lost sales.

Weekly forecasts are recursive. Later days use earlier **unbuffered demand predictions** as lag inputs when actual future sales are not yet available. Because of that, one-day holdout metrics should not be interpreted as seven-day forecast accuracy.

## Important Limitations

### Observed sales vs. true demand

No stockout history was supplied for the current dataset. If the restaurant sells out, recorded sales can be lower than actual customer demand.

The model therefore estimates **observed sales**, not fully uncensored demand.

### Incomplete seasonal history

The current prepared sample spans less than a complete multi-year seasonal history. Additional data should improve evaluation of year-over-year seasonality and unusual calendar periods.

### Closures and partial days

Partial days are excluded from model targets and historical quantities.

Daily rows with zero checks are treated as probable closures by default and excluded from targets/history. This is an operational inference and should be confirmed rather than blindly assumed.

### Events and promotions

Known future holidays, promotions, local events, and menu changes are not included in the core model in this version.

### Holdout reuse

The holdout is used for model comparison and selection, so its reported score is not a fully independent post-selection performance estimate. Rolling or multiple chronological validation windows are a natural next step.

## Data Privacy

**No restaurant transaction or sales export is included in this repository.**

Raw restaurant data, generated operational reports, model outputs, local environments, and secrets remain outside version control through `.gitignore`.

The code is designed to run against local data stored in `data/`, while generated files are written to `outputs/`. Both directories are ignored except for placeholder `.gitkeep` files.

## Tech Stack

- Python 3.11+
- pandas
- NumPy
- scikit-learn
- matplotlib

## Project Structure

```text
RibModel/
├── data/                  # local inputs; ignored by Git
├── outputs/               # generated reports; ignored by Git
├── src/
│   ├── rib_forecast.py
│   └── weather_bias.py
├── tests/
│   ├── test_forecast.py
│   └── test_weather.py
├── .gitignore
├── README.md
├── requirements.txt
└── requirements-lock.txt
```

## Setup

Create a local Python virtual environment.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### Windows Command Prompt

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -r requirements.txt
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Use `deactivate` to exit the environment. In VS Code, select `.venv` as the Python interpreter.

For exact verified dependency versions:

```bash
python -m pip install -r requirements-lock.txt
```

## Data and Usage

### Generic transaction export

Put a local export in `data/sales.csv`.

Inspect an unfamiliar export first:

```powershell
python src/rib_forecast.py --input data/sales.csv --inspect
```

Run the forecast:

```powershell
python src/rib_forecast.py --input data/sales.csv
```

Explicit mappings can be supplied when column names differ:

```powershell
python src/rib_forecast.py --input data/sales.csv --date-column "Transaction Date" --item-column "Menu Item" --quantity-column "Qty" --rib-keyword "rib" --prep-buffer 15 --buffer-type percent
```

Mappings are inferred for common date/item/quantity names, while explicit arguments take priority.

Quantities must be finite and nonnegative. Refunds should be resolved before modeling. Timestamps should use consistent local-date formatting, and all rib quantities must be converted to a consistent unit before fitting.

### Prepared daily export

The current operational workflow can consume a normalized daily file:

```powershell
python src/rib_forecast.py --input data/rib_sales_daily.csv --item-mapping data/item_mapping.csv --prep-buffer 12
```

Daily format is detected by `total_quarter_racks`; use `--input-format daily` to specify it explicitly.

For the prepared daily workflow:

- target units are quarter racks
- full rack = 4 quarters
- half rack = 2 quarters
- combo/modifier conversions are already included in prepared totals
- the item mapping audits conversion arithmetic and reconciles totals

The example command uses a fixed 12-quarter / 3-full-rack operational buffer. That number is configurable and is not a learned model parameter.

## Missing and Zero Days

At least 35 observed days are required, though several months covering all weekdays are preferable.

Days containing other restaurant transactions but no matching rib items are observed zero-sales days.

Dates entirely absent from an export remain unknown by default and are excluded from training/evaluation targets.

Use:

```text
--missing-days zero
```

only after confirming that missing dates truly represent zero sales.

Optional `--stockout-column SoldOut` accepts true/false, yes/no, or 1/0 values and adds stockout flags to diagnostics. The software does not fabricate demand beyond observed sales.

## Outputs

The `outputs/` directory can contain:

- `daily_rib_sales.csv`
- `model_comparison.csv`
- held-out predictions
- per-model worst-day tables
- sales-over-time plot
- weekday/month/distribution plots
- actual-vs-predicted plot
- model-selection notes
- `prep_recommendation.json`
- `weekly_prep_sheet.csv`
- `weekly_prep_sheet.pdf`
- `weekly_prep_sheet.png`
- weather experiment diagnostics

Generated outputs remain ignored by Git.

## Weekly Prep Sheet

A normal forecast run also creates a seven-day prep sheet:

```powershell
python src/rib_forecast.py --input data/rib_sales_daily.csv --item-mapping data/item_mapping.csv --prep-buffer 12
```

For a chosen week or known closure:

```powershell
python src/rib_forecast.py --input data/rib_sales_daily.csv --item-mapping data/item_mapping.csv --prep-buffer 12 --week-start 2026-11-02 --closed-date 2026-11-05
```

Closed dates receive zero predicted demand and zero prep.

Refresh the sheet with completed sales whenever possible rather than treating a seven-day forecast as fixed.

## Optional Weather Scenario

The optional weather workflow accepts:

- `temp_max_f`
- `temp_min_f`
- `precip_in`
- `snowfall_in`

Example:

```powershell
python src/rib_forecast.py --input data/rib_sales_daily.csv --item-mapping data/item_mapping.csv --prep-buffer 12 --weather-high 45 --weather-low 32 --weather-rain 0.30 --weather-snow 0 --weather-bias auto
```

The application does not perform an automatic weather lookup. Supply forecast conditions that would actually have been available before prep.

The adjustment is bounded between 0.75 and 1.25 as a conservative business rule, not a learned confidence interval.

Because the current historical weather experiment worsened holdout performance, `auto` leaves weather adjustment disabled.

For several future days, provide a local CSV:

```csv
date,temp_max_f,temp_min_f,precip_in,snowfall_in
```

and run with `--weather-forecast data/weather_forecast.csv`.

## Verification

Run the test suite with:

```powershell
python -m unittest discover -s tests -v
```

Tests use synthetic sales and verify items including:

- past-only historical features
- missing-day handling
- partial/closed-day behavior
- daily mapping reconciliation
- recursive weekly forecasts
- weekly prep generation
- weather validation
- end-to-end output generation

Synthetic test scores do not represent restaurant performance.

## Next Steps

Likely future improvements include:

- rolling chronological validation
- explicit stockout / sellout-time capture
- calibration of the prep buffer against waste and stockout costs
- additional full seasonal cycles
- verified promotions and local-event features
- daily operational error tracking
- comparison of forecast accuracy with actual waste and lost-sales outcomes

## License

No license has been specified. All rights remain with the repository owner.
