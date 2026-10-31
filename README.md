# RibModel

A small Python project that forecasts next-day restaurant rib sales, compares a weekday baseline, Poisson regression and histogram gradient boosting, and recommends a preparation quantity.

## Setup (Python 3.11+)

From this folder in Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Command Prompt activation: `.venv\Scripts\activate.bat`.
macOS/Linux: `python3 -m venv .venv`, then `source .venv/bin/activate`.
Use `deactivate` to exit. In VS Code select `.venv` as the Python interpreter.

## Data and usage

Put your export in `data/sales.csv`. Raw restaurant data and generated outputs are ignored by Git.
Inspect an unfamiliar export first:

```powershell
python src/rib_forecast.py --input data/sales.csv --inspect
python src/rib_forecast.py --input data/sales.csv
python src/rib_forecast.py --input data/sales.csv --date-column "Transaction Date" --item-column "Menu Item" --quantity-column "Qty" --rib-keyword "rib" --prep-buffer 15 --buffer-type percent
```

Mappings are inferred for common date/item/quantity names; explicit arguments take priority. Keywords are literal case-insensitive substrings; repeated `--rib-keyword` arguments replace defaults. Review matching item names, especially combo meals. Quantities must be finite and nonnegative. Resolve refunds before running. Parse timestamps in the restaurant's local timezone with consistent formatting; ambiguous date formats should be normalized upstream. All quantities must use the same unit: convert half/full racks to equivalent units upstream. The tool never assumes quantities mean individual ribs or racks.

At least 35 observed days are required; several months covering all weekdays are better. Days containing other restaurant transactions but no matching rib items are observed zero sales. Dates entirely absent from the export remain unknown by default and are excluded as training/evaluation targets. Use `--missing-days zero` only after confirming they are true zero-sale days. Missing historical inputs are imputed using training medians for Poisson; boosting handles missing inputs directly. Optional `--stockout-column SoldOut` accepts true/false, yes/no or 1/0 and adds flags to diagnostics. No fabricated demand correction is applied. Sales on stockout days underestimate true demand; absent flags are an explicit limitation. An export containing only rib lines cannot distinguish zero rib sales from missing dates.

## Features and evaluation

Calendar features: weekday, month, day of month, ISO week, weekend. Historical features: previous day, previous week, trailing 7/28-day mean, previous four same weekdays. Every historical feature shifts before rolling; preprocessing is fitted only on training dates.

The first 80% of known dates train the models; the last 20% evaluate them (`--test-fraction` changes this). Evaluation simulates **one-day-ahead predictions**: each day's observed sales are available before forecasting the following day. This is not a fixed-origin multi-day forecast. Models remain fixed during evaluation. The weekday baseline uses training-only averages. MAE is the average absolute error in source units; RMSE penalizes large misses more heavily.

Poisson regression uses regularization and a log link, with encoded weekday/month and scaled numeric inputs. Count-like nonnegative demand makes it a useful interpretable comparison, but the Poisson equal-mean/variance assumption is often imperfect. Boosting captures nonlinear effects and interactions with a small fixed configuration; there is no hyperparameter search.

The automatic choice starts with the baseline, then accepts Poisson and boosting only when each reduces current MAE by at least 5%. This is a transparent heuristic, not proof of superiority. Review RMSE, worst days, complexity and maintenance before choosing. Use `--model weekday_baseline`, `--model poisson_regression` or `--model gradient_boosting` to override. The holdout helps choose a model, so its reported score is not an independent estimate after model selection. Additional chronological validation windows are needed to establish stability.

## Outputs and prep

`outputs/` contains daily sales, the MAE/RMSE comparison, held-out predictions, per-model worst-day tables (with weekday and optional stockout flags), sales-over-time/weekday/month/distribution plots, actual-versus-predicted plot, model-selection notes and `prep_recommendation.json`.

The selected model is refitted on all observed dates for the next calendar day. Fixed buffer (default 3): `ceil(prediction + buffer)`. Percentage buffer: `ceil(prediction * (1 + buffer / 100))`. The recommendation retains the input quantity units. Tune the buffer against waste, stockout and lost-sales costs; it is not a calibrated uncertainty interval. Confirm the restaurant is open on the recommended day. Holidays, promotions, changing menus, weather and future events are not modeled in this first version.

## ML Concepts Practiced

- Feature engineering and lagged calendar-day history.
- Time-based validation and avoiding future-data leakage.
- Baselines before complexity.
- Poisson regression / GLMs and their assumptions.
- Gradient boosting for nonlinear relationships.
- MAE versus RMSE.
- Model complexity versus interpretability and stability.
- Business tradeoffs: waste, censored demand, lost sales and operational prep.

## Verification

```powershell
python -m unittest discover -s tests -v
```

Tests use synthetic sales and verify past-only features, missing-day handling and end-to-end output generation; their scores do not represent restaurant performance.

## GitHub

Intended repository: https://github.com/brianwemett/RibModel
Do not commit restaurant exports, local environments, secrets or generated reports. Authentication is handled by your Git credential manager or SSH configuration.

For the exact verified dependency versions, install with: `python -m pip install -r requirements-lock.txt`.


## Prepared daily exports (Good Smoke BBQ)

The uploaded `data/rib_sales_daily.csv` is already normalized by business date and rack portion. Run from the project root:

```powershell
python src/rib_forecast.py --input data/rib_sales_daily.csv --item-mapping data/item_mapping.csv --prep-buffer 12
```

Daily format is detected by `total_quarter_racks`; use `--input-format daily` to specify it explicitly. The target is **quarter racks**, not menu item counts. Full rack = 4 quarters; half rack = 2. Do not multiply combo/modifier counts again: the prepared totals already contain those conversions. The mapping is used to audit conversion arithmetic and reconcile its full-period total against daily totals, not to recount sales. It must cover the same export period. Daily component totals and `total_racks_equiv` are checked when present. Raw transaction mode remains supported (`--input-format transactions`).

The command above uses a fixed 12-quarter buffer (3 full racks). JSON output includes expected quarter racks, rack equivalents and the number of whole racks needed after buffering. The generic CLI default remains 3 input units, so explicitly set your chosen buffer. Use `--buffer-type percent --prep-buffer 15` for a 15% buffer.

`partial=1` days are retained in the calendar audit but excluded from targets and historical quantities. A partial last day still determines the next calendar date; its incomplete total never becomes a next-day lag. The recommendation reports the latest complete observation and number of missing recent sales days, and is provisional when recent data is incomplete.

By default, daily rows with zero checks are treated as probable closures and excluded from targets/history; `--closed-days keep` retains them as zeros for calendar-day sales forecasting. This inference needs operational confirmation. One-check/near-zero days remain included, including Easter; do not infer closure from low demand alone. Predictions under the default policy describe an assumed open service day. Check the opening schedule before prep.

`outputs/data_quality.json` lists units, usable days, excluded dates and reconciled totals. `daily_rib_sales.csv` retains reported quantities and exclusion reasons. Worst-day tables include check counts, holiday names and event labels for retrospective investigation only; event labels may be unverified. Contextual columns, same-day net sales and manually supplied holiday fields are not core-model predictors. Weather can enter the optional scenario adjustment described below. No online event verification is performed.

The November 2025-October 2026 upload has less than a complete annual cycle. Rib Teaser portion sizes and ad-hoc item assumptions remain as documented in the data README. No stockout history was supplied, so the forecast estimates observed sales rather than uncensored demand. The data README references upstream build/weather files that are not present in this folder; this project consumes the prepared CSVs directly. All uploaded data and generated results remain excluded from Git.


## Weekly prep sheet

Each forecast run also creates `outputs/weekly_prep_sheet.pdf` (one printable landscape letter page), `weekly_prep_sheet.csv` (editable daily figures), and a PNG preview. By default the sheet covers the next seven calendar days after the latest input date. It shows forecast volume, daily buffered prep, a weekly total, and blanks for actual prep, sold, leftovers and notes. For quarter-rack inputs, the kitchen sheet shows demand in full-rack equivalents and prep/actuals in whole racks. The weekly total is the sum of each day's rounded whole-rack prep; it assumes no carryover. Adjust prep for usable inventory and confirmed events.

```powershell
python src/rib_forecast.py --input data/rib_sales_daily.csv --item-mapping data/item_mapping.csv --prep-buffer 12
```

For a chosen week or known closure, add `--week-start 2026-11-02` and repeat `--closed-date YYYY-MM-DD` for each closed date. Week start must follow the input period and be within 90 days of its end. Closed dates must be between the next forecast date and the sheet's end; they receive zero demand and zero prep, with no buffer. Other dates are assumed open. Future closures are left unknown in lag features to match excluded historical closures. A later week start forecasts intervening days too, without showing them on the sheet. Do not use a future start to skip unavailable history silently: refresh sales first when possible.

The selected model is refitted once using complete known sales. Later forecasts recursively use earlier **unbuffered demand predictions**, never prep quantities or unavailable actuals. Existing missing/partial historical dates stay unknown. The baseline keeps its historical weekday averages. One-day held-out metrics do not measure seven-day accuracy; the weekly plan is provisional, especially when the latest input day is partial. Refresh the sheet daily with completed sales. The buffer remains a business assumption rather than an uncertainty interval.


## Optional next-day weather adjustment

The daily export's `temp_max_f`, `temp_min_f`, `precip_in`, and `snowfall_in` can train an optional weather adjustment. Inputs use **Fahrenheit and inches**; precipitation means expected amount, not percentage chance of rain. Use forecast conditions available before prep, not today's conditions as a substitute for tomorrow's forecast. There is no automatic weather lookup or external service.

For example, the following **hypothetical** conditions illustrate the CLI (replace with the actual next-day forecast):

```powershell
python src/rib_forecast.py --input data/rib_sales_daily.csv --item-mapping data/item_mapping.csv --prep-buffer 12 --weather-high 45 --weather-low 32 --weather-rain 0.30 --weather-snow 0 --weather-bias auto
```

`auto` is the default. A regularized Poisson model learns weather associations while controlling for calendar and prior sales. For a given day, its expected demand at supplied conditions is divided by its expected demand at reference weather, holding other features fixed. Reference weather is the historical median of each field. This multiplier adjusts the selected core model's forecast before the safety buffer, bounded between **0.75 and 1.25**. This cap is a conservative business assumption, not a calibrated interval or a learned threshold. Weather associations are not causal effects.

The chronological comparison fits on training dates only and uses training-only reference medians. Adjustment turns on automatically only with at least 60 complete training-weather days, 14 complete test-weather days, 80% test coverage and at least 5% holdout MAE improvement. Test days without all four observed fields retain the base forecast. This gate is exploratory: historic observations are what happened, not archived forecasts available the day before. Forecast errors can reduce real operational performance. After comparison, the weather model and reference medians are refitted using all known historical target dates.

On the current 350-day prepared sample, core Poisson MAE is **10.19 quarters (2.55 racks)** versus **11.71 quarters (2.93 racks)** after weather adjustment. This worsens error by roughly 15%, so **auto leaves weather adjustment off**. `--weather-bias off` disables it explicitly. `--weather-bias on` is an explicit exploratory override that applies the learned multiplier despite this result; it still requires sufficient historical data. An override should not be treated as evidence of improvement.

For several days, create a local CSV and pass `--weather-forecast data/weather_forecast.csv`:

```csv
date,temp_max_f,temp_min_f,precip_in,snowfall_in
```

Add dated rows using the weather forecast available at the time you plan prep. Dates must be unique and within the upcoming forecast period. CLI weather flags set or override the next-day row. A blank field is unknown and uses reference weather for that field; an entirely blank or absent day causes no adjustment. An explicit zero means dry/no snow, not missing. Temperature highs must be at least their corresponding lows. Negative precipitation, invalid numbers, duplicate dates and out-of-period forecasts are rejected. No historical missing weather is automatically treated as zero.

`outputs/weather_forecast_template.csv` provides seven blank dated rows for the current input period. Copy it into `data/`, fill in real forecast values, and use that file. Tomorrow's recommendation and the weekly sheet use the same adjustment logic. Later weekly forecasts use earlier adjusted demand in their lags; the buffer is never fed back as demand. Closed dates always receive zero demand/prep regardless of weather.

Audit files:

- `weather_report.json`: comparison, gate decision, reference weather and limitations.
- `weather_comparison.csv` and `weather_test_predictions.csv`: experimental historic results (when history is sufficient).
- `weather_coefficients.csv`: standardized numeric coefficients, not raw-unit effects.
- `weather_inputs.csv`: the supplied future weather, saved for review.
- `prep_recommendation.json`: base demand, multiplier, supplied conditions and adjusted prep.
- `weekly_prep_sheet.csv`: daily base demand, multiplier, conditions and adjusted forecast; the PDF annotates applied changes in Notes.

No supplied future weather means no adjustment, even when enabled. Update complete sales and weather forecasts daily. The same stockout, incomplete-sales and unverified-event limitations still apply.
