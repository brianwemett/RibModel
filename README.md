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
