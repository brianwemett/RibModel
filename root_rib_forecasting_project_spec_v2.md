# Restaurant Rib Demand Forecasting Project

## Goal

Build a small, practical Python machine-learning project that predicts how many ribs the restaurant should prepare for a given day.

The main purpose is to:

1. Create a useful restaurant forecasting tool.
2. Practice a realistic end-to-end ML workflow.
3. Get hands-on familiarity with:
   - pandas
   - NumPy
   - scikit-learn
   - matplotlib
4. Compare a simple baseline, an interpretable GLM-style count model, and a tree-based model.
5. Produce code that is easy to understand and explain in a technical interview.

This is intentionally a small-data project. Do **not** add PySpark, TensorFlow, PyTorch, or unnecessary infrastructure.

---

## Business Problem

The restaurant needs to decide how many ribs to prepare each day.

Cooking too few ribs can cause:

- stockouts
- lost sales
- poor guest experience

Cooking too many can cause:

- waste
- unnecessary food cost
- labor inefficiency

The model should predict expected daily rib demand using historical sales data.

A secondary goal is to make the result operationally useful by producing a recommended prep quantity.

---

## Important Modeling Caveat

Historical **sales are not always the same as true demand**.

If the restaurant sold out of ribs on a given day, recorded sales may understate how many customers actually wanted ribs.

If the dataset contains any indication of:

- stockouts
- sold-out times
- inventory limits
- item availability

use that information.

If no stockout information exists, document this clearly as a limitation.

---

## Expected Input

The input will be a restaurant sales export in CSV format.

The exact schema is not yet known.

The script should inspect the available columns and make it easy to configure or map the following fields:

- transaction date/time
- item name or item identifier
- quantity sold
- gross/net sales if available
- discounts if available
- order type if available
- category if available

The user should be able to specify which item names correspond to ribs.

If the raw export is order-line level, aggregate it to one row per calendar day.

---

## Project Structure

Prefer a simple structure such as:

```text
rib_forecast/
    .venv/
    data/
    outputs/
    src/
        rib_forecast.py
    .gitignore
    README.md
    requirements.txt
```

A single well-organized Python script is acceptable for the first version.

---

## Virtual Environment Setup

Create a local Python virtual environment named `.venv` in the project root before installing dependencies.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Windows Command Prompt

```cmd
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Add the virtual environment to `.gitignore`:

```text
.venv/
```

The README should explain how to create, activate, and deactivate the environment.

To deactivate:

```bash
deactivate
```

If using VS Code, select the Python interpreter from the `.venv` environment for this project.

---

## Python Requirements

Use:

- Python 3.11+
- pandas
- numpy
- scikit-learn
- matplotlib

Optional:

- statsmodels, only if it adds useful GLM diagnostics without making the project unnecessarily complex

Do not use:

- PySpark
- TensorFlow
- PyTorch
- deep learning
- cloud services
- external databases

Create a `requirements.txt` containing the required libraries.

---

## Workflow

### 1. Load and Inspect the Data

Load the CSV with pandas.

Print or log:

- shape
- column names
- date range
- null counts
- example rows
- unique or most-common item names

Make the script fail gracefully if required columns cannot be identified.

---

### 2. Filter Rib Sales

Create a configurable list of rib-related item names or keywords.

Examples might include:

- ribs
- half rack
- full rack
- baby back ribs
- beef ribs

Do not hard-code assumptions beyond a clearly editable configuration section.

Aggregate rib quantities by calendar day.

The resulting modeling table should contain at least:

```text
date
rib_quantity
```

---

### 3. Create Calendar and Historical Features

Create useful features such as:

- day_of_week
- day_of_month
- week_of_year
- month
- is_weekend

Create lagged or rolling features when enough history exists:

- ribs sold previous day
- ribs sold 7 days earlier
- average ribs sold over previous 7 days
- average ribs sold over previous 28 days
- average of previous same weekdays if practical

All lagged and rolling features must use **past data only**.

Do not leak future information into predictors.

Optional features if the data supports them:

- holiday indicator
- promotion indicator
- special-event indicator
- dine-in vs takeout mix
- total restaurant sales
- total covers/orders

Keep the first version simple.

---

## Exploratory Analysis

Produce useful plots such as:

- rib sales over time
- average rib sales by day of week
- average rib sales by month
- histogram of daily rib sales

Save plots to the `outputs/` directory.

Keep visualizations simple and readable.

---

## Train/Test Strategy

This is time-series-like business data.

Do **not** use a simple random train/test split.

Use a chronological split.

For example:

- first 80% of dates = training
- final 20% of dates = test

The script should print:

- training date range
- test date range
- number of observations in each

This should simulate predicting future demand from past data.

---

## Models

Build and compare three approaches.

### Model 0: Simple Baseline

Create at least one simple baseline.

Preferred baseline:

- average historical rib sales for the same day of week

Possible alternative:

- average of the previous 4 same weekdays

The baseline is important because a more complex model should only be preferred if it meaningfully improves on a simple operational rule.

---

### Model 1: Poisson Regression

Use:

```python
sklearn.linear_model.PoissonRegressor
```

Why:

- daily rib demand is count-like data
- it is interpretable
- it provides practice with a GLM-style model

Use appropriate preprocessing.

For categorical calendar features, use one-hot encoding if needed.

Consider regularization using the model's `alpha` parameter.

Keep tuning simple.

Explain in comments:

- why Poisson regression is reasonable
- what assumptions it makes
- why it is useful as an interpretable baseline

---

### Model 2: Tree-Based Model

Use one of:

```python
HistGradientBoostingRegressor
```

or:

```python
GradientBoostingRegressor
```

Prefer `HistGradientBoostingRegressor` if practical.

Why:

- handles nonlinear relationships
- can capture interactions
- provides a useful comparison against the simpler GLM

Do not over-tune.

Use a small, defensible set of hyperparameters.

---

## Evaluation

Evaluate all models on the same held-out time period.

Report:

- MAE
- RMSE

Optional:

- MAPE, only if zero-demand days are handled carefully

Also compare:

- actual daily rib sales
- predicted daily rib sales

Create a comparison table such as:

```text
model                 MAE     RMSE
weekday_baseline      ...
poisson_regression    ...
gradient_boosting     ...
```

Save the results to CSV.

---

## Diagnostic Analysis

For each model, inspect where the largest errors occur.

Create a table containing:

```text
date
actual
prediction
absolute_error
```

Show the worst prediction days.

If possible, investigate whether large errors correspond to:

- weekends
- holidays
- promotions
- unusually high or low restaurant volume
- sellout days
- seasonal changes

This is important because the goal is not just to produce a metric, but to understand why the model fails.

---

## Model Selection

Do not automatically choose the model with the smallest error.

The final comparison should discuss:

- predictive performance
- interpretability
- model complexity
- stability
- ease of maintenance
- operational usefulness

A slightly worse Poisson model may still be attractive if it is much easier to understand and maintain.

A tree model should only be preferred if the performance improvement is meaningful.

---

## Prep Recommendation

After selecting a model, create a simple operational recommendation.

Example:

```text
Predicted demand: 22 racks
Recommended prep: 25 racks
```

The safety buffer should be configurable.

For the first version, a simple buffer is acceptable, such as:

- fixed number of racks
- percentage above prediction

Document that the correct buffer depends on the relative cost of:

- waste
- stockouts
- lost sales

---

## Output Files

Save useful outputs to `outputs/`, including:

```text
daily_rib_sales.csv
model_comparison.csv
test_predictions.csv
rib_sales_over_time.png
sales_by_weekday.png
actual_vs_predicted.png
```

Optional:

```text
feature_importance.csv
```

---

## Code Quality

The code should be:

- readable
- commented
- modular
- easy to explain in an interview

Prefer functions such as:

```python
load_data()
filter_rib_sales()
build_daily_dataset()
create_features()
split_data()
build_baseline()
train_poisson_model()
train_tree_model()
evaluate_model()
plot_results()
main()
```

Avoid excessive abstraction.

Do not build unnecessary classes unless they clearly improve readability.

---

## Command-Line Usage

Prefer simple CLI usage such as:

```bash
python src/rib_forecast.py --input data/sales.csv
```

Optional arguments:

```bash
--date-column
--item-column
--quantity-column
--rib-keyword
--test-fraction
--prep-buffer
```

If the export schema is unknown, allow the first run to print detected columns and exit cleanly so mappings can be configured.

---

## README

Create a README explaining:

1. Business problem.
2. Dataset assumptions.
3. How to create and activate the `.venv`.
4. How to install dependencies.
5. How to run the script.
6. Features used.
7. Models compared.
8. Evaluation approach.
9. Known limitations.
10. How the prep recommendation is calculated.

Also include a short section titled:

## ML Concepts Practiced

Cover:

- feature engineering
- time-based validation
- avoiding leakage
- baseline models
- Poisson regression / GLMs
- gradient boosting
- MAE vs RMSE
- model complexity vs interpretability
- production/business tradeoffs

---

## Important Interview-Learning Goal

Write the code so the user can comfortably explain:

1. Why a time-based split was used instead of a random split.
2. Why a simple baseline was created first.
3. Why Poisson regression is appropriate for count-like demand.
4. Why a tree-based model might outperform a GLM.
5. Why a more complicated model is not automatically better.
6. How MAE and RMSE differ.
7. How leakage can occur with rolling features.
8. Why historical sales may underestimate true demand on stockout days.
9. How model performance connects to the actual cost of food waste and lost sales.

---

## First Implementation Priority

Build the simplest correct end-to-end version first:

```text
CSV
  ↓
clean/filter rib sales
  ↓
aggregate by day
  ↓
create calendar + lag features
  ↓
chronological train/test split
  ↓
weekday baseline
  ↓
PoissonRegressor
  ↓
HistGradientBoostingRegressor
  ↓
MAE/RMSE comparison
  ↓
plots + predictions + recommended prep quantity
```

Then iterate only where additional complexity provides clear value.
