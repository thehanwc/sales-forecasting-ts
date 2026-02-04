# sales-forecasting-ts
A LightGBM Store Sales Time Series Forecasting Pipeline

Forecast daily unit sales for product families across Favorita stores using a global ML model with time-series feature engineering.

## Project Overview
This repository contains an end-to-end pipeline (implemented in `model.ipynb`) for the Kaggle **Store Sales - Time Series Forecasting** competition. The goal is to predict **sales** for the dates in `test.csv` (the test period immediately follows the last date in `train.csv`).  
The competition is evaluated using **RMSLE (Root Mean Squared Logarithmic Error)**, which motivates training on `log1p(sales)` and converting predictions back with `expm1`.  

## Key Ideas / Method
This solution uses a **global LightGBM regressor** trained over all `(store_nbr, family)` series, with:
- **Date features**: year, month, day, day-of-week, week-of-year, day-of-year, month boundary flags, plus cyclic encodings (sin/cos for DOW and DOY).
- **Exogenous joins**:
  - `stores.csv` for store metadata (city/state/type/cluster)
  - `oil.csv` merged on date
  - `holidays_events.csv` transformed into national/regional/local “event counters” by type (holiday/event/additional/bridge/workday)
- **Lag/rolling features**:
  - Promotion lags: `promo_lag_{1,7,14,28}`, `promo_rollmean_{7,28}`
  - Sales lags (on log scale): `sales_lag_{1,7,14,28}`, `sales_rollmean_{7,28}`
- **Walk-forward validation**:
  - Rolling-origin folds with contiguous validation blocks
  - Recursive (auto-regressive) prediction during validation and for test horizon
- **Hyperparameter tuning**:
  - Optuna tuning over recent data slice for speed, then train final LightGBM on all features.

## Repository Contents
- `model.ipynb` — Main notebook: loading, preprocessing, EDA, feature engineering, training, Optuna tuning, and submission creation.
- `data/` — Local copy of Kaggle data files (recommended to keep out of git for public repos).

## Data
Expected files (from the Kaggle competition):
- `train.csv` — training target `sales` with `date`, `store_nbr`, `family`, `onpromotion`
- `test.csv` — same keys without `sales`
- `stores.csv` — store metadata
- `oil.csv` — daily oil price (WTI)
- `holidays_events.csv` — holidays and events (with locale granularity)
- `transactions.csv` — daily transactions per store (optional; not used in the current notebook unless added)
- `sample_submission.csv` — submission format

## Reproduce Results
1. Place Kaggle data in `./data/` (or update the paths in the notebook).
2. Open and run the notebook:
   - `model.ipynb`
3. The notebook will generate a submission by:
   - training a final LightGBM model on `log1p(sales)`
   - producing recursive predictions over the test horizon
   - writing `submission.csv` in the required format

## Output
- `submission.csv` — final predictions aligned to `sample_submission.csv` schema.

## Implementation Notes
- **Target transform**: train on `log1p(sales)`; predictions are clipped at 0 and inverted using `expm1`.
- **Missing oil values**: forward/back filled and/or median-imputed.
- **Holiday engineering**: events are converted into count features at national / regional (state) / local (city) levels.
- **Categoricals**: `store_nbr`, `family`, and store metadata are treated as categorical features for LightGBM.
