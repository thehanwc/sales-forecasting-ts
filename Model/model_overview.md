# Model Overview — LightGBM

This document describes the modeling approach implemented in `model.ipynb`.

## 1) Problem Framing

We model the task as a **supervised regression** problem over a multi-entity daily time series.

**Goal:** predict `sales` for each row in `test.csv`, where a row is defined by:

- `date`
- `store_nbr`
- `family`
- `onpromotion`

The forecast horizon is **16 consecutive days** immediately after the last training date.

## 2) Core Modeling Choices

### A) Loss / Target Transform

The competition is evaluated with **RMSLE**, so the notebook trains in log space:

- Train target: `y_log = log1p(sales)`
- Validation metric: RMSE on `y_log` (equivalent to RMSLE on original scale)
- Inference: `sales_pred = expm1(y_pred_log)` then clip to `>= 0`

### B) Validation Strategy (Time-Based)

A strict time split is used:

- Training fold: all dates **before** the last 16 training days
- Validation fold: the **last 16 days** of training

This matches the competition’s forecasting horizon and avoids random leakage across time.

### C) Model Family

**LightGBM (GBDT)** is used because:

- It handles mixed feature types well
- It supports **native categorical features**
- It is strong on tabular time series when paired with lag/rolling features

## 3) Data Assembly & Cleaning

### A) Concatenate Train + Test (for feature engineering)

The notebook tags rows:

- `is_train_flag = 1` for train
- `is_train_flag = 0` for test

Then concatenates to build features consistently.

### B) Oil price cleaning

`oil.csv` contains missing values in `dcoilwtico`:

- Forward-fill by date (`ffill`)
- Any leading NaNs are set to `-1.0`

## 4) Feature Engineering (Leakage-Safe by Design)

The approach is explicitly **horizon-aware** to avoid using unknown future sales/transactions.

### A) Store metadata (static)

Joined from `stores.csv`:

- `city`, `state`, `type`, `cluster` (categoricals)

### B) Holiday features (calendar + locale-aware)

From `holidays_events.csv`:

- Exclude `transferred == True`
- Build flags at multiple scopes:
  - National (by date)
  - Regional (by date + state)
  - Local (by date + city)

Features include:

- `holiday_nat`, `holiday_reg`, `holiday_loc`
- `workday_nat`, `workday_reg`, `workday_loc`
- `is_holiday_any`, `is_workday_any`
- plus categorical descriptors (`*_type`, `*_desc`) where available

### C) Oil features (external signal)

From cleaned oil series:

- Lags: `oil_lag_1`, `oil_lag_7`, `oil_lag_14`
- Rollings (shifted to avoid same-day leakage):
  - `oil_roll_mean_7`
  - `oil_roll_mean_28`
  - `oil_roll_std_28`

### D) Transactions features (store-level demand proxy)

Transactions are **not provided for test dates** in this bundle, so the notebook uses only features that remain valid:

- Reindex to a full (store × date) grid
- Fill missing transactions with 0 (and track missingness with `txn_was_missing`)
- **But** raw `transactions` and `txn_was_missing` are dropped before training
- Only lag/anchored features are used

Lags (store-level, >= 16 days):

- `txn_lag_16`, `txn_lag_17`, `txn_lag_21`, `txn_lag_28`, `txn_lag_35`, `txn_lag_42`, `txn_lag_56`, `txn_lag_84`

Rolling stats anchored at `t-16`:

- `txn_roll_mean_{7,14,28}_at16`
- `txn_roll_std_{7,14,28}_at16`

### E) Date / time features

Generated from `date`:

- `year`, `month`, `day`, `dayofweek`, `dayofyear`, `weekofyear`
- `is_weekend`, `is_month_start/end`, `is_quarter_start/end`
- pay-day proxies: `is_payday_15`, `is_payday_eom`
- `time_idx` (days since first date)
- cyclical encodings:
  - `dow_sin`, `dow_cos`
  - `doy_sin`, `doy_cos`

### F) Sales history features (store_nbr × family)

Sales features must avoid test leakage. Because the test window is exactly the next 16 days, features are built to only reference values at least 16 days back.

Lags (>= 16 days):

- `sales_lag_{16,17,21,28,35,42,56,84,112,140,168,365}`

Rolling stats anchored at `t-16`:

- `sales_roll_mean_{7,14,28,56}_at16`
- `sales_roll_std_{7,14,28,56}_at16`

### G) Promotion features (known in test)

Promotions are available in test, so shorter lags are allowed:

- `promo_lag_{1,7,14,28}`
- rolling sums/means from `onpromotion` shifted by 1 day:
  - `promo_roll_sum_{7,14,28}`
  - `promo_roll_mean_{7,14,28}`

## 5) Categorical Handling

LightGBM is configured to use categorical splits via pandas `category` dtype.

To avoid category mismatch between train/test, the notebook:

- Unions category levels between `X` and `X_test`
- Ensures both share the same category set

## 6) Hyperparameter Tuning (Optuna)

The notebook runs a manual Optuna loop:

- **Sampler:** TPE (seeded)
- **Pruner:** MedianPruner (warmup steps)
- **Early stopping:** enabled during training trials
- **Constraint:** `num_leaves <= 2^max_depth - 1` to keep tree shape consistent

Parameters searched include:

- `max_depth`, `num_leaves`
- `min_data_in_leaf`
- `learning_rate`
- `feature_fraction`, `bagging_fraction`, `bagging_freq`
- `lambda_l1`, `lambda_l2`
- `min_gain_to_split`
- `extra_trees` (boolean)

The best trial yields:

- best RMSE on validation in log space
- best boosting iteration (used for final training)

## 7) Final Training & Inference

### Final training

- Train on **all training rows** using best tuned parameters
- Use the best iteration discovered during tuning

### Predictions

- `y_pred_log = model.predict(X_test)`
- `sales_pred = expm1(y_pred_log)`
- clip to `>= 0`
- write `submission.csv` with columns: `id`, `sales`
