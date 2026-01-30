# Data Overview — Store Sales (Kaggle Competition Bundle)

This repository includes the dataset archive:

- `store-sales-time-series-forecasting.zip`

After extracting, you should have the following CSV files:

```text
train.csv
test.csv
stores.csv
oil.csv
holidays_events.csv
transactions.csv
sample_submission.csv
```

## Date Ranges (Observed in This Bundle)

- **train.csv:** 2013-01-01 → 2017-08-15
- **test.csv:**  2017-08-16 → 2017-08-31 (16 days)

Auxiliary tables:

- **oil.csv:** 2013-01-01 → 2017-08-31
- **transactions.csv:** 2013-01-01 → 2017-08-15
- **holidays_events.csv:** spans beyond the modeling range (includes 2017 dates)

## File-by-File Dictionary

### 1) `train.csv`

Main training table.

**Columns**

- `id` (int): unique row identifier
- `date` (date): daily timestamp
- `store_nbr` (int/categorical): store identifier
- `family` (categorical): product family label
- `sales` (float): target (unit sales)
- `onpromotion` (int): number of items in the family on promotion

**Notes**

- Multi-entity time series: one row per (date, store_nbr, family)
- Used to build lag/rolling history features

### 2) `test.csv`

Rows to predict.

**Columns**

- `id`, `date`, `store_nbr`, `family`, `onpromotion`

**Notes**

- Same schema as train **minus** `sales`
- Forecast horizon is the next 16 days after the last train date

### 3) `sample_submission.csv`

Submission template.

**Columns**

- `id`
- `sales`

### 4) `stores.csv`

Store metadata.

**Columns**

- `store_nbr`
- `city` (categorical)
- `state` (categorical)
- `type` (categorical): store type label
- `cluster` (int/categorical): grouping of similar stores

**Usage**

- Join onto train/test by `store_nbr`

### 5) `oil.csv`

Daily oil price series.

**Columns**

- `date`
- `dcoilwtico` (float): daily oil price (WTI)

**Notes**

- Contains missing values (NaNs)
- Common practice: forward-fill by date

### 6) `holidays_events.csv`

Holiday and events calendar.

**Columns**

- `date`
- `type`: holiday type label (e.g., Holiday, Event, Work Day, etc.)
- `locale`: scope (National / Regional / Local)
- `locale_name`: name for the scope (country, state, or city label depending on `locale`)
- `description`: event/holiday description
- `transferred` (bool): whether the holiday was moved/shifted

**Usage notes**

- Many solutions exclude `transferred == True` and rely on the non-transferred entry
- Locale-aware joins allow building separate national/regional/local flags

### 7) `transactions.csv`

Store-level daily transaction counts.

**Columns**

- `date`
- `store_nbr`
- `transactions` (int): count of transactions for that store/day

**Important note**

- Transaction counts are not provided for test dates in this bundle (they end at the train end date).
- Leakage-safe modeling typically uses only lags/anchored features for transactions.

## Expected Modeling Implications

- **Sales are heavy-tailed** across families; log transforms are common.
- **Promotions** are a strong short-horizon driver and are known in test.
- **Oil price** can act as an external regressor for macro conditions.
- **Transactions** help as a store-traffic proxy, but require careful feature design due to missing test coverage.
