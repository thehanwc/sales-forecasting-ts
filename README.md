# sales-forecasting-ts
*A LightGBM Time Series Forecasting Pipeline for Retail Store Sales*

* * *

## Overview

This repository contains an end-to-end time series forecasting workflow built for Kaggle’s **Store Sales - Time Series Forecasting** competition:

https://www.kaggle.com/competitions/store-sales-time-series-forecasting

The goal is to predict daily sales for thousands of **store × product-family** combinations for **Corporación Favorita** (Ecuador).  
This repo focuses on a leakage-aware feature engineering pipeline and a strong tabular baseline using **LightGBM** with **Optuna** hyperparameter tuning.

* * *

## Why This Project Is Useful

Time series competitions (especially “panel” time series with many entities) reward strong fundamentals:

- **Leakage-safe feature engineering** (lags/rolling windows aligned to the forecast horizon)
- **Time-based validation** that mirrors the test setup
- **Hybrid features** combining calendar signals, metadata, external drivers (oil), and historical behavior
- **Fast iteration** via a single notebook pipeline that produces a valid Kaggle submission

This project demonstrates those competencies in a compact, reproducible structure.

* * *

## What You’ll Find

### 1. `Model/`

Contains everything related to modeling and training.

- `[1] model.ipynb`  
  Jupyter Notebook covering:
  - Data ingestion
  - Cleaning
  - Feature engineering
  - Time-based validation
  - Optuna tuning + LightGBM training
  - Test inference + submission file generation

- `[2] model_overview.md`  
  Documentation covering:
  - Model design choices
  - Feature set details
  - Validation strategy
  - Tuning/training workflow

### 2. `Data/`

Contains the competition dataset bundle and dataset documentation.

- `[1] store-sales-time-series-forecasting.zip`  
  Kaggle dataset bundle (CSV files inside).

- `[2] data_overview.md`  
  Documentation covering:
  - File inventory + schema
  - Keys/relationships
  - Date ranges + row counts
  - Data quality notes and ingestion guidance

* * *

## Getting Started

1. **Prepare data**
   - Place the Kaggle dataset zip in `Data/` (or download it from Kaggle if your repo does not include it).
   - Extract the zip so the CSVs are accessible locally.

2. **Install dependencies**
   - Python libraries used in the notebook:
     - `pandas`, `numpy`
     - `lightgbm`
     - `optuna`
     - `jupyter`

3. **Run the notebook**
   - Open `Model/model.ipynb`
   - Update the `DATA_DIR` path in the notebook to point to your extracted CSVs
   - Execute cells top-to-bottom
   - The notebook writes a `submission.csv` in `DATA_DIR`

* * *

## Tools & Technologies

- Python
  - pandas, numpy
  - LightGBM
  - Optuna
- Jupyter Notebook

* * *

## Getting Help

If you encounter issues or want to extend the work:

- Open an issue in this repository’s Issues section.

* * *

## Maintainer & Contributor

Author / Maintainer:  
Han Wei Chang

Focus areas:
- Time Series Forecasting
- Feature Engineering
- Gradient Boosting Models
- Practical ML Workflows

This repository represents the author’s individual analytical contribution and is intended for portfolio / learning purposes.

* * *

## Citation

If you reference this work in academic or technical contexts, please cite it as:

> Chang, H. W. (2026). Store Sales - Time Series Forecasting (LightGBM baseline). GitHub repository.

* * *

## Notes on Data Usage

The dataset originates from Kaggle’s competition. Please ensure your usage complies with Kaggle’s dataset and competition terms.
