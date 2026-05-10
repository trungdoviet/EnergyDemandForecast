# Forecasting Taiwan's Monthly Electricity Demand


| | |
|---|---|
| Task | Regression — predict next month's electricity demand for Taiwan (TWh) |
| Baseline model | **Lasso** (`LassoCV`) — RMSE **0.586** TWh, MAPE **1.92 %**, DirAcc **94.9 %** ← best |
| Advanced model | **LightGBM** (TimeSeriesSplit walk-forward CV) — RMSE 0.936, MAPE 3.19 %, DirAcc 88.1 % |
| Reference | Seasonal-Naive (lag-12) — RMSE 0.886, MAPE 2.88 % |
| Test window | 60 months, 2021-01 → 2026-03 |
| Data sources | Ember (electricity) + IMF (macro) + NASA POWER (satellite temperature) |

## Files

```
EnergyDemandForecast/
├── README.md                    ← this file
├── notebook.ipynb               ← executed notebook, 0 errors, 37 cells
├── results_summary.csv          ← model comparison table (auto-generated)
├── test_predictions.csv         ← per-month test forecasts + errors (auto-generated)
├── fig_eda.png                  ← trend / seasonality / temperature-vs-demand
├── fig_split.png                ← time-aware train/test split visualisation
├── fig_forecast.png             ← actual vs Lasso vs LightGBM on test window
├── fig_lasso_coef.png           ← Lasso retained-feature coefficients
├── fig_lgbm_imp.png             ← LightGBM gain importance
├── fig_errors.png               ← error analysis by calendar month + over time
└── data/
    ├── _prepare_clean_data.py
    ├── tw_electricity_monthly.csv
    ├── tw_temperature_monthly.csv
    ├── tw_macro_annual.csv
    ├── ember_monthly_full.csv → ../../V2/data/...   (symlink, 67 MB)
    └── 6× raw JSON downloads (IMF + NASA)
```

## Reproduce

```bash
# 1) Install dependencies
python3 -m pip install --user pandas numpy scikit-learn matplotlib seaborn lightgbm jupyter
brew install libomp                  # macOS only

# 2) (Optional) rebuild clean CSVs from raw downloads
cd data && python3 _prepare_clean_data.py && cd ..

# 3) Run the notebook
jupyter nbconvert --to notebook --execute notebook.ipynb \
                  --output notebook.ipynb \
                  --ExecutePreprocessor.timeout=300
```

## How the notebook maps to the rubric

| Rubric requirement | Notebook section |
|---|---|
| Problem framing (target, horizon, regression vs classification, economic relevance) | §1 |
| At least 2 data types | §3 (three sources: Ember + IMF + NASA POWER) |
| Time-aware split, no look-ahead | §8 (chronological 80/20 + visualisation) |
| 1 baseline model | §10 (Lasso) |
| 1 advanced model | §11 (LightGBM with walk-forward CV) |
| Justify model selection & compare | §10, §11, §12 |
| Regression metrics (MAE, RMSE, MAPE) | §9 (helper) and §12 (table) |
| Feature importance / coefficients | §14 (Lasso coefs + LightGBM gain) |
| Overfitting assessment | §16.1 (CV vs test comparison) |
| Structural-break discussion | §16.2 |
| Ethical implications | §16.4 |
| Limitations of AI in economics | §16.3 |

## Headline takeaway

> Lasso wins (MAPE 1.92 %) because monthly national electricity demand is genuinely linear-additive in the engineered features (lag-12 seasonality + cooling-degree temperature + rolling trend). LightGBM is not overfitting — its CV and test errors agree — it is simply the wrong inductive bias for this signal shape. The right model for an economic forecasting task is the one whose inductive bias matches the data-generating process.
