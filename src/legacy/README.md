# Legacy modules (yfinance + RandomForest)

These target the previous design: 8 yfinance spot coins, 6 factors, a two-layer
RandomForest. They are kept for reference only and **will not run against the
current `src/config.py`**, which no longer defines `LAYERS`, `TICKERS`,
`RF_PARAMS` or `TIMING_THRESHOLD`.

Superseded by, respectively:

| legacy | current |
|---|---|
| `data_processing.py` | `binance_data.py` |
| `features.py` | `factors.py` |
| `model_pipeline.py` | `models.py` |
| `backtest.py` | `perp_backtest.py` |
| `daily_inference.py` | `perp_inference.py` |
| `trading_plan.py` | `perp_plan.py` |
| `pipeline.py` | `perp_pipeline.py` |

The old spot CSVs are still in `data/*.csv` and the old models in
`models/rf_*.joblib` / `models/production_rf_model.joblib`. Nothing reads them.
