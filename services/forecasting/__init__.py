"""Forecasting models package.

    services/forecasting/
        lag_regression.py   StandardScaler + LASSO + LinearRegression
        arima_model.py       ADF test + automatic (p, d, q) selection
        lstm_model.py         multi-feature LSTM (seq_len=30)

Each module exposes the same four functions: ``train(df)``, ``save``,
``load``, and ``predict_next`` — see services/model_selector.py for the
training orchestration that calls all three per ticker and persists
everything under models/ + prediction_cache/.

Training only ever runs inside the automated pipeline (run_pipeline.py,
triggered by .github/workflows/update_pipeline.yml) — never inside
Streamlit, which only loads what model_selector.py already cached.
"""
from __future__ import annotations

from . import arima_model, lag_regression, lstm_model

MODEL_LABELS = {
    "lag_reg": "Lag-Informed Regression",
    "arima": "ARIMA",
    "lstm": "LSTM",
    "naive": "Naive baseline",
}

__all__ = ["MODEL_LABELS", "lag_regression", "arima_model", "lstm_model"]
