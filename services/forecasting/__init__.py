"""Forecasting models package.

    services/forecasting/
        lag_regression.py   StandardScaler + LASSO + LinearRegression
        arima_model.py       ADF test + automatic (p, d, q) selection
        lstm_model.py         multi-feature LSTM (seq_len=30)

Each module exposes the same four functions: ``train(df)``, ``save``,
``load``, and ``predict_next`` — see model_selector.py for the training
orchestration that calls all three per ticker and persists everything
under models/.

``run_all_models`` is kept for backward compatibility with any code (or
notebook) that still wants to train + evaluate all three models in one
call without going through the persistence layer — this is what the
dashboard used to call directly, before model training was moved out of
Streamlit and into the pipeline (see services/model_selector.py and
ui/data.py).
"""
from __future__ import annotations

import pandas as pd

from services.evaluation import evaluate_naive

from . import arima_model, lag_regression, lstm_model

MODEL_LABELS = {
    "lag_reg": "Lag-Informed Regression",
    "arima": "ARIMA",
    "lstm": "LSTM",
    "naive": "Naive baseline",
}


def run_all_models(df: pd.DataFrame) -> dict:
    """Runs all three models plus the naive baseline on a validated OHLCV
    dataframe. Returns metrics + next-day predictions + backtest series.

    This trains on the fly and does not persist anything — prefer
    services.model_selector.train_and_select_all() for the production
    pipeline, which saves models to disk and caches these results for the
    dashboard to load without retraining.
    """
    lag_artifact, lag_metrics, lag_next, lag_backtest = lag_regression.train(df)
    _, _, arima_metrics, arima_next, arima_backtest = arima_model.train(df)
    lstm_artifact, lstm_metrics, lstm_next, lstm_backtest = lstm_model.train(df)
    naive_metrics = evaluate_naive(df)

    return {
        "metrics": {
            "lag_reg": lag_metrics,
            "arima": arima_metrics,
            "lstm": lstm_metrics,
            "naive": naive_metrics,
        },
        "next_close": {
            "lag": round(lag_next, 2),
            "arima": round(arima_next, 2),
            "lstm": round(lstm_next, 2),
        },
        "backtest30": lag_backtest[-30:] if len(lag_backtest) >= 30 else lag_backtest,
        "backtest_by_model": {
            "Lag-Informed Regression": lag_backtest,
            "ARIMA": arima_backtest,
            "LSTM": lstm_backtest,
        },
    }


__all__ = ["run_all_models", "MODEL_LABELS", "lag_regression", "arima_model", "lstm_model"]
