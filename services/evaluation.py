"""Shared model-evaluation utilities.

One metrics implementation used by every forecasting model (Lag-Informed
Regression, ARIMA, LSTM) and by the naive baseline, so RMSE/MAE/MAPE/R²
are always computed the same way regardless of which model produced the
predictions.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

log = logging.getLogger(__name__)


def _naive_mae(y_true: np.ndarray) -> float:
    """Denominator for MASE: the mean absolute error of a naive one-step
    (yesterday's value) forecast over the same series."""
    y_true = np.asarray(y_true, dtype=float)
    if len(y_true) < 2:
        return 1e-8
    return float(np.mean(np.abs(np.diff(y_true)))) or 1e-8


def compute_metrics(y_true, y_pred) -> dict:
    """RMSE, MAE, MAPE, R² (plus MASE, kept for the existing dashboard's
    "vs. naive" comparisons) — the four headline metrics requested by the
    capstone methodology, all as formatted strings for direct display.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    mase = mae / _naive_mae(y_true)

    nonzero = y_true != 0
    if nonzero.any():
        mape = float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100)
    else:
        mape = 0.0

    r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else 0.0

    return {
        "rmse": f"{rmse:.4f}",
        "mae": f"{mae:.4f}",
        "mase": f"{mase:.4f}",
        "mape": f"{mape:.4f}",
        "r2": f"{r2:.4f}",
    }


def evaluate_naive(df: pd.DataFrame) -> dict:
    """Baseline: predict tomorrow's close = today's close."""
    close = df["Close"].values
    y_true = close[1:]
    y_pred = close[:-1]
    return compute_metrics(y_true, y_pred)


def build_comparison_table(metrics_by_model: dict[str, dict], labels: dict[str, str] | None = None) -> pd.DataFrame:
    """Turn {"lag_reg": {...}, "arima": {...}, "lstm": {...}, "naive": {...}}
    into a tidy comparison table — one row per model, ranked by RMSE.
    """
    labels = labels or {}
    rows = []
    for key, metrics in metrics_by_model.items():
        rows.append({
            "Model": labels.get(key, key),
            "RMSE": float(metrics["rmse"]),
            "MAE": float(metrics["mae"]),
            "MAPE": float(metrics["mape"]),
            "R2": float(metrics["r2"]),
        })
    table = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    return table


def select_best_model(metrics_by_model: dict[str, dict], candidate_keys: list[str]) -> str:
    """Returns the key (from candidate_keys) with the lowest RMSE."""
    return min(candidate_keys, key=lambda k: float(metrics_by_model[k]["rmse"]))
