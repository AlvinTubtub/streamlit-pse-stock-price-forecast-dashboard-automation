"""Real model training/inference — replaces the original dashboard's
Math.random() mock generators.

Three models, matching the capstone's methodology:
  - Lag-Informed Regression : scikit-learn LinearRegression on lag/MA features
  - ARIMA                   : statsmodels ARIMA on the Close series
  - LSTM                    : a small PyTorch LSTM over a sliding window

Swap-in point for your own trained models
------------------------------------------
If you already have `.pkl` / `.pth` files from your actual capstone
training run, load them in `load_pretrained_models()` below instead of
calling `train_lag_regression` / `train_arima` / `train_lstm` on the fly.
Everything downstream (metrics, predictions, chart data) expects the same
return shapes documented on each function, so the rest of the app does not
need to change.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .feature_engineering import build_lag_features, train_test_split_frame

try:
    from statsmodels.tsa.arima.model import ARIMA

    HAS_STATSMODELS = True
except ImportError:  # pragma: no cover
    HAS_STATSMODELS = False

try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except ImportError:  # pragma: no cover
    HAS_TORCH = False


def _naive_mae(y_true: np.ndarray) -> float:
    """Denominator for MASE: the mean absolute error of a naive one-step
    (yesterday's value) forecast over the same series."""
    y_true = np.asarray(y_true, dtype=float)
    if len(y_true) < 2:
        return 1e-8
    return float(np.mean(np.abs(np.diff(y_true)))) or 1e-8


def _metrics_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    mase = mae / _naive_mae(y_true)
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else 0.0
    return {
        "rmse": f"{rmse:.4f}",
        "mae": f"{mae:.4f}",
        "mase": f"{mase:.4f}",
        "r2": f"{r2:.4f}",
    }


# --------------------------------------------------------------------------
# Naive baseline: predict tomorrow's close = today's close
# --------------------------------------------------------------------------
def evaluate_naive(df: pd.DataFrame) -> dict:
    close = df["Close"].values
    y_true = close[1:]
    y_pred = close[:-1]
    return _metrics_dict(y_true, y_pred)


# --------------------------------------------------------------------------
# Lag-Informed Regression
# --------------------------------------------------------------------------
def train_lag_regression(df: pd.DataFrame):
    features = build_lag_features(df)
    train, test = train_test_split_frame(features)

    x_cols = [c for c in features.columns if c != "target"]
    model = LinearRegression()
    model.fit(train[x_cols], train["target"])

    test_pred = model.predict(test[x_cols])
    metrics = _metrics_dict(test["target"].values, test_pred)

    # Refit on all available rows so the next-day forecast uses the most
    # recent data too.
    full = features.dropna()
    model.fit(full[x_cols], full["target"])
    last_row = features.iloc[[-1]][x_cols].ffill()
    next_close = float(model.predict(last_row)[0])

    # Backtest series for the "actual vs predicted" chart (last N test points)
    backtest = model.predict(features[x_cols].bfill())

    return {"metrics": metrics, "next_close": next_close, "backtest": backtest.tolist()}


# --------------------------------------------------------------------------
# ARIMA
# --------------------------------------------------------------------------
def train_arima(df: pd.DataFrame, order=(5, 1, 0)):
    close = df["Close"]
    n_test = max(1, int(round(len(close) * 0.15)))
    train_series = close.iloc[: len(close) - n_test]
    test_series = close.iloc[len(close) - n_test :]

    if not HAS_STATSMODELS:
        # Fallback: naive-ish drift model, clearly documented, so the app
        # still runs in environments without statsmodels installed.
        y_pred = train_series.iloc[-1] + np.cumsum(
            np.full(len(test_series), train_series.diff().mean())
        )
        metrics = _metrics_dict(test_series.values, y_pred)
        next_close = float(close.iloc[-1] + train_series.diff().mean())
        backtest = close.shift(1).bfill().tolist()
        return {"metrics": metrics, "next_close": next_close, "backtest": backtest}

    model = ARIMA(train_series, order=order).fit()
    forecast = model.forecast(steps=n_test)
    metrics = _metrics_dict(test_series.values, forecast.values)

    full_model = ARIMA(close, order=order).fit()
    next_close = float(full_model.forecast(steps=1).iloc[0])

    # In-sample one-step-ahead fitted values, for the backtest chart
    backtest = full_model.predict(start=1, end=len(close) - 1, typ="levels")
    backtest = pd.concat([pd.Series([close.iloc[0]]), backtest]).reset_index(drop=True)

    return {"metrics": metrics, "next_close": next_close, "backtest": backtest.tolist()}


# --------------------------------------------------------------------------
# LSTM
# --------------------------------------------------------------------------
class _LSTMNet(nn.Module if HAS_TORCH else object):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def _make_windows(series: np.ndarray, window: int):
    xs, ys = [], []
    for i in range(len(series) - window):
        xs.append(series[i : i + window])
        ys.append(series[i + window])
    return np.array(xs), np.array(ys)


def train_lstm(df: pd.DataFrame, window: int = 10, epochs: int = 60):
    close = df["Close"].values.astype("float32")

    if not HAS_TORCH or len(close) < window + 20:
        # Fallback for environments without torch, or too little history
        y_pred_next = float(close[-1])
        metrics = evaluate_naive(df)
        backtest = np.concatenate([[close[0]], close[:-1]]).tolist()
        return {"metrics": metrics, "next_close": y_pred_next, "backtest": backtest}

    mean, std = close.mean(), close.std() + 1e-8
    norm = (close - mean) / std

    X, y = _make_windows(norm, window)
    n_test = max(1, int(round(len(X) * 0.15)))
    X_train, X_test = X[: len(X) - n_test], X[len(X) - n_test :]
    y_train, y_test = y[: len(y) - n_test], y[len(y) - n_test :]

    X_train_t = torch.tensor(X_train).unsqueeze(-1)
    y_train_t = torch.tensor(y_train).unsqueeze(-1)
    X_test_t = torch.tensor(X_test).unsqueeze(-1)

    torch.manual_seed(42)
    model = _LSTMNet()
    optim = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        optim.zero_grad()
        pred = model(X_train_t)
        loss = loss_fn(pred, y_train_t)
        loss.backward()
        optim.step()

    model.eval()
    with torch.inference_mode():
        test_pred_norm = model(X_test_t).squeeze(-1).numpy()
        all_pred_norm = model(torch.tensor(X).unsqueeze(-1)).squeeze(-1).numpy()

        last_window = torch.tensor(norm[-window:]).float().unsqueeze(0).unsqueeze(-1)
        next_norm = model(last_window).item()

    test_pred = test_pred_norm * std + mean
    y_test_actual = y_test * std + mean
    metrics = _metrics_dict(y_test_actual, test_pred)

    next_close = float(next_norm * std + mean)

    backtest_vals = (all_pred_norm * std + mean).tolist()
    backtest = [float(close[0])] * window + backtest_vals  # pad head to align lengths

    return {"metrics": metrics, "next_close": next_close, "backtest": backtest}


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def run_all_models(df: pd.DataFrame) -> dict:
    """Runs all three models plus the naive baseline on a validated OHLCV
    dataframe. Returns metrics + next-day predictions + backtest series."""
    lag = train_lag_regression(df)
    arima = train_arima(df)
    lstm = train_lstm(df)
    naive_metrics = evaluate_naive(df)

    return {
        "metrics": {
            "lag_reg": lag["metrics"],
            "arima": arima["metrics"],
            "lstm": lstm["metrics"],
            "naive": naive_metrics,
        },
        "next_close": {
            "lag": round(lag["next_close"], 2),
            "arima": round(arima["next_close"], 2),
            "lstm": round(lstm["next_close"], 2),
        },
        "backtest30": lag["backtest"][-30:] if len(lag["backtest"]) >= 30 else lag["backtest"],
        # Per-model backtest series (same length as df), used by the
        # "actual vs predicted" chart so every model — not just lag reg —
        # can be plotted against the real actual/naive series.
        "backtest_by_model": {
            "Lag-Informed Regression": lag["backtest"],
            "ARIMA": arima["backtest"],
            "LSTM": lstm["backtest"],
        },
    }

