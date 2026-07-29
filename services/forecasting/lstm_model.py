"""Multi-feature LSTM forecasting.

Replaces the original Close-only LSTM with a model trained on multiple
inputs: Open, High, Low, Close, Volume, RSI, MACD, EMA, and moving
averages — all built by services/feature_engineering.py so the feature
definitions match the Lag-Informed Regression model exactly.

Requirements implemented:
  - Sequence length = 30
  - Chronological train/val/test split (no shuffling)
  - Early stopping on validation loss
  - Checkpointing of the best-validation-loss weights
  - Persistence to a single .pth file (state dict + normalization stats +
    architecture metadata, so ``load`` can reconstruct the network without
    any external context)

Training and inference are separate: ``train()`` fits + evaluates,
``save``/``load`` persist the artifact, and ``predict_next`` runs a
forward pass only — no retraining — which is what the dashboard calls.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from services.evaluation import compute_metrics
from services.feature_engineering import build_full_features

log = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except ImportError:  # pragma: no cover
    HAS_TORCH = False

SEQ_LEN = 30
FEATURE_COLUMNS = [
    "Open", "High", "Low", "Close", "Volume",
    "rsi_14", "macd", "ema_10", "ema_20", "ma_5", "ma_10",
]


class _LSTMNet(nn.Module if HAS_TORCH else object):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.1 if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def _build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Combine raw OHLCV with the shared technical-indicator set, keeping
    only the columns this model needs, and dropping warm-up rows with NaNs."""
    engineered = build_full_features(df)
    indicator_cols = [c for c in FEATURE_COLUMNS if c not in ("Open", "High", "Low", "Close", "Volume")]
    frame = df[["Open", "High", "Low", "Close", "Volume"]].join(engineered[indicator_cols])
    return frame.dropna().reset_index(drop=True)


def _make_sequences(X: np.ndarray, y: np.ndarray, seq_len: int):
    xs, ys = [], []
    for i in range(len(X) - seq_len):
        xs.append(X[i : i + seq_len])
        ys.append(y[i + seq_len])
    return np.array(xs), np.array(ys)


def _fallback_result(df: pd.DataFrame):
    """Used when torch isn't installed, or there isn't enough history for a
    30-step sequence model plus train/val/test splits."""
    close = df["Close"].values.astype("float32")
    metrics = compute_metrics(close[1:], close[:-1])
    next_close = float(close[-1])
    backtest = np.concatenate([[close[0]], close[:-1]]).tolist()
    return None, metrics, next_close, backtest


def train(df: pd.DataFrame, seq_len: int = SEQ_LEN, epochs: int = 200, patience: int = 12):
    """Returns (artifact, metrics, next_close, backtest_series).

    ``artifact`` is a plain dict (state dict + normalization stats +
    architecture metadata) rather than the live nn.Module, so it can be
    persisted with ``torch.save`` and reconstructed later without needing
    this exact object's class already instantiated.
    """
    frame = _build_feature_frame(df)

    if not HAS_TORCH or len(frame) < seq_len + 40:
        return _fallback_result(df)

    X_raw = frame[FEATURE_COLUMNS].values.astype("float32")
    y_raw = frame["Close"].values.astype("float32")

    x_mean, x_std = X_raw.mean(axis=0), X_raw.std(axis=0) + 1e-8
    y_mean, y_std = float(y_raw.mean()), float(y_raw.std()) + 1e-8

    X_norm = (X_raw - x_mean) / x_std
    y_norm = (y_raw - y_mean) / y_std

    X_seq, y_seq = _make_sequences(X_norm, y_norm, seq_len)

    n = len(X_seq)
    n_test = max(1, int(round(n * 0.15)))
    n_val = max(1, int(round(n * 0.15)))
    n_train = n - n_test - n_val
    if n_train < 10:
        return _fallback_result(df)

    X_train, y_train = X_seq[:n_train], y_seq[:n_train]
    X_val, y_val = X_seq[n_train : n_train + n_val], y_seq[n_train : n_train + n_val]
    X_test, y_test = X_seq[n_train + n_val :], y_seq[n_train + n_val :]

    torch.manual_seed(42)
    model = _LSTMNet(input_size=len(FEATURE_COLUMNS))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    X_train_t = torch.tensor(X_train)
    y_train_t = torch.tensor(y_train).unsqueeze(-1)
    X_val_t = torch.tensor(X_val)
    y_val_t = torch.tensor(y_val).unsqueeze(-1)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train_t)
        loss = loss_fn(pred, y_train_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.inference_mode():
            val_loss = loss_fn(model(X_val_t), y_val_t).item()

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}  # checkpoint
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:  # early stopping
                log.info("LSTM early stopping after %d epochs without improvement", patience)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.inference_mode():
        test_pred_norm = model(torch.tensor(X_test)).squeeze(-1).numpy()
        all_pred_norm = model(torch.tensor(X_seq)).squeeze(-1).numpy()
        last_window = torch.tensor(X_norm[-seq_len:]).unsqueeze(0)
        next_norm = model(last_window).item()

    test_pred = test_pred_norm * y_std + y_mean
    y_test_actual = y_test * y_std + y_mean
    metrics = compute_metrics(y_test_actual, test_pred)

    next_close = float(next_norm * y_std + y_mean)

    backtest_vals = (all_pred_norm * y_std + y_mean).tolist()
    backtest = [float(frame["Close"].iloc[0])] * seq_len + backtest_vals  # pad head to align lengths

    artifact = {
        "state_dict": model.state_dict(),
        "input_size": len(FEATURE_COLUMNS),
        "feature_columns": FEATURE_COLUMNS,
        "seq_len": seq_len,
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
    }
    return artifact, metrics, next_close, backtest


def save(artifact, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if artifact is None or not HAS_TORCH:
        # No-torch fallback path: nothing to persist.
        return
    torch.save(artifact, path)


def load(path):
    return torch.load(path, weights_only=False)


def predict_next(artifact, df: pd.DataFrame) -> float:
    """Forward pass only from an already-trained artifact — no
    retraining, used by the dashboard."""
    model = _LSTMNet(input_size=artifact["input_size"])
    model.load_state_dict(artifact["state_dict"])
    model.eval()

    frame = _build_feature_frame(df)
    seq_len = artifact["seq_len"]
    X_raw = frame[artifact["feature_columns"]].values.astype("float32")[-seq_len:]
    X_norm = (X_raw - artifact["x_mean"]) / artifact["x_std"]

    with torch.inference_mode():
        next_norm = model(torch.tensor(X_norm).unsqueeze(0)).item()

    return float(next_norm * artifact["y_std"] + artifact["y_mean"])
