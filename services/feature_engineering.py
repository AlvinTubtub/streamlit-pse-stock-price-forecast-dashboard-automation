"""Lag-feature engineering for next-day close-price forecasting.

Shared by the Lag-Informed Regression model and by the LSTM's input
windowing, so both models see a consistent feature definition.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LAG_COLUMNS = ["lag_1", "lag_2", "lag_3", "lag_5", "ma_5", "ma_10", "volume_ma_5"]


def build_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Given a validated OHLCV dataframe (sorted by Date), return a dataframe
    of lag/moving-average features plus the target (next day's Close).

    Rows without enough history for the longest lag/window are dropped, and
    the last row (no known next-day target) is dropped from the *training*
    frame — callers that need to predict the next unseen day should build
    features on the full series and take the final row separately.
    """
    out = pd.DataFrame(index=df.index)
    close = df["Close"]

    out["lag_1"] = close.shift(1)
    out["lag_2"] = close.shift(2)
    out["lag_3"] = close.shift(3)
    out["lag_5"] = close.shift(5)
    out["ma_5"] = close.shift(1).rolling(5).mean()
    out["ma_10"] = close.shift(1).rolling(10).mean()
    out["volume_ma_5"] = df["Volume"].shift(1).rolling(5).mean()
    out["target"] = close

    return out


def train_test_split_frame(features: pd.DataFrame, test_frac: float = 0.15):
    """Chronological split — never shuffles, to avoid look-ahead leakage."""
    features = features.dropna()
    n = len(features)
    n_test = max(1, int(round(n * test_frac)))
    train = features.iloc[: n - n_test]
    test = features.iloc[n - n_test :]
    return train, test
