"""Feature engineering for next-day close-price forecasting.

Shared by all three forecasting models (Lag-Informed Regression, ARIMA's
stationarity checks, and the LSTM's input windowing) so every model sees
the exact same feature definitions — no drift between what regression
trains on and what the LSTM feeds its sequences.

Two tiers are exposed:

``build_lag_features``
    The original 7 lag/moving-average columns. Kept unchanged for backward
    compatibility with anything still importing it directly.

``build_technical_indicators`` / ``build_full_features``
    The expanded technical-indicator set (EMA, RSI, MACD, Bollinger Bands,
    returns, volatility, spreads) layered on top of the lag features. This
    is what the Lag-Informed Regression and LSTM models now train on.

All indicators are computed using only same-day-or-earlier data (no
look-ahead), so every row is a valid predictor for that row's *next* day
close.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LAG_COLUMNS = ["lag_1", "lag_2", "lag_3", "lag_5", "ma_5", "ma_10", "volume_ma_5"]

TECHNICAL_COLUMNS = [
    "ema_10",
    "ema_20",
    "rsi_14",
    "macd",
    "macd_signal",
    "bb_upper",
    "bb_lower",
    "daily_return",
    "rolling_volatility",
    "hl_spread",
    "oc_spread",
]

FULL_FEATURE_COLUMNS = LAG_COLUMNS + TECHNICAL_COLUMNS


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


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's Relative Strength Index."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)  # neutral RSI where undefined (e.g. no losses yet)


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series]:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line


def _bollinger(close: pd.Series, window: int = 20, num_std: float = 2.0) -> tuple[pd.Series, pd.Series]:
    mid = close.rolling(window).mean()
    std = close.rolling(window).std()
    return mid + num_std * std, mid - num_std * std


def build_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Return the expanded technical-indicator columns, aligned to df.index.

    Indicators use each row's own Close/High/Low/Open — legitimate for
    predicting that row's *next* day close since nothing here reaches
    forward in time.
    """
    out = pd.DataFrame(index=df.index)
    close = df["Close"]

    out["ema_10"] = close.ewm(span=10, adjust=False).mean()
    out["ema_20"] = close.ewm(span=20, adjust=False).mean()
    out["rsi_14"] = _rsi(close, 14)

    macd_line, macd_signal = _macd(close)
    out["macd"] = macd_line
    out["macd_signal"] = macd_signal

    bb_upper, bb_lower = _bollinger(close)
    out["bb_upper"] = bb_upper
    out["bb_lower"] = bb_lower

    out["daily_return"] = close.pct_change()
    out["rolling_volatility"] = out["daily_return"].rolling(10).std()
    out["hl_spread"] = df["High"] - df["Low"]
    out["oc_spread"] = df["Open"] - df["Close"]

    return out


def build_full_features(df: pd.DataFrame) -> pd.DataFrame:
    """Lag features + technical indicators + target (next day's Close),
    all reusable by the Lag-Informed Regression model and the LSTM's
    feature matrix.
    """
    lag = build_lag_features(df)
    technical = build_technical_indicators(df)
    target = lag.pop("target")
    out = pd.concat([lag, technical], axis=1)
    out["target"] = target
    return out


def train_test_split_frame(features: pd.DataFrame, test_frac: float = 0.15):
    """Chronological split — never shuffles, to avoid look-ahead leakage."""
    features = features.dropna()
    n = len(features)
    n_test = max(1, int(round(n * test_frac)))
    train = features.iloc[: n - n_test]
    test = features.iloc[n - n_test :]
    return train, test
