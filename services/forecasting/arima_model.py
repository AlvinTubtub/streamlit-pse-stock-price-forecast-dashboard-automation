"""ARIMA forecasting with automatic (p, d, q) selection.

Replaces the old hardcoded ``ARIMA(5, 1, 0)`` with:

  1. An Augmented Dickey-Fuller test to check stationarity of the Close
     series (informs the differencing order ``d``).
  2. Automatic order selection — via ``pmdarima.auto_arima`` when the
     optional ``pmdarima`` dependency is installed, otherwise a small
     AIC-ranked grid search over (p, d, q) using statsmodels directly, so
     the pipeline still runs end-to-end without the extra dependency.

As with the other models, training and inference are separate: ``train()``
fits + evaluates, ``save``/``load`` persist the fitted statsmodels results
object with joblib, and ``predict_next`` forecasts one step ahead from an
already-fitted model with no retraining.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from services.evaluation import compute_metrics

log = logging.getLogger(__name__)

try:
    import joblib
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller

    HAS_STATSMODELS = True
except ImportError:  # pragma: no cover
    HAS_STATSMODELS = False

try:
    import pmdarima as pm

    HAS_PMDARIMA = True
except ImportError:  # pragma: no cover
    HAS_PMDARIMA = False

DEFAULT_ORDER = (5, 1, 0)  # last-resort fallback, matches the original hardcoded order
MAX_P = 5
MAX_Q = 5
MAX_D = 2


def is_stationary(series: pd.Series, alpha: float = 0.05) -> bool:
    """Augmented Dickey-Fuller test: True if the series is already
    stationary (p-value below alpha), meaning d=0 is a reasonable start."""
    if not HAS_STATSMODELS:
        return False
    try:
        _, pvalue, *_ = adfuller(series.dropna())
        return bool(pvalue < alpha)
    except Exception:  # pragma: no cover - defensive
        log.warning("ADF test failed; assuming non-stationary.", exc_info=True)
        return False


def _select_order(train_series: pd.Series) -> tuple[int, int, int]:
    """Automatic (p, d, q) selection."""
    if HAS_PMDARIMA:
        try:
            auto_model = pm.auto_arima(
                train_series,
                start_p=0, start_q=0, max_p=MAX_P, max_q=MAX_Q, max_d=MAX_D,
                seasonal=False, stepwise=True, suppress_warnings=True, error_action="ignore",
            )
            return tuple(auto_model.order)
        except Exception:  # pragma: no cover - defensive
            log.warning("pmdarima.auto_arima failed; falling back to grid search.", exc_info=True)

    # Fallback: small AIC-ranked grid search, using the ADF test only to
    # decide which differencing order to start from.
    d_guess = 0 if is_stationary(train_series) else 1
    best_order, best_aic = DEFAULT_ORDER, np.inf
    for d in {d_guess, min(d_guess + 1, MAX_D)}:
        for p in range(0, MAX_P + 1):
            for q in range(0, MAX_Q + 1):
                if p == 0 and q == 0:
                    continue
                try:
                    fitted = ARIMA(train_series, order=(p, d, q)).fit()
                except Exception:
                    continue
                if fitted.aic < best_aic:
                    best_aic, best_order = fitted.aic, (p, d, q)
    return best_order


def train(df: pd.DataFrame):
    """Returns (fitted_model, order, metrics, next_close, backtest_series)."""
    close = df["Close"]
    n_test = max(1, int(round(len(close) * 0.15)))
    train_series = close.iloc[: len(close) - n_test]
    test_series = close.iloc[len(close) - n_test :]

    if not HAS_STATSMODELS:
        # Fallback: naive-ish drift model, so the app still runs in
        # environments without statsmodels installed.
        y_pred = train_series.iloc[-1] + np.cumsum(np.full(len(test_series), train_series.diff().mean()))
        metrics = compute_metrics(test_series.values, y_pred)
        next_close = float(close.iloc[-1] + train_series.diff().mean())
        backtest = close.shift(1).bfill().tolist()
        return None, DEFAULT_ORDER, metrics, next_close, backtest

    order = _select_order(train_series)
    log.info("Selected ARIMA order %s", order)

    model = ARIMA(train_series, order=order).fit()
    forecast = model.forecast(steps=n_test)
    metrics = compute_metrics(test_series.values, forecast.values)

    full_model = ARIMA(close, order=order).fit()
    next_close = float(full_model.forecast(steps=1).iloc[0])

    # In-sample one-step-ahead fitted values, for the backtest chart
    backtest = full_model.predict(start=1, end=len(close) - 1, typ="levels")
    backtest = pd.concat([pd.Series([close.iloc[0]]), backtest]).reset_index(drop=True).tolist()

    return full_model, order, metrics, next_close, backtest


def save(model, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if model is None:
        # No-statsmodels fallback path: nothing to persist.
        return
    joblib.dump(model, path)


def load(path):
    return joblib.load(path)


def predict_next(model) -> float:
    """Forecast one step ahead from an already-fitted model — no
    retraining, used by the dashboard."""
    return float(model.forecast(steps=1).iloc[0])
