"""Lag-Informed Regression: StandardScaler -> LASSO feature selection ->
LinearRegression, trained on the full lag + technical-indicator feature set
(services/feature_engineering.py), predicting next-day Close.

Training and inference are deliberately separate (see model_selector.py /
services/pdf_pipeline): ``train()`` fits and evaluates a model, ``save()``
/``load()`` persist it with joblib, and ``predict_next()`` produces a
forecast from an already-trained artifact without any retraining — the
Streamlit dashboard only ever calls ``load`` + ``predict_next``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.preprocessing import StandardScaler

from services.evaluation import compute_metrics
from services.feature_engineering import build_full_features, train_test_split_frame

log = logging.getLogger(__name__)


@dataclass
class LagRegressionArtifact:
    """Everything needed to reproduce a prediction without retraining."""

    scaler: StandardScaler
    model: LinearRegression
    all_features: list[str] = field(default_factory=list)
    selected_features: list[str] = field(default_factory=list)


def _select_features_lasso(X_train_scaled: np.ndarray, y_train: np.ndarray, feature_names: list[str]) -> list[str]:
    """LASSO (cross-validated alpha) picks the feature subset with non-zero
    coefficients. Falls back to the full feature set if LASSO zeroes out
    everything (can happen on very short/noisy histories)."""
    try:
        lasso = LassoCV(cv=5, random_state=42, max_iter=10_000).fit(X_train_scaled, y_train)
        mask = lasso.coef_ != 0
    except Exception:  # pragma: no cover - defensive: never let selection crash training
        log.warning("LassoCV feature selection failed; falling back to all features.", exc_info=True)
        mask = np.ones(len(feature_names), dtype=bool)

    if not mask.any():
        mask = np.ones(len(feature_names), dtype=bool)

    return [name for name, keep in zip(feature_names, mask) if keep]


def train(df: pd.DataFrame) -> tuple[LagRegressionArtifact, dict, float, list[float]]:
    """Returns (artifact, metrics, next_close, backtest_series)."""
    features = build_full_features(df)
    train_df, test_df = train_test_split_frame(features)
    x_cols = [c for c in features.columns if c != "target"]

    scaler = StandardScaler().fit(train_df[x_cols])
    X_train_scaled = scaler.transform(train_df[x_cols])
    X_test_scaled = scaler.transform(test_df[x_cols])

    selected = _select_features_lasso(X_train_scaled, train_df["target"].values, x_cols)
    sel_idx = [x_cols.index(c) for c in selected]

    model = LinearRegression().fit(X_train_scaled[:, sel_idx], train_df["target"])
    test_pred = model.predict(X_test_scaled[:, sel_idx])
    metrics = compute_metrics(test_df["target"].values, test_pred, y_train=train_df["target"].values)

    # Refit on all available rows so the persisted model (and the next-day
    # forecast) uses the most recent data too.
    full = features.dropna()
    X_full_scaled = scaler.transform(full[x_cols])
    model.fit(X_full_scaled[:, sel_idx], full["target"])

    last_row = features.iloc[[-1]][x_cols].ffill()
    last_scaled = scaler.transform(last_row)
    next_close = float(model.predict(last_scaled[:, sel_idx])[0])

    backtest_scaled = scaler.transform(features[x_cols].bfill())
    backtest = model.predict(backtest_scaled[:, sel_idx]).tolist()

    artifact = LagRegressionArtifact(
        scaler=scaler, model=model, all_features=x_cols, selected_features=selected,
    )
    return artifact, metrics, next_close, backtest


def save(artifact: LagRegressionArtifact, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)


def load(path) -> LagRegressionArtifact:
    return joblib.load(path)


def predict_next(artifact: LagRegressionArtifact, df: pd.DataFrame) -> float:
    """Predict next-day Close from an already-trained artifact — no
    retraining, used by the dashboard."""
    features = build_full_features(df)
    last_row = features.iloc[[-1]][artifact.all_features].ffill()
    scaled = artifact.scaler.transform(last_row)
    sel_idx = [artifact.all_features.index(c) for c in artifact.selected_features]
    return float(artifact.model.predict(scaled[:, sel_idx])[0])
