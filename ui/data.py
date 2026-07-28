"""Cached orchestration between services/ (real data + real models) and the
page modules. Nothing here touches HTML — it returns plain dicts/DataFrames.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import streamlit as st

from services.data_loader import load_companies
from services.data_validator import validate_ohlcv_csv
from services.forecasting import run_all_models

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"

MODEL_LABELS = {
    "lag_reg": "Lag-Informed Regression",
    "arima": "ARIMA",
    "lstm": "LSTM",
    "naive": "Naive baseline",
}


@st.cache_data(show_spinner=False)
def _train_symbol(symbol: str, _mtime: float) -> dict:
    df = validate_ohlcv_csv(DATA_DIR / f"{symbol}.csv")
    return run_all_models(df)


@st.cache_data(show_spinner=False)
def get_dashboard_data():
    """Returns (companies, dataframes, results_by_symbol, missing_symbols)."""
    companies, dataframes, missing = load_companies()
    results = {}
    for symbol, df in dataframes.items():
        mtime = (DATA_DIR / f"{symbol}.csv").stat().st_mtime
        results[symbol] = _train_symbol(symbol, mtime)
    return companies, dataframes, results, missing


def aggregate_metrics(results: dict) -> dict:
    """Mean of each numeric metric across all trained companies, per model."""
    aggregate = {}
    for model_id in ["lag_reg", "arima", "lstm", "naive"]:
        rows = [results[s]["metrics"][model_id] for s in results]
        if rows:
            aggregate[model_id] = {
                k: f"{np.mean([float(r[k]) for r in rows]):.4f}"
                for k in ["rmse", "mae", "mase", "r2"]
            }
        else:
            aggregate[model_id] = {"rmse": "0", "mae": "0", "mase": "0", "r2": "0"}
    return aggregate
