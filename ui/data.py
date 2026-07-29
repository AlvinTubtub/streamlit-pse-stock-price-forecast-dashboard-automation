"""Cached orchestration between services/ (real data + real models) and the
page modules. Nothing here touches HTML — it returns plain dicts/DataFrames.

Model training now happens outside Streamlit entirely (services/pdf_pipeline
-> services/model_selector, triggered after every PDF ingestion). This
module's job is just to load whatever the pipeline already trained and
cached in models/predictions/<SYMBOL>.json — the dashboard never retrains.

If a symbol has real CSV data but hasn't been through the training
pipeline yet (e.g. a fresh checkout before the first "Update Data" run),
we fall back to training it live via services.forecasting.run_all_models
so the app is still fully explorable out of the box — but this fallback
is clearly logged as a fallback, not the production path.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import streamlit as st

from services.data_loader import load_companies
from services.data_validator import validate_ohlcv_csv
from services.model_selector import PREDICTIONS_DIR

log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"

MODEL_LABELS = {
    "lag_reg": "Lag-Informed Regression",
    "arima": "ARIMA",
    "lstm": "LSTM",
    "naive": "Naive baseline",
}


@st.cache_data(show_spinner=False)
def _load_cached_results(symbol: str, _cache_mtime: float) -> dict:
    """Loads the pipeline's cached training results for one symbol.
    ``_cache_mtime`` is only there to bust Streamlit's cache when the
    underlying JSON file changes (e.g. after a fresh pipeline run)."""
    path = PREDICTIONS_DIR / f"{symbol}.json"
    return json.loads(path.read_text())


@st.cache_data(show_spinner=False)
def _train_symbol_live(symbol: str, _mtime: float) -> dict:
    """Fallback only: trains on the fly for a symbol the pipeline hasn't
    cached results for yet. Not used once services/model_selector.py has
    run at least once for this symbol."""
    from services.forecasting import run_all_models

    log.warning(
        "No cached predictions for %s — training live as a fallback. "
        "Run the Update Data pipeline to persist real models.", symbol,
    )
    df = validate_ohlcv_csv(DATA_DIR / f"{symbol}.csv")
    return run_all_models(df)


@st.cache_data(show_spinner=False)
def get_dashboard_data():
    """Returns (companies, dataframes, results_by_symbol, missing_symbols)."""
    companies, dataframes, missing = load_companies()
    results = {}
    for symbol, df in dataframes.items():
        cache_path = PREDICTIONS_DIR / f"{symbol}.json"
        if cache_path.exists():
            results[symbol] = _load_cached_results(symbol, cache_path.stat().st_mtime)
        else:
            csv_mtime = (DATA_DIR / f"{symbol}.csv").stat().st_mtime
            results[symbol] = _train_symbol_live(symbol, csv_mtime)
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
