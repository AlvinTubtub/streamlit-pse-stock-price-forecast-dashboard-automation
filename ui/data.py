"""Read-only data access layer between services/ (real data + real models)
and the page modules. Nothing here touches HTML — it returns plain
dicts/DataFrames.

Streamlit is a pure presentation layer: this module only ever *loads*
what services/pdf_pipeline -> services/model_selector already produced
and committed to the repo (data/raw/*.csv, models/, prediction_cache/,
best_models.json, latest_processed.json). It never downloads PDFs,
processes data, trains/retrains models, or writes anything back to the
repo — all of that happens exclusively in the automated pipeline
(run_pipeline.py, triggered by .github/workflows/update_pipeline.yml).

A symbol with a CSV but no cached prediction yet (e.g. a ticker added
before its first pipeline run) simply has no entry in ``results`` — its
historical chart still renders, but forecast/model-performance sections
show a "not processed yet" message instead of training anything locally.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import streamlit as st

from services.data_loader import load_companies
from services.model_selector import BEST_MODELS_PATH, PREDICTION_CACHE_DIR
from services.pdf_pipeline.config import LATEST_PROCESSED_PATH

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
    underlying JSON file changes (e.g. after the next pipeline run)."""
    path = PREDICTION_CACHE_DIR / f"{symbol}.json"
    return json.loads(path.read_text())


@st.cache_data(show_spinner=False)
def get_dashboard_data():
    """Returns (companies, dataframes, results_by_symbol, missing_symbols).

    ``results`` only contains symbols the pipeline has already produced a
    prediction_cache/<SYMBOL>.json for — no live training happens here.
    """
    companies, dataframes, missing = load_companies()
    results = {}
    for symbol in dataframes:
        cache_path = PREDICTION_CACHE_DIR / f"{symbol}.json"
        if cache_path.exists():
            results[symbol] = _load_cached_results(symbol, cache_path.stat().st_mtime)
    return companies, dataframes, results, missing


def get_best_models() -> dict:
    """Loads best_models.json ({ticker: winning model label}), written by
    the automated pipeline. Returns {} if it doesn't exist yet."""
    if not BEST_MODELS_PATH.exists():
        return {}
    return json.loads(BEST_MODELS_PATH.read_text())


def get_latest_processed() -> dict | None:
    """Loads latest_processed.json — metadata about the most recent
    automated pipeline run (written by services/pdf_pipeline/pipeline.py).
    Returns None if the pipeline hasn't run yet in this environment."""
    if not LATEST_PROCESSED_PATH.exists():
        return None
    return json.loads(LATEST_PROCESSED_PATH.read_text())


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
