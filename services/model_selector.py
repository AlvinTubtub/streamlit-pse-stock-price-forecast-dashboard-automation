"""Training orchestration + best-model selection.

This is the one place that trains all three forecasting models for every
ticker, evaluates them, saves each trained model to disk, caches the full
results (metrics/predictions/backtests) for the dashboard to load without
retraining, and writes ``best_models.json`` mapping each ticker to
whichever model had the lowest test-set RMSE.

Called automatically after every successful PDF ingestion (see
services/pdf_pipeline/pipeline.py) and by the standalone
``python -m services.model_selector`` / run_pipeline.py CLI entrypoints —
never from inside the Streamlit dashboard itself.

Directory layout produced:

    models/
        lag_regression/<TICKER>.pkl
        arima/<TICKER>.pkl
        lstm/<TICKER>.pth
        predictions/<TICKER>.json   # cached metrics + predictions for ui/data.py
    best_models.json                # {"BDO": "LSTM", "MER": "ARIMA", ...}
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from services.data_validator import CSVValidationError, validate_ohlcv_csv
from services.evaluation import evaluate_naive, select_best_model
from services.forecasting import MODEL_LABELS, arima_model, lag_regression, lstm_model

log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
MODELS_DIR = BASE_DIR / "models"
LAG_MODELS_DIR = MODELS_DIR / "lag_regression"
ARIMA_MODELS_DIR = MODELS_DIR / "arima"
LSTM_MODELS_DIR = MODELS_DIR / "lstm"
PREDICTIONS_DIR = MODELS_DIR / "predictions"
BEST_MODELS_PATH = BASE_DIR / "best_models.json"


def _ensure_dirs() -> None:
    for d in (LAG_MODELS_DIR, ARIMA_MODELS_DIR, LSTM_MODELS_DIR, PREDICTIONS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def train_symbol(symbol: str, df: pd.DataFrame) -> dict:
    """Trains + evaluates all three models for one ticker, saves each to
    disk, and returns the results dict cached for the dashboard (same
    shape as the legacy ``run_all_models`` return value)."""
    log.info("Training models for %s (%d rows)...", symbol, len(df))

    lag_artifact, lag_metrics, lag_next, lag_backtest = lag_regression.train(df)
    lag_regression.save(lag_artifact, LAG_MODELS_DIR / f"{symbol}.pkl")

    arima_fitted, order, arima_metrics, arima_next, arima_backtest = arima_model.train(df)
    arima_model.save(arima_fitted, ARIMA_MODELS_DIR / f"{symbol}.pkl")
    log.info("%s ARIMA order selected: %s", symbol, order)

    lstm_artifact, lstm_metrics, lstm_next, lstm_backtest = lstm_model.train(df)
    lstm_model.save(lstm_artifact, LSTM_MODELS_DIR / f"{symbol}.pth")

    naive_metrics = evaluate_naive(df)

    return {
        "metrics": {
            "lag_reg": lag_metrics,
            "arima": arima_metrics,
            "lstm": lstm_metrics,
            "naive": naive_metrics,
        },
        "next_close": {
            "lag": round(lag_next, 2),
            "arima": round(arima_next, 2),
            "lstm": round(lstm_next, 2),
        },
        "backtest30": lag_backtest[-30:] if len(lag_backtest) >= 30 else lag_backtest,
        "backtest_by_model": {
            "Lag-Informed Regression": lag_backtest,
            "ARIMA": arima_backtest,
            "LSTM": lstm_backtest,
        },
    }


def train_and_select_all(raw_dir: Path = RAW_DIR) -> dict[str, str]:
    """Trains + saves models for every ticker CSV in ``raw_dir``, caches
    each ticker's results for the dashboard, and writes best_models.json.

    Returns the {symbol: best_model_label} mapping. Bad/unreadable CSVs
    are logged and skipped rather than aborting the whole run.
    """
    _ensure_dirs()
    best_models: dict[str, str] = {}

    csv_paths = sorted(raw_dir.glob("*.csv"))
    if not csv_paths:
        log.warning("No CSVs found in %s — nothing to train.", raw_dir)
        return best_models

    for csv_path in csv_paths:
        symbol = csv_path.stem
        try:
            df = validate_ohlcv_csv(csv_path)
        except CSVValidationError as exc:
            log.warning("Skipping %s — failed OHLCV validation: %s", symbol, exc)
            continue
        except Exception:
            log.exception("Skipping %s — unexpected error while loading CSV.", symbol)
            continue

        try:
            result = train_symbol(symbol, df)
        except Exception:
            log.exception("Training failed for %s — skipping.", symbol)
            continue

        best_key = select_best_model(result["metrics"], ["lag_reg", "arima", "lstm"])
        best_models[symbol] = MODEL_LABELS[best_key]

        (PREDICTIONS_DIR / f"{symbol}.json").write_text(json.dumps(result, indent=2))
        log.info("%s best model: %s (RMSE %s)", symbol, MODEL_LABELS[best_key], result["metrics"][best_key]["rmse"])

    BEST_MODELS_PATH.write_text(json.dumps(best_models, indent=2, sort_keys=True))
    log.info("Wrote %s (%d tickers)", BEST_MODELS_PATH, len(best_models))
    return best_models


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    mapping = train_and_select_all()
    print(json.dumps(mapping, indent=2, sort_keys=True))
