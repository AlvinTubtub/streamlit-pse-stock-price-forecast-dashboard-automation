# ForecastPH — Native Streamlit Stock Forecast Dashboard

ForecastPH is a native Streamlit dashboard for selected PSE-listed companies. It provides historical price visualization, model comparison, and next-day forecasting using OHLCV data.

This repository is the fully Python-based version of the project. The previous HTML/Tailwind/ApexCharts implementation has been replaced with a Streamlit + Plotly architecture.

**Streamlit is a pure, read-only presentation layer.** It has no data-update, upload, or retraining capability of any kind — every number on screen comes from files already committed to the repo by the automated pipeline described below. See [Architecture](#architecture) for why, and exactly what Streamlit is and isn't allowed to do.

## Features

- Home dashboard with project overview
- Company list and sector browsing
- Company details with historical charts, next-day forecast, and actual-vs-predicted backtests
- Model comparison across forecasting methods (RMSE/MAE/MASE/R²)
- Educational section explaining OHLCV and forecasting models
- About page for project context and capstone background
- A "data last refreshed" indicator sourced directly from the automated pipeline's own run metadata

## Architecture

```text
Cron-job.org (Mon–Fri, 4:00 PM Philippine Time)
        │  POST repository_dispatch
        ▼
GitHub Actions — .github/workflows/update_pipeline.yml
        │
        ▼
run_pipeline.py
  1. Download latest PSE EDGE disclosures
  2. Extract and validate PDF tables
  3. Update OHLCV datasets (data/raw/<SYMBOL>.csv)
  4. Feature engineering
  5. Retrain Lag-Informed Regression, ARIMA, LSTM
  6. Evaluate (RMSE, MAE, MASE, R²)
  7. Select the best model per company
  8. Update prediction_cache/ + best_models.json
  9. Update latest_processed.json
        │
        ▼
Commit changed artifacts only (idempotent — no-op if nothing changed)
        │
        ▼
Streamlit Community Cloud auto-redeploys from the new commit
        │
        ▼
Dashboard reflects the latest data — no user interaction required
```

Streamlit (`app.py`, `pages_app/`, `ui/`) only ever:
- loads `data/raw/*.csv`, `models/`, `prediction_cache/`, `best_models.json`, `latest_processed.json`
- displays the Company List, Company Details, Forecast Results, Model Performance, charts, and dashboard metrics built from what it loaded

Streamlit never downloads PDFs, processes data, retrains models, executes any forecasting pipeline, or writes anything back to the repository. There is no "Update Data" page, no upload widget, and no button anywhere in the app that triggers processing — the only way data changes is a commit from the automated pipeline landing in the repo.

## Tech Stack

- Python
- Streamlit
- Pandas
- NumPy
- Plotly
- scikit-learn
- Statsmodels + pmdarima (automatic ARIMA order selection)
- Torch
- Joblib (model persistence)
- pdfplumber, requests (PDF ingestion pipeline)

## Model Training Pipeline

Training happens exclusively inside the automated pipeline (never in Streamlit):

```text
PSE EDGE PDF -> PDF Extraction -> CSV Generation -> Data Validation
    -> Feature Engineering -> Model Training (x3) -> Model Evaluation
    -> Best Model Selection -> Saved Models -> Streamlit Dashboard (read-only)
```

- `services/feature_engineering.py` — lag features + technical indicators
  (EMA 10/20, RSI 14, MACD/Signal, Bollinger Bands, daily return, rolling
  volatility, High-Low and Open-Close spreads), shared by every model.
- `services/forecasting/lag_regression.py` — StandardScaler -> LASSO
  feature selection -> LinearRegression.
- `services/forecasting/arima_model.py` — ADF stationarity test +
  automatic (p, d, q) selection (via `pmdarima.auto_arima`, or an
  AIC-ranked grid search fallback).
- `services/forecasting/lstm_model.py` — multi-feature LSTM
  (Open/High/Low/Close/Volume + indicators), sequence length 30,
  chronological train/val/test split, early stopping, checkpointing.
- `services/evaluation.py` — shared RMSE/MAE/MASE/R² metrics (MASE is
  scaled against the in-sample naive one-step forecast).
- `services/model_selector.py` — orchestrates training all three models
  per ticker, saves them under `models/`, caches predictions under
  `prediction_cache/`, and writes `best_models.json`.

Triggered exclusively by `services/pdf_pipeline/pipeline.py`, called only
from `run_pipeline.py` (the CLI entrypoint `.github/workflows/update_pipeline.yml`
runs). Can also be run directly for local development:

```bash
python -m services.model_selector          # train + save models for every data/raw/*.csv
python run_pipeline.py --no-train           # ingest new data only, skip retraining (local/dev only)
```

## Project Structure

```text
pse-streamlit-2/
├── app.py
├── run_pipeline.py            # headless CLI entrypoint — the ONLY way the pipeline runs
├── requirements.txt
├── requirements-pipeline.txt  # deps for run_pipeline.py / CI
├── README.md
├── .github/
│   └── workflows/
│       └── update_pipeline.yml   # single orchestration layer; triggered by Cron-job.org
├── data/
│   ├── raw/                # <TICKER>.csv — the data the dashboard reads
│   ├── pdf_reports/        # staged PSE EDGE EOD PDFs (gitignored, except bundled samples)
│   └── pdf_pipeline/       # intermediate ETL artifacts + pipeline.log (gitignored)
├── models/
│   ├── lag_regression/     # <TICKER>.pkl
│   ├── arima/              # <TICKER>.pkl
│   └── lstm/                # <TICKER>.pth
├── prediction_cache/        # <TICKER>.json — cached metrics/predictions the dashboard loads
├── best_models.json         # {"<TICKER>": "<best model label>"} per ticker, lowest RMSE
├── latest_processed.json    # metadata about the most recent automated pipeline run
├── pages_app/
│   ├── about.py
│   ├── companies.py
│   ├── compare.py
│   ├── details.py
│   ├── home.py
│   └── learn.py
├── services/
│   ├── data_loader.py
│   ├── data_validator.py
│   ├── feature_engineering.py   # lag + technical-indicator features, shared by all models
│   ├── evaluation.py            # shared RMSE/MAE/MASE/R² metrics
│   ├── model_selector.py        # trains all 3 models per ticker, saves them, picks the best
│   ├── forecasting/
│   │   ├── __init__.py
│   │   ├── lag_regression.py
│   │   ├── arima_model.py
│   │   └── lstm_model.py
│   └── pdf_pipeline/       # PDF ingestion pipeline (download, parser, cleaner, validator, merge)
│       ├── config.py
│       ├── downloader.py
│       ├── parser.py
│       ├── cleaner.py
│       ├── validator.py
│       ├── merge.py
│       └── pipeline.py     # single orchestration layer: ingestion -> training -> metadata
└── ui/
```

## Requirements

- Python 3.10 or newer
- pip

## Installation (running the dashboard locally)

Clone or download the repository, then open a terminal in the project folder.

```bash
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

If the `streamlit` command isn't available: `python -m streamlit run app.py`.

Then open the local URL shown in the terminal, usually `http://localhost:8501`.

Since Streamlit never generates data itself, a fresh clone shows whatever
`data/raw/`, `models/`, and `prediction_cache/` were last committed —
normally the output of the most recent automated pipeline run.

## Data Format

Each company CSV in `data/raw/` contains: `Date, Open, High, Low, Close, Volume`, named by ticker symbol (e.g. `ALI.csv`, `BPI.csv`, ...). The full list of 15 tracked tickers: ALI, APX, BPI, GLO, ICT, JFC, MBT, MEG, MER, NIKL, PGOLD, SCC, SECB, SHLPH, SMPH.

## Notes

- `SECB.csv` replaces `BDO.csv` throughout the project.
- `services/data_loader.py` maps ticker symbols to the company metadata used in the dashboard.
- If a company CSV or trained model is missing, the app shows a placeholder or an in-app "not processed yet" message instead of failing — see [Architecture](#architecture).

## Automated Pipeline

`.github/workflows/update_pipeline.yml` is the single orchestration layer for the whole project. It has **no GitHub-native cron schedule** — it's triggered externally by [Cron-job.org](https://cron-job.org), Monday–Friday at **4:00 PM Philippine Time**, via a `repository_dispatch` API call.

### Setting up the Cron-job.org trigger

1. Create a GitHub Personal Access Token with `repo` + `workflow` scope (a fine-grained token scoped to just this repo's contents+actions permissions also works).
2. In Cron-job.org, create a new job with:
   - **Schedule**: Monday–Friday, 16:00 (4:00 PM) — set the job's timezone to `Asia/Manila`.
   - **Request type**: Custom HTTP request (`POST`)
   - **URL**: `https://api.github.com/repos/<OWNER>/<REPO>/dispatches`
   - **Headers**:
     - `Accept: application/vnd.github+json`
     - `Authorization: Bearer <YOUR_GITHUB_PAT>`
     - `X-GitHub-Api-Version: 2022-11-28`
   - **Body**: `{"event_type": "run-pipeline"}`
3. Save. Cron-job.org will now POST to GitHub on that schedule, which fires the `repository_dispatch` trigger and starts the workflow — no polling, no GitHub Actions schedule minute-drift.

**Never commit the PAT to this repository.** Store it only in Cron-job.org's own encrypted request-header field.

### What the workflow does

1. Checks out the repo and installs `requirements-pipeline.txt`.
2. Runs `python run_pipeline.py --download`, which downloads new EOD reports, extracts, cleans, validates, and merges them into `data/raw/`, then retrains all three models per ticker, evaluates them, selects the best model per company, and writes `prediction_cache/`, `best_models.json`, and `latest_processed.json`.
3. Verifies at least one non-empty CSV exists in `data/raw/` — if not, the job fails loudly instead of silently pushing nothing.
4. Stages `data/raw/`, `models/`, `prediction_cache/`, `best_models.json`, and `latest_processed.json`, then checks `git diff --cached`. If nothing changed (e.g. a market holiday, or the pipeline already ran for that data), the job **finishes successfully without committing** — the workflow is idempotent, no empty commits ever.
5. If something changed, commits and pushes.
6. Uploads `data/pdf_pipeline/pipeline.log` as a build artifact either way, for troubleshooting.
7. Streamlit Community Cloud picks up the new commit and redeploys automatically — no separate step needed on this repo's side.

Only `contents: write` permission is granted — nothing else.

### Manual / ops trigger (workflow_dispatch)

For a maintainer testing or backfilling outside the Cron-job.org schedule: **Actions → PSE Data & Model Pipeline → Run workflow**, with optional `start_date` / `end_date` inputs (YYYY-MM-DD). This is an operator action taken directly in GitHub, entirely outside the deployed Streamlit app — the app itself has no equivalent capability.

### Running it locally

```bash
pip install -r requirements-pipeline.txt   # or requirements.txt if you also want to run the app
python run_pipeline.py                     # fetch new reports, process, train, evaluate, select
python run_pipeline.py --no-download       # only process what's already in data/pdf_reports/
python run_pipeline.py --no-train          # skip retraining (only refresh data/raw/ CSVs)
python run_pipeline.py --start-date 2026-07-01 --end-date 2026-07-27
```

Exit code `0` means success (including "nothing new to do"); exit code `1` means a real failure — check `data/pdf_pipeline/pipeline.log`.

### Disabling automation

- In GitHub: **Actions → PSE Data & Model Pipeline → ⋯ → Disable workflow**.
- Or delete/rename `.github/workflows/update_pipeline.yml`.
- Independently, pause or delete the job in your Cron-job.org account — that alone stops new runs from being triggered, without touching anything in this repo.

### Idempotency / duplicate-run protection

Re-running the pipeline on data it already has is safe and a no-op at the commit layer: `merge_into_raw()` upserts by date (identical rows produce an identical file), retraining on unchanged data reproduces the same models bit-for-bit-equivalent results, and the workflow's `git diff --cached` check means an unchanged working tree never produces a commit — including two accidental triggers on the same day.

## About the Forecasting Models

The dashboard compares three forecasting approaches:

- Lag-Informed Regression
- ARIMA
- LSTM

against a naive (yesterday's close) baseline, using:

- RMSE
- MAE
- MASE
- R²

## Disclaimer

This dashboard is intended for academic, educational, and analytical decision-support purposes only. It is not financial advice and should not be used as the sole basis for investment decisions.

## License

For academic and internal project use.
