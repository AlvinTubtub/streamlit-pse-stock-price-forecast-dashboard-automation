# ForecastPH — Native Streamlit Stock Forecast Dashboard

ForecastPH is a native Streamlit dashboard for selected PSE-listed companies. It provides historical price visualization, model comparison, and next-day forecasting using OHLCV CSV files.

This repository is the fully Python-based version of the project. The previous HTML/Tailwind/ApexCharts implementation has been replaced with a Streamlit + Plotly architecture.

## Features

- Home dashboard with project overview
- Company list and sector browsing
- Company details with historical charts
- Model comparison across forecasting methods
- **Update Data**: native PDF ingestion pipeline that fetches PSE EDGE End-of-Day quotation PDFs (or processes ones already staged locally), extracts/cleans/validates OHLCV rows, and merges them into `data/raw/` — runs automatically via GitHub Actions daily at 3:00 PM, 4:00 PM, and 5:00 PM PHT, or on demand from the app
- **Automated model training + selection**: after every successful ingestion, all three forecasting models are retrained per ticker, evaluated (RMSE/MAE/MASE/R²), saved to `models/`, and the lowest-RMSE model per ticker is recorded in `best_models.json` — see [Model Training Pipeline](#model-training-pipeline) below
- Educational section explaining OHLCV and forecasting models
- About page for project context and capstone background

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

Training is completely separate from the dashboard — Streamlit only loads
already-trained models, it never fits anything itself.

```text
PSE EDGE PDF -> PDF Extraction -> CSV Generation -> Data Validation
    -> Feature Engineering -> Model Training (x3) -> Model Evaluation
    -> Best Model Selection -> Saved Models -> Streamlit Dashboard
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
- `services/evaluation.py` — shared RMSE/MAE/MASE/R² metrics.
- `services/model_selector.py` — orchestrates training all three models
  per ticker, saves them under `models/`, and writes `best_models.json`.

Triggered automatically by `services/pdf_pipeline/pipeline.py` after every
successful ingestion (both from the Streamlit "Update Data" page and from
`run_pipeline.py` / the scheduled GitHub Action), or run directly:

```bash
python -m services.model_selector          # train + save models for every data/raw/*.csv
python run_pipeline.py --no-train           # ingest new data only, skip retraining
```

## Project Structure

```text
pse-streamlit-2/
├── app.py
├── run_pipeline.py         # headless CLI entrypoint for the PDF pipeline (used by CI)
├── requirements.txt
├── requirements-pipeline.txt  # lightweight deps for run_pipeline.py / CI only
├── README.md
├── .github/
│   └── workflows/
│       └── update_data.yml # scheduled + manual data-update automation
├── data/
│   ├── raw/
│   │   ├── ALI.csv
│   │   ├── APX.csv
│   │   ├── BPI.csv
│   │   ├── GLO.csv
│   │   ├── ICT.csv
│   │   ├── JFC.csv
│   │   ├── MBT.csv
│   │   ├── MEG.csv
│   │   ├── MER.csv
│   │   ├── NIKL.csv
│   │   ├── PGOLD.csv
│   │   ├── SCC.csv
│   │   ├── SECB.csv
│   │   ├── SHLPH.csv
│   │   └── SMPH.csv
│   ├── pdf_reports/        # staged PSE EDGE EOD PDFs (gitignored, except bundled samples)
│   └── pdf_pipeline/       # intermediate ETL artifacts + pipeline.log (gitignored)
├── models/                 # trained model artifacts (gitignored contents; structure tracked)
│   ├── lag_regression/     # <TICKER>.pkl
│   ├── arima/              # <TICKER>.pkl
│   ├── lstm/                # <TICKER>.pth
│   └── predictions/        # <TICKER>.json — cached metrics/predictions the dashboard loads
├── best_models.json        # {"<TICKER>": "<best model label>"} per ticker, lowest RMSE
├── pages_app/
│   ├── about.py
│   ├── companies.py
│   ├── compare.py
│   ├── data_pipeline.py    # "Update Market Data" page
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
│   │   ├── __init__.py          # run_all_models() — legacy on-the-fly training, no persistence
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
│       └── pipeline.py     # calls services.model_selector after every successful merge
├── ui/
```

Requirements
Python 3.10 or newer
pip
Installation

Clone or download the repository, then open a terminal in the project folder.

1. Create a virtual environment
python3 -m venv .venv
2. Activate the virtual environment

On macOS/Linux:

source .venv/bin/activate

On Windows:

.venv\Scripts\activate
3. Install dependencies
pip install -r requirements.txt
Run the App
streamlit run app.py

If the streamlit command is not available:

python -m streamlit run app.py

Then open the local URL shown in the terminal, usually:

http://localhost:8501
Data Format

The app expects each company CSV to contain these columns:

Date
Open
High
Low
Close
Volume

Files should be named using the ticker symbol, for example:

ALI.csv
APX.csv
BPI.csv
GLO.csv
ICT.csv
JFC.csv
MBT.csv
MEG.csv
MER.csv
NIKL.csv
PGOLD.csv
SCC.csv
SECB.csv
SHLPH.csv
SMPH.csv
Notes
SECB.csv replaces BDO.csv throughout the project.
services/data_loader.py maps the ticker symbols to the company metadata used in the dashboard.
If a company CSV is missing, the app may show a placeholder or a warning instead of failing.
The forecasting page trains or loads model outputs from the CSV data depending on the current implementation in services/forecasting.py.

Update Market Data (PDF Pipeline)

The Update Data page (services/pdf_pipeline/) turns official PSE EDGE End-of-Day quotation report PDFs into the same data/raw/<SYMBOL>.csv files everything else in the app reads — no separate project or manual conversion step needed.

Two ways to feed it reports:
- Fetch from PSE EDGE: pick a date range and it downloads directly from documents.pse.com.ph (weekends/holidays 404 and are skipped automatically).
- Staged Reports: process whatever's already sitting in data/pdf_reports/ (the repo ships with sample reports for 2026-07-01 through 2026-07-27 pre-staged there).

Pipeline steps, each reusable on its own:
1. download_reports() — fetch PDFs (services/pdf_pipeline/downloader.py)
2. extract_reports() — parse the 15 target companies' OHLCV rows out of each PDF (services/pdf_pipeline/parser.py)
3. clean_quotes() — dedupe, coerce types, drop invalid rows (services/pdf_pipeline/cleaner.py)
4. validate_quotes() — structural + sanity checks, shown as pass/fail in the UI (services/pdf_pipeline/validator.py)
5. merge_into_raw() — upsert by date into data/raw/<SYMBOL>.csv, then re-validated with the app's own services/data_validator.py before anything downstream sees it (services/pdf_pipeline/merge.py)

After a successful run, the dashboard's cached data is cleared automatically, so Company List, Details, and Model Performance immediately reflect the new data — no restart needed.

## Automated Data Updates

Every day, `data/raw/<SYMBOL>.csv` is updated automatically, three times a day — no one has to open the app or click anything.

```text
GitHub Actions
        │
        ▼
run_pipeline.py
        │
        ▼
PDF Pipeline (services/pdf_pipeline)
        │
        ▼
CSV Outputs (data/raw/<SYMBOL>.csv)
        │
        ▼
Automatic Git Commit
        │
        ▼
Streamlit Dashboard
```

### Schedule

`.github/workflows/update_data.yml` runs on three cron schedules — `0 7 * * *`, `0 8 * * *`, `0 9 * * *` (UTC) — daily at **3:00 PM, 4:00 PM, and 5:00 PM Philippine Time** (UTC+8, no DST), so a report that publishes late in one run is picked up by the next one that same afternoon.

### Manual run (workflow_dispatch)

From the GitHub repo: **Actions → Update PSE Market Data → Run workflow**. Optional `start_date` / `end_date` inputs (YYYY-MM-DD) let you backfill a specific range instead of "everything since the last commit."

### What the workflow does

1. Checks out the repo and installs `requirements-pipeline.txt` — a minimal dependency set (pandas, numpy, pdfplumber, requests) so the job doesn't need to install the full Streamlit/torch/statsmodels stack just to fetch and merge CSVs.
2. Runs `python run_pipeline.py`, which downloads any new EOD reports, extracts, cleans, validates, and merges them into `data/raw/`.
3. Verifies at least one non-empty CSV exists in `data/raw/` — if not, the job fails loudly instead of silently pushing nothing.
4. Checks `git diff --cached` against `data/raw/` only. If nothing changed (data already up to date, or nothing new was published — e.g. a market holiday), the job **finishes successfully without committing** — no empty commits, ever.
5. If something did change, commits with message `Auto-update PSE data (YYYY-MM-DD)` and pushes.
6. Uploads `data/pdf_pipeline/pipeline.log` as a build artifact either way, for troubleshooting.

Only `contents: write` permission is granted — nothing else.

### Running it locally

```bash
pip install -r requirements-pipeline.txt   # or requirements.txt if you also want to run the app
python run_pipeline.py                     # fetch new reports + process everything staged
python run_pipeline.py --no-download       # only process what's already in data/pdf_reports/
python run_pipeline.py --start-date 2026-07-01 --end-date 2026-07-27
```

Exit code `0` means success (including "nothing new to do"); exit code `1` means a real failure — check `data/pdf_pipeline/pipeline.log`.

### Disabling automation

- Quickest: in the GitHub UI, **Actions → Update PSE Market Data → ⋯ → Disable workflow**.
- Or delete/rename `.github/workflows/update_data.yml`.
- `workflow_dispatch` still works independently of the schedule, so you can disable the cron trigger alone by removing the `schedule:` block and keeping `workflow_dispatch:`.

### Duplicate-commit protection

Re-running the pipeline (scheduled or manual) on data it already has is safe and a no-op at every layer: `merge_into_raw()` upserts by date (identical rows produce an identical file), and the workflow's `git diff --cached` check means an unchanged `data/raw/` never produces a commit — including two accidental runs on the same day, or a manual run right after the scheduled one.

About the Forecasting Models

The dashboard compares three forecasting approaches:

Lag-Informed Regression
ARIMA
LSTM

Model performance is evaluated using common regression metrics such as:

RMSE
MAE
MASE
R²
Disclaimer

This dashboard is intended for academic, educational, and analytical decision-support purposes only. It is not financial advice and should not be used as the sole basis for investment decisions.

License

For academic and internal project use.


If you want, I can also turn this into a cleaner README with badges, screenshots, and a more polished capstone