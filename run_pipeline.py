#!/usr/bin/env python3
"""Headless runner for the PSE PDF ingestion pipeline.

Runs the exact same orchestrator the "Update Data" Streamlit page uses
(services/pdf_pipeline.run_pipeline) with no Streamlit dependency at all,
so it can run in GitHub Actions (or any cron/CI environment) and drive
data/raw/<SYMBOL>.csv updates automatically.

Usage:
    python run_pipeline.py                  # download new reports, then process everything staged
    python run_pipeline.py --no-download     # only process whatever's already in data/pdf_reports/
    python run_pipeline.py --start-date 2026-07-01 --end-date 2026-07-27

Exit codes:
    0  success — new data merged, nothing to do (already up to date), or
       merged with non-critical warnings
    1  failure — nothing usable could be extracted, or an unexpected error
       occurred (see data/pdf_pipeline/pipeline.log for details)
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import date, datetime

from services.pdf_pipeline import run_pipeline

# Statuses that still count as a successful CI run — "no_files" means the
# pipeline correctly found nothing new to do (e.g. a market holiday, or the
# data is already current), which is a normal outcome, not a failure.
SUCCESS_STATUSES = {"ok", "merged_with_warnings", "no_files"}


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the PSE PDF ingestion pipeline headlessly.")
    parser.add_argument(
        "--no-download", dest="download", action="store_false",
        help="Skip fetching new reports from PSE EDGE; only process PDFs already in data/pdf_reports/.",
    )
    parser.add_argument("--start-date", type=_parse_date, default=None, help="YYYY-MM-DD, defaults to the day after the newest data on file.")
    parser.add_argument("--end-date", type=_parse_date, default=None, help="YYYY-MM-DD, defaults to today.")
    parser.set_defaults(download=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    started = time.monotonic()

    print("=" * 60)
    print("Starting pipeline...")
    print("=" * 60)

    if args.download:
        print("Downloading reports...")
    print("Extracting...")
    print("Cleaning...")
    print("Validating...")
    print("Merging...")

    result = run_pipeline(download=args.download, start_date=args.start_date, end_date=args.end_date)

    elapsed = round(time.monotonic() - started, 2)
    status = result["status"]

    print("-" * 60)
    if result["download"] is not None:
        dl = result["download"]
        print(f"Download: {len(dl.downloaded)} new, {len(dl.skipped)} already had, "
              f"{len(dl.not_found)} not published yet, {len(dl.errors)} failed")
    print(f"PDFs processed : {result['pdf_count']} ({result['parsed_count']} parsed OK)")
    print(f"Records extracted: {result['record_count']}")
    if result["parse_warnings"]:
        print(f"Warnings        : {len(result['parse_warnings'])}")
    if result["merge_summaries"]:
        print(f"Symbols updated : {len(result['merge_summaries'])}")
    if result["post_validation_errors"]:
        print(f"Post-validation failures: {len(result['post_validation_errors'])}")
    print(f"Status          : {status}")
    print(f"Elapsed         : {elapsed}s")
    print("-" * 60)

    if status in SUCCESS_STATUSES:
        print("Finished successfully.")
        return 0

    print(f"Finished with a failure status: {status}")
    if result.get("error"):
        print(f"Error: {result['error']}")
    print("See data/pdf_pipeline/pipeline.log for details.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
