"""Update Market Data page — native UI for the PDF ingestion pipeline.

Lets the user fetch PSE EDGE EOD quotation PDFs automatically, or process
whatever's already staged locally, then runs download -> extract -> clean
-> validate -> merge (services/pdf_pipeline) and refreshes the cached
dashboard data so results show up immediately on the other pages.
"""
from __future__ import annotations

from datetime import date, timedelta

import streamlit as st

from services.pdf_pipeline import latest_raw_date, list_staged_pdfs, run_pipeline
from ui.components import card_close, card_open, section_header
from ui.data import get_dashboard_data


def _show_result(result: dict) -> None:
    status = result["status"]

    if status == "no_files":
        st.warning("No PDF reports to process — fetch new ones first.")
        return
    if status == "no_rows":
        st.error("PDFs were read, but no quotation rows could be extracted from them.")
        return
    if status == "error":
        st.error(f"Pipeline failed: {result['error']}")
        return

    dl = result["download"]
    if dl is not None:
        st.markdown("###### Download")
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("New", len(dl.downloaded))
        d2.metric("Already had", len(dl.skipped))
        d3.metric("Not published yet", len(dl.not_found))
        d4.metric("Errors", len(dl.errors))
        for d, msg in dl.errors:
            st.caption(f"⚠️ {d}: {msg}")

    if result["parse_errors"]:
        with st.expander(f"⚠️ {len(result['parse_errors'])} file(s) failed to parse", expanded=False):
            for path, msg in result["parse_errors"]:
                st.write(f"**{path.name}** — {msg}")

    validation = result["validation"]
    if validation is not None:
        st.markdown("###### Validation")
        icon = "✅" if validation["ok"] else "⚠️"
        st.write(f"{icon} {validation['row_count']} rows checked")
        for check in validation["checks"]:
            mark = "✅" if check["passed"] else "❌"
            st.caption(f"{mark} {check['label']}" + (f" — {check['detail']}" if check["detail"] else ""))

    if result["merge_summaries"]:
        st.markdown("###### Merged into `data/raw/`")
        st.dataframe(
            [
                {
                    "Symbol": s["symbol"],
                    "Rows added": s["rows_added"],
                    "Total rows": s["rows_after"],
                    "Latest date": s["latest_date"],
                }
                for s in result["merge_summaries"]
            ],
            width="stretch", hide_index=True,
        )

    if result["post_validation_errors"]:
        st.error("Some merged files failed the app's own OHLCV validation — they were NOT used by the dashboard:")
        for symbol, msg in result["post_validation_errors"]:
            st.caption(f"❌ {symbol}: {msg}")

    if status in ("ok", "merged_with_warnings") and result["merge_summaries"]:
        get_dashboard_data.clear()
        st.success("Dashboard data refreshed — the Company List, Details, and Model Performance pages now use the updated data.")


def render() -> None:
    section_header("🔄 Update Market Data", "Ingest new PSE EDGE end-of-day quotation reports (PDF) into the dashboard.")

    st.info(
        "This pipeline parses official PSE EDGE End-of-Day PDF reports for the 15 tracked companies, "
        "cleans and validates the extracted rows, then merges them into the same "
        "`data/raw/<SYMBOL>.csv` files the forecasting and dashboard modules already read — "
        "no other part of the app needs to change."
    )

    latest = latest_raw_date()
    card_open()
    st.markdown(
        f"**Current data freshness:** latest trading day on file is "
        f"**{latest.strftime('%B %d, %Y') if latest else 'unknown'}**."
    )
    card_close()

    tab_fetch, tab_staged = st.tabs(["📡 Fetch from PSE EDGE", "📁 Staged Reports"])

    with tab_fetch:
        st.caption("Downloads directly from documents.pse.com.ph. Weekends/holidays return 404 and are skipped automatically.")
        default_start = (latest + timedelta(days=1)) if latest else date(2026, 7, 1)
        c1, c2 = st.columns(2)
        start = c1.date_input("From", value=min(default_start, date.today()))
        end = c2.date_input("To", value=date.today())
        if st.button("Fetch & Process", type="primary", key="fetch_btn"):
            with st.spinner("Downloading and processing reports..."):
                result = run_pipeline(download=True, start_date=start, end_date=end)
            _show_result(result)

    with tab_staged:
        staged = list_staged_pdfs()
        st.caption(f"{len(staged)} PDF report(s) currently staged in `data/pdf_reports/`.")
        if staged:
            st.dataframe([{"File": p.name} for p in staged], width="stretch", hide_index=True)
            if st.button("Process All Staged Reports", type="primary", key="staged_btn"):
                with st.spinner("Processing staged reports..."):
                    result = run_pipeline(pdf_paths=staged)
                _show_result(result)
        else:
            st.write("No PDFs staged yet — use the Fetch tab.")
