"""ForecastPH — fully native Streamlit rebuild.

This replaces the previous approach of embedding the original dashboard's
HTML/CSS/JS via st.iframe(). Every page, chart, and interaction below is
built with native Streamlit + Plotly, styled to match the original's dark
"glass" theme (see ui/theme.py) — same design language and features, no
embedded HTML file, no JS.

Streamlit is a pure presentation layer here: it only ever reads
data/raw/*.csv, models/, prediction_cache/, best_models.json, and
latest_processed.json — all produced by the fully automated pipeline
(run_pipeline.py, triggered on a schedule by Cron-job.org via
.github/workflows/update_pipeline.yml). It never downloads, processes,
or trains anything itself; there is no in-app way to trigger any of that.
"""
from __future__ import annotations

import traceback

import streamlit as st

from pages_app import about, companies, compare, details, home, learn
from ui.components import educational_banner
from pages_app import about, companies, compare, details, home, learn
from ui.components import educational_banner
from ui.data import get_dashboard_data, get_latest_processed
from ui.nav import init_state, top_nav
from ui.theme import inject_css

from ui.nav import init_state, top_nav
from ui.theme import inject_css

st.set_page_config(
    page_title="ForecastPH | Educational Stock Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

inject_css()
init_state()

companies_data, dataframes, results, missing = get_dashboard_data()

st.divider()
st.subheader("DEBUG")

st.write("BPI DataFrame tail:")
st.dataframe(dataframes["BPI"].tail())

st.write("Latest dataframe row:")
st.write(dataframes["BPI"].iloc[-1])

company = next(c for c in companies if c["symbol"] == "BPI")

st.write("Latest company object:")
st.json({
    "latestClose": company["latestClose"],
    "lastOHLCV": company["ohlcv"][-1]
})

top_nav(companies_data)

latest_run = get_latest_processed()
if latest_run and latest_run.get("last_run_at"):
    try:
        run_date, run_time = latest_run["last_run_at"].split("T")
        st.caption(f"📡 Data last refreshed by the automated pipeline: {run_date} {run_time[:5]} UTC (status: {latest_run['status']})")
    except (ValueError, KeyError):
        pass

educational_banner()

if missing:
    with st.expander(f"⚠️ Using placeholder data for {len(missing)} companies — added automatically once the pipeline processes them", expanded=False):
        st.write(", ".join(missing))
        st.caption("These companies will populate automatically after the next scheduled pipeline run — no manual action needed.")

page = st.session_state.page

if page == "home":
    home.render(companies_data)
elif page == "companies":
    companies.render(companies_data)
elif page == "details":
    details.render(companies_data, dataframes, results)
elif page == "compare":
    compare.render(companies_data, results)
elif page == "learn":
    learn.render()
elif page == "about":
    about.render()
else:
    home.render(companies_data)

st.markdown(
    '<p style="text-align:center; color:#64748b; font-size:0.8rem; margin-top:3rem;">'
    "© 2026 BSIT Data Analytics Capstone Project. All rights reserved.</p>",
    unsafe_allow_html=True,
)
