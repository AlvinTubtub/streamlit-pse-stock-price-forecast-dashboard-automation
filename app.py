"""ForecastPH — fully native Streamlit rebuild.

This replaces the previous approach of embedding the original dashboard's
HTML/CSS/JS via st.iframe(). Every page, chart, and interaction below is
built with native Streamlit + Plotly, styled to match the original's dark
"glass" theme (see ui/theme.py) — same design language and features, no
embedded HTML file, no JS.

Still no database: CSV in -> validate -> preprocess -> forecast -> display,
per Streamlit run.
"""
from __future__ import annotations

import streamlit as st

from pages_app import about, companies, compare, data_pipeline, details, home, learn
from ui.components import educational_banner
from ui.data import get_dashboard_data
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

top_nav(companies_data)
educational_banner()

if missing:
    with st.expander(f"⚠️ Using placeholder data for {len(missing)} companies — add real CSVs to unlock them", expanded=False):
        st.write(", ".join(missing))
        st.caption(f"Drop real OHLCV CSVs into `data/raw/<SYMBOL>.csv`, or use the **🔄 Update Data** page to ingest official PSE EDGE PDF reports.")

page = st.session_state.page

if page == "home":
    home.render(companies_data)
elif page == "companies":
    companies.render(companies_data)
elif page == "details":
    details.render(companies_data, dataframes, results)
elif page == "compare":
    compare.render(companies_data, results)
elif page == "data":
    data_pipeline.render()
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
