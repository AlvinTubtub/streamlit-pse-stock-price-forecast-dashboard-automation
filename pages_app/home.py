from __future__ import annotations

import streamlit as st

from ui.components import stat_card
from ui.nav import go_to

SECTORS = ["Financials", "Industrial", "Property", "Services", "Mining and Oil"]


def render(companies: list[dict]) -> None:
    st.markdown(
        """
        <div class="fph-hero">
            <span class="fph-badge fph-badge-brand" style="margin-bottom:1rem;display:inline-block;">
                🎓 Educational Dashboard
            </span>
            <h1 class="fph-title" style="font-size:2.6rem; margin:0.5rem 0 1rem 0;">
                Cross-Sector Next-Day<br>Stock Price Forecasting
            </h1>
            <p class="fph-dim" style="font-size:1.1rem; max-width:640px; line-height:1.6;">
                Explore historical Philippine stock market data, compare machine learning and
                statistical models, and understand next-day price prediction techniques.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Explore Forecasts →", width="stretch", type="primary"):
            go_to("companies")
    with c2:
        if st.button("Learn About Forecasting", width="stretch"):
            go_to("learn")

    st.write("")
    cols = st.columns(4)
    with cols[0]:
        stat_card("🏢", "Total Companies", str(len(companies)), "Across 5 PSE Sectors", "rgba(59,130,246,0.1)", "#60a5fa")
    with cols[1]:
        stat_card("🧠", "Forecasting Models", "3", "Lag-Informed Regression, ARIMA, LSTM", "rgba(168,85,247,0.1)", "#c084fc")
    with cols[2]:
        stat_card("📅", "Prediction Horizon", "Next Trading Day", "t+1 Forecasting", "rgba(74,222,128,0.1)", "#4ade80")
    with cols[3]:
        stat_card("🗄️", "Data Source", "PSE EDGE", "Real historical OHLCV data", "rgba(251,146,60,0.1)", "#fb923c")

    st.write("")
    st.markdown('<h3 class="fph-title" style="font-size:1.3rem;">Sectors Covered</h3>', unsafe_allow_html=True)
    sector_cols = st.columns(5)
    for col, sector in zip(sector_cols, SECTORS):
        count = sum(1 for c in companies if c["sector"] == sector)
        with col:
            st.markdown(
                f"""
                <div class="fph-card" style="text-align:center; padding:1rem;">
                    <span style="font-size:0.9rem; color:{'#cbd5e1'};">{sector}</span><br>
                    <span class="fph-faint">{count} Companies</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("View", key=f"sector_{sector}", width="stretch"):
                st.session_state.sector_filter = sector
                go_to("companies")
