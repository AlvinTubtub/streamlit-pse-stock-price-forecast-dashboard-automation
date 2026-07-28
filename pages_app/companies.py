from __future__ import annotations

import streamlit as st

from ui.components import section_header
from ui.nav import go_to

SECTORS = ["Financials", "Industrial", "Property", "Services", "Mining and Oil"]


def render(companies: list[dict]) -> None:
    section_header("Included Companies", "Select a company to view historical data and forecast results.")

    default_sector = st.session_state.pop("sector_filter", "All Sectors")
    options = ["All Sectors"] + SECTORS
    sector = st.selectbox(
        "Sector", options,
        index=options.index(default_sector) if default_sector in options else 0,
        label_visibility="collapsed",
    )

    filtered = [c for c in companies if sector == "All Sectors" or c["sector"] == sector]

    cols = st.columns(3)
    for i, company in enumerate(filtered):
        with cols[i % 3]:
            has_data = bool(company.get("ohlcv"))
            status = "Real data ready" if has_data else "Placeholder (no CSV yet)"
            st.markdown(
                f"""
                <div class="fph-card" style="cursor:default;">
                    <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                        <div>
                            <h3 class="fph-title" style="font-size:1.5rem; margin:0;">{company['symbol']}</h3>
                            <span class="fph-badge fph-badge-slate" style="margin-top:6px; display:inline-block;">{company['sector']}</span>
                        </div>
                        <div style="background:#0f172a;border:1px solid #334155;padding:8px;border-radius:8px;">📈</div>
                    </div>
                    <p class="fph-dim" style="font-size:0.9rem; margin:1rem 0 1.5rem 0;">{company['name']}</p>
                    <div style="display:flex; justify-content:space-between; border-top:1px solid #334155; padding-top:0.75rem;">
                        <span class="fph-faint">{status}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("View Details →", key=f"co_{company['symbol']}", width="stretch"):
                go_to("details", company["symbol"])
