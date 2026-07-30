"""Native-Streamlit replacement for the original dashboard's nav() router.

The original SPA swapped innerHTML based on a `state.currentView` string
set by onclick handlers. Here that becomes st.session_state["page"], set
by real Streamlit buttons, with st.rerun() taking the place of the JS
re-render.
"""
from __future__ import annotations

import streamlit as st

PAGES = [
    ("home", "🏠 Home"),
    ("companies", "🏢 Company List"),
    ("compare", "📊 Model Performance"),
    ("learn", "📖 Learn Stocks"),
    ("about", "ℹ️ About"),
]


def init_state() -> None:
    if "page" not in st.session_state:
        st.session_state.page = "home"
    if "selected_symbol" not in st.session_state:
        st.session_state.selected_symbol = None


def go_to(page: str, symbol: str | None = None) -> None:
    st.session_state.page = page
    if symbol is not None:
        st.session_state.selected_symbol = symbol
    st.rerun()


def top_nav(companies: list[dict]) -> None:
    logo_col, *nav_cols, search_col = st.columns([2] + [1] * len(PAGES) + [2])

    with logo_col:
        st.markdown(
            '<div style="display:flex;align-items:center;gap:8px;height:100%;">'
            '<span style="background:#2563eb;color:white;padding:6px 8px;border-radius:8px;">📈</span>'
            '<span class="fph-title" style="font-size:1.2rem;">Forecast<span class="fph-brand">PH</span></span>'
            "</div>",
            unsafe_allow_html=True,
        )

    for col, (page_id, label) in zip(nav_cols, PAGES):
        with col:
            is_active = st.session_state.page == page_id
            if st.button(label, key=f"nav_{page_id}", width="stretch",
                         type="primary" if is_active else "secondary"):
                go_to(page_id)

    with search_col:
        query = st.text_input("Search", placeholder="Search symbol or name...", label_visibility="collapsed", key="global_search")
        if query:
            matches = [
                c for c in companies
                if query.lower() in c["symbol"].lower() or query.lower() in c["name"].lower()
            ][:5]
            if matches:
                with st.container():
                    for c in matches:
                        if st.button(f"{c['symbol']} — {c['name']}", key=f"search_{c['symbol']}", width="stretch"):
                            go_to("details", c["symbol"])

    st.markdown(f"<hr style='margin-top:0;border-color:#334155;'>", unsafe_allow_html=True)
