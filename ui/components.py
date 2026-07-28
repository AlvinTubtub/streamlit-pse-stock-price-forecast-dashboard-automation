"""Small HTML-snippet helpers so page modules stay readable. These render
static markup only — every interactive element (buttons, selects, file
uploaders) uses native Streamlit widgets instead, per the project's
hybrid-architecture decision.
"""
from __future__ import annotations

import streamlit as st


def badge(text: str, kind: str = "slate") -> str:
    return f'<span class="fph-badge fph-badge-{kind}">{text}</span>'


def stat_card(icon: str, label: str, value: str, sub: str, icon_bg: str, icon_color: str) -> None:
    st.markdown(
        f"""
        <div class="fph-card" style="display:flex; gap:1rem; align-items:flex-start;">
            <div style="background:{icon_bg}; color:{icon_color}; padding:0.75rem; border-radius:0.75rem; font-size:1.4rem; line-height:1;">{icon}</div>
            <div>
                <div class="fph-dim" style="font-size:0.85rem; font-weight:500;">{label}</div>
                <div class="fph-title" style="font-size:1.5rem;">{value}</div>
                <div class="fph-faint" style="margin-top:2px;">{sub}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(title: str, subtitle: str | None = None) -> None:
    st.markdown(f'<h1 class="fph-title" style="font-size:2rem;">{title}</h1>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<p class="fph-dim">{subtitle}</p>', unsafe_allow_html=True)


def card_open(extra_style: str = "") -> None:
    st.markdown(f'<div class="fph-card" style="{extra_style}">', unsafe_allow_html=True)


def card_close() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


def educational_banner() -> None:
    st.markdown(
        """
        <div style="background:rgba(30,58,138,0.35); border:1px solid rgba(59,130,246,0.3);
                    border-radius:0.5rem; padding:0.6rem 1rem; text-align:center;
                    font-size:0.85rem; color:#dbeafe; margin-bottom:1.5rem;">
            ℹ️ <strong>Educational Purpose Only:</strong> Forecasts use real historical OHLCV data
            and trained models, but should not be used as financial advice.
        </div>
        """,
        unsafe_allow_html=True,
    )
