"""Color palette + global CSS, ported 1:1 from the original dashboard's
Tailwind config (see assets/dashboard_template.html) so the native
Streamlit rebuild looks like the same product, not a new one.
"""
from __future__ import annotations

import streamlit as st

COLORS = {
    "bg": "#0f172a",
    "card": "#1e293b",
    "border": "#334155",
    "brand_50": "#eff6ff",
    "brand_100": "#dbeafe",
    "brand_400": "#60a5fa",
    "brand_500": "#3b82f6",
    "brand_600": "#2563eb",
    "brand_900": "#1e3a8a",
    "brand_950": "#172554",
    "text": "#cbd5e1",
    "text_dim": "#94a3b8",
    "text_faint": "#64748b",
    "green": "#4ade80",
    "red": "#f87171",
    "purple": "#c084fc",
    "yellow": "#facc15",
}


def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', sans-serif;
        }}
        .stApp {{
            background-color: {COLORS['bg']};
            color: {COLORS['text']};
        }}
        section[data-testid="stSidebar"] {{
            background-color: {COLORS['card']};
            border-right: 1px solid {COLORS['border']};
        }}
        #MainMenu, footer {{visibility: hidden;}}

        /* ---- Cards ---- */
        .fph-card {{
            background: {COLORS['card']};
            border: 1px solid {COLORS['border']};
            border-radius: 1rem;
            padding: 1.5rem;
            margin-bottom: 1rem;
        }}
        .fph-card-glow {{
            background: {COLORS['card']};
            border: 2px solid {COLORS['brand_500']};
            border-radius: 1rem;
            padding: 2rem;
            box-shadow: 0 0 30px rgba(59,130,246,0.15);
            text-align: center;
        }}
        .fph-hero {{
            background: linear-gradient(135deg, {COLORS['brand_950']} 0%, {COLORS['bg']} 100%);
            border: 1px solid {COLORS['border']};
            border-radius: 1.25rem;
            padding: 2.5rem;
            margin-bottom: 1.5rem;
        }}

        /* ---- Text helpers ---- */
        .fph-title {{ color: white; font-weight: 700; letter-spacing: -0.02em; }}
        .fph-dim {{ color: {COLORS['text_dim']}; }}
        .fph-faint {{ color: {COLORS['text_faint']}; font-size: 0.8rem; }}
        .fph-brand {{ color: {COLORS['brand_400']}; }}

        /* ---- Badges ---- */
        .fph-badge {{
            display: inline-block; font-size: 0.65rem; font-weight: 700;
            padding: 2px 8px; border-radius: 6px; letter-spacing: 0.04em;
            text-transform: uppercase;
        }}
        .fph-badge-brand {{ background: rgba(59,130,246,0.15); color: {COLORS['brand_400']}; border: 1px solid rgba(59,130,246,0.3); }}
        .fph-badge-green {{ background: rgba(74,222,128,0.15); color: {COLORS['green']}; border: 1px solid rgba(74,222,128,0.3); }}
        .fph-badge-red {{ background: rgba(248,113,113,0.15); color: {COLORS['red']}; border: 1px solid rgba(248,113,113,0.3); }}
        .fph-badge-slate {{ background: rgba(148,163,184,0.15); color: {COLORS['text_dim']}; border: 1px solid {COLORS['border']}; }}

        /* ---- Nav buttons ---- */
        div[data-testid="stHorizontalBlock"] .stButton > button {{
            background: transparent;
            border: 1px solid transparent;
            color: {COLORS['text_dim']};
            font-weight: 500;
            border-radius: 8px;
            transition: all 0.15s ease;
        }}
        div[data-testid="stHorizontalBlock"] .stButton > button:hover {{
            background: rgba(255,255,255,0.05);
            color: white;
            border-color: {COLORS['border']};
        }}

        /* Primary CTA buttons */
        .stButton > button[kind="primary"] {{
            background: {COLORS['brand_600']} !important;
            border: none !important;
            box-shadow: 0 4px 14px rgba(59,130,246,0.3);
        }}
        .stButton > button[kind="primary"]:hover {{
            background: {COLORS['brand_500']} !important;
        }}

        hr {{ border-color: {COLORS['border']}; }}
        </style>
        """,
        unsafe_allow_html=True,
    )
