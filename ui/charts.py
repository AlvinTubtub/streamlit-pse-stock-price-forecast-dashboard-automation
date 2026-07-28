"""Plotly figure builders — the native-Python replacement for the original
dashboard's ApexCharts calls. Same information, same dark theme, same
color roles (brand blue for the primary series, muted slate for grid/axes).
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from .theme import COLORS

_LAYOUT_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=COLORS["text_dim"], family="Inter, sans-serif"),
    margin=dict(l=10, r=10, t=30, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(color=COLORS["text"])),
    hovermode="x unified",
)

_GRID = dict(gridcolor=COLORS["border"], zerolinecolor=COLORS["border"], color=COLORS["text_dim"])


def history_line_chart(df: pd.DataFrame, height: int = 400) -> go.Figure:
    """Full OHLCV chart: Open/High/Low/Close as lines on the primary axis,
    Volume as a filled area on a secondary axis — matching the prototype's
    combined historical OHLCV chart."""
    fig = go.Figure()
    price_traces = [
        ("Open", "#93c5fd", 1.25),
        ("High", COLORS["green"], 1.25),
        ("Low", COLORS["red"], 1.25),
        ("Close", COLORS["brand_500"], 2.25),
    ]
    for col, color, width in price_traces:
        if col in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["Date"], y=df[col], mode="lines", name=col,
                    line=dict(color=color, width=width),
                )
            )
    if "Volume" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["Date"], y=df["Volume"], mode="none", name="Volume",
                fill="tozeroy", fillcolor="rgba(96,165,250,0.15)", yaxis="y2",
            )
        )
    fig.update_layout(
        **_LAYOUT_BASE,
        height=height,
        yaxis2=dict(overlaying="y", side="right", showgrid=False, showticklabels=False, range=[0, df["Volume"].max() * 4] if "Volume" in df.columns else None),
    )
    fig.update_xaxes(**_GRID)
    fig.update_yaxes(**_GRID, tickprefix="₱")
    return fig


def actual_vs_predicted_chart(dates, actual, naive, model_series: dict[str, list[float]], height: int = 350) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=actual, mode="lines", name="Actual", line=dict(color="white", width=2.5)))
    fig.add_trace(go.Scatter(x=dates, y=naive, mode="lines", name="Naive baseline", line=dict(color=COLORS["text_faint"], width=1.5, dash="dot")))
    palette = {"Lag-Informed Regression": "#7dd3fc", "ARIMA": "#a78bfa", "LSTM": COLORS["brand_400"]}
    for name, series in model_series.items():
        fig.add_trace(go.Scatter(x=dates, y=series, mode="lines", name=name, line=dict(color=palette.get(name, COLORS["brand_400"]), width=2)))
    fig.update_layout(**_LAYOUT_BASE, height=height)
    fig.update_xaxes(**_GRID)
    fig.update_yaxes(**_GRID, tickprefix="₱")
    return fig


def forecast_error_chart(dates, actual: list[float], model_series: dict[str, list[float]], height: int = 350) -> go.Figure:
    """Daily prediction error (Predicted - Actual) in Pesos. Positive values
    indicate overprediction, negative values indicate underprediction; a
    model that hugs the zero reference line is more accurate."""
    fig = go.Figure()
    fig.add_hline(y=0, line=dict(color=COLORS["text_faint"], width=1, dash="dot"))
    palette = {"Lag-Informed Regression": "#7dd3fc", "ARIMA": "#a78bfa", "LSTM": COLORS["brand_400"]}
    actual_arr = list(actual)
    for name, series in model_series.items():
        errors = [round(p - a, 4) for p, a in zip(series, actual_arr)]
        fig.add_trace(go.Scatter(x=dates, y=errors, mode="lines", name=name, line=dict(color=palette.get(name, COLORS["brand_400"]), width=2)))
    fig.update_layout(**_LAYOUT_BASE, height=height)
    fig.update_xaxes(**_GRID)
    fig.update_yaxes(**_GRID, tickprefix="₱", zeroline=True)
    return fig


def error_metrics_bar(categories: list[str], rmse: list[float], mae: list[float], height: int = 320) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(name="RMSE", x=categories, y=rmse, marker_color=COLORS["brand_400"]))
    fig.add_trace(go.Bar(name="MAE", x=categories, y=mae, marker_color="#93c5fd"))
    fig.update_layout(**_LAYOUT_BASE, height=height, barmode="group")
    fig.update_xaxes(**_GRID)
    fig.update_yaxes(**_GRID)
    return fig


def r2_bar(categories: list[str], values: list[float], height: int = 320) -> go.Figure:
    colors = ["#7dd3fc", "#a78bfa", COLORS["brand_400"], COLORS["text_faint"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=categories, y=values, marker_color=colors[: len(categories)], showlegend=False))
    fig.update_layout(**_LAYOUT_BASE, height=height, showlegend=False)
    fig.update_xaxes(**_GRID)
    fig.update_yaxes(**_GRID, range=[0, 1])
    return fig
