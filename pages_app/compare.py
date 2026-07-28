from __future__ import annotations

import pandas as pd
import streamlit as st

from ui.charts import error_metrics_bar, r2_bar
from ui.data import MODEL_LABELS, aggregate_metrics

CATEGORIES = ["Lag-Informed Regression", "ARIMA", "LSTM", "Naive baseline"]
_MODEL_KEY_ORDER = ["lag_reg", "arima", "lstm", "naive"]


def render(companies: list[dict], results: dict) -> None:
    st.markdown('<h1 class="fph-title" style="font-size:2rem;">Model Performance</h1>', unsafe_allow_html=True)

    symbols_with_data = [s for s in results]
    options = ["All Companies (Aggregate)"] + symbols_with_data
    choice = st.selectbox("Company", options, label_visibility="collapsed")

    if choice == "All Companies (Aggregate)":
        metrics = aggregate_metrics(results)
        subtitle = f"Aggregate across {len(symbols_with_data)} trained companies"
    else:
        metrics = results[choice]["metrics"]
        company = next(c for c in companies if c["symbol"] == choice)
        subtitle = f"{choice} — {company['name']} · {company['sector']}"

    st.caption(subtitle)

    competing = {k: metrics[k] for k in ["lag_reg", "arima", "lstm"]}
    best_rmse_key = min(competing, key=lambda k: float(competing[k]["rmse"]))
    best_mae_key = min(competing, key=lambda k: float(competing[k]["mae"]))
    best_mase_key = min(competing, key=lambda k: float(competing[k]["mase"]))
    best_r2_key = max(competing, key=lambda k: float(competing[k]["r2"]))

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Best Model RMSE ↓", competing[best_rmse_key]["rmse"], help="Average error in Pesos, penalizes big misses more. Lower is better.")
        st.caption(MODEL_LABELS[best_rmse_key])
    with c2:
        st.metric("Best Model MAE ↓", competing[best_mae_key]["mae"], help="Average absolute miss in Pesos. Lower is better.")
        st.caption("₱ per prediction")
    with c3:
        st.metric("Best Model MASE ↓", competing[best_mase_key]["mase"], help="Error scaled against a naive (yesterday's price) benchmark. Below 1.0 beats naive guessing.")
        st.caption("Scale-free")
    with c4:
        st.metric("Best Model R² ↑", competing[best_r2_key]["r2"], help="Share of price movement explained by the model. Higher is better.")
        st.caption(f"Naive baseline R² = {metrics['naive']['r2']}")

    st.info(
        f"**Overall best model:** {MODEL_LABELS[best_rmse_key]} demonstrated the strongest ability "
        "to capture patterns for this selection, achieving the lowest overall prediction error (RMSE)."
    )

    st.markdown("##### RMSE and MAE by model, with naive baseline (lower = better)")
    st.plotly_chart(
        error_metrics_bar(CATEGORIES, [float(metrics[k]["rmse"]) for k in _MODEL_KEY_ORDER], [float(metrics[k]["mae"]) for k in _MODEL_KEY_ORDER]),
        width="stretch", config={"displayModeBar": False},
    )

    st.markdown("##### R² by model, with naive baseline (higher = better)")
    st.plotly_chart(
        r2_bar(CATEGORIES, [float(metrics[k]["r2"]) for k in _MODEL_KEY_ORDER]),
        width="stretch", config={"displayModeBar": False},
    )

    st.markdown("##### Detailed Performance Comparison")
    naive_rmse = float(metrics["naive"]["rmse"])
    rows = []
    ranked = sorted(competing.keys(), key=lambda k: float(competing[k]["rmse"]))
    for rank, key in enumerate(ranked, start=1):
        m = competing[key]
        rows.append({
            "Model": MODEL_LABELS[key],
            "RMSE ↓": m["rmse"],
            "MAE ↓": m["mae"],
            "MASE ↓": m["mase"],
            "R² ↑": m["r2"],
            "Rank": "🏆 Best" if rank == 1 else ("2nd" if rank == 2 else "3rd"),
            "vs. naive": "✅ beats naive" if float(m["rmse"]) < naive_rmse else "⚠️ below naive",
        })
    rows.append({
        "Model": "Naive baseline",
        "RMSE ↓": metrics["naive"]["rmse"],
        "MAE ↓": metrics["naive"]["mae"],
        "MASE ↓": metrics["naive"]["mase"],
        "R² ↑": metrics["naive"]["r2"],
        "Rank": "reference",
        "vs. naive": "—",
    })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.caption(
        "The naive baseline just guesses yesterday's price. A good model must beat this baseline "
        "to prove it's actually useful."
    )
