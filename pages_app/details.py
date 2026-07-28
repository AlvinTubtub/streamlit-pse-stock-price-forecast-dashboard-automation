from __future__ import annotations

import streamlit as st

from ui.charts import actual_vs_predicted_chart, forecast_error_chart, history_line_chart
from ui.nav import go_to


def render(companies: list[dict], dataframes: dict, results: dict) -> None:
    symbol = st.session_state.selected_symbol
    company = next((c for c in companies if c["symbol"] == symbol), None)

    if st.button("← Back to Companies"):
        go_to("companies")

    if company is None:
        st.error("Company not found.")
        return

    df = dataframes.get(symbol)
    result = results.get(symbol)

    latest_close = company.get("latestClose")
    top1, top2 = st.columns([3, 1])
    with top1:
        st.markdown(
            f"""
            <div class="fph-card">
                <div style="display:flex; align-items:center; gap:10px;">
                    <h1 class="fph-title" style="font-size:2.2rem; margin:0;">{company['symbol']}</h1>
                    <span class="fph-badge fph-badge-brand">{company['sector']}</span>
                </div>
                <p class="fph-dim" style="font-size:1.05rem;">{company['name']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with top2:
        date_label = df["Date"].iloc[-1].strftime("%Y-%m-%d") if df is not None else "—"
        st.markdown(
            f"""
            <div class="fph-card" style="text-align:right;">
                <p class="fph-faint" style="margin:0;">Latest Close</p>
                <p class="fph-title" style="font-size:1.8rem; margin:0;">₱{latest_close}</p>
                <p class="fph-faint" style="margin:0;">Date: {date_label}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if df is None or result is None:
        st.warning(
            f"No validated CSV found for **{symbol}** yet. Add `data/raw/{symbol}.csv` "
            "(Date, Open, High, Low, Close, Volume) to unlock real charts and forecasts for this company."
        )
        return

    left, right = st.columns([2, 1])

    with left:
        st.markdown('<div class="fph-card">', unsafe_allow_html=True)
        st.markdown("##### 📈 Historical OHLCV Price Chart")
        st.plotly_chart(history_line_chart(df), width="stretch", config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

        n = min(30, len(df) - 1)
        tail = df.tail(n)
        dates = tail["Date"]
        actual = tail["Close"].tolist()
        naive = df["Close"].shift(1).tail(n).tolist()
        model_series = {name: series[-n:] for name, series in result["backtest_by_model"].items()}

        st.markdown('<div class="fph-card">', unsafe_allow_html=True)
        st.markdown(f"##### 📈 Actual vs. Predicted — {symbol} test set, last 30 days shown")
        st.plotly_chart(
            actual_vs_predicted_chart(dates, actual, naive, model_series),
            width="stretch", config={"displayModeBar": False},
        )
        st.caption(
            "A model line that hugs the white 'actual' line more tightly than the dotted naive "
            "baseline is genuinely adding information over just guessing yesterday's price."
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="fph-card">', unsafe_allow_html=True)
        st.markdown("##### 📉 Forecast Error Over Time")
        st.plotly_chart(
            forecast_error_chart(dates, actual, model_series),
            width="stretch", config={"displayModeBar": False},
        )
        st.caption(
            "Daily prediction error (Predicted − Actual) in Pesos. Positive values indicate "
            "overprediction, negative values indicate underprediction. A model that stays close "
            "to the zero line is more accurate."
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        preds = result["next_close"]
        best_model = min(preds, key=lambda k: float(result["metrics"][{"lag": "lag_reg", "arima": "arima", "lstm": "lstm"}[k]]["rmse"]))
        best_label = {"lag": "Lag-Informed Regression", "arima": "ARIMA", "lstm": "LSTM"}[best_model]
        pred_price = preds[best_model]
        direction = "Up" if pred_price >= latest_close else "Down"
        color = "#4ade80" if direction == "Up" else "#f87171"
        arrow = "▲" if direction == "Up" else "▼"

        st.markdown(
            f"""
            <div class="fph-card" style="background:linear-gradient(135deg, rgba(30,58,138,0.4), #1e293b); border-color:rgba(59,130,246,0.4);">
                <h4 class="fph-title" style="font-size:1.1rem;">Next-Day Forecast</h4>
                <p class="fph-brand" style="font-size:0.8rem;">Model: {best_label} (lowest RMSE)</p>
                <p class="fph-dim" style="margin:1rem 0 0.25rem 0; font-size:0.85rem;">Predicted Closing Price</p>
                <div style="display:flex; align-items:baseline; gap:10px;">
                    <span class="fph-title" style="font-size:2.3rem;">₱{pred_price}</span>
                    <span style="color:{color}; font-weight:600;">{arrow} {direction}</span>
                </div>
                <div style="border-top:1px solid rgba(255,255,255,0.1); margin-top:1rem; padding-top:0.75rem;">
                    <p style="font-size:0.85rem; color:#cbd5e1; font-style:italic;">
                        The {best_label} model predicts {symbol}'s closing price may
                        <strong style="color:{color};">slightly {direction.lower()}</strong> next trading day,
                        based on real historical patterns in the uploaded/bundled OHLCV data.
                    </p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="fph-card">
                <h5 class="fph-title" style="font-size:1rem;">ℹ️ Educational Note</h5>
                <p class="fph-dim" style="font-size:0.85rem;">
                    These forecasts are generated from real historical OHLCV data and real trained
                    models, but do not reflect live PSE connectivity or breaking news. They demonstrate
                    the capstone project's forecasting methodology only — not financial advice.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
