from __future__ import annotations

import streamlit as st


def render() -> None:
    st.markdown(
        '<h1 class="fph-title" style="font-size:2.2rem; text-align:center;">About the Research</h1>'
        '<p class="fph-dim" style="text-align:center;">BSIT Data Analytics Capstone Project Overview</p>',
        unsafe_allow_html=True,
    )
    st.write("")

    with st.container():
        st.markdown('<div class="fph-card">', unsafe_allow_html=True)
        st.markdown("#### Project Overview")
        st.markdown("##### The Problem")
        st.markdown(
            "Existing local studies on Philippine stock forecasting often focus on a single model or "
            "only track market indices, with outputs that are usually code-heavy and difficult for "
            "non-programmers to interpret. There is also a lack of comparative analysis to prove which "
            "model architecture works best for specific industries, since no single model performs "
            "perfectly across all market conditions."
        )
        st.markdown("##### How the System Works")
        st.markdown(
            "Historical OHLCV data (Open, High, Low, Close, and Volume) spanning several years are "
            "sourced primarily from the Official PSE Daily Quotations Reports. The dataset is then "
            "cleaned and preprocessed through data validation, feature engineering, lag feature "
            "generation, and scaling to prepare it for forecasting. The processed data are "
            "chronologically divided into an 85% Development Dataset and a 15% Hold-out Test Dataset, "
            "with Rolling-Origin Validation performed within the Development Dataset to preserve the "
            "temporal order of observations and prevent data leakage. Three forecasting models — "
            "Lag-Informed Regression, ARIMA, and LSTM — are subsequently developed, trained, and "
            "evaluated under identical experimental conditions. Each model predicts the next-day price "
            "change, which is then reconstructed into the predicted next-day closing price presented "
            "in the dashboard."
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="fph-card">', unsafe_allow_html=True)
        st.markdown("#### Methodology & Evaluation")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Target Users**")
            for item in ["Budding Traders (Primary Users)", "Students and Learning Users", "Researchers and Data Analytics Developers", "Academic Community", "Future Researchers"]:
                st.markdown(f"✅ {item}")
        with c2:
            st.markdown("**Statistical Rigor**")
            st.markdown(
                "RMSE (Root Mean Square Error) serves as the primary evaluation metric due to its "
                "sensitivity to large errors."
            )
            st.info(
                "**Significance Testing:** To ensure that the performance differences between models "
                "(Lag Reg vs ARIMA vs LSTM) are not just due to random chance, **Diebold-Mariano (DM) "
                "tests within companies and stock-level Friedman tests across companies** are applied."
            )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            """
            <div class="fph-card" style="text-align:center; background:linear-gradient(135deg,#1e293b,#0f172a);">
                <h4 class="fph-title">Responsible Use Disclaimer</h4>
                <p class="fph-dim" style="max-width:640px; margin:0 auto;">
                This dashboard is strictly an educational and analytical decision-support tool. It relies
                solely on historical market data and does not account for breaking news, economic shocks,
                geopolitical events, or unexpected market anomalies. The forecasts, model rankings, and
                insights provided do not constitute financial advice, buy-or-sell recommendations, automated
                trading signals, or guaranteed investment outcomes. Users should conduct their own research
                and consult qualified financial professionals before making investment decisions.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
