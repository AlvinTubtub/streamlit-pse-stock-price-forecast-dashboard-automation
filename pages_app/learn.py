from __future__ import annotations

import streamlit as st


def render() -> None:
    st.markdown(
        '<h1 class="fph-title" style="font-size:2.2rem; text-align:center;">Understanding the Forecasts</h1>'
        '<p class="fph-dim" style="text-align:center; max-width:640px; margin:0 auto;">'
        "A beginner-friendly guide to the data, models, and metrics used in this capstone research.</p>",
        unsafe_allow_html=True,
    )
    st.write("")

    st.markdown('<div class="fph-card">', unsafe_allow_html=True)
    st.markdown('<h2 class="fph-brand" style="font-size:1.4rem;">📋 What is OHLCV?</h2>', unsafe_allow_html=True)
    st.markdown(
        '<p class="fph-dim">Before we can forecast the future, we have to look at the past. Our models '
        "learn purely from numerical historical data — the basic building blocks of daily stock market "
        "records. We do not use news sentiment, rumors, or economic indicators.</p>",
        unsafe_allow_html=True,
    )
    cols = st.columns(5)
    items = [
        ("O", "Open", "Price when market opens.", "#60a5fa"),
        ("H", "High", "Highest price of the day.", "#4ade80"),
        ("L", "Low", "Lowest price of the day.", "#f87171"),
        ("C", "Close (target)", "Final price. What we predict.", "#93c5fd"),
        ("V", "Volume", "Total shares traded.", "#c084fc"),
    ]
    for col, (letter, label, desc, color) in zip(cols, items):
        with col:
            st.markdown(
                f"""
                <div style="background:#0f172a; border:1px solid #334155; border-radius:0.75rem; padding:1rem; text-align:center;">
                    <div style="color:{color}; font-weight:700; font-size:1.3rem;">{letter}</div>
                    <div style="color:white; font-size:0.85rem; font-weight:600;">{label}</div>
                    <div class="fph-faint">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<h2 class="fph-title" style="text-align:center; font-size:1.5rem;">The Forecasting Models</h2>', unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    models = [
        (m1, "🧮", "Lag Regression", '"The Pattern Spotter"', "An interpretable model. It studies recent prices and volume, uses PACF to find meaningful lag relationships, and LASSO regularization to keep only the strongest predictors."),
        (m2, "📈", "ARIMA", '"The Trend Tracker"', "A traditional statistical time-series standard. It analyzes trends, differencing (to stabilize data), and past forecasting errors to estimate the next day's close."),
        (m3, "🧠", "LSTM", '"The Deep Thinker"', "A deep-learning recurrent neural network. It studies sequences of historical prices, learning to 'remember' long-term patterns and 'forget' irrelevant noise."),
    ]
    for col, icon, name, tag, desc in models:
        with col:
            st.markdown(
                f"""
                <div class="fph-card">
                    <div style="font-size:1.6rem;">{icon}</div>
                    <h3 class="fph-title" style="font-size:1.2rem; margin:0.5rem 0 0 0;">{name}</h3>
                    <p class="fph-brand" style="font-size:0.85rem;">{tag}</p>
                    <p class="fph-dim" style="font-size:0.85rem;">{desc}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown('<h2 class="fph-title" style="text-align:center; font-size:1.5rem;">The Metrics: How Do We Know if a Model is Good?</h2>', unsafe_allow_html=True)
    st.markdown(
        '<p class="fph-dim" style="text-align:center; max-width:640px; margin:0 auto 1rem auto;">'
        "We grade each model by comparing its predictions against the stock's actual price.</p>",
        unsafe_allow_html=True,
    )
    metric_cols = st.columns(2)
    metrics = [
        ("RMSE", "Our Main Grading Score", "Average prediction error in Pesos, but heavily penalizes huge misses.", "Lower is better", "#60a5fa"),
        ("MAE", "The Average Miss", "The average difference between predicted and actual price, in Pesos.", "Lower is better", "#4ade80"),
        ("MASE", "The Scale-Free Miss", "Compares the model's error against a naive benchmark (like guessing yesterday's price). If MASE is below 1.0, the model is doing better than just blindly guessing the trend. Because it's a ratio, we can fairly compare models across cheap and expensive stocks.", "Lower is better", "#facc15"),
        ("R²", "The Explanation Score", "What percentage of the stock's actual movement the model successfully captured.", "Higher is better", "#c084fc"),
    ]
    for i, (name, tag, desc, badge, color) in enumerate(metrics):
        with metric_cols[i % 2]:
            st.markdown(
                f"""
                <div class="fph-card">
                    <h3 style="color:{color}; font-size:1.1rem; margin:0;">{name}</h3>
                    <p style="color:white; font-size:0.9rem; font-weight:600;">{tag}</p>
                    <p class="fph-dim" style="font-size:0.85rem;">{desc}</p>
                    <span class="fph-badge fph-badge-green">{badge}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.warning(
        "**Forecasts Are Not Financial Advice.** These models look only at numerical history. Real "
        "stock prices are affected by breaking news, economic reports, corporate earnings surprises, "
        "natural disasters, and human sentiment — none of which are captured by OHLCV data alone. Use "
        "this tool to learn about data analytics, not to risk real money."
    )
