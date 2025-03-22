import streamlit as st
import pandas as pd
import yfinance as yf
from engine.backtest_engine import run_backtest
from engine.metrics import calculate_metrics
import matplotlib.pyplot as plt

def performance_tab():
    st.title("📈 Portfolio Performance Backtest")

    st.subheader("🔁 Upload Portfolio Weights")
    uploaded_file = st.file_uploader("Upload Portfolio CSV", type=["csv"])

    if uploaded_file:
        weights_df = pd.read_csv(uploaded_file)
        st.success("Portfolio uploaded successfully!")
        st.dataframe(weights_df)

        benchmark = st.selectbox("Benchmark Ticker", ["SPY", "QQQ", "VTI"])
        start_date = st.date_input("Backtest Start", pd.to_datetime("2021-01-01"))
        end_date = st.date_input("Backtest End", pd.to_datetime("2024-01-01"))

        if st.button("Run Backtest"):
            with st.spinner("Running portfolio backtest..."):
                try:
                    perf_df, metrics = run_backtest(weights_df, start_date, end_date, benchmark)

                    st.subheader("📊 Performance Metrics")
                    for k, v in metrics.items():
                        st.metric(k, f"{v:.2f}" if isinstance(v, float) else v)

                    st.subheader("📈 Cumulative Return vs Benchmark")
                    st.line_chart(perf_df)

                    # Optional: Download performance results
                    csv = perf_df.to_csv(index=True).encode('utf-8')
                    st.download_button("Download Results", csv, file_name="performance.csv")

                except Exception as e:
                    st.error(f"Backtest failed: {e}")
    else:
        st.info("Please upload a CSV file containing tickers and weights.")
