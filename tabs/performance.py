import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from engine.data_loader import load_filtered_top_stocks

def performance_tab():
    st.title("📈 Portfolio Performance vs Benchmark")

    # Load stock + SPY data from Google Drive
    df = load_filtered_top_stocks()
    price_cols = df.columns[5:]

    # Detect portfolio from session (from analysis tab)
    if "optimized_portfolio" in st.session_state:
        st.success("📡 Using optimized portfolio from Analysis tab")
        weights_df = st.session_state["optimized_portfolio"]
    else:
        st.subheader("📤 Upload Optimized Portfolio")
        uploaded_file = st.file_uploader("Upload portfolio weights CSV", type=["csv"])
        if not uploaded_file:
            st.info("Please upload a portfolio file or run an optimization first.")
            return
        weights_df = pd.read_csv(uploaded_file)

    st.dataframe(weights_df)

    # Extract tickers and weights
    selected_tickers = weights_df['Ticker'].tolist()
    weights = dict(zip(weights_df['Ticker'], weights_df['Weight']))

    # Get prices for portfolio tickers
    df_portfolio = df[df['Ticker'].isin(selected_tickers)][["Ticker"] + list(price_cols)].copy()
    df_portfolio[price_cols] = df_portfolio[price_cols].replace('[\$,]', '', regex=True).astype(float)
    prices = df_portfolio.set_index("Ticker")[price_cols].T
    prices.index = pd.to_datetime(prices.index, errors='coerce')
    prices = prices.dropna(how="all")
    prices = prices.loc[:, prices.columns.intersection(selected_tickers)].dropna()

    # Get SPY benchmark data
    df_spy = df[df['Ticker'] == "SPY"][["Ticker"] + list(price_cols)].copy()
    df_spy[price_cols] = df_spy[price_cols].replace('[\$,]', '', regex=True).astype(float)
    spy_series = df_spy.set_index("Ticker")[price_cols].T.squeeze()
    spy_series.index = pd.to_datetime(spy_series.index, errors='coerce')
    spy_series = spy_series.dropna()
    spy_series = spy_series.loc[prices.index.intersection(spy_series.index)]  # Align dates

    # Align prices
    prices = prices.loc[spy_series.index]
    norm_prices = prices / prices.iloc[0]
    weighted_returns = norm_prices.multiply([weights[t] for t in norm_prices.columns], axis=1)
    portfolio = weighted_returns.sum(axis=1)

    # Normalize SPY
    spy_norm = spy_series / spy_series.iloc[0]

    # Plot
    st.subheader("📊 Portfolio vs S&P 500")
    fig, ax = plt.subplots()
    portfolio.plot(ax=ax, label="Optimized Portfolio", linewidth=2)
    spy_norm.plot(ax=ax, label="S&P 500 (SPY)", linewidth=2, linestyle="--")
    ax.set_title("Cumulative Return")
    ax.legend()
    st.pyplot(fig)

    # Metrics
    st.subheader("📋 Performance Metrics")

    def calc_metrics(series):
        returns = series.pct_change().dropna()
        total = (series.iloc[-1] - 1) * 100
        sharpe = returns.mean() / returns.std() * (252**0.5)
        drawdown = (series / series.cummax() - 1).min() * 100
        return total, sharpe, drawdown

    p_total, p_sharpe, p_dd = calc_metrics(portfolio)
    s_total, s_sharpe, s_dd = calc_metrics(spy_norm)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Return (Portfolio)", f"{p_total:.2f}%")
    col2.metric("Sharpe Ratio (Portfolio)", f"{p_sharpe:.2f}")
    col3.metric("Max Drawdown (Portfolio)", f"{p_dd:.2f}%")

    col4, col5, col6 = st.columns(3)
    col4.metric("Total Return (SPY)", f"{s_total:.2f}%")
    col5.metric("Sharpe Ratio (SPY)", f"{s_sharpe:.2f}")
    col6.metric("Max Drawdown (SPY)", f"{s_dd:.2f}%")
