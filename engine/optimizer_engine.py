import pandas as pd
import numpy as np
import yfinance as yf
import riskfolio as rp
import streamlit as st  # to show log info inside UI

def run_optimizer(sector_selection, risk_aversion, tracking_error_limit,
                  optimization_type, max_weight=0.2, max_holdings=15):
    
    # Load stock universe
    df = pd.read_csv("data/tickers_full_cleaned.csv")

    # Filter by selected sectors
    if sector_selection:
        df = df[df['FactSet Econ Sector'].isin(sector_selection)]

    if df.empty:
        raise ValueError("No tickers found for the selected sectors. Please check your filter.")

    tickers = df["FDS Symbol Ticker"].dropna().unique().tolist()
    sector_map = dict(zip(df["FDS Symbol Ticker"], df["FactSet Econ Sector"]))

    # Download price data
    st.caption(f"📡 Downloading price data for {len(tickers)} tickers...")
    price_data = yf.download(tickers, start="2021-01-01", end="2024-01-01")["Adj Close"]

    # Filter tickers: allow tickers with <10% missing data
    good_tickers = price_data.columns[price_data.isna().mean() < 0.1]
    price_data = price_data[good_tickers].dropna()

    st.caption(f"✅ {len(good_tickers)} tickers passed data quality check.")

    # Calculate returns
    returns = price_data.pct_change().dropna()

    if returns.shape[1] < 2:
        raise ValueError("Not enough tickers with valid data after filtering.")

    # Update sector mapping to only include valid tickers
    filtered_tickers = returns.columns.tolist()
    sector_map = {k: v for k, v in sector_map.items() if k in filtered_tickers}

    # Build portfolio
    port = rp.Portfolio(returns=returns)
    port.assets_stats(method_mu='hist', method_cov='ledoit')

    model = 'Classic'
    rm = 'MV'
    obj = 'Sharpe' if optimization_type == 'Max Sharpe' else 'MinRisk'

    weights = port.optimization(
        model=model,
        rm=rm,
        obj=obj,
        rf=0.02,
        l=risk_aversion,
        hist=True,
        upper_bounds=max_weight,
        lower_bounds=0.01,
        maxnumassets=min(max_holdings, len(filtered_tickers))
    )

    # Format weights
    weights_df = weights.reset_index()
    weights_df.columns = ['Ticker', 'Weight']
    weights_df['Sector'] = weights_df['Ticker'].map(sector_map)
    weights_df['Contribution'] = weights_df['Weight'] * 100

    return weights_df, port
