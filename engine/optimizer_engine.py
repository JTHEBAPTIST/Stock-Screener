import pandas as pd
import numpy as np
import riskfolio as rp
import streamlit as st
import requests
import io

from engine.data_loader import load_filtered_top_stocks

def run_optimizer(sector_selection, min_market_cap_bil, risk_aversion,
                  tracking_error_limit, optimization_type, max_weight=0.2, max_holdings=15):

    # 🔗 Load full dataset
    df = load_filtered_top_stocks()

    # --- Apply Filters ---
    if sector_selection:
        df = df[df['Sector'].isin(sector_selection)]
    df = df[df['Marketcap'] >= min_market_cap_bil * 1e9]

    if df.empty:
        st.warning("⚠️ No stocks found after filtering by sector and market cap.")
        raise ValueError("No tickers matched the selected filters.")

    # --- Extract and clean price data ---
    price_cols = df.columns[5:]  # Assumes first 5 cols are metadata
    df_prices = df[["Ticker"] + list(price_cols)].copy()
    df_prices[price_cols] = df_prices[price_cols].replace('[\$,]', '', regex=True).astype(float)

    prices = df_prices.set_index("Ticker")[price_cols].T
    prices.index = pd.to_datetime(prices.index, errors='coerce')
    prices = prices.dropna(how="all")

    st.info(f"📊 Initial price matrix: {prices.shape[0]} days × {prices.shape[1]} tickers")

    # --- Filter tickers by data quality ---
    valid_tickers = prices.columns[prices.isna().mean() < 0.3]  # ← less strict (30% missing OK)
    prices = prices[valid_tickers].dropna(axis=0, how="any")  # Drop dates with any NaNs

    st.info(f"🧼 Valid tickers after cleaning: {len(valid_tickers)}")

    if len(valid_tickers) < 2:
        st.error("❌ No tickers passed data quality check. Try selecting more sectors or lowering market cap filter.")
        raise ValueError("Not enough clean tickers to run optimization.")

    # --- Calculate returns ---
    returns = prices.pct_change().dropna()

    # --- Sector mapping ---
    sector_map = dict(zip(df['Ticker'], df['Sector']))

    # --- Build & run optimizer ---
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
        maxnumassets=min(max_holdings, len(returns.columns))
    )

    # --- Format weights output ---
    weights_df = weights.reset_index()
    weights_df.columns = ['Ticker', 'Weight']
    weights_df['Sector'] = weights_df['Ticker'].map(sector_map)
    weights_df['Contribution'] = weights_df['Weight'] * 100

    return weights_df, port
