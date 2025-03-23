import pandas as pd
import numpy as np
import riskfolio as rp
import streamlit as st

def run_optimizer(sector_selection, min_market_cap_bil, risk_aversion,
                  tracking_error_limit, optimization_type, max_weight=0.2, max_holdings=15):
    
    # Load full combined data
    df = pd.read_csv("data/filtered_top_stocks.csv")

    # Filter by selected sectors
    if sector_selection:
        df = df[df['Sector'].isin(sector_selection)]

    # Filter by market cap
    df = df[df['Marketcap'] >= min_market_cap_bil * 1e9]

    if df.empty:
        raise ValueError("No tickers matched the selected filters.")

    # Identify price columns
    price_cols = df.columns[5:]  # Assuming first 5 columns = Ticker, Company, Marketcap, Sector, Industry

    # Extract and clean price data
    df_prices = df[["Ticker"] + list(price_cols)].copy()
    df_prices[price_cols] = df_prices[price_cols].replace('[\$,]', '', regex=True).astype(float)

    # Transpose for Riskfolio format
    prices = df_prices.set_index("Ticker")[price_cols].T
    prices.index = pd.to_datetime(prices.index, errors='coerce')
    prices = prices.dropna(how="all")

    # Drop tickers with too much missing data
    valid_tickers = prices.columns[prices.isna().mean() < 0.1]
    prices = prices[valid_tickers].dropna()

    st.caption(f"✅ {len(valid_tickers)} tickers passed data quality check.")

    # Calculate returns
    returns = prices.pct_change().dropna()

    if returns.shape[1] < 2:
        raise ValueError("Not enough clean tickers to run optimization.")

    # Sector map for final output
    sector_map = dict(zip(df['Ticker'], df['Sector']))

    # Build Riskfolio portfolio
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

    # Finalize weights DataFrame
    weights_df = weights.reset_index()
    weights_df.columns = ['Ticker', 'Weight']
    weights_df['Sector'] = weights_df['Ticker'].map(sector_map)
    weights_df['Contribution'] = weights_df['Weight'] * 100

    return weights_df, port
