import pandas as pd
import numpy as np
import riskfolio as rp
import streamlit as st

def run_optimizer(sector_selection, min_market_cap_bil, risk_aversion,
                  tracking_error_limit, optimization_type, max_weight=0.2, max_holdings=15):
    
    # Load full data
    df = pd.read_csv("data/filtered_top_stocks.csv")
    
    # Filter: sector & market cap
    if sector_selection:
        df = df[df['Sector'].isin(sector_selection)]
    df = df[df['Marketcap'] >= min_market_cap_bil * 1e9]

    if df.empty:
        raise ValueError("No tickers matched the selected filters.")

    # Extract metadata and price data
    metadata_cols = ['Ticker', 'Sector']
    price_cols = df.columns[5:]  # Assuming first 5 columns are metadata

    price_data = df[["Ticker"] + list(price_cols)].copy()
    price_data[price_cols] = price_data[price_cols].replace('[\$,]', '', regex=True).astype(float)

    # Reshape: rows = dates, columns = tickers
    df_prices = price_data.set_index("Ticker")[price_cols].T
    df_prices.index = pd.to_datetime(df_prices.index, errors='coerce')
    df_prices = df_prices.dropna(how="all")

    # Drop tickers with too much missing data
    valid_tickers = df_prices.columns[df_prices.isna().mean() < 0.1]
    price_data = df_prices[valid_tickers].dropna()

    st.caption(f"✅ {len(valid_tickers)} tickers passed data quality check.")

    # Calculate returns
    returns = price_data.pct_change().dropna()

    if returns.shape[1] < 2:
        raise ValueError("Not enough clean tickers to run optimization.")

    sector_map = dict(zip(df['Ticker'], df['Sector']))

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
        maxnumassets=min(max_holdings, len(returns.columns))
    )

    # Format results
    weights_df = weights.reset_index()
    weights_df.columns = ['Ticker', 'Weight']
    weights_df['Sector'] = weights_df['Ticker'].map(sector_map)
    weights_df['Contribution'] = weights_df['Weight'] * 100

    return weights_df, port
