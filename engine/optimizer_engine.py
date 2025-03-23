import pandas as pd
import numpy as np
import riskfolio as rp
import streamlit as st
import requests
import io

# 🔗 Google Drive CSV File Loader (with YOUR file ID)
@st.cache_data(show_spinner="📥 Loading data from Google Drive...")
def load_portfolio_csv_from_drive():
    file_id = "1XyOn6UwAvKvvxMeHkOGYrY4txIOpFwdy"
    url = f"https://drive.google.com/uc?id={file_id}"
    r = requests.get(url)
    if r.status_code != 200:
        raise ValueError("❌ Failed to fetch data from Google Drive.")
    return pd.read_csv(io.StringIO(r.text))

def run_optimizer(sector_selection, min_market_cap_bil, risk_aversion,
                  tracking_error_limit, optimization_type,
                  max_weight=0.2, max_holdings=15):

    # Load Google Drive-hosted CSV
    df = load_portfolio_csv_from_drive()

    # Filter by Sector and Market Cap
    if sector_selection:
        df = df[df['Sector'].isin(sector_selection)]
    df = df[df['Marketcap'] >= min_market_cap_bil * 1e9]

    if df.empty:
        raise ValueError("❌ No tickers match the selected filters.")

    # Extract price data
    price_cols = df.columns[5:]  # First 5 columns are: Ticker, Company, Marketcap, Sector, Industry
    df_prices = df[['Ticker'] + list(price_cols)].copy()
    df_prices[price_cols] = df_prices[price_cols].replace('[\$,]', '', regex=True).astype(float)

    # Transpose to get (dates × tickers)
    prices = df_prices.set_index("Ticker")[price_cols].T
    prices.index = pd.to_datetime(prices.index, errors='coerce')
    prices = prices.dropna(how="all")

    st.info(f"📊 Initial price matrix: {prices.shape[0]} days × {prices.shape[1]} tickers")

    # Filter tickers with less than 30% missing values
    valid_tickers = prices.columns[prices.isna().mean() < 0.3]
    prices = prices[valid_tickers].dropna()

    st.info(f"🧼 Tickers after cleaning: {len(valid_tickers)}")

    if len(valid_tickers) < 2:
        raise ValueError("❌ Not enough valid tickers to run optimization.")

    # Calculate returns
    returns = prices.pct_change().dropna()

    # Build Riskfolio-Lib Portfolio
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

    # Final formatting
    sector_map = dict(zip(df['Ticker'], df['Sector']))
    weights_df = weights.reset_index()
    weights_df.columns = ['Ticker', 'Weight']
    weights_df['Sector'] = weights_df['Ticker'].map(sector_map)
    weights_df['Contribution'] = weights_df['Weight'] * 100

    return weights_df, port
