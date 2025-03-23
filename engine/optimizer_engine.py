import pandas as pd
import numpy as np
import riskfolio as rp
import streamlit as st
import requests
import io

# ✅ NEW Google Sheets Loader (converted to direct CSV export)
@st.cache_data(show_spinner="📥 Loading portfolio data from Google Sheets...")
def load_portfolio_csv_from_drive():
    url = "https://docs.google.com/spreadsheets/d/1b116oKivSrDqi6UkrALD5-FiFcsun2nn/export?format=csv"
    r = requests.get(url)
    if r.status_code != 200:
        raise ValueError("❌ Failed to fetch Google Sheets CSV.")
    return pd.read_csv(io.StringIO(r.text))


def run_optimizer(sector_selection, min_market_cap_bil, risk_aversion,
                  tracking_error_limit, optimization_type,
                  max_weight=0.2, max_holdings=15):

    # Load cleaned full dataset from Google Sheets
    df = load_portfolio_csv_from_drive()

    # --- Filter by sector ---
    if sector_selection:
        df = df[df['Sector'].isin(sector_selection)]

    # --- Filter by market cap ---
    df = df[df['Marketcap'] >= min_market_cap_bil * 1e9]

    if df.empty:
        raise ValueError("❌ No tickers match the selected filters.")

    # --- Clean and parse price data ---
    price_cols = df.columns[5:]  # assumes first 5 = Ticker, Company, Marketcap, Sector, Industry
    df_prices = df[['Ticker'] + list(price_cols)].copy()
    df_prices[price_cols] = df_prices[price_cols].replace(r'[\$,]', '', regex=True).astype(float)

    prices = df_prices.set_index("Ticker")[price_cols].T
    prices.index = pd.to_datetime(prices.index, errors='coerce')
    prices = prices.dropna(how="all")

    st.info(f"📊 Price matrix loaded: {prices.shape[0]} days × {prices.shape[1]} tickers")

    # --- Filter tickers with <30% missing values ---
    valid_tickers = prices.columns[prices.isna().mean() < 0.3]
    prices = prices[valid_tickers].dropna(axis=1)

    st.info(f"🧼 Tickers after cleaning: {len(valid_tickers)}")

    if len(valid_tickers) < 2:
        raise ValueError("❌ Not enough valid tickers to run optimization.")

    # --- Calculate returns ---
    returns = prices.pct_change().dropna()

    # --- Run Riskfolio-Lib optimizer ---
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

    # --- Format results ---
    sector_map = dict(zip(df['Ticker'], df['Sector']))
    weights_df = weights.reset_index()
    weights_df.columns = ['Ticker', 'Weight']
    weights_df['Sector'] = weights_df['Ticker'].map(sector_map)
    weights_df['Contribution'] = weights_df['Weight'] * 100

    return weights_df, port
