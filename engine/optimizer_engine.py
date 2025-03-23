import pandas as pd
import numpy as np
import riskfolio as rp
import streamlit as st
import requests
import io
import os

# ✅ Primary: Try Google Sheets
@st.cache_data(show_spinner="📥 Loading portfolio data...")
def load_portfolio_csv_from_drive():
    try:
        url = "https://docs.google.com/spreadsheets/d/1b116oKivSrDqi6UkrALD5-FiFcsun2nn/export?format=csv"
        r = requests.get(url)
        if r.status_code != 200:
            raise ValueError("Google Sheets not available.")
        return pd.read_csv(io.StringIO(r.text))
    except Exception as e:
        st.warning("⚠️ Google Sheets load failed. Using local Excel fallback.")
        local_path = "C:/Users/Admin/Downloads/stock_Fund_price.xlsx"
        if not os.path.exists(local_path):
            raise FileNotFoundError("❌ Local Excel file not found. Please check path.")
        return pd.read_excel(local_path)

# ✅ Main Optimization Function
def run_optimizer(sector_selection, min_market_cap_bil, risk_aversion,
                  tracking_error_limit, optimization_type,
                  max_weight=0.2, max_holdings=15):

    # Load from either Sheets or fallback Excel
    df = load_portfolio_csv_from_drive()

    # --- Column validation ---
    required_cols = ["Ticker", "Company", "Marketcap", "Sector", "Industry"]
    if not all(col in df.columns for col in required_cols[:3]):
        raise ValueError("Missing key columns: Ticker / Marketcap / Company")

    # --- Filter by sector and market cap ---
    if "Sector" in df.columns and sector_selection:
        df = df[df['Sector'].isin(sector_selection)]
    if "Marketcap" in df.columns:
        df = df[df['Marketcap'] >= min_market_cap_bil * 1e9]

    if df.empty:
        raise ValueError("❌ No tickers match your filters.")

    # --- Extract price data ---
    price_cols = df.columns[5:]
    df_prices = df[['Ticker'] + list(price_cols)].copy()
    df_prices[price_cols] = df_prices[price_cols].replace(r'[\$,]', '', regex=True).astype(float)

    prices = df_prices.set_index("Ticker")[price_cols].T
    prices.index = pd.to_datetime(prices.index, errors='coerce')
    prices = prices.dropna(how="all")

    st.info(f"📊 Price matrix: {prices.shape[0]} days × {prices.shape[1]} tickers")

    # --- Filter clean tickers ---
    valid_tickers = prices.columns[prices.isna().mean() < 0.3]
    prices = prices[valid_tickers].dropna(axis=1)

    st.info(f"🧼 Tickers after cleaning: {len(valid_tickers)}")

    if len(valid_tickers) < 2:
        raise ValueError("❌ Not enough clean tickers for optimization.")

    # --- Calculate returns ---
    returns = prices.pct_change().dropna()

    # --- Riskfolio Optimization ---
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

    # --- Final Output ---
    sector_map = dict(zip(df['Ticker'], df.get('Sector', 'Unknown')))
    weights_df = weights.reset_index()
    weights_df.columns = ['Ticker', 'Weight']
    weights_df['Sector'] = weights_df['Ticker'].map(sector_map)
    weights_df['Contribution'] = weights_df['Weight'] * 100

    return weights_df, port
