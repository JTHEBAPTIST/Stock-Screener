import pandas as pd
import numpy as np
import riskfolio as rp
import streamlit as st
import requests
import io

# ✅ Load static company metadata from GitHub repo (data folder)
@st.cache_data(show_spinner="📥 Loading static metadata...")
def load_static_metadata():
    return pd.read_csv("data/Static data.csv")

# ✅ Load time series price data from Google Drive CSV
@st.cache_data(show_spinner="📈 Loading time series price data...")
def load_price_data_from_drive():
    file_id = "1k2zUxdD5OcdeINR2gWJYGAo5GxBmkbPr"
    url = f"https://drive.google.com/uc?id={file_id}&export=download"
    r = requests.get(url)
    if r.status_code != 200:
        raise ValueError("❌ Failed to load price data from Google Drive.")
    df = pd.read_csv(io.StringIO(r.text))
    return df

# ✅ Main Optimizer
def run_optimizer(sector_selection, min_market_cap_bil, risk_aversion,
                  tracking_error_limit, optimization_type,
                  max_weight=0.2, max_holdings=15):

    # Load data
    df_prices = load_price_data_from_drive()
    df_meta = load_static_metadata()

    # Merge price data with metadata on 'Ticker'
    df = pd.merge(df_prices, df_meta, on="Ticker", how="inner")

    # Validate structure
    required = ["Ticker", "Sector"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # --- Filter by Sector ---
    if sector_selection:
        df = df[df["Sector"].isin(sector_selection)]

    # --- Filter by Market Cap (if available) ---
    if "Marketcap" in df.columns:
        df = df[df["Marketcap"] >= min_market_cap_bil * 1e9]

    if df.empty:
        raise ValueError("❌ No tickers match your filters.")

    # --- Clean Price Data ---
    price_cols = df.columns.difference(["Ticker", "Company", "Sector", "Industry", "Marketcap"])
    df_prices_clean = df[["Ticker"] + list(price_cols)].copy()
    df_prices_clean[price_cols] = df_prices_clean[price_cols].replace(r'[\$,]', '', regex=True).astype(float)

    # Transpose to (dates × tickers)
    prices = df_prices_clean.set_index("Ticker")[price_cols].T
    prices.index = pd.to_datetime(prices.index, errors='coerce')
    prices = prices.dropna(how="all")

    st.info(f"📊 Price matrix: {prices.shape[0]} days × {prices.shape[1]} tickers")

    # --- Drop columns with too much missing data ---
    valid_tickers = prices.columns[prices.isna().mean() < 0.3]
    prices = prices[valid_tickers].dropna(axis=1)
    st.info(f"🧼 Tickers after cleaning: {len(valid_tickers)}")

    if len(valid_tickers) < 2:
        raise ValueError("❌ Not enough valid tickers after cleaning.")

    # --- Returns ---
    returns = prices.pct_change().dropna()

    # --- Riskfolio Optimizer ---
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
    sector_map = dict(zip(df["Ticker"], df["Sector"]))
    weights_df = weights.reset_index()
    weights_df.columns = ["Ticker", "Weight"]
    weights_df["Sector"] = weights_df["Ticker"].map(sector_map)
    weights_df["Contribution"] = weights_df["Weight"] * 100

    return weights_df, port
