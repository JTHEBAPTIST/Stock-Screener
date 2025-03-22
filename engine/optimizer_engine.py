import pandas as pd
import numpy as np
import yfinance as yf
import riskfolio as rp

def run_optimizer(sector_selection, risk_aversion, tracking_error_limit,
                  optimization_type, max_weight=0.2, max_holdings=15):
    
    # Load tickers and sector info from your cleaned CSV
    df = pd.read_csv("data/tickers_full_cleaned.csv")

    # Filter by selected sectors
    if sector_selection:
        df = df[df['FactSet Econ Sector'].isin(sector_selection)]

    tickers = df["FDS Symbol Ticker"].dropna().unique().tolist()
    sector_map = dict(zip(df["FDS Symbol Ticker"], df["FactSet Econ Sector"]))

    # Download historical price data
    price_data = yf.download(tickers, start="2021-01-01", end="2024-01-01")["Adj Close"]
    price_data = price_data.dropna(axis=1)

    # Calculate daily returns
    returns = price_data.pct_change().dropna()
    if returns.shape[1] < 2:
        raise ValueError("Not enough tickers with valid data after filtering.")

    # Re-map sector info after dropping missing tickers
    filtered_tickers = returns.columns.tolist()
    sector_map = {k: v for k, v in sector_map.items() if k in filtered_tickers}

    # Create portfolio object
    port = rp.Portfolio(returns=returns)
    port.assets_stats(method_mu='hist', method_cov='ledoit')

    # Select objective
    model = 'Classic'
    rm = 'MV'
    obj = 'Sharpe' if optimization_type == 'Max Sharpe' else 'MinRisk'

    # Run optimizer
    weights = port.optimization(
        model=model,
        rm=rm,
        obj=obj,
        rf=0.02,
        l=risk_aversion,
        hist=True,
        upper_bounds=max_weight,
        lower_bounds=0.01,
        maxnumassets=max_holdings
    )

    # Return formatted weights + portfolio object
    weights_df = weights.reset_index()
    weights_df.columns = ['Ticker', 'Weight']
    weights_df['Sector'] = weights_df['Ticker'].map(sector_map)
    weights_df['Contribution'] = weights_df['Weight'] * 100  # Placeholder for now

    return weights_df, port
