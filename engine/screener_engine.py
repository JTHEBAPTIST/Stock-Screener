# engine/screener_engine.py

import pandas as pd

def run_screener(exchanges, sectors, min_market_cap):
    # Load the full cleaned dataset
    df = pd.read_csv("data/tickers_full_cleaned.csv")

    # Apply exchange filter if selected
    if exchanges:
        df = df[df['Exchange Name (VND)'].isin(exchanges)]

    # Apply sector filter if selected
    if sectors:
        df = df[df['FactSet Econ Sector'].isin(sectors)]

    # Optional market cap filter — placeholder until market cap is available in data
    if '5000L by MCAP ($MM)' in df.columns:
        df = df[df['5000L by MCAP ($MM)'] > (min_market_cap * 1000)]

    return df.reset_index(drop=True)

