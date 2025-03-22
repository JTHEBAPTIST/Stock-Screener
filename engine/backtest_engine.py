import yfinance as yf
import pandas as pd
import numpy as np
from engine.metrics import calculate_metrics

def run_backtest(weights_df, start_date, end_date, benchmark="SPY"):
    tickers = weights_df['Ticker'].tolist()
    weights = weights_df.set_index("Ticker")["Weight"]

    # Download historical data
    price_data = yf.download(tickers + [benchmark], start=start_date, end=end_date)["Adj Close"]
    price_data = price_data.dropna()

    # Normalize prices and calculate portfolio return
    norm_prices = price_data / price_data.iloc[0]
    portfolio_prices = (norm_prices[tickers] * weights).sum(axis=1)
    benchmark_prices = norm_prices[benchmark]

    perf_df = pd.DataFrame({
        "Portfolio": portfolio_prices,
        "Benchmark": benchmark_prices
    })

    metrics = calculate_metrics(perf_df)
    return perf_df, metrics
