def calculate_metrics(df):
    returns = df.pct_change().dropna()

    sharpe = returns["Portfolio"].mean() / returns["Portfolio"].std() * (252 ** 0.5)
    drawdown = (df["Portfolio"] / df["Portfolio"].cummax() - 1).min()
    total_return = df["Portfolio"].iloc[-1] - 1
    excess_return = (returns["Portfolio"] - returns["Benchmark"]).mean() * 252

    return {
        "Total Return (%)": total_return * 100,
        "Sharpe Ratio": sharpe,
        "Max Drawdown (%)": drawdown * 100,
        "Alpha (vs Benchmark)": excess_return
    }
