import streamlit as st
import pandas as pd
from engine.optimizer_engine import run_optimizer
import matplotlib.pyplot as plt
import riskfolio as rp

def analysis_tab():
    st.title("📊 Strategy Builder & Optimizer")

    st.subheader("🧠 Strategy Parameters")
    col1, col2, col3 = st.columns(3)

    with col1:
        optimization_type = st.selectbox("Optimization Type", ["Mean-Variance", "Max Sharpe"])
    with col2:
        selected_universe = st.multiselect("Choose Sectors", [
            "Technology", "Healthcare", "Finance", "Consumer", "Energy", "Retail", "Utilities", "Industrials"
        ])
    with col3:
        risk_aversion = st.slider("Risk Aversion", 0.0, 10.0, 2.0)

    st.markdown("### 🔧 Constraints")
    col4, col5 = st.columns(2)
    with col4:
        max_weight = st.slider("Max Weight per Stock (%)", 5, 100, 20)
    with col5:
        max_holdings = st.slider("Max Number of Holdings", 5, 50, 15)

    tracking_error_limit = st.number_input("Tracking Error Limit (future use)", value=0.05)

    if st.button("🚀 Run Optimization"):
        with st.spinner("Optimizing portfolio using Riskfolio-Lib..."):
            try:
                weights_df, port_obj = run_optimizer(
                    sector_selection=selected_universe,
                    risk_aversion=risk_aversion,
                    tracking_error_limit=tracking_error_limit,
                    optimization_type=optimization_type,
                    max_weight=max_weight / 100,
                    max_holdings=max_holdings
                )

                st.success("Portfolio optimized successfully!")

                st.subheader("📋 Optimized Portfolio Allocation")
                st.dataframe(weights_df)

                # Download button
                csv = weights_df.to_csv(index=False).encode("utf-8")
                st.download_button("Download Portfolio as CSV", csv, file_name="optimized_portfolio.csv")

                # Efficient frontier
                st.subheader("📈 Efficient Frontier")
                fig = plot_efficient_frontier(port_obj)
                st.pyplot(fig)

            except Exception as e:
                st.error(f"Optimization failed: {e}")

# 🔧 Efficient Frontier Plot
def plot_efficient_frontier(portfolio_object):
    try:
        frontier = portfolio_object.efficient_frontier(model='Classic', rm='MV', points=50)
        fig, ax = plt.subplots(figsize=(10, 5))
        rp.plot_frontier(portfolio_object, frontier=frontier, ax=ax, rm='MV', showfig=False)
        ax.set_title("Efficient Frontier")
        return fig
    except Exception as e:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f"Failed to plot frontier:\n{e}", ha='center')
        return fig
