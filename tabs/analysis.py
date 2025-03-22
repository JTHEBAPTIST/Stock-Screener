# tabs/analysis.py

import streamlit as st
from engine.optimizer_engine import run_optimizer


def analysis_tab():
    st.title("📐 Strategy Builder & Optimizer")

    # ----- Strategy Builder Panel -----
    st.subheader("Build Your Strategy")

    col1, col2 = st.columns(2)

    with col1:
        opt_type = st.selectbox("Select Optimization Type", [
            "Mean-Variance", "Max Sharpe", "Custom"
        ])

        selected_universe = st.multiselect("Choose Sectors/Industries", [
            "Technology", "Healthcare", "Finance", "Consumer", "Energy", "Industrials"
        ])

        risk_aversion = st.slider("Risk Aversion", 0.0, 10.0, 2.0)

    with col2:
        tracking_error_limit = st.number_input("Tracking Error Limit (%)", value=5.0)
        sector_cap = st.slider("Sector Cap (%)", min_value=0, max_value=100, value=25)
        min_holdings = st.number_input("Min Holdings", min_value=1, max_value=50, value=5)
        max_holdings = st.number_input("Max Holdings", min_value=5, max_value=100, value=20)

    # ----- Run Optimization -----
    if st.button("Run Optimization"):
        st.success("Running optimization...")
        weights = run_optimizer(
            universe=selected_universe,
            risk_aversion=risk_aversion,
            te_limit=tracking_error_limit,
            sector_cap=sector_cap,
            min_holdings=min_holdings,
            max_holdings=max_holdings,
            method=opt_type
        )

        if weights is not None and not weights.empty:
            st.subheader("📊 Optimized Portfolio")
            st.dataframe(weights)

            st.download_button(
                "Download Portfolio CSV",
                data=weights.to_csv(index=False),
                file_name="optimized_portfolio.csv",
                mime='text/csv'
            )
        else:
            st.warning("No optimized portfolio returned.")

    # ----- (Optional) Efficient Frontier Placeholder -----
    with st.expander("📈 Show Efficient Frontier (Coming Soon)"):
        st.info("Efficient frontier visualization will be added here.")
