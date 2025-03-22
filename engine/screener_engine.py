# tabs/screener.py

import streamlit as st
import pandas as pd
from engine.screener_engine import run_screener

def screener_tab():
    st.title("Equity Screener")

    # ----- Filters Panel (Top) -----
    st.subheader("Select Screening Criteria")
    col1, col2, col3 = st.columns(3)

    with col1:
        exchange = st.multiselect(
            "Exchange",
            ["NASDAQ", "NYSE", "AMEX"],
            default=["NASDAQ", "NYSE", "AMEX"]
        )
    
    with col2:
        sector = st.multiselect(
            "Sector",
            ["Technology", "Healthcare", "Finance", "Consumer", "Energy", "Retail", "Utilities", "Industrials"],
            default=[]
        )

    with col3:
        min_market_cap = st.number_input(
            "Min Market Cap (Billions USD)",
            value=10.0,
            step=1.0
        )

    st.markdown("---")

    # ----- Optional: Search Bar & Toggle -----
    ticker_search = st.text_input("Search for a Ticker or Company Name")
    show_all_columns = st.checkbox("Show All Columns", value=False)

    df = run_screener(exchange, sector, min_market_cap)

    if ticker_search:
        df = df[df['FDS Symbol Ticker'].str.contains(ticker_search, case=False, na=False) |
                df['Company Name'].str.contains(ticker_search, case=False, na=False)]

    default_cols = [
        "Company Name", "CUSIP", "Company Sedol", "FactSet Econ Sector", "FactSet Ind",
        "Gen Sec Type Desc", "Nation", "Curncy Name", "Exchange Name (VND)", "Latest Price",
        "180D Annualized Std Dev.", "Simple Tot Ret (USD) Last Mo", "LTM Total Return",
        "LTM Total Return S&P 500", "Last 12 Month Excess Return", "3y ALPHA Rel to Loc Idx",
        "In Buy List", "S&P 500 60M Std Dev", "Bid Price", "Ask Price", "22D ADV ($MM)",
        "5000L by MCAP ($MM)", "Max Score", "Min Score", "1 Mo Fwd Return",
        "V&M Model Score", "V&M Score (IQR)", "PEG Model Score (W)", "PEG Model Score (IQR)",
        "Multi Factor Model Score (W)", "Multi Factor Model Score (IQR)",
        "N(0,1) Model Score", "N(0,Sigma) Model Score"
    ]

    if show_all_columns:
        display_df = df.copy()
    else:
        display_columns = st.multiselect("Select Columns to Display", df.columns.tolist(), default=default_cols)
        display_df = df[display_columns] if display_columns else df

    st.markdown("---")
    st.subheader(f"Model Overview: {len(display_df)} Stocks")
    st.dataframe(display_df)

    # ----- Expandable Model Sections -----
    with st.expander("🔍 PERFORMANCE"):
        perf_cols = [col for col in df.columns if "Return" in col or "Alpha" in col or "Std Dev" in col]
        st.dataframe(df[["FDS Symbol Ticker", "Company Name"] + perf_cols])

    with st.expander("📈 MODEL 1: VAL & MOM"):
        vm_cols = [col for col in df.columns if "V&M" in col]
        st.dataframe(df[["FDS Symbol Ticker", "Company Name"] + vm_cols])

    with st.expander("📊 MODEL 2: PEG"):
        peg_cols = [col for col in df.columns if "PEG" in col]
        st.dataframe(df[["FDS Symbol Ticker", "Company Name"] + peg_cols])

    with st.expander("🧠 MODEL 3: MULTI FACTOR"):
        mf_cols = [col for col in df.columns if "Multi Factor" in col]
        st.dataframe(df[["FDS Symbol Ticker", "Company Name"] + mf_cols])

    with st.expander("🧪 MODEL 4: NORM SCORE"):
        norm_cols = [col for col in df.columns if "Model Score" in col and "Multi" not in col and "V&M" not in col and "PEG" not in col]
        st.dataframe(df[["FDS Symbol Ticker", "Company Name"] + norm_cols])

    # ----- CSV Download -----
    csv = display_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Displayed Data as CSV", data=csv, file_name="screened_stocks.csv", mime='text/csv')
