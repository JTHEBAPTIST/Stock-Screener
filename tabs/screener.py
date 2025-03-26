# tabs/screener.py

import streamlit as st
import pandas as pd
from engine.screener_engine import run_screener

def screener_tab():
    st.title("Equity Screener")

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
