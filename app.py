# app.py

import streamlit as st
from tabs import screener  # You can import performance, analysis later

# Define available tabs
TABS = {
    "Screener": screener.screener_tab,
    # "Performance": performance.performance_tab,
    # "Analysis": analysis.analysis_tab
}

# Page config
st.set_page_config(
    page_title="Quant Equity Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main title and sidebar
st.title("📈 Quantitative Equity Dashboard")
st.sidebar.title("Navigation")

# Tab selector
selected_tab = st.sidebar.radio("Choose a Tab", list(TABS.keys()))

# Run selected tab
TABS[selected_tab]()
