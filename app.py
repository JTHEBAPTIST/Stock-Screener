import streamlit as st
from tabs import screener, analysis, performance  # Now including performance

# Define all available tabs
TABS = {
    "Screener": screener.screener_tab,
    "Analysis": analysis.analysis_tab,
    "Performance": performance.performance_tab  # ✅ Included here!
}

# Streamlit page config
st.set_page_config(page_title="Quantitative Equity Dashboard", layout="wide")

# Sidebar
st.sidebar.title("Navigation")
selected_tab = st.sidebar.radio("Choose a Tab", list(TABS.keys()))

# Render selected tab
TABS[selected_tab]()
