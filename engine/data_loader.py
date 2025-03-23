import pandas as pd
import requests
import io
import streamlit as st

@st.cache_data(show_spinner="📦 Loading data from Google Drive...")
def load_filtered_top_stocks():
    file_id = "1XyOn6UwAvKvvxMeHkOGYrY4txIOpFwdy"
    url = f"https://drive.google.com/uc?id={file_id}"
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError("❌ Failed to download file from Google Drive.")
    return pd.read_csv(io.StringIO(response.text))
