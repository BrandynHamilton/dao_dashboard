import pandas as pd
import requests
import yfinance as yf
import streamlit as st


@st.cache_data
def fetch_data_from_api(api_url, params=None):
    response = requests.get(api_url, params=params)
    if response.status_code == 200:
        data = response.json()
        if 'rows' in data['result']:
            return pd.DataFrame(data['result']['rows'])
        return data
    else:
        print(f"Failed to retrieve data: {response.status_code}")
        return pd.DataFrame()  # or an empty dict
    
# API URLs and parameters
api_key = "NB6wsLRGqoVVQaLVuwk9mFB3kDssFVGK"

# MakerDAO API call
mkrdao_url = "https://api.dune.com/api/v1/query/2840463/results/"
mkrdao_params = {"api_key": api_key}
mkrdao = fetch_data_from_api(mkrdao_url, mkrdao_params)

st.write(mkrdao_df)