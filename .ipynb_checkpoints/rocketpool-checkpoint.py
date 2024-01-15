import requests 
import pandas as pd
import streamlit as st
from makerdao import dpi_history
from Lido import tbill
from formulas import *

api_key_cg = st.secrets["api_key_cg"]

@st.cache_data(ttl=86400)  # Corrected decorator for Streamlit
def get_rpl_historical_data(api_key):
    rpl_historical_api = "https://api.coingecko.com/api/v3/coins/rocket-pool/market_chart"
    params = {
        'vs_currency': 'usd',
        'days': '1152',
        'interval': 'daily',
        'x_cg_demo_api_key': api_key  # Add the API key as a parameter
    }
    response = requests.get(rpl_historical_api, params=params)  # Corrected API URL variable
    rpl_history = pd.DataFrame()
    rpl_market_cap = pd.DataFrame()

    if response.status_code == 200:
        data = response.json()
        rpl_history = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
        rpl_market_cap = pd.DataFrame(data['market_caps'], columns=['timestamp', 'marketcap'])

        # Convert 'timestamp' to datetime and set as index
        rpl_history['date'] = pd.to_datetime(rpl_history['timestamp'], unit='ms')
        rpl_market_cap['date'] = pd.to_datetime(rpl_market_cap['timestamp'], unit='ms')
        rpl_history.set_index('date', inplace=True)
        rpl_market_cap.set_index('date', inplace=True)

        # Drop the original 'timestamp' columns
        rpl_history.drop(columns='timestamp', inplace=True)
        rpl_market_cap.drop(columns='timestamp', inplace=True)

        return rpl_history, rpl_market_cap
    else:
        print(f"Failed to retrieve data: {response.status_code}")
        return rpl_history, rpl_market_cap  # Return empty DataFrames in case of failure

rpl_history, rpl_market_cap = get_rpl_historical_data(api_key_cg)

@st.cache_data(ttl=86400)  # Corrected decorator for Streamlit
def fetch_market_data(api_url, api_key):
    # Initialize variables to None
    market_value = None
    current_price = None
    supply = None

    # Add the API key as a query parameter
    params = {'x_cg_demo_api_key': api_key}
    response = requests.get(api_url, params=params)

    if response.status_code == 200:
        data = response.json()
        market_value = data['market_data']['market_cap']['usd']
        current_price = data['market_data']['current_price']['usd']
        supply = data['market_data']['circulating_supply']
    else:
        print(f"Failed to retrieve data: {response.status_code}")

    return market_value, current_price, supply

rplmrktapi = "https://api.coingecko.com/api/v3/coins/rocket-pool"
rpl_market_value, rpl_current_price, rpl_supply = fetch_market_data(rplmrktapi, api_key_cg)

rpl_history['rpl_daily_returns'] = rpl_history['price'].pct_change()

merged = rpl_history.merge(dpi_history, left_index=True, right_index=True)

merged = merged.dropna()

x = merged['daily_returns'].values.reshape(-1, 1)
y = merged['rpl_daily_returns'].values.reshape(-1, 1)

beta = calculate_beta(x,y)

beta_num = beta[0]

rpl_cagr = calculate_historical_returns(rpl_history)

print(rpl_cagr)

rpl_annual_returns = rpl_history.groupby(rpl_history.index.year).apply(calculate_annual_return)
rpl_annual_returns = pd.DataFrame(rpl_annual_returns)

excess_returns = rpl_annual_returns[0] - tbill['value']

rpl_avg_excess_return = excess_returns.mean()

print('rocketpool avg return:',rpl_avg_excess_return)
print('rocketpool beta:', beta)
print('rocketpool cagr:', rpl_cagr)


