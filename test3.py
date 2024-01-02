import pandas as pd
import streamlit as st
import requests

# Modified function to return only ldo_history and ldo_market_cap
@st.cache_data(ttl=86400)
def get_ldo_historical_data():
    ldo_historical_api = "https://api.coingecko.com/api/v3/coins/lido-dao/market_chart?vs_currency=usd&days=1152&interval=daily"
    response = requests.get(ldo_historical_api)
    ldo_history = pd.DataFrame()
    ldo_market_cap = pd.DataFrame()

    if response.status_code == 200:
        data = response.json()
        ldo_history = pd.DataFrame(data['prices'], columns=['date', 'price'])
        ldo_market_cap = pd.DataFrame(data['market_caps'], columns=['date', 'marketcap'])

    return ldo_history, ldo_market_cap

# Call the function
ldo_history, ldo_market_cap = get_ldo_historical_data()

# Convert 'date' columns to datetime and set as index
ldo_history['date'] = pd.to_datetime(ldo_history['date'], unit='ms')
ldo_market_cap['date'] = pd.to_datetime(ldo_market_cap['date'], unit='ms')
ldo_history.set_index('date', inplace=True)
ldo_market_cap.set_index('date', inplace=True)

# Calculate ldo_supply
ldo_supply = ldo_market_cap['marketcap'] / ldo_history['price']

# Now you can use ldo_history, ldo_market_cap, and ldo_supply outside the function
# Example: print the first few rows of each DataFrame
st.write(ldo_history.head())
st.write(ldo_market_cap.head())
st.write(ldo_supply.head())
