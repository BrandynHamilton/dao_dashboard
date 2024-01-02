import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import requests

def get_ldo_historical_data():
    ldo_historical_api = "https://api.coingecko.com/api/v3/coins/lido-dao/market_chart?vs_currency=usd&days=1152&interval=daily"
    response = requests.get(ldo_historical_api)
    ldo_history = pd.DataFrame()
    ldo_market_cap = pd.DataFrame()

    if response.status_code == 200:
        data = response.json()
        ldo_history = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
        ldo_market_cap = pd.DataFrame(data['market_caps'], columns=['timestamp', 'marketcap'])

        # Convert 'timestamp' to datetime and set as index
        ldo_history['date'] = pd.to_datetime(ldo_history['timestamp'], unit='ms')
        ldo_market_cap['date'] = pd.to_datetime(ldo_market_cap['timestamp'], unit='ms')
        ldo_history.set_index('date', inplace=True)
        ldo_market_cap.set_index('date', inplace=True)

        # Drop the original 'timestamp' columns
        ldo_history.drop(columns='timestamp', inplace=True)
        ldo_market_cap.drop(columns='timestamp', inplace=True)

    return ldo_history, ldo_market_cap

# Call the function
ldo_history, ldo_market_cap = get_ldo_historical_data()

# Assuming further processing happens here...
ldo_supply = ldo_market_cap['marketcap'] / ldo_history['price']


print(ldo_history)
print(ldo_market_cap)
print(ldo_supply)
