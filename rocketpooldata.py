#!/usr/bin/env python
# coding: utf-8

# # 

# In[147]:


import requests
import pandas as pd
import numpy as np


# In[148]:


rpl_assets_url = 'https://api.dune.com/api/v1/query/3351658/results?api_key=fKXF51FhPIqzpGn96BnOcEgw3P9rcJzG'
response = requests.get(rpl_assets_url)


# In[149]:


data = response.json()


# In[150]:


data_useful = pd.DataFrame(data['result']['rows'])


# In[151]:


print(data_useful)


# In[152]:


data_useful['day'] = pd.to_datetime(data_useful['day'])


# In[153]:


data_useful.set_index('day', inplace=True)


# In[154]:


import pandas as pd

# Assuming 'df' is your existing DataFrame and it has a DateTimeIndex
# Example: df = pd.DataFrame({'Cumulative_Net_Eth': [32, 96, ...], 'net_eth_added': [32, 64, ...]}, index=pd.to_datetime(['2021-11-02', '2021-11-05', ...]))

# Step 1: Create a date range
start_date = data_useful.index.min()
end_date = data_useful.index.max()
date_range = pd.date_range(start=start_date, end=end_date, freq='D')

# Step 2: Reindex your DataFrame
data_useful_reindexed = data_useful.reindex(date_range)

# Step 3: Forward fill missing values
data_useful_filled = data_useful_reindexed.fillna(method='ffill')

# Now 'df_filled' is your DataFrame with missing dates filled in


# In[155]:


data_useful_filled


# In[156]:


assets = data_useful_filled['Cumulative_Net_Eth'].to_frame('net_eth')

assets


# In[157]:


api_key = 'CG-jTsiV2rsyVSHHULoNSWHU493'


# In[158]:


def fetch_rpl_historical_data(api_url, api_key):
    # Use the API key either as a query parameter or in the headers
    params = {'vs_currency': 'usd', 'days': 'max', 'interval': 'daily', 'x_cg_demo_api_key': api_key}
    headers = {'x-cg-demo-api-key': api_key}  # Alternatively, use this header

    response = requests.get(api_url, params=params, headers=headers)

    if response.status_code == 200:
        # Parse the JSON response
        mkr_historical_pricedata = response.json()
        # Extract the 'prices' and 'market_caps' data
        mkr_historical_price = mkr_historical_pricedata['prices']
        mkr_market_cap = pd.DataFrame(mkr_historical_pricedata['market_caps'], columns=['date', 'marketcap'])

        # Convert the 'timestamp' column from UNIX timestamps in milliseconds to datetime objects
        mkr_history = pd.DataFrame(mkr_historical_price, columns=['timestamp', 'price'])
        mkr_history['date'] = pd.to_datetime(mkr_history['timestamp'], unit='ms')
        mkr_history.set_index('date', inplace=True)
        mkr_history.drop(columns='timestamp', inplace=True)
        return mkr_history, mkr_market_cap
    else:
        print(f"Failed to retrieve data: {response.status_code}")
        return pd.DataFrame(), pd.DataFrame()


# In[159]:


eth_historical_api = "https://api.coingecko.com/api/v3/coins/ethereum/market_chart"

eth_history, historical_mk = fetch_rpl_historical_data(eth_historical_api, api_key)


# In[160]:


assets.index = assets.index.tz_localize(None)
eth_history['price'].index = eth_history['price'].index.tz_localize(None)

assets_merged = assets.merge(eth_history['price'], left_index=True, right_index=True)


# In[161]:


assets_merged = assets.merge(eth_history['price'], left_index=True, right_index=True)


# In[162]:


assets_merged['assets_usd'] = assets_merged['net_eth'] * assets_merged['price']


# In[163]:


assets_merged['assets_usd'].plot()


# In[164]:


assets_merged['assets_usd_formatted'] = assets_merged['assets_usd'].apply(lambda x: f"{x:,.2f}")
assets_merged['assets_usd_formatted']


# In[165]:


rpl_liabilities_url = 'https://api.dune.com/api/v1/query/1282424/results?api_key=fKXF51FhPIqzpGn96BnOcEgw3P9rcJzG'
response = requests.get(rpl_liabilities_url)


# In[166]:


data2 = response.json()

data_useful2 = pd.DataFrame(data2['result']['rows'])

data_useful2


# In[167]:


# Convert 'time' to datetime and set as index
data_useful2['time'] = pd.to_datetime(data_useful2['time'])
data_useful2.set_index('time', inplace=True)

# Now perform the groupby operation



# In[168]:


time_series_reth = data_useful2.resample('D').mean()


# In[169]:


time_series_reth.fillna(0, inplace=True)


# In[170]:


reth_historical_api = "https://api.coingecko.com/api/v3/coins/rocket-pool-eth/market_chart"

reth_history, historical_mk_reth = fetch_rpl_historical_data(reth_historical_api, api_key)


# In[171]:


reth_history


# In[172]:


time_series_reth.index = time_series_reth.index.tz_localize(None)
reth_history['price'].index = reth_history['price'].index.tz_localize(None)


# In[173]:


merged_liabilities = time_series_reth.merge(reth_history['price'], left_index=True, right_index=True)


# In[174]:


merged_liabilities['liabilities_usd'] = merged_liabilities['total_reth'] * merged_liabilities['price']
merged_liabilities['formated_liabilities'] = merged_liabilities['liabilities_usd'].apply(lambda x: f"{x:,.2f}")


# In[175]:


merged_liabilities


# In[176]:


liabilities = merged_liabilities['liabilities_usd'].to_frame('liabilities')
assets_df = assets_merged['assets_usd'].to_frame('assets')

balance_sheet = liabilities.merge(assets_df, left_index=True, right_index=True)


# In[177]:


balance_sheet['equity'] = balance_sheet['assets'] - balance_sheet['liabilities']


# In[178]:


balance_sheet['equity_formatted'] = balance_sheet['equity'].apply(lambda x: f"{x:,.2f}")


# In[179]:


balance_sheet


# In[185]:


historical_mk_reth.index = reth_history.index


# In[188]:


historical_mk_reth.index = historical_mk_reth.index.normalize()


# In[193]:


historical_mk_reth


# In[195]:


historical_mk_reth.index


# In[197]:


liabilities_df = balance_sheet['liabilities'].to_frame('liabilities')

liabilities_df


# In[199]:


enterprise_value = historical_mk_reth.merge(liabilities_df, left_index=True, right_index=True)


# In[201]:


enterprise_value['ev_historical'] = enterprise_value['marketcap'] + enterprise_value['liabilities']


# In[203]:


enterprise_value['ev_historical']


rpl_revenue_url = 'https://api.dune.com/api/v1/query/3354037/results?api_key=fKXF51FhPIqzpGn96BnOcEgw3P9rcJzG'
response = requests.get(rpl_revenue_url)


data3 = response.json()

data_useful3 = pd.DataFrame(data3['result']['rows'])

data_useful3

data_useful3.set_index('day', inplace=True)
data_useful3.index = pd.to_datetime(data_useful3.index)

rpl_revenue = data_useful3['rpl_usd'].to_frame('rpl_usd')

start_date1 = rpl_revenue.index.min()
end_date1 = rpl_revenue.index.max()
date_range1 = pd.date_range(start=start_date, end=end_date, freq='W-MON')

# Step 2: Reindex your DataFrame
rpl_revenue_reindexed = rpl_revenue.reindex(date_range)

# Step 3: Fill missing values with 0
rpl_revenue_reindexed['rpl_usd'].fillna(0, inplace=True)

monthly_revenue = rpl_revenue_reindexed.resample('M').sum()

monthly_revenue.index = monthly_revenue.index.tz_localize(None)

ev_df = enterprise_value.merge(monthly_revenue, left_index=True, right_index=True)
ev_df['ev_to_rev'] = ev_df['ev_historical'] / ev_df['rpl_usd']





