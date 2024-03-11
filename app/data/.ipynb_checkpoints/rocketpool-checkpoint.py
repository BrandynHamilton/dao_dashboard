import requests 
import pandas as pd
import streamlit as st
from .makerdao import dpi_history, current_risk_free, average_yearly_risk_premium, cumulative_risk_premium as dpi_cumulative_risk_premium
from .Lido import tbill, tbilldf_after2020
from .formulas import *

api_key_cg = st.secrets["api_key_cg"]
api_key = st.secrets["api_key"]

@st.cache_data()  # Corrected decorator for Streamlit
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

@st.cache_data()  # Corrected decorator for Streamlit
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

tbill_decimals = tbilldf_after2020 / 100

excess_returns = rpl_annual_returns[0] - tbill_decimals['value']

rpl_avg_excess_return = excess_returns.mean()

print('rocketpool avg return:',rpl_avg_excess_return)
print('rocketpool beta:', beta)
print('rocketpool cagr:', rpl_cagr)

@st.cache_data()
def get_api_data(url, params):
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()  # Return the parsed JSON data
    else:
        return None  # Return None or raise an exception if the request failed


rpl_assets_url = 'https://api.dune.com/api/v1/query/3380184/results/'
rpl_params = {"api_key": api_key}

data = get_api_data(rpl_assets_url, rpl_params)


# In[150]:


data_useful = pd.DataFrame(data['result']['rows'])


# In[151]:


print(data_useful)


# In[152]:


data_useful['day'] = pd.to_datetime(data_useful['day'])


# In[153]:


data_useful.set_index('day', inplace=True)


# In[154]:


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


assets = data_useful_filled['assets_usd'].to_frame('net_eth')

assets


# In[157]:




# In[158]:

@st.cache_data(ttl=86400)
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

eth_history, historical_mk = fetch_rpl_historical_data(eth_historical_api, api_key_cg)


# In[160]:


assets.index = assets.index.tz_localize(None)
eth_history['price'].index = eth_history['price'].index.tz_localize(None)

assets_merged = assets.merge(eth_history['price'], left_index=True, right_index=True)


# In[161]:


assets_merged = assets.merge(eth_history['price'], left_index=True, right_index=True)


# In[162]:


assets_merged['assets_usd'] = assets_merged['net_eth'] 


# In[163]:


assets_merged['assets_usd'].plot()


# In[164]:


assets_merged['assets_usd_formatted'] = assets_merged['assets_usd'].apply(lambda x: f"{x:,.2f}")
assets_merged['assets_usd_formatted']


# In[165]:


rpl_liabilities_url = 'https://api.dune.com/api/v1/query/3380184/results/'


# In[166]:


data2 = get_api_data(rpl_liabilities_url, rpl_params)

data_useful2 = pd.DataFrame(data2['result']['rows'])

data_useful2


# In[167]:


# Convert 'time' to datetime and set as index
data_useful2['day'] = pd.to_datetime(data_useful2['day'])
data_useful2.set_index('day', inplace=True)

# Now perform the groupby operation



# In[168]:


time_series_reth = abs(data_useful2['liabilities_usd']).resample('D').mean()


# In[169]:


time_series_reth.fillna(0, inplace=True)


# In[170]:


reth_historical_api = "https://api.coingecko.com/api/v3/coins/rocket-pool-eth/market_chart"

reth_history, historical_mk_reth = fetch_rpl_historical_data(reth_historical_api, api_key_cg)


# In[171]:


reth_history


# In[172]:


time_series_reth.index = time_series_reth.index.tz_localize(None)
reth_history['price'].index = reth_history['price'].index.tz_localize(None)


# In[173]:


merged_liabilities = time_series_reth.to_frame('liabilities_usd').merge(reth_history['price'], left_index=True, right_index=True)


# In[174]:


merged_liabilities['liabilities_usd'] = merged_liabilities['liabilities_usd']
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


#balance_sheet['equity_formatted'] = balance_sheet['equity'].apply(lambda x: f"{x:,.2f}")


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


rpl_revenue_url = 'https://api.dune.com/api/v1/query/3510307/results/'


data3 = get_api_data(rpl_revenue_url, rpl_params)

data_useful3 = pd.DataFrame(data3['result']['rows'])

data_useful3

data_useful3.set_index('day', inplace=True)
data_useful3.index = pd.to_datetime(data_useful3.index)

rpl_revenue = data_useful3['rpl_usd'].to_frame('rpl_usd')

start_date1 = rpl_revenue.index.min()
end_date1 = rpl_revenue.index.max()
date_range1 = pd.date_range(start=start_date1, end=end_date1, freq='W-MON')

# Step 2: Reindex your DataFrame
rpl_revenue_reindexed = rpl_revenue.reindex(date_range1)

# Step 3: Fill missing values with 0
rpl_revenue_reindexed['rpl_usd'].fillna(0, inplace=True)



monthly_revenue = rpl_revenue_reindexed.resample('M').sum()

monthly_revenue.index = monthly_revenue.index.tz_localize(None)

ev_df = enterprise_value.merge(monthly_revenue, left_index=True, right_index=True)

ttm_rev = ev_df['rpl_usd'].tail(12).sum()
ev_df['ev_to_rev'] = ev_df['ev_historical'] / ev_df['rpl_usd']

current_ratio = balance_sheet['assets'] / balance_sheet['liabilities']

debt_ratio = balance_sheet['liabilities'] / balance_sheet['assets']
debt_to_equity = balance_sheet['liabilities'] / balance_sheet['equity']
#rev_per_share = 

cost_of_debt = 0.0314

current_risk_free

cost_equity = current_risk_free + beta * average_yearly_risk_premium
long_cost_equity = current_risk_free + beta * dpi_cumulative_risk_premium

live_liabilities = balance_sheet['liabilities'].iloc[-1]

e = rpl_market_value
d = live_liabilities
re = cost_equity
long_re = long_cost_equity
v = e + d
rd = cost_of_debt

wacc = ((e/v) * re) + ((d/v) * rd)
long_wacc = ((e/v) * long_re) + ((d/v) * rd)


ttm_ev_rev = ev_df['ev_historical'].iloc[-1] / ttm_rev


current_equity = balance_sheet['equity'].iloc[-1]

bookval = current_equity / rpl_supply

market_to_book = rpl_current_price / bookval

