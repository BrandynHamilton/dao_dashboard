#!/usr/bin/env python
# coding: utf-8

# In[18]:

"""
get_ipython().system('pip install yfinance')
"""

# In[19]:


import pandas as pd
import requests
import yfinance as yf
import streamlit as st
"""
import import_ipynb
"""

# In[20]:


# Define the API endpoint and parameters
url = "https://api.dune.com/api/v1/query/2840463/results/"
params = {
    "api_key": "NB6wsLRGqoVVQaLVuwk9mFB3kDssFVGK"
}

# Send a GET request to the API endpoint
response = requests.get(url, params=params)

# Check for a valid response
if response.status_code == 200:
    # Parse the JSON response
    datas = response.json()

    # Extract the 'rows' list from the JSON data
    rows_data = datas['result']['rows']

    # Convert the data into a DataFrame
    mkrdao = pd.DataFrame(rows_data)

    # Print the DataFrame to the console
    print(mkrdao.head())  # Print the first 5 rows
else:
    print(f"Failed to retrieve data: {response.status_code}")



# In[21]:


revenueapi = "https://api.dune.com/api/v1/query/2641549/results"
revparams = {
    "api_key": "NB6wsLRGqoVVQaLVuwk9mFB3kDssFVGK"
}

revresponse = requests.get(revenueapi, params=revparams)

if revresponse.status_code == 200:
    revdata = revresponse.json()
    
   
    
    revrows = revdata['result']['rows']
    
    revdf = pd.DataFrame(revrows)
    
monthly_stats = revdf.groupby(['period']).sum()

item_month = revdf.groupby(['period','item']).sum()




# In[22]:


# Define the API endpoint and parameters
makerapi = "https://api.coingecko.com/api/v3/coins/maker"


# Send a GET request to the API endpoint
response2 = requests.get(makerapi)

# Check for a valid response
#if response2.status_code == 200:
    # Parse the JSON response
makerpricedata = response2.json()
   # Extract the 'rows' list from the JSON data
market_value = makerpricedata['market_data']['market_cap']['usd']
current_price = makerpricedata['market_data']['current_price']['usd']
supply = makerpricedata['market_data']['circulating_supply']
    


print(market_value, current_price, supply)



# In[23]:


#beta - we will use DPI as benchmark, an index of various defi assets

# Define the API endpoint and parameters
dpiapi = "https://api.coingecko.com/api/v3/coins/defipulse-index"


# Send a GET request to the API endpoint
response3 = requests.get(dpiapi)

print(response3)

# Check for a valid response
if response3.status_code == 200:
    # Parse the JSON response
    dpipricedata = response3.json()
   # Extract the 'rows' list from the JSON data
    dpi_market_value = dpipricedata['market_data']['market_cap']['usd']
    dpi_current_price = dpipricedata['market_data']['current_price']['usd']
    dpi_supply = dpipricedata['market_data']['circulating_supply']
    
print(dpi_current_price)


# In[24]:


#beta - we will use DPI as benchmark, an index of various defi assets

# Define the API endpoint and parameters
dpi_historical_api = "https://api.coingecko.com/api/v3/coins/defipulse-index/market_chart?vs_currency=usd&days=1152&interval=daily"


# Send a GET request to the API endpoint
response4 = requests.get(dpi_historical_api)

print(response4)

# Check for a valid response
    # Parse the JSON response
dpi_historical_pricedata = response4.json()
   # Extract the 'rows' list from the JSON data
dpi_historical_price = dpi_historical_pricedata['prices']
dpi_history = pd.DataFrame(dpi_historical_price, columns=['date','price'])
    
dpi_history['daily_returns'] = dpi_history['price'].pct_change().dropna()

dpi_history = dpi_history.iloc[1:] #first day of trading nothing to get for return

# Convert the 'date' column from UNIX timestamps in milliseconds to datetime objects
dpi_history['date'] = pd.to_datetime(dpi_history['date'], unit='ms')

dpi_history.set_index('date', inplace=True)


# In[ ]:





# In[25]:


# Define the API endpoint and parameters
mkr_historical_api = "https://api.coingecko.com/api/v3/coins/maker/market_chart?vs_currency=usd&days=1152&interval=daily"


# Send a GET request to the API endpoint
response5 = requests.get(mkr_historical_api)

print(response5)

# Check for a valid response
if response5.status_code == 200:
    # Parse the JSON response
    mkr_historical_pricedata = response5.json()
   # Extract the 'rows' list from the JSON data
    mkr_historical_price = mkr_historical_pricedata['prices']
    mkr_history = pd.DataFrame(mkr_historical_price, columns=['date','price'])
   
mkr_history['daily_returns'] = mkr_history['price'].pct_change().dropna()

mkr_history = mkr_history.iloc[1:]

# Convert the 'date' column from UNIX timestamps in milliseconds to datetime objects
mkr_history['date'] = pd.to_datetime(mkr_history['date'], unit='ms')

# Check the converted dates
print(mkr_history)





# In[26]:


# Define the API endpoint and parameters
mkr_vault_url = "https://api.dune.com/api/v1/query/17338/results/"
params_vault = {
    "api_key": "NB6wsLRGqoVVQaLVuwk9mFB3kDssFVGK"
}

# Send a GET request to the API endpoint
response7 = requests.get(mkr_vault_url, params=params_vault)

# Check for a valid response
if response7.status_code == 200:
    # Parse the JSON response
    mkr_vault_data = response7.json()
    # Extract the 'rows' list from the JSON data
    mkr_vault_rows_data = mkr_vault_data['result']['rows']
    mkr_vault_df = pd.DataFrame(mkr_vault_rows_data)


print(mkr_vault_df)



# In[27]:


lidobs_url = "https://api.dune.com/api/v1/query/2484656/results/"
params_lidobs = {
    "api_key": "NB6wsLRGqoVVQaLVuwk9mFB3kDssFVGK"
}

# Send a GET request to the API endpoint
response8 = requests.get(lidobs_url, params=params_lidobs)

# Check for a valid response
if response8.status_code == 200:
    # Parse the JSON response
    lidobs_data = response8.json()
    # Extract the 'rows' list from the JSON data
    lidobs_rows_data = lidobs_data['result']['rows']
    lidobs_df = pd.DataFrame(lidobs_rows_data)


lidobs_df.columns







# In[28]:


# Define the API endpoint and parameters
tbill_historical_api = "https://api.stlouisfed.org/fred/series/observations?series_id=TB3MS&api_key=af3aeb14543cb05941f1b87abc3e3b7b&file_type=json"


# Send a GET request to the API endpoint
response6 = requests.get(tbill_historical_api)

print(response6)

# Check for a valid response
if response6.status_code == 200:
    # Parse the JSON response
    tbill_historical_pricedata = response6.json()
    
    
""" #Extract the 'rows' list from the JSON data
    tbill_historical_price = tbill_historical_pricedata['prices']
    mkr_history = pd.DataFrame(mkr_historical_price, columns=['date','price'])
    
print(mkr_history)
print(type(mkr_history['date']))
"""
tbilldf = pd.DataFrame(tbill_historical_pricedata["observations"])
tbilldf


# In[29]:


# Define the API endpoint and parameters
tbill_historical_api = "https://api.stlouisfed.org/fred/series/observations?series_id=TB3MS&api_key=af3aeb14543cb05941f1b87abc3e3b7b&file_type=json"


# Send a GET request to the API endpoint
response6 = requests.get(tbill_historical_api)

print(response6)

# Check for a valid response
if response6.status_code == 200:
    # Parse the JSON response
    tbill_historical_pricedata = response6.json()
    
    
""" #Extract the 'rows' list from the JSON data
    tbill_historical_price = tbill_historical_pricedata['prices']
    mkr_history = pd.DataFrame(mkr_historical_price, columns=['date','price'])
    
print(mkr_history)
print(type(mkr_history['date']))
"""
tbilldf = pd.DataFrame(tbill_historical_pricedata["observations"])

tbilldf["date"] = pd.to_datetime(tbilldf["date"])

tbilldf.set_index('date', inplace=True)

tbilldf['value'] = tbilldf['value'].astype(float)


# In[30]:


BTC = yf.Ticker("BTC-USD")

# get all stock infov
BTC_history = BTC.history(period='max', interval='1mo')

BTC_history.index = pd.to_datetime(BTC_history.index)

filtered_BTC_history = BTC_history[(BTC_history.index >= "2019-12") & (BTC_history.index < "2023-11")]

filtered_BTC_history.tail()


# In[31]:


lidobs_url = "https://api.dune.com/api/v1/query/2484656/results/"
params_lidobs = {
    "api_key": "NB6wsLRGqoVVQaLVuwk9mFB3kDssFVGK"
}

# Send a GET request to the API endpoint
response8 = requests.get(lidobs_url, params=params_lidobs)

# Check for a valid response
if response8.status_code == 200:
    # Parse the JSON response
    lidobs_data = response8.json()
    # Extract the 'rows' list from the JSON data
    lidobs_rows_data = lidobs_data['result']['rows']
    lidobs_df = pd.DataFrame(lidobs_rows_data)


lidobs_df.head()


# In[32]:


mkr_supply_url = "https://api.dune.com/api/v1/query/482349/results"
params_mkr_supply = {
    "api_key": "NB6wsLRGqoVVQaLVuwk9mFB3kDssFVGK"
}

# Send a GET request to the API endpoint
response11 = requests.get(mkr_supply_url, params=params_mkr_supply)

# Check for a valid response
if response11.status_code == 200:
    # Parse the JSON response
    mkr_supply_data = response11.json()
    # Extract the 'rows' list from the JSON data
    mkrsupply_rows_data = mkr_supply_data['result']['rows']
    mkrsupply_df = pd.DataFrame(mkrsupply_rows_data)


mkrsupply_df.head()




# In[33]:


import pandas as pd

# Assuming mkrsupply_df is your DataFrame
# Convert 'period' to datetime if it's not already
mkrsupply_df['period'] = pd.to_datetime(mkrsupply_df['period'])

# Set 'period' as the index
mkrsupply_df.set_index('period', inplace=True)

# Group by year and month, and sum the values
sum_mkrsupply_df = mkrsupply_df.groupby([pd.Grouper(freq='M')]).mean(numeric_only=True)

# Reset index if you want 'period' back as a column
sum_mkrsupply_df.reset_index(inplace=True)

# Display the resulting DataFrame
ttm_supply = sum_mkrsupply_df['balance'].head(12).mean()

ttm_supply


# In[34]:


mkrsupply_df


# In[ ]:




