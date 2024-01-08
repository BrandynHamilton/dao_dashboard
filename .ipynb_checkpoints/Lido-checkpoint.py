#!/usr/bin/env python
# coding: utf-8

# In[2]:

"""
get_ipython().system('pip install import_ipynb')
"""

# In[3]:


import pandas as pd
import requests
import numpy as np
import seaborn as sns
import yfinance as yf
sns.set_theme()
# Set the float display format to suppress scientific notation
#pd.set_option('display.float_format', lambda x: '%.3f' % x)

#import apis
import makerdao
import formulas


import pandas as pd
import requests
import yfinance as yf
import streamlit as st

@st.cache_data(ttl=86400)
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


@st.cache_data(ttl=86400)
def fetch_market_data(api_url, api_key):
    params = {'api_key': api_key}  # Add the API key as a parameter
    response = requests.get(api_url, params=params)

    if response.status_code == 200:
        data = response.json()
        market_value = data['market_data']['market_cap']['usd']
        current_price = data['market_data']['current_price']['usd']
        supply = data['market_data']['circulating_supply']
        return market_value, current_price, supply
    else:
        print(f"Failed to retrieve data: {response.status_code}")
        return None, None, None  # Return None for each value if the API call fails


@st.cache_data(ttl=86400)
def fetch_dpi_historical_data(api_url):
    response = requests.get(api_url)
    if response.status_code == 200:
        # Parse the JSON response
        dpi_historical_pricedata = response.json()
        # Extract the 'prices' data
        dpi_historical_price = dpi_historical_pricedata['prices']
        # Create a DataFrame
        dpi_history = pd.DataFrame(dpi_historical_price, columns=['date', 'price'])
        # Convert the 'date' column from UNIX timestamps in milliseconds to datetime objects
        dpi_history['date'] = pd.to_datetime(dpi_history['date'], unit='ms')
        return dpi_history
    else:
        print(f"Failed to retrieve data: {response.status_code}")
        return pd.DataFrame()  # Return an empty DataFrame in case of failure
    
# Decorate your function with st.cache to enable caching
@st.cache_data(ttl=86400)
def get_lidobs_data():
    lidobs_url = "https://api.dune.com/api/v1/query/2484656/results/"
    params_lidobs = {
        "api_key": "Vti1XpoLF3ulDjZuSbyLXblt4I1JGoVu"
    }

    # Send a GET request to the API endpoint
    response = requests.get(lidobs_url, params=params_lidobs)

    # Print the response status code
    print(f"Response Status Code: {response.status_code}")

    # Initialize an empty DataFrame
    lidobs_df = pd.DataFrame()

    # Check for a valid response
    if response.status_code == 200:
        # Parse the JSON response
        lidobs_data = response.json()
        # Extract the 'rows' list from the JSON data
        lidobs_rows_data = lidobs_data['result']['rows']
        lidobs_df = pd.DataFrame(lidobs_rows_data)
    else:
        print(f"Failed to retrieve data: HTTP {response.status_code}")

    return lidobs_df
    
api_key = "Vti1XpoLF3ulDjZuSbyLXblt4I1JGoVu"
api_key_cg = "CG-jTsiV2rsyVSHHULoNSWHU493"


# Use the function in your Streamlit app
lidobs_df = get_lidobs_data()

assets = lidobs_df['>Assets ($)']
assets.index = lidobs_df['period']
liabilities = lidobs_df['>Liabilities ($)'] 
liabilities.index = lidobs_df['period']

equity = assets - liabilities
equity.index=assets.index



equity.head()





# In[5]:


current_ratio = assets / liabilities

current_ratio.iloc[0]


# In[6]:

lidoincome_url = "https://api.dune.com/api/v1/query/2464243/results/"
params_lidoincome = {"api_key": api_key}
lidoincome_df = fetch_data_from_api(lidoincome_url, params_lidoincome)
lidoincome_df['period'] = pd.to_datetime(lidoincome_df['period'])


df_ttm = lidoincome_df.head(12)

ttm_metric = pd.DataFrame(df_ttm.sum(numeric_only=True))

ttm_metrics = ttm_metric.transpose()

ttm_metrics.columns


# In[9]:


ttm_net_income = ttm_metrics['($) >>Total Protocol Income']
ttm_revenue = ttm_metrics['($) >Net Revenue']



net_profit_margin = ttm_net_income / ttm_revenue

ttm_average_assets = assets.head(12).mean()

roa = ttm_net_income / ttm_average_assets

ttm_average_equity = equity.head(12).mean()

roe = ttm_net_income / ttm_average_equity

print(roe)


# In[ ]:





# In[10]:


# Define the API endpoint and parameters
lidomrktapi = "https://api.coingecko.com/api/v3/coins/lido-dao"
ldo_market_value, ldo_current_price, ldo_supply = fetch_market_data(lidomrktapi, api_key_cg)

print('supply is:',ldo_supply)

# In[11]:


debt_to_equity = liabilities / equity

debt_ratio = liabilities / assets 

eps = ttm_net_income.iloc[0] / ldo_supply

print(eps)


# In[12]:


price_to_earnings = ldo_current_price/eps

print(price_to_earnings)


# In[13]:


bookval = equity / ldo_supply

market_to_book = ldo_current_price / bookval

print(market_to_book.iloc[0])


# In[68]:


# Modified function to return only ldo_history and ldo_market_cap


@st.cache_data(ttl=86400)
def get_ldo_historical_data(api_key):
    ldo_historical_api = "https://api.coingecko.com/api/v3/coins/lido-dao/market_chart"
    params = {
        'vs_currency': 'usd',
        'days': '1152',
        'interval': 'daily',
        'api_key': api_key  # Add the API key as a parameter
    }
    response = requests.get(ldo_historical_api, params=params)
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
    else:
        print(f"Failed to retrieve data: {response.status_code}")
        return ldo_history, ldo_market_cap  # Return empty DataFrames in case of failure

# Usage


# Call the function
ldo_history, ldo_market_cap = get_ldo_historical_data(api_key_cg)

# Assuming further processing happens here...


# Calculate ldo_supply
ldo_supply = ldo_market_cap['marketcap'] / ldo_history['price']

# Now groupby should work
ldo_fixed = ldo_supply.groupby([pd.Grouper(freq='M')]).mean()

# Calculate the trailing twelve-month (TTM) average supply
ttm_supply = ldo_fixed.tail(12).mean()

# Display the result in Streamlit
ttm_supply

# In[15]:


ldo_history['daily_returns'] = ldo_history['price'].pct_change().dropna()

#ldo_history['date'] = pd.to_datetime(ldo_history['date'], unit='ms')

ldo_history


# In[53]:





# In[16]:


from makerdao import dpi_history



dpi_history['daily_returns'] = dpi_history['price'].pct_change().dropna()

dpi_history_filtered = dpi_history[dpi_history.index >= '2021-01-05']

dpi_history_filtered


# In[17]:


from sklearn.linear_model import LinearRegression

# Merge the two dataframes on the date
merged_df = dpi_history_filtered.merge(ldo_history, on='date', suffixes=('_dpi', '_ldo'))

# Drop rows with missing values in daily_returns for either dataframe
merged_df.dropna(subset=['daily_returns_dpi', 'daily_returns_ldo'], inplace=True)

# Prepare the data for regression
X = merged_df['daily_returns_dpi'].values.reshape(-1, 1)
Y = merged_df['daily_returns_ldo'].values

# Create and fit the model
model = LinearRegression()
model.fit(X, Y)

# Extract the beta coefficient
beta = model.coef_[0]

print(f"Beta coefficient: {beta}")



# In[18]:


print(beta)


# In[19]:


ldo_history['daily_returns'].dropna()


# In[20]:


dpi_history_filtered['daily_returns'].dropna()


# In[21]:


import streamlit as st
import requests
import pandas as pd

# Decorate your function with st.cache to enable caching
@st.cache_data(ttl=86400)
def get_lidoyield_data():
    lidoyield_url = "https://api.dune.com/api/v1/query/570874/results/"
    params_lidoyield = {
        "api_key": "Vti1XpoLF3ulDjZuSbyLXblt4I1JGoVu"
    }

    # Send a GET request to the API endpoint
    response = requests.get(lidoyield_url, params=params_lidoyield)

    # Initialize an empty DataFrame
    lidoyield_df = pd.DataFrame()

    # Check for a valid response
    if response.status_code == 200:
        # Parse the JSON response
        lidoyield_data = response.json()
        # Extract the 'rows' list from the JSON data
        lidoyield_rows_data = lidoyield_data['result']['rows']
        lidoyield_df = pd.DataFrame(lidoyield_rows_data)

    return lidoyield_df

# Use the function in your Streamlit app
lidoyield_df = get_lidoyield_data()



# In[22]:


lido_cost_debt = lidoyield_df['Lido staking APR(instant)'] / 100
cost_of_debt = lido_cost_debt.iloc[0]

cost_of_debt


# In[23]:


from formulas import calculate_annual_return

#ldo_history.set_index('date', inplace=True)

# Group by year and apply the function
#annual_returns = ldo_history.groupby(ldo_history.index.year).apply(calculate_annual_return)

#print(annual_returns)


# In[24]:


from makerdao import tbilldf

# Assuming tbilldf.index has been correctly set to a datetime format
tbilldf['decimal'] = tbilldf['value'] / 100
tbill_yearly = tbilldf.groupby(tbilldf.index.year).mean(numeric_only=True)

tbill_filtered = tbill_yearly['decimal'][-3:]

tbill_filtered


# In[25]:


print(beta)


# In[26]:


from makerdao import average_yearly_risk_premium, current_risk_free


cost_equity = current_risk_free + beta * average_yearly_risk_premium


# In[27]:


print(cost_equity)


# In[28]:


formatted_market_value = "{:,.0f}".format(ldo_market_value)

formatted_market_value


# In[29]:


live_liabilities = "{:,.0f}".format(liabilities[0])

live_liabilities


# In[30]:


cost_equity


# In[31]:


cost_of_debt


# In[32]:


value_financing = ldo_market_value + liabilities[0]
formatted_financing = "{:,.0f}".format(value_financing)

formatted_financing



# In[33]:


print('market value of equity:', formatted_market_value)
print('value of liabilities:', live_liabilities)
print('cost of equity:', cost_equity)
print('cost of debt:', cost_of_debt)
print('proportion equity financed:', (ldo_market_value / value_financing) * 100)
print('proportion debt financed:', (liabilities[0] / value_financing) * 100)
      


# In[34]:


e = ldo_market_value
d = liabilities
re = cost_equity
v = e + d
rd = cost_of_debt

wacc = ((e/v) * re) + ((d/v) * rd)

print('wacc is:',wacc[0])



# In[ ]:


def calculate_historical_returns(prices):
    prices['date'] = prices.index
    starting_value = prices.iloc[0]['price']
    ending_value = prices.iloc[-1]['price']
    number_of_years = (prices.iloc[-1]['date'] - prices.iloc[0]['date']).days / 365.25
    cagr = (ending_value/starting_value) ** (1/number_of_years) - 1
    return cagr

lido_cagr = calculate_historical_returns(ldo_history)


from makerdao import tbilldf_after2020

tbill = tbilldf_after2020[tbilldf_after2020.index > 2020] / 100

tbill

ldo_annual_returns = ldo_history.groupby(ldo_history.index.year).apply(calculate_annual_return)
ldo_annual_returns = pd.DataFrame(ldo_annual_returns)

ldo_annual_returns

excess_returns = ldo_annual_returns[0] - tbill['value']
excess_returns

ldo_avg_excess_return = excess_returns.mean()

ldo_avg_excess_return

current_ratio_df = current_ratio.to_frame(name='current_ratio')

net_profit_margin_history = lidoincome_df['($) >>Total Protocol Income'] / lidoincome_df['($) >Net Revenue']

net_profit_margin_history.index = lidoincome_df['period']

# Convert indices to datetime if they are not already
lidoincome_df.index = pd.to_datetime(lidoincome_df['period'])
assets.index = pd.to_datetime(assets.index)

# Reindex or resample one DataFrame to match the other's index
# Here, reindexing assets to match lidoincome_df's index
assets_reindexed = assets.reindex(lidoincome_df.index)

# Now perform the division
roa_history = lidoincome_df['($) >>Total Protocol Income'] / assets_reindexed

equity.index = pd.to_datetime(equity.index)
equity_reindex = equity.reindex(lidoincome_df.index)

# Now perform the division
roe_history = lidoincome_df['($) >>Total Protocol Income'] / equity_reindex

liabilities.index = pd.to_datetime(liabilities.index)

debt_ratio_history = liabilities/assets

ldo_market_cap.index = pd.to_datetime(ldo_history['date'])

monthly_cap = ldo_market_cap.resample('M').mean()

# Convert the index of both DataFrames to the same type if necessary
monthly_cap.index = pd.to_datetime(monthly_cap.index)
lidoincome_df.index = pd.to_datetime(lidoincome_df.index)

# Normalize lidoincome_df index to the end of the month to match monthly_cap
lidoincome_df.index = lidoincome_df.index.to_period('M').to_timestamp('M')

# Reindex lidoincome_df to match monthly_cap's index
cap_reindexed = monthly_cap.reindex(lidoincome_df.index)

# Calculate the Price to Sales ratio
price_to_sales_ratio = cap_reindexed['marketcap'] / lidoincome_df['($) >Net Revenue']

# Replace infinities and fill NaNs
price_to_sales_ratio = price_to_sales_ratio.replace([np.inf, -np.inf], np.nan).fillna(0)

# Normalize the index of bookval to the end of the month
bookval.index = pd.to_datetime(bookval.index)
bookval.index = bookval.index.to_period('M').to_timestamp('M')

# Ensure price_to_sales_ratio index is also normalized to the end of the month
price_to_sales_ratio.index = price_to_sales_ratio.index.to_period('M').to_timestamp('M')

debt_to_equity.index = pd.to_datetime(debt_to_equity.index)
current_ratio_df.index = pd.to_datetime(current_ratio_df.index)

# Normalize the index of bookval to the end of the month
bookval.index = bookval.index.to_period('M').to_timestamp('M')

# Ensure price_to_sales_ratio index is also normalized to the end of the month
price_to_sales_ratio.index = price_to_sales_ratio.index.to_period('M').to_timestamp('M')

# Combine into a single DataFrame
combined_df = pd.DataFrame({
    'price_to_sales_ratio': price_to_sales_ratio,
    'bookval': bookval
})

# Normalize the indices of other series to end-of-month
roa_history.index = roa_history.index.to_period('M').to_timestamp('M')
roe_history.index = roe_history.index.to_period('M').to_timestamp('M')
debt_to_equity.index = debt_to_equity.index.to_period('M').to_timestamp('M')
debt_ratio_history.index = debt_ratio_history.index.to_period('M').to_timestamp('M')
current_ratio_df.index = current_ratio_df.index.to_period('M').to_timestamp('M')
net_profit_margin_history.index = net_profit_margin_history.index.to_period('M').to_timestamp('M')

# Add these series to combined_df
combined_df['roa_history'] = roa_history
combined_df['roe_history'] = roe_history
combined_df['debt_to_equity'] = debt_to_equity
combined_df['debt_ratio_history'] = debt_ratio_history
combined_df['current_ratio'] = current_ratio_df['current_ratio']
combined_df['net_profit_margin_history'] = net_profit_margin_history

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Replace inf/-inf with NaN
combined_df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Impute missing values (e.g., NaNs)
imputer = SimpleImputer(strategy='mean')
imputed_combined_df = pd.DataFrame(imputer.fit_transform(combined_df), columns=combined_df.columns, index=combined_df.index)

# Scale the metrics in combined_df
standard_scaler = StandardScaler()
metrics_standard_scaled = pd.DataFrame(standard_scaler.fit_transform(imputed_combined_df), 
                                       columns=imputed_combined_df.columns, 
                                       index=imputed_combined_df.index)



# Assuming 'combined_df' is your DataFrame
standard_scaler = StandardScaler()
metrics_standard_scaled = pd.DataFrame(standard_scaler.fit_transform(combined_df), 
                                       columns=combined_df.columns, 
                                       index=combined_df.index)

# Define the function to score each metric based on its Z-score
def score_metric(value, is_higher_better=True, max_score=3):
    if is_higher_better:
        return max(1, min(max_score, (value + 3) / 6 * max_score))
    else:
        return max(1, min(max_score, (-value + 3) / 6 * max_score))

# Define the original weights for each metric in 'combined_df'
weights = {
    'price_to_sales_ratio': 0.10,  # Lower is better
    'bookval': 0.15,
    'roa_history': 0.15,
    'roe_history': 0.15,
    'debt_to_equity': 0.15,  # Lower is better
    'debt_ratio_history': 0.10,  # Lower is better
    'current_ratio': 0.15,
    'net_profit_margin_history': 0.15
}

# Define which metrics are considered 'higher is better'
higher_is_better_metrics = [
    'bookval',
    'roa_history',
    'roe_history',
    'current_ratio',
    'net_profit_margin_history'
]

# Score and calculate weighted scores
for metric in weights:
    higher_is_better = metric in higher_is_better_metrics
    metrics_standard_scaled[metric + '_score'] = metrics_standard_scaled[metric].apply(
        score_metric, args=(higher_is_better,))
    metrics_standard_scaled[metric + '_weighted_score'] = metrics_standard_scaled[metric + '_score'] * weights[metric]

# Compute overall financial health score
metrics_standard_scaled['financial_health_score'] = metrics_standard_scaled[
    [col for col in metrics_standard_scaled.columns if 'weighted_score' in col]
].sum(axis=1)

# Normalize and categorize the score
max_possible_score = sum(weights.values()) * 3
metrics_standard_scaled['normalized_financial_health_score'] = (
    metrics_standard_scaled['financial_health_score'] / max_possible_score * 3
)

def categorize_score(score):
    if score >= 2.5:
        return 'good'
    elif score >= 1.5:
        return 'okay'
    else:
        return 'bad'

metrics_standard_scaled['financial_health_category'] = metrics_standard_scaled[
    'normalized_financial_health_score'
].apply(categorize_score)

# Display the results
print(metrics_standard_scaled['financial_health_category'])

print(ttm_metrics.columns)


# Create a dictionary with the components of the income statement
income_statement_data = {
    'Cost of Revenue': ttm_metrics['($) >Cost of Revenue'],
    'Liquidity Expenses': ttm_metrics['($) >Liquidity Expenses'],
    'Operating Expenses': ttm_metrics['($) >Operating Expenses'],
    'Net Revenue': ttm_metrics['($) >Net Revenue'],
    'Net Income': ttm_metrics['($) >>Total Protocol Income']
}

# Create a DataFrame from the dictionary
consolidated_income_statement = pd.DataFrame(income_statement_data)

# Assuming 'consolidated_income_statement' is your DataFrame
consolidated_income_statement = consolidated_income_statement.applymap(lambda x: f"${x:,.2f}")
consolidated_income_statement = consolidated_income_statement.transpose()

# Rename the column to 'Amount'
consolidated_income_statement = consolidated_income_statement.rename(columns={0: 'Amount'})

# Now the column should be renamed, and you can print to check
print(consolidated_income_statement)

enterprise_value = ldo_market_value + liabilities[0] - assets[0]

ev_to_rev = enterprise_value/ttm_revenue