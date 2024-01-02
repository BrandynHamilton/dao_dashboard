

# In[ ]:





# Here we import libraries and api data for makerdao balance sheet

# In[2]:


import import_ipynb
import pandas as pd
import requests
import numpy as np
import seaborn as sns
import yfinance as yf
import streamlit as st

# Set the float display format to suppress scientific notation
#pd.set_option('display.float_format', lambda x: '%.3f' % x)





# makerdao revenue statement 

# In[3]:


from makerdao import monthly_stats

monthly_stats.tail()



# In[4]:


var_in_cf = monthly_stats['net_income'].pct_change()

# Perform operation and assign to a new variable
cleaned_var_in_cf = pd.DataFrame(var_in_cf.iloc[:-1].dropna())

# Use 'cleaned_var_in_cf' for further operations

cleaned_var_in_cf['net_income'].tail() 


# In[5]:


from formulas import month_to_quarter
from makerdao import revdf

revdf['quarter'] = revdf['month'].apply(month_to_quarter)


# In[6]:


quarterly_stats = revdf.groupby(['year', 'quarter']).sum().reset_index()
last_four_quarters = quarterly_stats.iloc[-4:]
ttm_data = last_four_quarters.sum()

ttm_revenue = ttm_data['lending_income'] + ttm_data['liquidation_income'] + ttm_data['trading_income']


# HERE is the API for maker price data

# In[7]:


from apis import mkrdao

accounts = mkrdao['item'].unique()

print(accounts)


# In[8]:


cryptoloans = mkrdao.where(mkrdao['item'].isin(['Crypto-Loans'])).dropna().groupby('period').sum()
rwa = mkrdao.where(mkrdao['item'].isin(['Real-World Assets'])).dropna().groupby('period').sum()
stablecoins = mkrdao.where(mkrdao['item'].isin(['Stablecoins'])).dropna().groupby('period').sum()
otherassets = mkrdao.where(mkrdao['item'].isin(['Others assets'])).dropna().groupby('period').sum()
dsr = mkrdao.where(mkrdao['item'].isin(['DSR'])).dropna().groupby('period').sum()
dai = mkrdao.where(mkrdao['item'].isin(['DAI'])).dropna().groupby('period').sum()
mkrequity = mkrdao.where(mkrdao['item'].isin(['Equity'])).dropna().groupby('period').sum()

#Most recent crypto loan balances 
rwa.head()


# In[13]:


assets = cryptoloans['balance'].fillna(0) + rwa['balance'].fillna(0) + stablecoins['balance'].fillna(0) + otherassets['balance'].fillna(0)
liabilities = dsr['balance'] + dai['balance']
equity = mkrequity['balance']


equity


# In[14]:


ttm_data


# In[15]:


from apis import supply, current_price, market_value

bookval = abs(equity) / supply

market_to_book = current_price / bookval

print(market_to_book)


# In[16]:


from apis2 import mkrsupply_df, historical_mk, mkr_history
#monthly_stats

historical_mk



# In[17]:


historical_mk['date'] = pd.to_datetime(historical_mk['date'], unit='ms')
historical_mk


# In[18]:


historical_supply =  historical_mk['marketcap'] / mkr_history['price']
historical_supply.index = historical_mk['date']


# In[19]:


historical_supply_clean = historical_supply.dropna()

historical_supply_clean


# In[20]:


filtered_equity = abs(equity[equity.index > '2020-10-28'])

new_filtered_equity = pd.DataFrame(filtered_equity)
new_historical_supply = pd.DataFrame(historical_supply_clean)


# In[21]:


filtered_equity


# In[22]:


historical_supply_clean


# In[23]:


import pandas as pd

# Assuming filtered_equity and historical_supply_clean are already loaded

# Standardize the datetime indices to a common format
filtered_equity.index = pd.to_datetime(filtered_equity.index).normalize()
historical_supply_clean.index = pd.to_datetime(historical_supply_clean.index).normalize()

# Perform a left merge on the index
metrics = filtered_equity.to_frame(name='equity').merge(historical_supply_clean.to_frame(name='supply'), 
                                                        left_index=True, right_index=True, 
                                                        how='left')

# Display the merged DataFrame

metrics


# In[24]:


metrics['bookval'] = metrics['equity'] / metrics['supply']
metrics


# In[20]:


mkrsupply_df['balance'].info()


# In[21]:


equity.info()


# In[22]:


from apis2 import ttm_supply

ttm_net_income = ttm_data['net_income']

eps = ttm_net_income / ttm_supply

print(eps)


# In[24]:


# Ensure the index is a DateTimeIndex
#metrics.index = pd.to_datetime(metrics.index.start_time)

# Group by month and calculate averages
monthly_averages = metrics.resample('M').mean()

# Display the resulting DataFrame
print(monthly_averages)


# In[25]:


hist_net_income = monthly_stats['net_income'].to_frame()
hist_net_income.index = pd.to_datetime(hist_net_income.index)
#hist_net_income
hist_net_income.tail()


# In[26]:


# Drop the specified columns from the DataFrame
#metrics = metrics.drop(columns=['net income_x', 'net income_y', 'net income'])
hist_price = mkr_history['price'].to_frame()
hist_price.index = mkr_history['date']


# In[27]:


# Display the DataFrame after dropping the columns
hist_price.index = mkr_history['date']
hist_price.index = hist_price.index.normalize()
#metrics['eps'] = metrics['net_income'] / metrics['supply']
metrics.index = pd.to_datetime(metrics.index)
print(metrics.index)
print(hist_price.index)


# In[28]:


# Resample to get the month-end closing price
monthly_closing_price = hist_price.resample('M').last()

# Display the resulting DataFrame



# In[29]:


metricsfiltered3 = monthly_averages.merge(monthly_closing_price, left_index=True, right_index=True, how='left')
metricsfiltered3.tail()


# In[30]:


hist_net_income.tail()


# In[31]:


# Adjust hist_net_income index to end-of-month
hist_net_income.index = hist_net_income.index + pd.offsets.MonthEnd(0)

# Merge the DataFrames
mf4 = metricsfiltered3.merge(hist_net_income, left_index=True, right_index=True, how='left')

mf4['eps'] = mf4['net_income'] / mf4['supply']
mf4['price_to_earnings'] = mf4['price'] / mf4['eps']
mf4['roe'] = mf4['net_income'] / mf4['equity']
# Display the merged DataFrame
print(mf4)


# In[32]:


# Assuming 'period' is a DatetimeIndex in all dataframes

# Step 1: Create a comprehensive index
all_dates = cryptoloans.index.union(rwa.index).union(stablecoins.index).union(otherassets.index)

# Step 2: Reindex each dataframe
cryptoloans_reindexed = cryptoloans.reindex(all_dates).fillna(0)
rwa_reindexed = rwa.reindex(all_dates).fillna(0)
stablecoins_reindexed = stablecoins.reindex(all_dates).fillna(0)
otherassets_reindexed = otherassets.reindex(all_dates).fillna(0)

# Step 3: Perform the addition
assets = cryptoloans_reindexed['balance'] + rwa_reindexed['balance'] + stablecoins_reindexed['balance'] + otherassets_reindexed['balance']

# Verify the results
print(assets)


# In[33]:


assets.index = pd.to_datetime(assets.index)
assets = assets.resample('M').mean()


# In[34]:


mf5 = mf4.merge(assets, left_index=True, right_index=True, how='left')

mf5 = mf5.rename(columns={'balance': 'average_assets'})


# In[35]:


mf5['roa'] = mf5['net_income'] / mf5['average_assets']
mf5


# In[36]:


price_to_earnings = current_price/eps

print(price_to_earnings)


# In[37]:


ROE = ttm_net_income / abs(equity)

print(ROE)


# In[38]:


ROA = ttm_net_income / assets

print(ROA)


# In[39]:


debt_to_equity = abs(liabilities) / abs(equity)

print(debt_to_equity)


# In[40]:


debt_to_equity.index = pd.to_datetime(debt_to_equity.index)
debt_to_equity = debt_to_equity.resample('M').mean()
mf6 = mf5.merge(debt_to_equity, left_index=True, right_index=True, how='left')
mf6 = mf6.rename(columns={'balance':'debt_to_equity'})


# In[41]:


liabilities.index = pd.to_datetime(liabilities.index)
monthly_liabilities = liabilities.resample('M').mean()


# In[42]:


debt_ratio = abs(monthly_liabilities) / assets

mf7 = mf6.merge(debt_ratio, left_index=True, right_index=True, how='left')
mf7 = mf7.rename(columns={'balance':'debt_ratio'})


# In[43]:


print(accounts)



# In[44]:


# Step 3: Perform the addition
current_assets = cryptoloans_reindexed['balance'] + stablecoins_reindexed['balance'] + otherassets_reindexed['balance']

current_liabilities = dai['balance'] + dsr['balance']

current_ratio = current_assets / abs(current_liabilities)

current_ratio.index = pd.to_datetime(current_ratio.index)
current_ratio = current_ratio.resample('M').mean()
mf8 = mf7.merge(current_ratio, left_index=True, right_index=True, how='left')
mf8 = mf8.rename(columns={'balance':'current_ratio'})

mf8


# In[45]:


revenue = monthly_stats['lending_income'] + monthly_stats['liquidation_income'] + monthly_stats['trading_income']
revenue.index = pd.to_datetime(monthly_stats.index)
revenue.index = revenue.index + pd.offsets.MonthEnd(0)
mf9 = mf8.merge(revenue.to_frame(), left_index=True, right_index=True, how='left')
mf9 = mf9.rename(columns={ 0 :'revenue'})
mf9.tail()


# In[46]:


mf9['net_profit_margin'] = mf9['net_income'] / mf9['revenue']
mf9['price_to_sales'] = (mf9['supply'] * mf9['price']) / mf9['revenue']

mf9.tail()


# In[47]:


net_profit_margin = ttm_net_income / ttm_revenue
print(net_profit_margin)


# In[48]:


ttm_revenue


# In[49]:


price_to_sales = market_value / ttm_revenue

print(price_to_sales)


# In[50]:


assetsdf = pd.DataFrame(cryptoloans + rwa + stablecoins + otherassets).dropna().drop(columns = 'normalized')
assetsdf.index = pd.to_datetime(assetsdf.index)
assetsdf.index = assetsdf.index.normalize()

assetsdf.describe


# In[51]:


std_of_assets = assetsdf['balance'].pct_change().dropna().std(axis=0) 

std_of_assets


# In[52]:


assetsdf['balance'].pct_change().max()*100


# In[53]:


liabilitiesdf = pd.DataFrame(dai + dsr).dropna().drop(columns='normalized')
#plt.gca().ticklabel_format(style='plain', axis='both', useOffset=False, useLocale=False)

liabilitiesdf.index = pd.to_datetime(liabilitiesdf.index)
liabilitiesdf.index = liabilitiesdf.index.normalize()



# In[54]:


equitydf = pd.DataFrame(mkrequity).dropna().drop(columns='normalized')

equitydf.index = pd.to_datetime(equitydf.index)
equitydf.index = equitydf.index.normalize()


# **Here lets focus on getting to what we need for model; for DCF we need WACC; for WACC we need SML for cost of equity, and Weighted average stability fee - DSR expense rate for the cost of debt**

# Cost of Equity

# In[55]:


from apis2 import dpi_history

dpi_history.tail()


# In[56]:


from apis import mkr_history

mkr_history.head()


# In[57]:


#Now lets calculate the beta

from sklearn.linear_model import LinearRegression

X = dpi_history['daily_returns'].values.reshape(-1, 1)
Y = mkr_history['daily_returns'].values

model = LinearRegression()
model.fit(X, Y)

beta = model.coef_[0]
print(beta)


# In[58]:


dpi_history


# In[59]:


mkr_history


# In[60]:


import pandas as pd
from sklearn.linear_model import LinearRegression

# Define the rolling window size
window_size = 30  # For example, 30 days

# Initialize an empty Series to store rolling beta values
rolling_betas = pd.Series(index=dpi_history.index[window_size - 1:])

# Calculate rolling beta for each window
for start in range(len(dpi_history) - window_size + 1):
    end = start + window_size

    X_window = dpi_history['daily_returns'][start:end].values.reshape(-1, 1)
    Y_window = mkr_history['daily_returns'][start:end].values

    model = LinearRegression()
    model.fit(X_window, Y_window)

    # Store the beta coefficient
    rolling_betas.iloc[start] = model.coef_[0]

# Display the rolling beta series
print(rolling_betas)


# In[61]:

#mkr_history = mkr_history.set_index('date')

rolling_betas.index = mkr_history[29:].index
rolling_betas


# In[62]:


dpi_history


# In[63]:


mkr_history


# In[64]:


#dpi_history.set_index('date', inplace=True)


# In[67]:


from formulas import calculate_annual_return
# Convert the 'date' column to datetime
#dpi_history['date'] = pd.to_datetime(dpi_history['date'])

# Set the 'date' column as the index
#dpi_history.set_index('date', inplace=True)

# Group by year and apply the function
annual_returns = dpi_history.groupby(dpi_history.index.year).apply(calculate_annual_return)

print(annual_returns)


# In[68]:


from apis2 import tbilldf

tbilldf.tail()


# In[69]:


# Assuming 'day' is a column in your DataFrame and you want to filter based on this
tbill_timeseries = tbilldf[pd.to_datetime(tbilldf.index) >= '2019-12-01']


# In[70]:


tbill_decimals = pd.DataFrame(tbill_timeseries['value'] / 100)

tbill_decimals.tail()


# In[71]:


monthly_stats.index = pd.to_datetime(monthly_stats.index)

filtered_stats = monthly_stats[(monthly_stats.index >= "2019-12") & (monthly_stats.index < "2023-12")]



# In[72]:


#linear regression and r value for net income and interest rates

from sklearn.linear_model import LinearRegression
#We do .iloc[:-1] because is API and month is not done; we use previous full month
x = tbill_decimals['value'].iloc[:-1].values.reshape(-1, 1)
y = filtered_stats['net_income'].iloc[:-1].values

model = LinearRegression()
model.fit(x, y)


r_squared = model.score(x, y)

print("R^2:", r_squared)



# In[73]:


import matplotlib.pyplot as plt

# Create a scatter plot of the original data
plt.scatter(x, y, color='blue', label='Data points')

# Predict y values for the given x values
y_pred = model.predict(x)

# Plot the regression line
plt.plot(x, y_pred, color='red', label='Regression line')

# Add labels and title (optional)
plt.xlabel('tbill rate')
plt.ylabel('net income')
plt.title('Linear Regression Analysis')

# Show the legend
plt.legend()

# Display the plot
plt.show()


# In[74]:


y2 = filtered_stats['net_income'].iloc[:-1].values
x2 = tbill_decimals['value'].iloc[:-1].values

from scipy.stats import pearsonr

correlation_coefficient, _ = pearsonr(x2, y2)
print(correlation_coefficient)


# DAO income sensitive to broader crypto market volume (demand for services), and interest rate hikes (interest income)

# In[75]:


tbilldf_yearly = tbill_timeseries.groupby(tbill_timeseries.index.year).mean()

tbilldf_yearly


tbilldf_after2020 = tbilldf_yearly[tbilldf_yearly.index >= 2020]

tbilldf_after2020_dec = tbilldf_after2020 / 100




# In[76]:


current_risk_free = tbilldf['value'].iloc[-1] / 100


# In[77]:


tbilldf['value'] = tbilldf['value'] / 100
monthly_riskfree = tbilldf['value'].resample('M').mean()
monthly_riskfree_filtered = monthly_riskfree[monthly_riskfree.index > '2020-09-30']
monthly_riskfree_filtered


# In[79]:


#mkr_history['date'] = pd.to_datetime(mkr_history['date'])  # Convert the date column to datetime
#mkr_history.set_index('date', inplace=True)  # Set the date column as the index


# Convert daily returns to monthly returns
monthly_returns = (1 + mkr_history['daily_returns']).resample('M').prod() - 1

monthly_returns


# In[80]:


monthly_risk_premium = monthly_returns - monthly_riskfree_filtered
monthly_risk_premium.fillna(method='ffill')


# In[81]:


current_risk_free 


# In[82]:


annual_returns = pd.DataFrame(annual_returns)
annual_returns


# In[83]:


tbilldf_after2020_dec['value']


# In[84]:


yearly_risk_premium = annual_returns[0] - tbilldf_after2020_dec['value']
     

yearly_risk_premium


# In[85]:


average_yearly_risk_premium = yearly_risk_premium.mean()

average_yearly_risk_premium


# In[86]:


dpi_history['daily_returns']
 


# In[87]:


mkr_history.set_index(dpi_history.index, inplace=True)


# In[ ]:





# cumulative approach

# In[88]:


# Assuming df is your DataFrame and it's sorted by date
initial_value = dpi_history['price'].iloc[0]
final_value = dpi_history['price'].iloc[-1]
number_of_years = (dpi_history.index[-1] - dpi_history.index[0]).days / 365.25

cagr = (final_value / initial_value) ** (1 / number_of_years) - 1
cagr_percentage = cagr * 100

print(f"The CAGR is {cagr_percentage:.2f}%")


# In[89]:


cumulative_risk_premium = cagr - current_risk_free

print(cumulative_risk_premium)


# Cost of Equity=Risk Free Rate+β×(Market Return−Risk Free Rate)

# In[90]:


short_term_makerdao_cost_equity = current_risk_free + beta * average_yearly_risk_premium
long_term_makerdao_cost_equity = current_risk_free + beta * cumulative_risk_premium


# In[102]:


monthly_betas = rolling_betas.resample('M').mean()


# In[105]:


short_re_historical = monthly_riskfree_filtered + monthly_betas * monthly_risk_premium

short_re_historical


# In[106]:


print(short_term_makerdao_cost_equity, long_term_makerdao_cost_equity)


# Now, the cost of debt
# 
# 
# 

# In[107]:


stability_fees = {
     "ETH-A" : 0.0525,
     "ETH-B" : 0.0575,
     "ETH-C" : 0.05,
     "WSTETH-A" : 0.0525,
     "WBTC-A" : 0.0586,
     "Others" : sum([0.045,0.04,0.05,0.0466,0.025,0.028,0.0011,0.026,0.04,0.04,0.0561,0.0636,0.03,0.07]) / 14
    }

print(stability_fees)


# In[108]:


from apis2 import mkr_vault_df

mkr_vault_df = mkr_vault_df.groupby(['period','collateral']).sum()

mkr_vault_df.index


# In[109]:


total_weighted_fee = 0
total_revenue = mkr_vault_df['revenues'].sum()

for index, row in mkr_vault_df.iterrows():
    collateral_type = row.name[1]  # Access the 'collateral' part of the MultiIndex
    fee = stability_fees.get(collateral_type, stability_fees['Others'])  # Use the collateral type to get the fee
    weighted_fee = fee * row['revenues']
    total_weighted_fee += weighted_fee

weighted_average_stability_fee = total_weighted_fee / total_revenue

print("Weighted Average Stability Fee:", weighted_average_stability_fee)


# In[110]:


dsr_rate = 0.05

mkrdao_cost_debt = weighted_average_stability_fee - dsr_rate

mkrdao_cost_debt


# In[112]:


mf10 = mf9.merge(monthly_betas.to_frame(), left_index=True, right_index=True, how='left')
mf10 = mf10.rename(columns={0:'beta'})
mf10['beta'] = mf10['beta'].fillna(method='bfill')
mf10


# In[113]:


e1 = mf10['price'] * mf10['supply']
d1 = abs(monthly_liabilities)
v1 = e1 + d1
short_re1 = short_re_historical
rd1 = mkrdao_cost_debt

monthly_wacc = ((e1/v1) * short_re1) + ((d1/v1) * rd1)

print(monthly_wacc)


# In[ ]:





# In[140]:


e = market_value
d = abs(liabilities)
v = e + d
short_re = short_term_makerdao_cost_equity
long_re = long_term_makerdao_cost_equity # need to recalculate maybe with other benchmark, or maybe short_re better.  This also impacts beta
rd = mkrdao_cost_debt

wacc = ((e/v) * short_re) + ((d/v) * rd)

print(wacc.tail())


# In[116]:


print('market value of equity:', e)
print('market value of debt:', d)
print('value of financing:', v)
print('cost of equity:', short_re)
print('cost of debt:', rd)
print('proportion of debt financing:', (d / v)*100)
print('proportion of equity financing:', (e / v) * 100)
print('wacc:',wacc)


# In[117]:


#linear regression and r value for net income and interest rates

from sklearn.linear_model import LinearRegression
#We do .iloc[:-1] because is API and month is not done; we use previous full month
x = tbill_decimals['value'].iloc[:-1].values.reshape(-1, 1)
y = filtered_stats['lending_income'].iloc[:-1].values

model = LinearRegression()
model.fit(x, y)


r_squared = model.score(x, y)

print("R^2:", r_squared)



# In[118]:


monthly_growth_rate = monthly_stats.select_dtypes(include='number').pct_change()


monthly_growth_rate['trading_income'] = monthly_growth_rate['trading_income'].fillna(0)
monthly_growth_rate.replace([np.inf, -np.inf], np.nan, inplace=True)
monthly_growth_rate['liquidation_income'] = monthly_growth_rate['liquidation_income'].fillna(0)


monthly_growth_rate.tail()


# Liquidity Ratios

# In[119]:


from apis2 import filtered_BTC_history

# Convert the tz-aware index to tz-naive
filtered_BTC_history.index = filtered_BTC_history.index.tz_localize(None)

# Now try merging
merged_data = pd.merge(filtered_BTC_history, tbill_decimals, left_index=True, right_index=True)

merged_data.tail()


# In[120]:


filtered_stats['net_income'].tail()


# In[121]:


#linear regression and r value for net income and interest rates

from sklearn.linear_model import LinearRegression
#We do .iloc[:-1] because is API and month is not done; we use previous full month
x = merged_data[['Close', 'value']].values#.reshape(-1, 1)
y = filtered_stats['net_income'].iloc[:-1].values

model = LinearRegression()
model.fit(x, y)


r_squared = model.score(x, y)

print("R^2:", r_squared)



# In[122]:


"""
correlation_coefficient, _ = pearsonr(x, y)
print(correlation_coefficient)
"""


# In[123]:


ttm_data


# In[124]:


mkr_history


# In[125]:


mkr_annual_returns = mkr_history.groupby(mkr_history.index.year).apply(calculate_annual_return)

excess_return = mkr_annual_returns - tbilldf_after2020_dec['value']

mkr_avg_excess_return = excess_return.mean()


# In[126]:


mkr_avg_excess_return


# In[127]:


def fetch_dpi_historical_data(api_url):
    response = requests.get(api_url)
    dpi_history = pd.DataFrame()

    if response.status_code == 200:
        # Parse the JSON response
        dpi_historical_pricedata = response.json()
        # Extract the 'prices' data
        dpi_historical_price = dpi_historical_pricedata['prices']
        # Create a DataFrame with a 'timestamp' column instead of 'date'
        dpi_history = pd.DataFrame(dpi_historical_price, columns=['timestamp', 'price'])
        # Convert the 'timestamp' column from UNIX timestamps in milliseconds to datetime objects
        dpi_history['date'] = pd.to_datetime(dpi_history['timestamp'], unit='ms')
        # Set the 'date' column as the index of the DataFrame
        dpi_history.set_index('date', inplace=True)
        # Drop the original 'timestamp' column
        dpi_history.drop(columns='timestamp', inplace=True)
        return dpi_history
    else:
        print(f"Failed to retrieve data: {response.status_code}")
        return dpi_history  # Return an empty DataFrame in case of failure


# In[128]:


# Define the API endpoint and parameters
dpi_historical_api = "https://api.coingecko.com/api/v3/coins/defipulse-index/market_chart?vs_currency=usd&days=1152&interval=daily"

dpi_history = fetch_dpi_historical_data(dpi_historical_api)

dpi_history


# In[129]:


dpi_history['daily_returns'] = dpi_history['price'].pct_change().dropna()

dpi_history = dpi_history.iloc[1:] #first day of trading nothing to get for return

# Convert the 'date' column from UNIX timestamps in milliseconds to datetime objects
#dpi_history['date'] = pd.to_datetime(dpi_history['date'], unit='ms')


# In[130]:


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



# In[131]:


api_key = "NB6wsLRGqoVVQaLVuwk9mFB3kDssFVGK"

url='https://api.dune.com/api/v1/query/907852/results'
params = {"api_key": api_key }

daidf = fetch_data_from_api(url, params)


# In[132]:


daidf


# In[133]:


average_block_time_seconds = 15
# Assuming 1 block is a very short period, like 15 seconds
# Convert this to a fraction of a day (e.g., 15 seconds / total seconds in a day)
average_block_time_days = average_block_time_seconds / (24 * 60 * 60)

# Convert maturity to a numeric value (e.g., days)
maturity_days = {
    '1-day': 1,
    '1-week': 7,
    '1-month': 30,  # Approximate
    '1-year': 365,
    '1-block': average_block_time_days  # Short-term, but meaningful for financial analysis
}

daidf['maturity_days'] = daidf['maturity'].map(maturity_days)

# Calculate weight for each entry
daidf['weight'] = daidf['outflow'] / daidf['total_period']

# Calculate weighted maturity for each entry
daidf['weighted_maturity'] = daidf['maturity_days'] * daidf['weight']

# Group by date and calculate the WAM for each date
grouped = daidf.groupby('dt')
wam_by_date = grouped.apply(lambda x: (x['weighted_maturity'].sum()))

print(wam_by_date)


# In[134]:


daidf


# In[ ]:





# In[135]:


# Ensure 'dt' is a DateTime index
daidf['dt'] = pd.to_datetime(daidf['dt'])
daidf.set_index('dt', inplace=True)

# Now perform the groupby operation
wam_by_date = daidf.groupby(daidf.index.date).sum()

# Display the result
print(wam_by_date.iloc[-1])


# In[136]:


std_of_mkr = mkr_history['price'].pct_change().dropna().std(axis=0) 
std_of_mkr


# In[137]:


import pandas as pd
import numpy as np
from scipy.stats import norm
import requests

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

api_key = "NB6wsLRGqoVVQaLVuwk9mFB3kDssFVGK"

dai_maturity_url='https://api.dune.com/api/v1/query/907852/results'
dmuparams = {"api_key": api_key }

daidf = fetch_data_from_api(dai_maturity_url, dmuparams)

daidf['dt'] = pd.to_datetime(daidf['dt'])
daidf.set_index('dt', inplace=True)


# In[138]:


average_block_time_seconds = 15
# Assuming 1 block is a very short period, like 15 seconds
# Convert this to a fraction of a day (e.g., 15 seconds / total seconds in a day)
average_block_time_days = average_block_time_seconds / (24 * 60 * 60)

# Convert maturity to a numeric value (e.g., days)
maturity_days = {
    '1-day': 1,
    '1-week': 7,
    '1-month': 30,  # Approximate
    '1-year': 365,
    '1-block': average_block_time_days  # Short-term, but meaningful for financial analysis
}

daidf['maturity_days'] = daidf['maturity'].map(maturity_days)

# Calculate weight for each entry
daidf['weight'] = daidf['outflow'] / daidf['total_period']

# Calculate weighted maturity for each entry
daidf['weighted_maturity'] = daidf['maturity_days'] * daidf['weight']

# Group by date and calculate the WAM for each date
grouped = daidf.groupby('dt')
wam_by_date = grouped.apply(lambda x: (x['weighted_maturity'].sum()))

print(wam_by_date.iloc[-1])


# In[145]:


# Display the result
current_wam = wam_by_date.iloc[-1]

std_of_mkr = mkr_history['price'].pct_change().dropna().std(axis=0) 


# BSM valuation of equity as a call option
def equity_as_call_option(S, K, T, r, sigma, q=0):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_value = (S * np.exp(-q * T) * norm.cdf(d1)) - (K * np.exp(-r * T) * norm.cdf(d2))
    return call_value

# Example inputs
debt_obligation = abs(liabilities.iloc[-1]) 
time_to_maturity = float(current_wam/365)  # Time to maturity in years
risk_free_rate = float(current_risk_free)  # Risk-free interest rate
volatility = float(std_of_assets)  # Estimated volatility of the DAO's token price

# Calculate equity value as a call option
value_of_assets = assets.iloc[-1]
equity_value = equity_as_call_option(value_of_assets, debt_obligation, time_to_maturity, risk_free_rate, volatility)

print(f"The value of equity as a call option on assets is: {equity_value}")
print(f"The market value of equity is: {market_value}")


# In[149]:


price_per_bsm = equity_value / mf10['supply'].iloc[-1]

price_per_bsm

bsm_premium = mf10['price'].iloc[-1] - price_per_bsm

bsm_premium


# In[144]:


print(value_of_assets, 
debt_obligation, 
time_to_maturity,
risk_free_rate, 
volatility) 


# In[150]:


current_assets


# In[ ]:


#monthly_betas = rolling_betas.resample('M').mean()


# In[170]:


mf11 = mf10.merge(monthly_wacc.to_frame(), left_index=True, right_index=True, how='left')
mf11 = mf11.rename(columns={0:'wacc'})

mf11['wacc'] = mf11['wacc'].fillna(method='bfill')

full_metrics = mf11.fillna(method='ffill')

full_metrics


# In[202]:


import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Assuming 'full_metrics' is your DataFrame with the financial metrics
min_max_scaler = MinMaxScaler()
metrics_min_max_scaled = pd.DataFrame(min_max_scaler.fit_transform(full_metrics), columns=full_metrics.columns)

standard_scaler = StandardScaler()
metrics_standard_scaled = pd.DataFrame(standard_scaler.fit_transform(full_metrics), columns=full_metrics.columns, index=full_metrics.index)

# Define the function to score each metric based on its Z-score
def score_metric(value, is_higher_better=True, max_score=3):  # Changed max_score to 3
    if is_higher_better:
        return max(1, min(max_score, (value + 3) / 6 * max_score))
    else:
        return max(1, min(max_score, (-value + 3) / 6 * max_score))

# Define the original weights for each metric
weights = {
    'current_ratio': 0.20,  # High importance for liquidity
    'debt_to_equity': 0.15,  # Financial leverage and long-term solvency
    'debt_ratio': 0.10,  # Solvency and financial structure
    'roa': 0.15,  # Operational efficiency
    'net_profit_margin': 0.15,  # Profitability efficiency
    'bookval': 0.10,  # Net asset value
    'net_income': 0.10,  # Overall profitability
    'revenue': 0.05,  # Operational scale and market position
    # Reduced focus or excluded metrics
    'equity': 0.0,  # Less emphasis
    'supply': 0.0,  # Less emphasis
    'price': 0.0,  # Less emphasis
    'eps': 0.0,  # Less emphasis
    'price_to_earnings': 0.0,  # Less emphasis
    'roe': 0.0,  # Less emphasis
    'price_to_sales': 0.0,  # Less emphasis
    'beta': 0.0,  # Less emphasis
    'wacc': 0.0   # Less emphasis
}

# Calculating the total sum of weights
total_weight = sum(weights.values())

# Scaling down each weight so that total sum becomes 1.0
scaled_weights = {k: v / total_weight for k, v in weights.items()}

# Define which metrics are considered 'higher is better'
higher_is_better_metrics = [
    'roa',
    'current_ratio',
    'revenue',
    'net_profit_margin',
    'bookval',
    'net_income'
]

# Score each metric using Z-scores and the scaled weights
for metric in scaled_weights:
    higher_is_better = metric in higher_is_better_metrics
    metrics_standard_scaled[metric + '_score'] = metrics_standard_scaled[metric].apply(score_metric, args=(higher_is_better,))

# Multiply scores by their scaled weights and calculate the overall financial health score
for metric in scaled_weights:
    metrics_standard_scaled[metric + '_weighted_score'] = metrics_standard_scaled[metric + '_score'] * scaled_weights[metric]

metrics_standard_scaled['financial_health_score'] = metrics_standard_scaled[[col for col in metrics_standard_scaled.columns if 'weighted_score' in col]].sum(axis=1)

# Normalize the overall score to a 1 to 3 scale
max_possible_score = sum(scaled_weights.values()) * 3
metrics_standard_scaled['normalized_financial_health_score'] = metrics_standard_scaled['financial_health_score'] / max_possible_score * 3

# Categorize the normalized score into 3 categories
def categorize_score(score):
    if score >= 2.5:
        return 'good'
    elif score >= 1.5:
        return 'okay'
    else:
        return 'bad'

metrics_standard_scaled['financial_health_category'] = metrics_standard_scaled['normalized_financial_health_score'].apply(categorize_score)

# Display the results
print(metrics_standard_scaled[['normalized_financial_health_score', 'financial_health_category']])


# In[200]:


# Verify the total sum of weights
sum_of_weights = sum(weights.values())
print(f"Total sum of weights: {sum_of_weights}")
print(scaled_weights)


# In[169]:


metrics_standard_scaled.columns


# In[176]:


metrics_standard_scaled.to_excel("metrics.xlsx", index=True)


# In[ ]:




