import pandas as pd
import requests
import numpy as np
import seaborn as sns
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
    params = {'api_key': api_key}  # Add API key as a parameter
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
def fetch_mkr_historical_data(api_url, api_key):
    params = {'api_key': api_key}  # Add the API key as a parameter
    response = requests.get(api_url, params=params)

    if response.status_code == 200:
        # Parse the JSON response
        mkr_historical_pricedata = response.json()
        # Extract the 'prices' data
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
        return mkr_history, mkr_market_cap  # Return an empty DataFrame in case of failure

    
def fetch_dpi_historical_data(api_url, api_key):
    params = {'api_key': api_key}  # Add the API key as a parameter
    response = requests.get(api_url, params=params)
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


@st.cache_data(ttl=86400)
def fetch_and_process_api_data(api_url, data_key, date_column, value_column, date_format='datetime'):
    response = requests.get(api_url)
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data[data_key])
        
        if date_format == 'datetime':
            df[date_column] = pd.to_datetime(df[date_column])
        
        df.set_index(date_column, inplace=True)
        df[value_column] = df[value_column].astype(float)
        return df
    else:
        print(f"Failed to retrieve data: {response.status_code}")
        return pd.DataFrame()  # Return an empty DataFrame in case of failure    
# API URLs and parameters
api_key = "Vti1XpoLF3ulDjZuSbyLXblt4I1JGoVu"
api_key_cg = "CG-jTsiV2rsyVSHHULoNSWHU493"

# MakerDAO API call
mkrdao_url = "https://api.dune.com/api/v1/query/2840463/results/"
mkrdao_params = {"api_key": api_key}
mkrdao = fetch_data_from_api(mkrdao_url, mkrdao_params)

#Income Statement Call
revenueapi = "https://api.dune.com/api/v1/query/2641549/results"
revparams = {"api_key": api_key}
revdf = fetch_data_from_api(revenueapi, revparams)
    
monthly_stats = revdf.groupby(['period']).sum()
item_month = revdf.groupby(['period','item']).sum()

#Market Data Call

makerapi = "https://api.coingecko.com/api/v3/coins/maker"
market_value, current_price, supply = fetch_market_data(makerapi, api_key_cg)

#DPI market data call 

# dpi current info
dpiapi = "https://api.coingecko.com/api/v3/coins/defipulse-index"
dpi_market_value, dpi_current_price, dpi_supply = fetch_market_data(dpiapi, api_key_cg)

# dpi historical info
dpi_historical_api = "https://api.coingecko.com/api/v3/coins/defipulse-index/market_chart?vs_currency=usd&days=1152&interval=daily"
dpi_history = fetch_dpi_historical_data(dpi_historical_api, api_key_cg)
dpi_history['daily_returns'] = dpi_history['price'].pct_change().dropna()
dpi_history = dpi_history.iloc[1:] #first day of trading nothing to get for return

# Maker Historical Price Data
mkr_historical_api = "https://api.coingecko.com/api/v3/coins/maker/market_chart?vs_currency=usd&days=1152&interval=daily"
mkr_history, historical_mk = fetch_mkr_historical_data(mkr_historical_api, api_key_cg)
historical_mk['date'] = pd.to_datetime(historical_mk['date'], unit='ms')
historical_mk.set_index('date', inplace=True)
historical_supply =  historical_mk['marketcap'] / mkr_history['price']
mkr_history['daily_returns'] = mkr_history['price'].pct_change().dropna()
mkr_history = mkr_history.iloc[1:]

# MKR vault data
mkr_vault_url = "https://api.dune.com/api/v1/query/17338/results/"
mkrvault_params = {"api_key": api_key}
mkr_vault_df = fetch_data_from_api(mkr_vault_url, mkrvault_params)

#Lido Balance Sheet
lidobs_url = "https://api.dune.com/api/v1/query/2484656/results/"
params_lidobs = {"api_key": api_key}
lidobs_df = fetch_data_from_api(lidobs_url, params_lidobs)

# tbill api for risk-free rate 
tbill_historical_api = "https://api.stlouisfed.org/fred/series/observations?series_id=TB3MS&api_key=af3aeb14543cb05941f1b87abc3e3b7b&file_type=json"
tbilldf = fetch_and_process_api_data(tbill_historical_api, "observations", "date", "value")

#MKR Supply 
mkr_supply_url = "https://api.dune.com/api/v1/query/482349/results"
params_mkr_supply = {"api_key": api_key}
mkrsupply_df = fetch_data_from_api(mkr_supply_url, params_mkr_supply)



#more cleaning

mkrsupply_df['period'] = pd.to_datetime(mkrsupply_df['period'])
mkrsupply_df.set_index('period', inplace=True)
sum_mkrsupply_df = mkrsupply_df.groupby([pd.Grouper(freq='M')]).mean(numeric_only=True)
sum_mkrsupply_df.reset_index(inplace=True)
ttm_supply = sum_mkrsupply_df['balance'].head(12).mean()

var_in_cf = monthly_stats['net_income'].pct_change()
cleaned_var_in_cf = pd.DataFrame(var_in_cf.iloc[:-1].dropna())

cleaned_var_in_cf['net_income'].tail() 

#calculating quarterly numbers
from formulas import month_to_quarter

revdf['quarter'] = revdf['month'].apply(month_to_quarter)
quarterly_stats = revdf.groupby(['year', 'quarter']).sum().reset_index()
last_four_quarters = quarterly_stats.iloc[-4:]
ttm_data = last_four_quarters.sum()

ttm_revenue = ttm_data['lending_income'] + ttm_data['liquidation_income'] + ttm_data['trading_income']

accounts = mkrdao['item'].unique()

#Getting each account from the makerdao balance sheet
cryptoloans = mkrdao.where(mkrdao['item'].isin(['Crypto-Loans'])).dropna().groupby('period').sum()
rwa = mkrdao.where(mkrdao['item'].isin(['Real-World Assets'])).dropna().groupby('period').sum()
stablecoins = mkrdao.where(mkrdao['item'].isin(['Stablecoins'])).dropna().groupby('period').sum()
otherassets = mkrdao.where(mkrdao['item'].isin(['Others assets'])).dropna().groupby('period').sum()
dsr = mkrdao.where(mkrdao['item'].isin(['DSR'])).dropna().groupby('period').sum()
dai = mkrdao.where(mkrdao['item'].isin(['DAI'])).dropna().groupby('period').sum()
mkrequity = mkrdao.where(mkrdao['item'].isin(['Equity'])).dropna().groupby('period').sum()

#calculating assets, liabilities, equity
assets = cryptoloans['balance'].iloc[-1] + rwa['balance'].iloc[-1] + stablecoins['balance'].iloc[-1] + otherassets['balance'].iloc[-1] 
liabilities = dsr['balance'].iloc[-1] + dai['balance'].iloc[-1]
equity = mkrequity['balance'].iloc[-1]
equity_history = mkrequity['balance'].to_frame()
liability_history = dsr['balance'] + dai['balance']

#financial metrics calculations
bookval = abs(equity) / supply

market_to_book = current_price / bookval

ttm_net_income = ttm_data['net_income']

eps = ttm_net_income / ttm_supply

price_to_earnings = current_price/eps

ROE = ttm_net_income / abs(equity)

ROA = ttm_net_income / assets

debt_to_equity = abs(liabilities) / abs(equity)

debt_to_equity_history = abs(liability_history) / abs(equity_history)

debt_ratio = abs(liabilities) / assets

enterprise_value = market_value + abs(dsr['balance'].iloc[-1]) - stablecoins['balance'].iloc[-1]

ev_multiple = enterprise_value / ttm_net_income

current_assets = cryptoloans['balance'].iloc[-1] + stablecoins['balance'].iloc[-1] + otherassets['balance'].iloc[-1] 
current_liabilities = dai['balance'].iloc[-1] + dsr['balance'].iloc[-1]

current_ratio = current_assets / abs(current_liabilities)

cash_ratio = stablecoins['balance'].iloc[-1] / abs(current_liabilities)

net_profit_margin = ttm_net_income / ttm_revenue

price_to_sales = market_value / ttm_revenue

net_working_capital = current_assets + current_liabilities

print(net_working_capital)


# In[73]:


assetsdf = pd.DataFrame(cryptoloans + rwa + stablecoins + otherassets).dropna().drop(columns = 'normalized')
assetsdf.index = pd.to_datetime(assetsdf.index)
assetsdf.index = assetsdf.index.normalize()

assetsdf.tail()


# In[74]:


std_of_assets = assetsdf['balance'].pct_change().dropna().std(axis=0) 

std_of_assets


# In[75]:


assetsdf['balance'].pct_change().max()*100


# In[76]:


liabilitiesdf = pd.DataFrame(dai + dsr).dropna().drop(columns='normalized')
#plt.gca().ticklabel_format(style='plain', axis='both', useOffset=False, useLocale=False)

liabilitiesdf.index = pd.to_datetime(liabilitiesdf.index)
liabilitiesdf.index = liabilitiesdf.index.normalize()



# In[77]:


equitydf = pd.DataFrame(mkrequity).dropna().drop(columns='normalized')

equitydf.index = pd.to_datetime(equitydf.index)
equitydf.index = equitydf.index.normalize()


# **Here lets focus on getting to what we need for model; for DCF we need WACC; for WACC we need SML for cost of equity, and Weighted average stability fee - DSR expense rate for the cost of debt**

#Now lets calculate the beta

#instead of dpi, lets try eth as benchmark

eth = yf.Ticker('ETH-USD')

eth_history = eth.history(period='max', interval='1mo')

eth_history.index = pd.to_datetime(eth_history.index)

filtered_eth = eth_history[(eth_history.index >= "2019-12") & (eth_history.index < "2023-11")]


from sklearn.linear_model import LinearRegression

X = dpi_history['daily_returns'].values.reshape(-1, 1)
Y = mkr_history['daily_returns'].values

model = LinearRegression()
model.fit(X, Y)

beta = model.coef_[0]
print(beta)


# In[82]:




# In[83]:


from formulas import calculate_annual_return

# Group by year and apply the function
annual_returns = dpi_history.groupby(dpi_history.index.year).apply(calculate_annual_return)

print(annual_returns)

# Assuming 'day' is a column in your DataFrame and you want to filter based on this
tbill_timeseries = tbilldf[pd.to_datetime(tbilldf.index) >= '2019-12-01']

tbill_decimals = pd.DataFrame(tbill_timeseries['value'] / 100)

tbill_decimals.tail()


monthly_stats.index = pd.to_datetime(monthly_stats.index)

filtered_stats = monthly_stats[(monthly_stats.index >= "2019-12") & (monthly_stats.index < "2023-12")]



# In[88]:



# DAO income sensitive to broader crypto market volume (demand for services), and interest rate hikes (interest income)

# In[91]:


tbilldf_yearly = tbill_timeseries.groupby(tbill_timeseries.index.year).mean(numeric_only=True)

tbilldf_yearly


tbilldf_after2020 = tbilldf_yearly[tbilldf_yearly.index >= 2020]

tbilldf_after2020_dec = tbilldf_after2020 / 100


current_risk_free = tbilldf['value'].iloc[-1] / 100


current_risk_free 



annual_returns = pd.DataFrame(annual_returns)
annual_returns


tbilldf_after2020_dec['value']



yearly_risk_premium = annual_returns[0] - tbilldf_after2020_dec['value']
     

yearly_risk_premium


average_yearly_risk_premium = yearly_risk_premium.mean()

average_yearly_risk_premium


mkr_history.set_index(dpi_history.index, inplace=True)

# Assuming df is your DataFrame and it's sorted by date
initial_value = dpi_history['price'].iloc[0]
final_value = dpi_history['price'].iloc[-1]
number_of_years = (dpi_history.index[-1] - dpi_history.index[0]).days / 365.25

cagr = (final_value / initial_value) ** (1 / number_of_years) - 1
cagr_percentage = cagr * 100

print(f"The CAGR is {cagr_percentage:.2f}%")


# In[101]:


cumulative_risk_premium = cagr - current_risk_free

print(cumulative_risk_premium)


# Cost of Equity=Risk Free Rate+β×(Market Return−Risk Free Rate)

short_term_makerdao_cost_equity = current_risk_free + beta * average_yearly_risk_premium
long_term_makerdao_cost_equity = current_risk_free + beta * cumulative_risk_premium

print('cost of equity short term and long term:', short_term_makerdao_cost_equity, long_term_makerdao_cost_equity)


# Now, the cost of debt

#I do not have an api for the stability fees, I check makerburn.com regularly to update if needed
stability_fees = {
     "ETH-A" : 0.0525,
     "ETH-B" : 0.0575,
     "ETH-C" : 0.05,
     "WSTETH-A" : 0.0525,
     "WBTC-A" : 0.0586,
     "Others" : 0.0384
    }

mkr_vault_df = mkr_vault_df.groupby(['period','collateral']).sum()

total_weighted_fee = 0
total_revenue = mkr_vault_df['revenues'].sum()

for index, row in mkr_vault_df.iterrows():
    collateral_type = row.name[1]  # Access the 'collateral' part of the MultiIndex
    fee = stability_fees.get(collateral_type, stability_fees['Others'])  # Use the collateral type to get the fee
    weighted_fee = fee * row['revenues']
    total_weighted_fee += weighted_fee

weighted_average_stability_fee = total_weighted_fee / total_revenue

print("Weighted Average Stability Fee:", weighted_average_stability_fee)

#another static data point 
dsr_rate = 0.05

mkrdao_cost_debt = weighted_average_stability_fee - dsr_rate

mkrdao_cost_debt


e = market_value
d = abs(liabilities)
v = e + d
short_re = short_term_makerdao_cost_equity
long_re = long_term_makerdao_cost_equity 
rd = mkrdao_cost_debt

wacc = ((e/v) * short_re) + ((d/v) * rd)

print(wacc)


print('market value of equity:', e)
print('market value of debt:', d)
print('value of financing:', v)
print('cost of equity:', short_re)
print('cost of debt:', rd)
print('proportion of debt financing:', (d / v)*100)
print('proportion of equity financing:', (e / v) * 100)
print('wacc:',wacc)



monthly_growth_rate = monthly_stats.select_dtypes(include='number').pct_change()

monthly_growth_rate['trading_income'] = monthly_growth_rate['trading_income'].fillna(0)
monthly_growth_rate.replace([np.inf, -np.inf], np.nan, inplace=True)
monthly_growth_rate['liquidation_income'] = monthly_growth_rate['liquidation_income'].fillna(0)

# In[115]:
def calculate_historical_returns(prices):
    prices['date'] = prices.index
    starting_value = prices.iloc[0]['price']
    ending_value = prices.iloc[-1]['price']
    number_of_years = (prices.iloc[-1]['date'] - prices.iloc[0]['date']).days / 365.25
    cagr = (ending_value/starting_value) ** (1/number_of_years) - 1
    return cagr

mkr_cagr = calculate_historical_returns(mkr_history)

mkr_annual_returns = mkr_history.groupby(mkr_history.index.year).apply(calculate_annual_return)

excess_return = mkr_annual_returns - tbilldf_after2020_dec['value']

mkr_avg_excess_return = excess_return.mean()

dai_maturity_url='https://api.dune.com/api/v1/query/907852/results'
dmuparams = {"api_key": api_key }

daidf = fetch_data_from_api(dai_maturity_url, dmuparams)

daidf['dt'] = pd.to_datetime(daidf['dt'])
daidf.set_index('dt', inplace=True)

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

# Display the result
current_wam = wam_by_date.iloc[-1]

std_of_mkr = mkr_history['price'].pct_change().dropna().std(axis=0) 

from scipy.stats import norm

# BSM valuation of equity as a call option
def equity_as_call_option(S, K, T, r, sigma, q=0):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_value = (S * np.exp(-q * T) * norm.cdf(d1)) - (K * np.exp(-r * T) * norm.cdf(d2))
    return call_value

# Example inputs
debt_obligation = abs(liabilities) 
time_to_maturity = float(current_wam/365)  # Time to maturity in years
risk_free_rate = float(current_risk_free)  # Risk-free interest rate
volatility = float(std_of_assets)  # Estimated volatility of the DAO's token price

# Calculate equity value as a call option
value_of_assets = assets
equity_value = equity_as_call_option(value_of_assets, debt_obligation, time_to_maturity, risk_free_rate, volatility)

print(f"The value of equity as a call option on assets is: {equity_value}")
print(f"The market value of equity is: {market_value}")

filtered_equity = abs(equity_history)
equity_df = pd.DataFrame(filtered_equity)
historical_supply_clean = historical_supply.dropna()
historical_supply_clean.index = pd.to_datetime(historical_supply_clean.index).normalize()
equity_df.index = pd.to_datetime(equity_df.index).normalize()



metrics = equity_df.merge(historical_supply_clean.to_frame(name='supply'), 
                                                        left_index=True, right_index=True, 
                                                        how='left')
metrics = metrics.rename(columns={'balance':'equity'})
metrics['bookval'] = metrics['equity'] / metrics['supply']

monthly_averages = metrics.resample('M').mean()

hist_net_income = monthly_stats['net_income'].to_frame()
hist_net_income.index = pd.to_datetime(hist_net_income.index)

hist_price = mkr_history['price'].to_frame()
hist_price.index = mkr_history['date']
hist_price.index = hist_price.index.normalize()

monthly_closing_price = hist_price.resample('M').last()

metricsfiltered3 = monthly_averages.merge(monthly_closing_price, left_index=True, right_index=True, how='left')

hist_net_income.index = hist_net_income.index + pd.offsets.MonthEnd(0)

mf4 = metricsfiltered3.merge(hist_net_income, left_index=True, right_index=True, how='left')

mf4['eps'] = mf4['net_income'] / mf4['supply']
mf4['price_to_earnings'] = mf4['price'] / mf4['eps']
mf4['roe'] = mf4['net_income'] / mf4['equity']

all_dates = cryptoloans.index.union(rwa.index).union(stablecoins.index).union(otherassets.index)

# Step 2: Reindex each dataframe
cryptoloans_reindexed = cryptoloans.reindex(all_dates).fillna(0)
rwa_reindexed = rwa.reindex(all_dates).fillna(0)
stablecoins_reindexed = stablecoins.reindex(all_dates).fillna(0)
otherassets_reindexed = otherassets.reindex(all_dates).fillna(0)

# Step 3: Perform the addition
assets_timeseries = cryptoloans_reindexed['balance'] + rwa_reindexed['balance'] + stablecoins_reindexed['balance'] + otherassets_reindexed['balance']

# Verify the results
print(assets_timeseries)
assets_timeseries.index = pd.to_datetime(assets_timeseries.index)
assetstimeseries = assets_timeseries.resample('M').mean()

mf5 = mf4.merge(assetstimeseries, left_index=True, right_index=True, how='left')

mf5 = mf5.rename(columns={'balance': 'average_assets'})
mf5['roa'] = mf5['net_income'] / mf5['average_assets']

# Convert the liabilities and equity to absolute values if they are negative
liabilities_abs = liability_history.abs()
equity_abs = equity_history['balance'].abs()

# Calculate the debt-to-equity ratio
debt_to_equity_ratio = liabilities_abs / equity_abs

debt_to_equity_ratio.index = pd.to_datetime(debt_to_equity_history.index)
monthly_debt_to_equity = debt_to_equity_ratio.resample('M').mean()
mf6 = mf5.merge(monthly_debt_to_equity, left_index=True, right_index=True, how='left')
mf6 = mf6.rename(columns={'balance':'debt_to_equity'})

liabilities_abs.index = pd.to_datetime(liabilities_abs.index)

monthly_liabilities = liabilities_abs.resample('M').mean()

debt_ratio_history = monthly_liabilities / assetstimeseries

mf7 = mf6.merge(debt_ratio_history, left_index=True, right_index=True, how='left')
mf7 = mf7.rename(columns={'balance':'debt_ratio'})

current_assets = cryptoloans_reindexed['balance'] + stablecoins_reindexed['balance'] + otherassets_reindexed['balance']

current_liabilities = dai['balance'] + dsr['balance']

current_ratio = current_assets / abs(current_liabilities)

current_ratio.index = pd.to_datetime(current_ratio.index)
current_ratio = current_ratio.resample('M').mean()
mf8 = mf7.merge(current_ratio, left_index=True, right_index=True, how='left')
mf8 = mf8.rename(columns={'balance':'current_ratio'})

revenue = monthly_stats['lending_income'] + monthly_stats['liquidation_income'] + monthly_stats['trading_income']
revenue.index = pd.to_datetime(monthly_stats.index)
revenue.index = revenue.index + pd.offsets.MonthEnd(0)
mf9 = mf8.merge(revenue.to_frame(), left_index=True, right_index=True, how='left')
mf9 = mf9.rename(columns={ 0 :'revenue'})

mf9['net_profit_margin'] = mf9['net_income'] / mf9['revenue']
mf9['price_to_sales'] = (mf9['supply'] * mf9['price']) / mf9['revenue']

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


tbilldf['value'] = tbilldf['value'] / 100
monthly_riskfree = tbilldf['value'].resample('M').mean()
monthly_riskfree_filtered = monthly_riskfree[monthly_riskfree.index > '2020-09-30']

monthly_returns = (1 + mkr_history['daily_returns']).resample('M').prod() - 1

monthly_risk_premium = monthly_returns - monthly_riskfree_filtered

monthly_betas = rolling_betas.resample('M').mean()
short_re_historical = monthly_riskfree_filtered + monthly_betas * monthly_risk_premium

mf10 = mf9.merge(monthly_betas.to_frame(), left_index=True, right_index=True, how='left')
mf10 = mf10.rename(columns={0:'beta'})
mf10['beta'] = mf10['beta'].fillna(method='bfill')

full_metrics = mf10.fillna(method='ffill')
cleaned_metrics = full_metrics.drop(['beta', 'supply', 'revenue', 'average_assets', 'price','equity','net_income'], axis=1)

from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Assuming 'cleaned_metrics' is your DataFrame with the financial metrics
min_max_scaler = MinMaxScaler()
metrics_min_max_scaled = pd.DataFrame(min_max_scaler.fit_transform(cleaned_metrics), columns=cleaned_metrics.columns)

standard_scaler = StandardScaler()
metrics_standard_scaled = pd.DataFrame(standard_scaler.fit_transform(cleaned_metrics), columns=cleaned_metrics.columns, index=cleaned_metrics.index)

# Define the function to score each metric based on its Z-score
def score_metric(value, is_higher_better=True, max_score=3):
    if is_higher_better:
        return max(1, min(max_score, (value + 3) / 6 * max_score))
    else:
        return max(1, min(max_score, (-value + 3) / 6 * max_score))

# Define the original weights for each metric in 'cleaned_metrics'
weights = {
    'bookval': 0.20,
    'roe': 0.15,
    'roa': 0.15,
    'debt_to_equity': 0.15,
    'debt_ratio': 0.10,
    'current_ratio': 0.10,
    'net_profit_margin': 0.10,
    'price_to_sales': 0.05
    # Other metrics can be added if present in 'cleaned_metrics'
}

# Calculating the total sum of weights
total_weight = sum(weights.values())

# Scaling down each weight so that total sum becomes 1.0
scaled_weights = {k: v / total_weight for k, v in weights.items()}

# Define which metrics are considered 'higher is better'
higher_is_better_metrics = [
    'roa',
    'current_ratio',
    'net_profit_margin',
    'bookval',
    'roe'  # Assuming higher 'roe' is better
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

ev_to_rev = enterprise_value/ttm_revenue