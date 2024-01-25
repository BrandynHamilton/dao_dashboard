import streamlit as st
import pandas as pd
from data.makerdao import aligned_data, short_re as dpi_mkr_re, e as mkr_mk, d as mkr_liabilities, rd as mkr_rd, current_risk_free, average_yearly_risk_premium as dpi_market_premium, tbilldf, long_wacc as dpi_long_wacc, balance_sheet_time as mkr_bs, quarterly_df, cumulative_risk_premium, dpi_history
from data.rocketpool import eth_history 
from data.formulas import *
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np
import numpy_financial as npf
from data.Lido import liabilities as ldo_liabilities, ldo_history, e as ldo_mk, rd as ldo_rd, merged_df
from data.rocketpool import live_liabilities as rpl_liabilities, rpl_history, e as rpl_mk, rd as rpl_rd, merged


def wacc(e, d, re, rd):
    v = e + d
    return ((e/v) * re) + ((d/v) * rd)

def calculate_rd(risk_free, beta, market_premium):
    return risk_free + beta * market_premium

from scipy.optimize import newton

def calculate_irr(initial_investment, cash_flows):
    total_cash_flows = [initial_investment] + cash_flows
    return npf.irr(total_cash_flows)
    
def calculate_npv(rate, initial_investment, cash_flows):
    total_cash_flows = [initial_investment] + cash_flows
    periods = range(len(total_cash_flows))
    return sum(total_cash_flows[t] / (1 + rate) ** t for t in periods)

def calculate_npv_and_total_cash_flows(rate, initial_investment, cash_flows):
    total_cash_flows = [initial_investment] + cash_flows
    periods = range(len(total_cash_flows))
    return sum(total_cash_flows[t] / (1 + rate) ** t for t in periods)
    
def calculate_payback_period(initial_investment, cash_flows):
    cumulative_cash_flow = 0
    for i, cash_flow in enumerate(cash_flows, start=1):
        cumulative_cash_flow += cash_flow
        if cumulative_cash_flow >= -initial_investment:
            return i
    return None  # Indicates payback period is longer than the number of periods

def calculate_discounted_payback_period(rate, initial_investment, cash_flows):
    cumulative_cash_flow = 0
    for i, cash_flow in enumerate(cash_flows, start=1):
        discounted_cf = cash_flow / ((1 + rate) ** i)
        cumulative_cash_flow += discounted_cf
        if cumulative_cash_flow >= -initial_investment:
            return i
    return None  # Indicates payback period is longer than the number of periods

def calculate_profitability_index(rate, initial_investment, cash_flows):
    npv = calculate_npv(rate, initial_investment, cash_flows)
    return (npv + abs(initial_investment)) / abs(initial_investment)




eth_annual_returns = eth_history.groupby(eth_history.index.year).apply(calculate_annual_return)



tbill_timeseries = tbilldf[pd.to_datetime(tbilldf.index) >= '2015-12-01']
tbill_decimals = pd.DataFrame(tbill_timeseries['value'] / 100)
tbilldf_yearly = tbill_timeseries.groupby(tbill_timeseries.index.year).mean(numeric_only=True)

eth_history['daily_returns_eth'] = eth_history['price'].pct_change().dropna()

data_df = aligned_data.merge(eth_history['daily_returns_eth'], left_index=True, right_index=True)

x_eth = data_df['daily_returns_eth'].values.reshape(-1, 1)
x_dpi = data_df['daily_returns_dpi'].values.reshape(-1, 1)
y = data_df['daily_returns_mkr'].values

eth_mkr_beta = calculate_beta(x_eth,y)
dpi_mkr_beta = calculate_beta(x_dpi,y)


eth_yearly_risk_premium = eth_annual_returns.to_frame('annual_return').merge(tbilldf_yearly, left_index=True, right_index=True )
eth_yearly_risk_premium.drop(columns = ['decimal'], inplace=True)

eth_yearly_risk_premium = eth_yearly_risk_premium['annual_return'] - eth_yearly_risk_premium['value']

eth_market_premium = eth_yearly_risk_premium.mean()



eth_mkr_re = calculate_rd(current_risk_free, eth_mkr_beta, eth_market_premium)

eth_mkr_wacc = wacc(mkr_mk, mkr_liabilities, eth_mkr_re, mkr_rd)



eth_cagr = calculate_historical_returns(eth_history)

eth_cumulative_risk_premium = eth_cagr - current_risk_free

dpi_cumulative_risk_premium = cumulative_risk_premium



eth_mkr_long_re = calculate_rd(current_risk_free, eth_mkr_beta, eth_cumulative_risk_premium)

eth_mkr_long_wacc = wacc(mkr_mk, mkr_liabilities, eth_mkr_long_re, mkr_rd)





def main(dao_selection, benchmark_selection, time_frame_selection):
    st.title('NPV Project Analysis')
    st.subheader("WACC Calculation Settings")
    
    # Align data for beta calculation
    if dao_selection == 'MKR':
        e, d, rd = mkr_mk, mkr_liabilities, mkr_rd
        y = aligned_data['daily_returns_mkr'].dropna().values
    elif dao_selection == 'LDO':
        e, d, rd = ldo_mk, ldo_liabilities.iloc[0], ldo_rd
        y = merged_df['daily_returns_ldo'].dropna().values
    elif dao_selection == 'RPL':
        e, d, rd = rpl_mk, rpl_liabilities.iloc[0], rpl_rd
        y = merged['rpl_daily_returns'].dropna().values

    # Ensure x and y arrays are of equal length
    min_length = min(len(data_df['daily_returns_eth']), len(y))
    x_eth = data_df['daily_returns_eth'].iloc[:min_length].values.reshape(-1, 1)
    x_dpi = data_df['daily_returns_dpi'].iloc[:min_length].values.reshape(-1, 1)
    
    # Calculate beta
    selected_beta = calculate_beta(x_eth if benchmark_selection == 'ETH' else x_dpi, y[:min_length])
    
    # Calculate re and wacc
    market_premium = eth_market_premium if time_frame_selection == 'Short Term' else eth_cumulative_risk_premium if benchmark_selection == 'ETH' else dpi_cumulative_risk_premium
    re = calculate_rd(current_risk_free, selected_beta, market_premium)
    selected_wacc = wacc(e, d, re, rd)
    
    st.write(f"Selected WACC for {dao_selection}: {selected_wacc:.3%}")
    return selected_wacc

    # Additional code for NPV, IRR, and other financial metric calculations...




# Function to calculate beta, re, wacc for a specific DAO
def calculate_dao_metrics(dao_token, benchmark_selection, time_frame_selection):
    if dao_token == 'MKR':
        data = mkr_data
        liabilities = mkr_liabilities
        beta_function = calculate_beta
    elif dao_token == 'LDO':
        data = ldo_data
        liabilities = ldo_liabilities
        beta_function = calculate_beta
    elif dao_token == 'RPL':
        data = rpl_data
        liabilities = rpl_liabilities
        beta_function = calculate_beta

    # Calculation of Beta
    if benchmark_selection == 'ETH':
        beta = beta_function(data['eth_returns'], data['dao_returns'])
    elif benchmark_selection == 'DPI':
        beta = beta_function(data['dpi_returns'], data['dao_returns'])

    # Calculation of re
    if time_frame_selection == 'Short Term':
        market_premium = eth_market_premium if benchmark_selection == 'ETH' else dpi_market_premium
    else:
        market_premium = eth_cumulative_risk_premium if benchmark_selection == 'ETH' else dpi_cumulative_risk_premium

    re = calculate_rd(current_risk_free, beta, market_premium)
    
    # Calculation of wacc
    wacc_value = wacc(data['equity'], liabilities, re, data['rd'])

    return beta, re, wacc_value


def calculate_npv_and_total_cash_flows(rate, initial_investment, cash_flows):
    total_cash_flows = [initial_investment] + cash_flows
    periods = range(len(total_cash_flows))
    npv = sum(cf / (1 + rate) ** t for t, cf in enumerate(total_cash_flows))
    return npv, total_cash_flows

dao_selection = st.radio('Choose the DAO:', ('MKR', 'LDO', 'RPL'), key='dao_selection')
benchmark_selection = st.radio('Choose the benchmark:', ('ETH', 'DPI'), key='benchmark_selection')
time_frame_selection = st.radio('Choose the time frame:', ('Short Term', 'Long Term'), key='time_frame_selection')

selected_wacc = main(dao_selection, benchmark_selection, time_frame_selection)

# Inputs for NPV calculation
st.subheader("NPV Calculation")
initial_investment_input = st.number_input('Initial Investment', value=100000.0)
# Ensure the initial investment is negative
initial_investment = -abs(initial_investment_input)

# Assuming selected_wacc is defined earlier in your script
discount_rate = selected_wacc

st.write('Enter Cash Flows for Each Period:')
cash_flows = []
num_periods = st.number_input('Number of Periods', min_value=1, max_value=10, value=5)
for i in range(num_periods):
    cash_flows.append(st.number_input(f'Cash Flow for Period {i+1}', value=0.0))

# Calculate and display NPV
if st.button('Calculate Financial Metrics'):
    npv, total_cash_flows = calculate_npv_and_total_cash_flows(discount_rate, initial_investment, cash_flows)
    irr = calculate_irr(initial_investment, cash_flows)
    payback_period = calculate_payback_period(initial_investment, cash_flows)
    discounted_payback_period = calculate_discounted_payback_period(discount_rate, initial_investment, cash_flows)
    pi = calculate_profitability_index(discount_rate, initial_investment, cash_flows)

    st.write(f'Net Present Value (NPV): ${npv:,.2f}')
    st.write(f'Total Cashflows: {total_cash_flows}')
    if irr is not None:
        st.write(f'Internal Rate of Return (IRR): {irr * 100:.2f}%')
    else:
        st.write('Internal Rate of Return (IRR): Not calculable')
    
    st.write(f'Payback Period: {payback_period} periods' if payback_period else 'Payback Period: More than the number of periods')
    st.write(f'Discounted Payback Period: {discounted_payback_period} periods' if discounted_payback_period else 'Discounted Payback Period: More than the number of periods')
    st.write(f'Profitability Index (PI): {pi:.2f}')
