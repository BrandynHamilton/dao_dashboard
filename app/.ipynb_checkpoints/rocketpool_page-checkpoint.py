import streamlit as st
from data.rocketpool import *
from data.formulas import *
import numpy as np
import numpy_financial as npf
from data.makerdao import cumulative_risk_premium
from maker_page import average_eth_short_risk, eth_short_cumulative_risk_premium

rpl_mk = e
rpl_liabilities = d
rpl_rd = rd
dpi_market_premium = average_yearly_risk_premium
dpi_cumulative_risk_premium = cumulative_risk_premium



def calculate_wacc(e, d, re, rd):
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




tbilldf_yearly = tbill_decimals

eth_history['daily_returns_eth'] = eth_history['price'].pct_change().dropna()

data_df = merged.merge(eth_history['daily_returns_eth'], left_index=True, right_index=True)

x_eth = data_df['daily_returns_eth'].values.reshape(-1, 1)
x_dpi = data_df['daily_returns'].values.reshape(-1, 1)
y = data_df['rpl_daily_returns'].values

eth_rpl_beta = calculate_beta(x_eth,y)
dpi_rpl_beta = calculate_beta(x_dpi,y)

eth_market_premium = average_eth_short_risk
eth_cumulative_risk_premium = eth_short_cumulative_risk_premium


eth_rpl_long_re = calculate_rd(current_risk_free, eth_rpl_beta, eth_cumulative_risk_premium)

eth_rpl_long_wacc = calculate_wacc(rpl_mk, rpl_liabilities, eth_rpl_long_re, rpl_rd)

rpl_dpi_long_coste = calculate_rd(current_risk_free, dpi_rpl_beta, dpi_cumulative_risk_premium)
rpl_dpi_short_coste = calculate_rd(current_risk_free, dpi_rpl_beta, dpi_market_premium)
rpl_eth_long_coste = calculate_rd(current_risk_free, eth_rpl_beta, eth_cumulative_risk_premium)
rpl_eth_short_coste = calculate_rd(current_risk_free, eth_rpl_beta, eth_market_premium)
    
rpl_dpi_long_wacc = calculate_wacc(rpl_mk, rpl_liabilities, rpl_dpi_long_coste, rpl_rd)
rpl_dpi_short_wacc = calculate_wacc(rpl_mk, rpl_liabilities, rpl_dpi_short_coste, rpl_rd)
rpl_eth_long_wacc = calculate_wacc(rpl_mk, rpl_liabilities, rpl_eth_long_coste, rpl_rd)
rpl_eth_short_wacc = calculate_wacc(rpl_mk, rpl_liabilities, rpl_eth_short_coste, rpl_rd)




balancesheet_data = {
        'Assets': balance_sheet['assets'].iloc[-1],
        'Liabilities': balance_sheet['liabilities'].iloc[-1],
        'Equity': balance_sheet['equity'].iloc[-1]
    }
    
balancesheet = pd.DataFrame.from_dict(balancesheet_data, orient='index', columns=['Amount'])

def show_rocketpoolpage():

    
    
    
    st.title('Rocketpool (RPL)')

    with st.expander('Benchmark'):
        benchmark_selection = st.radio(
            'Choose the benchmark for WACC calculation:',
            ('DPI', 'ETH'),
            key='main_benchmark_selection'
        )
    with st.expander('Time Frame'):
        time_frame_selection = st.radio(
            'Choose the time frame for WACC calculation:',
            ('Short Term', 'Long Term'),
            key='main_time_frame_selection'
        )

    # Determine beta based on user selection
    if benchmark_selection == 'ETH':
        selected_beta = eth_rpl_beta
        market_premium = eth_market_premium if time_frame_selection == 'Short Term' else eth_cumulative_risk_premium
        re = calculate_rd(current_risk_free, selected_beta, market_premium)
    elif benchmark_selection == 'DPI':
        selected_beta = dpi_rpl_beta
        market_premium = dpi_market_premium if time_frame_selection == 'Short Term' else dpi_cumulative_risk_premium
        re = calculate_rd(current_risk_free, selected_beta, market_premium)

    selected_wacc = calculate_wacc(rpl_mk, rpl_liabilities, re, rpl_rd)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Selected WACC", f"{selected_wacc:.3%}")
    with col2:
        st.metric("Selected Beta", f"{selected_beta:.2f}")
    with col3:
        st.metric("Selected Cost of Equity", f"{re:.2%}")

    
    
    st.metric('Price', f"${rpl_current_price:,.2f}")

    #balance_sheet = balance_sheet.transpose()
    st.line_chart(rpl_history['price'])

    def generate_dynamic_summary():
        
    
        
        summary = (
            f"Rocketpool, currently trading at ${rpl_current_price:,.2f}, showcases a balanced financial profile. With a current ratio of {current_ratio.iloc[-1]:.2f}, it demonstrates adequate capability to meet its short-term liabilities with its current assets. However, a debt to equity ratio of {debt_to_equity.iloc[-1]:.2f} suggests a significant reliance on debt financing, which could pose risks in terms of financial leverage.")
        
        return summary
    
    
    with st.container():
        st.write(""" ### Executive Summary
        """)
        st.write(generate_dynamic_summary())
    
    st.subheader('Live Balance Sheet')
    st.table(balancesheet.style.format({"Amount": "${:,.2f}"}))

    
    
    

                 
    
       

    col1, col2 = st.columns(2)
    
    liquidity_df = pd.DataFrame({
    'Metric': ['Current Ratio'],
    'Value': [f"{current_ratio.iloc[-1]:.2f}"]
    })
    leverage_df = pd.DataFrame({
        'Metric': ['Debt Ratio', 'Debt to Equity Ratio'],
        'Value': [f"{debt_ratio.iloc[-1]:.2f}", f"{debt_to_equity.iloc[-1]:.2f}"]
    })
    market_value_df = pd.DataFrame({
        'Metric': ['Enterprise Value', 'EV to Revenue', "Book Value", "Market to Book"],
        'Value': [ f"${ev_df['ev_historical'].iloc[-1]:,.2f}", f"{ttm_ev_rev:.2f}", f"{bookval:.2f}", f"{market_to_book:.2f}"]
    })
    financial_metrics_df = pd.DataFrame({
        'Metric': ['Beta', 'WACC', 'CAGR', 'Average Excess Return', 'Cost of Debt', 'Cost of Equity'],
        'Value': [f"{selected_beta:.2f}", f"{selected_wacc:.2%}", f"{rpl_cagr:.2%}", f"{rpl_avg_excess_return:.2%}", f"{cost_of_debt:.2%}", f"{re:.2%}" ]
    })

    with col1:
        st.subheader('Liquidity Ratios')
        st.table(liquidity_df.set_index('Metric'))
    with col2:
        st.subheader('Leverage Ratios')
        st.table(leverage_df.set_index('Metric'))

    col3, col4 = st.columns(2)

    with col3:
        st.subheader('Market Value Metrics')
        st.table(market_value_df.set_index('Metric'))
    with col4:
        st.subheader('Financial Metrics')
        st.table(financial_metrics_df.set_index('Metric'))
    
        
       