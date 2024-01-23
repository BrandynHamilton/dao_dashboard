import streamlit as st
from data.makerdao import *
from data.makerev import *
import numpy as np
import numpy_financial as npf
from data.rocketpool import eth_history 
from data.formulas import *

mkr_mk = e
mkr_liabilities = d
mkr_rd = rd
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

eth_mkr_wacc = calculate_wacc(mkr_mk, mkr_liabilities, eth_mkr_re, mkr_rd)



eth_cagr = calculate_historical_returns(eth_history)

eth_cumulative_risk_premium = eth_cagr - current_risk_free



eth_mkr_long_re = calculate_rd(current_risk_free, eth_mkr_beta, eth_cumulative_risk_premium)

eth_mkr_long_wacc = calculate_wacc(mkr_mk, mkr_liabilities, eth_mkr_long_re, mkr_rd)

percentage_to_distribute = 0.90

quarterly_df['dividend'] = quarterly_df['net_income'] * percentage_to_distribute

quarterly_df['dividend_per_share'] = quarterly_df['dividend'] / quarterly_df['supply']

quarterly_df['dividend_per_share'] = quarterly_df['dividend_per_share'].apply(lambda x: 0 if x <= 0 else x)

def show_makerpage():
        
    
    

    
    
    
    
    st.title('MakerDAO (MKR)')

    with st.expander('Benchmark'):
        benchmark_selection = st.radio(
            'Choose the benchmark for WACC calculation:',
            ('ETH', 'DPI'),
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
        selected_beta = eth_mkr_beta
        market_premium = eth_market_premium if time_frame_selection == 'Short Term' else eth_cumulative_risk_premium
        re = calculate_rd(current_risk_free, selected_beta, market_premium)
    elif benchmark_selection == 'DPI':
        selected_beta = dpi_mkr_beta
        market_premium = dpi_market_premium if time_frame_selection == 'Short Term' else dpi_cumulative_risk_premium
        re = calculate_rd(current_risk_free, selected_beta, market_premium)

    selected_wacc = calculate_wacc(mkr_mk, mkr_liabilities, re, mkr_rd)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Selected WACC", f"{selected_wacc:.3%}")
    with col2:
        st.metric("Selected Beta", f"{selected_beta:.2f}")
    with col3:
        st.metric("Selected Cost of Equity", f"{re:.2%}")

    st.metric('marketcap:', mkr_mk)
    st.metric('liabilities:', mkr_liabilities)
    st.metric('cost of debt:',mkr_rd )
    

    
    
    st.metric('Price', f"${current_price:,.2f}")
    st.line_chart(mkr_history['price'])
    
    
    latest_health_score = metrics_standard_scaled['financial_health_category'].iloc[-2]
    color_map = {'bad': 'red', 'okay': 'yellow', 'good': 'green'}
    score_color = color_map.get(latest_health_score, 'black')
    st.markdown(f'<h3 style="color: white;">Financial Health: <span style="color: {score_color};">{latest_health_score.capitalize()}</span></h3>', unsafe_allow_html=True)
    
    def generate_dynamic_summary():
        if benchmark_selection == 'ETH':
            selected_beta = eth_mkr_beta
        elif benchmark_selection == 'DPI':
            selected_beta = dpi_mkr_beta
        
    
        
        summary = (
            f"MakerDAO (MKR) is currently priced at ${current_price:,.2f}. "
            f"The financial health is rated as '{latest_health_score.capitalize()}' with a net profit margin of {net_profit_margin:.2%}. "
            f"The current ratio of {current_ratio.iloc[-1]:.2f} and a debt to equity ratio of {debt_to_equity:.2f} reflect its financial stability."
        )
        return summary
    
    
    with st.container():
        st.write(""" ### Executive Summary
        """)
        st.write(generate_dynamic_summary())
    
    with st.expander("Financial Health and Analysis"):
        st.write("""
        **MakerDAO's Financial Overview:**"""
    
        f""" MakerDAO has realized a commendable net income of ${net_income:,.2f} TTM (Trailing Twelve Months). However, liquidity ratios raise flags about its short-term fiscal pressures. The current ratio stands at {current_ratio.iloc[-1]:.2f}, and a cash ratio of {cash_ratio:.2f}, suggesting potential challenges in meeting short-term obligations. Additionally, the leverage ratios indicate a substantial reliance on debt financing, as evidenced by a high debt to equity ratio of {debt_to_equity:.2f}. Despite these concerns, profitability metrics remain strong, underscoring the need for strategic financial management and careful consideration of liquidity and debt levels.
    
        **Market Position and Business Operations:**
    
        In the market, MakerDAO shows significant strength. With earnings per share at ${eps:,.2f} and a market to book ratio of {market_to_book:.2f}, investor confidence in the company is clear. A Compound Annual Growth Rate (CAGR) of {mkr_cagr:.2%} coupled with an average excess return of {mkr_avg_excess_return:.2%} further highlight its robust market standing. The Security Market Line (SML) analysis, considering MakerDAO's beta of {beta:.2f}, reveals a lower systemic risk relative to the overall market, indicating resilience against market volatility and a potentially safer investment. """)
    
    with st.expander("Management Outlook"):
        st.write("""
        Improve Liquidity: Prioritize bolstering cash reserves and managing liabilities to enhance fiscal responsiveness.
        
        Debt Management: Rebalance the capital structure to mitigate financial risk and improve the debt to equity standing.
        
        Expansion of Services: Diversify the Maker Protocol’s offerings and collateral assets to broaden user base and stabilize revenue.
        
        Community Engagement: Intensify community involvement in governance to maintain protocol responsiveness and drive innovation.
        
        Risk Management: Capitalize on the lower beta to highlight MakerDAO's relative market stability. Advance risk analytics and implement hedging strategies to safeguard against market downturns. Promote community risk education for more informed governance decisions.
        """)
    
    with st.expander("Investor Outlook"):
        st.write("""
        Investors should consider MakerDAO's profitability and innovative governance as indicative of a strong investment opportunity. However, attention must be given to the liquidity constraints and high leverage, which introduce elements of financial risk.
        """)
    
    
    
    
    
    
    
    
    
    incomestmt_data = {
        'Expenses': expenses,
        'Lending Income': lending_income,
        'Liquidation Income': liquidation_income,
        'Trading Income': trading_income,
        'Net Income': net_income
    }
    
    
    incomestmt = pd.DataFrame(list(incomestmt_data.items()), columns=['Item', 'Amount'])
    
    incomestmt = incomestmt.set_index('Item')
    
    incomestmt['Amount'] = pd.to_numeric(incomestmt['Amount'], errors='coerce').fillna(0)
    
    
   
    balancesheet_data = {
        'Assets': assets,
        'Liabilities': abs(liabilities),
        'Equity': abs(equity)
    }
    balancesheet = pd.DataFrame.from_dict(balancesheet_data, orient='index', columns=['Amount'])
    balancesheet.index.name = 'Item'
    
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader('TTM Consolidated Income Statement')
        st.table(incomestmt.style.format({"Amount": "${:,.2f}"}))
    
    with col2:
        st.subheader('Live Balance Sheet')
        st.table(balancesheet.style.format({"Amount": "${:,.2f}"}))
    
        
    
   
    col3, col4 = st.columns(2)
    
    liquidity_df = pd.DataFrame({
    'Metric': ['Current Ratio', 'Cash Ratio'],
    'Value': [f"{current_ratio.iloc[-1]:.2f}", f"{cash_ratio:.2f}"]
    })
    
    leverage_df = pd.DataFrame({
        'Metric': ['Debt Ratio', 'Debt to Equity Ratio'],
        'Value': [f"{debt_ratio:.2f}", f"{debt_to_equity:.2f}"]
    })
    
    profitability_df = pd.DataFrame({
        'Metric': ['Net Profit Margin', 'Return on Assets', 'Return on Equity', 'Capital Intensity Ratio'],
        'Value': [f"{net_profit_margin:.2%}", f"{ROA:.2%}", f"{ROE:.2%}", f"{capital_intensity_ratio:.2f}"]
    })
    
    market_value_df = pd.DataFrame({
        'Metric': ['Earnings per Share', 'Price to Earnings', 'Market to Book', 'Price to Sales'],
        'Value': [f"${eps:,.2f}", f"{price_to_earnings:.2f}", f"{market_to_book:.2f}", f"{price_to_sales:.2f}"]
    })
    
    enterprise_value_df = pd.DataFrame({
        'Metric': ['Enterprise Value', 'EV Multiple', 'Enterprise Value to Revenue'],
        'Value': [f"${enterprise_value:,.2f}", f"{ev_multiple:.2f}", f"{ev_to_rev:,.2f}"]
    })
    
    financial_metrics_df = pd.DataFrame({
        'Metric': ['Beta', 'WACC', 'Cost of Equity', 'Cost of Debt', 'CAGR', 'Average Excess Return'],
        'Value': [f"{selected_beta:.2f}", f"{selected_wacc:.2%}", f"{re:.2%}", f"{rd:.2%}*", f"{mkr_cagr:.2%}", f"{mkr_avg_excess_return:.2%}"]
    })
    
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader('Liquidity Ratios')
        st.table(liquidity_df.set_index('Metric'))
    
        st.subheader('Leverage Ratios')
        st.table(leverage_df.set_index('Metric'))
    
    with col4:
        st.subheader('Profitability Ratios')
        st.table(profitability_df.set_index('Metric'))
    
    st.subheader('Market Value Metrics')
    col5, col6 = st.columns(2)
    
    with col5:
        st.table(market_value_df.set_index('Metric'))
    
    with col6:
        st.table(enterprise_value_df.set_index('Metric'))
    
    st.subheader('Financial Metrics')
    col7, col8 = st.columns(2)
    
    with col7:
        st.table(financial_metrics_df.set_index('Metric').iloc[:3])
    
    with col8:
        st.table(financial_metrics_df.set_index('Metric').iloc[3:])

    
        
    
    
    st.markdown("""
---
\\* Cost of Debt calculated DSR - Stability Fees
""")
    st.markdown("""
     
    MakerDAO data and insights provided by [Steakhouse's MakerDAO dashboard on Dune Analytics](https://dune.com/steakhouse/makerdao).
    """, unsafe_allow_html=True)