import streamlit as st
import pandas as pd
import numpy as np
from makerdao import *
from sidebar import create_sidebar

create_sidebar()

st.title('MakerDAO (MKR)')

# Displaying current price with line chart
st.write(f"${current_price:,.2f}")
st.line_chart(mkr_history['price'])

# Display financial health score
latest_health_score = metrics_standard_scaled['financial_health_category'].iloc[-2]
color_map = {'bad': 'red', 'okay': 'yellow', 'good': 'green'}
score_color = color_map.get(latest_health_score, 'black')
st.markdown(f'<h3 style="color: white;">Financial Health: <span style="color: {score_color};">{latest_health_score.capitalize()}</span></h3>', unsafe_allow_html=True)

def generate_dynamic_summary():
    

    # Constructing the summary
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
    **MakerDAO's Financial Overview:**

    MakerDAO has realized a commendable net income of $76,051,021.51 TTM (Trailing Twelve Months). However, liquidity ratios raise flags about its short-term fiscal pressures. The current ratio stands at 0.53, and a cash ratio of 0.09, suggesting potential challenges in meeting short-term obligations. Additionally, the leverage ratios indicate a substantial reliance on debt financing, as evidenced by a high debt to equity ratio of 96.51. Despite these concerns, profitability metrics remain strong, underscoring the need for strategic financial management and careful consideration of liquidity and debt levels.

    **Market Position and Business Operations:**

    In the market, MakerDAO shows significant strength. With earnings per share at $84.05 and a market to book ratio of 27.17, investor confidence in the company is clear. A Compound Annual Growth Rate (CAGR) of 42.64% coupled with an average excess return of 111.15% further highlight its robust market standing. The Security Market Line (SML) analysis, considering MakerDAO's beta of 0.78, reveals a lower systemic risk relative to the overall market, indicating resilience against market volatility and a potentially safer investment.

    **Business Model and Governance:**

    At the core of MakerDAO's operation is its pioneering governance model through the MKR token, which fosters a high level of transparency and user engagement, aligning with the decentralized finance (DeFi) ethos. The operational success of the Maker Protocol, a key component of Ethereum-based decentralized applications (dapps), is reflective of the strategic effectiveness of the Maker Foundation's decentralization efforts. This governance model not only supports operational success but also ensures robust community involvement in decision-making processes.
    """)

with st.expander("Strategic Recommendations for Management"):
    st.write("""
    Improve Liquidity: Prioritize bolstering cash reserves and managing liabilities to enhance fiscal responsiveness.
    
    Debt Management: Rebalance the capital structure to mitigate financial risk and improve the debt to equity standing.
    
    Expansion of Services: Diversify the Maker Protocol’s offerings and collateral assets to broaden user base and stabilize revenue.
    
    Community Engagement: Intensify community involvement in governance to maintain protocol responsiveness and drive innovation.
    
    Risk Management: Capitalize on the lower beta to highlight MakerDAO's relative market stability. Advance risk analytics and implement hedging strategies to safeguard against market downturns. Promote community risk education for more informed governance decisions.
    """)

with st.expander("Investor Recommendations"):
    st.write("""
    Investors should consider MakerDAO's profitability and innovative governance as indicative of a strong investment opportunity. However, attention must be given to the liquidity constraints and high leverage, which introduce elements of financial risk.
    """)





# Accessing the data by label
expenses = ttm_data['expenses']
lending_income = ttm_data['lending_income']
liquidation_income = ttm_data['liquidation_income']
trading_income = ttm_data['trading_income']
net_income = ttm_data['net_income']

# Creating the income statement dictionary
incomestmt_data = {
    'Expenses': expenses,
    'Lending Income': lending_income,
    'Liquidation Income': liquidation_income,
    'Trading Income': trading_income,
    'Net Income': net_income
}

# Convert to DataFrame
incomestmt = pd.DataFrame(list(incomestmt_data.items()), columns=['Item', 'Amount'])

incomestmt = incomestmt.set_index('Item')

incomestmt['Amount'] = pd.to_numeric(incomestmt['Amount'], errors='coerce').fillna(0)


# Creating and displaying the balance sheet
balancesheet_data = {
    'Assets': assets,
    'Liabilities': abs(liabilities),
    'Equity': abs(equity)
}
balancesheet = pd.DataFrame.from_dict(balancesheet_data, orient='index', columns=['Amount'])
balancesheet.index.name = 'Item'

# Layout with columns
col1, col2 = st.columns(2)

with col1:
    st.subheader('TTM Consolidated Income Statement')
    st.table(incomestmt.style.format({"Amount": "${:,.2f}"}))

with col2:
    st.subheader('Live Balance Sheet')
    st.table(balancesheet.style.format({"Amount": "${:,.2f}"}))

    

# Layout with columns for financial information
col3, col4 = st.columns(2)

# Column 1: Liquidity and Leverage Ratios
with col3:
    st.subheader('Liquidity Ratios')
    st.write(f"Current Ratio: {current_ratio.iloc[-1]:.2f}")
    st.write(f"Cash Ratio: {cash_ratio:.2f}")

    st.subheader('Leverage Ratios')
    st.write(f"Debt Ratio: {debt_ratio:.2f}")
    st.write(f"Debt to Equity Ratio: {debt_to_equity:.2f}")

# Column 2: Profitability and Market Value Ratios
with col4:
    st.subheader('Profitability Ratios')
    st.write(f"Net Profit Margin: {net_profit_margin:.2%}")
    st.write(f"Return on Assets: {ROA:.2%}")
    st.write(f"Return on Equity: {ROE:.2%}")
    
st.subheader('Market Value Ratios')
col5, col6 = st.columns(2)

with col5:
    st.write(f"Earnings per Share: ${eps:,.2f}")
    st.write(f"Price to Earnings: {price_to_earnings:.2f}")
    st.write(f"Market to Book: {market_to_book:.2f}")
    st.write(f"Price to Sales: {price_to_sales:.2f}")
    
with col6:
    st.write(f"Enterprise Value: ${enterprise_value:,.2f}")
    st.write(f"EV Multiple: {ev_multiple:.2f}")
    st.write(f"Enterprise Value to Revenue: {ev_to_rev:,.2f}")

# Financial Metrics Section
st.subheader('Financial Metrics')
col7, col8 = st.columns(2)

with col7:
    st.write(f"Beta: {beta:.2f}")
    st.write(f"WACC: {wacc:.2%}")
    st.write(f"Cost of Equity: {short_re:.2%}")

with col8:
    st.write(f"Cost of Debt: {rd:.2%}")
    st.write(f"CAGR: {mkr_cagr:.2%}")
    st.write(f"Average Excess Return: {mkr_avg_excess_return:.2%}")
    



st.markdown("""
---
MakerDAO data and insights provided by [Steakhouse's MakerDAO dashboard on Dune Analytics](https://dune.com/steakhouse/makerdao).
""", unsafe_allow_html=True)