import streamlit as st
import pandas as pd
import numpy as np

# Assuming makerdao.py is correctly imported and necessary variables are available
from makerdao import *

st.title('MakerDAO (MKR)')

# Displaying current price with line chart
st.write(f"${current_price:,.2f}")
st.line_chart(mkr_history['price'])

# Display financial health score
latest_health_score = metrics_standard_scaled['financial_health_category'].iloc[-2]
color_map = {'bad': 'red', 'okay': 'yellow', 'good': 'green'}
score_color = color_map.get(latest_health_score, 'black')
st.markdown(f'<h3 style="color: white;">Financial Health: <span style="color: {score_color};">{latest_health_score.capitalize()}</span></h3>', unsafe_allow_html=True)

st.text("""
Executive Summary:
MakerDAO stands out as a dominant player in the decentralized finance (DeFi) sector, with its cutting-edge Multi-Collateral Dai (MCD) system and robust governance model. Exhibiting a net profit margin of 50.51% and an impressive return on equity of 141.04%, the company has proven its profitability and efficient capital utilization. An enterprise value of $2,455,136,429.26 and a price to earnings ratio of 18.96 showcases MakerDAO as an attractive proposition within the DeFi industry.

Financial Health and Analysis:
MakerDAO has realized a commendable net income of $76,051,021.51 TTM. Yet, liquidity ratios raise flags about its short-term fiscal pressures, with a current ratio of 0.53 and a cash ratio of 0.09. The leverage ratios underscore a substantial dependence on debt, evidenced by a debt to equity ratio of 96.51. While profitability metrics are strong, these ratios highlight areas for strategic financial management focus.

Market Position and Business Operations:
With earnings per share at $84.05 and a market to book ratio of 27.17, investor confidence is palpable. The company's CAGR of 42.64% and an average excess return of 111.15% underscore its robust market standing. The SML analysis, in light of MakerDAO's beta of 0.78, reveals a lower systemic risk relative to the market, indicating resilience against market volatility.

Business Model and Governance:
MakerDAO's pioneering governance via the MKR token fosters transparency and user engagement, integral to the DeFi ethos. The operational success of the Maker Protocol, a cornerstone of Ethereum-based dapps, reflects the strategic effectiveness of the Maker Foundation's decentralization efforts.

Strategic Recommendations for Management:

Improve Liquidity: Prioritize bolstering cash reserves and managing liabilities to enhance fiscal responsiveness.

Debt Management: Rebalance the capital structure to mitigate financial risk and improve the debt to equity standing.

Expansion of Services: Diversify the Maker Protocol’s offerings and collateral assets to broaden user base and stabilize revenue.

Community Engagement: Intensify community involvement in governance to maintain protocol responsiveness and drive innovation.

Risk Management: Capitalize on the lower beta to highlight MakerDAO's relative market stability. Advance risk analytics and implement hedging strategies to safeguard against market downturns. Promote community risk education for more informed governance decisions.

Investor Recommendations:

Investors should consider MakerDAO's profitability and innovative governance as indicative of a strong investment opportunity. However, attention must be given to the liquidity constraints and high leverage, which introduce elements of financial risk.

Conclusion:

MakerDAO's position in the DeFi marketplace is underscored by strong financial indicators and a distinctive governance model. The strategic focus on improving liquidity and debt management, coupled with an innovative approach to community engagement and risk management, will be vital for MakerDAO's sustainable growth and continued market leadership. Investors and management alike should take heed of the company's financial nuances and its potential within the evolving DeFi landscape.
""")



expenses = ttm_data.iloc[2]  # Adjust these indices based on your data structure
lending_income = ttm_data.iloc[3]
liquidation_income = ttm_data.iloc[4]
trading_income = ttm_data.iloc[7]
net_income = ttm_data.iloc[6]

# Creating and displaying the income statement
incomestmt_data = {
    'Expenses': expenses,
    'Lending Income': lending_income,
    'Liquidation Income': liquidation_income,
    'Trading Income': trading_income,
    'Net Income': net_income
}
incomestmt = pd.DataFrame.from_dict(incomestmt_data, orient='index', columns=['Amount'])
incomestmt.index.name = 'Item'

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
    st.table(balancesheet)

    

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
    st.write(f"Earnings per Share: ${eps:,.2f}")
    st.write(f"Price to Earnings: {price_to_earnings:.2f}")
    st.write(f"Market to Book: {market_to_book:.2f}")
    st.write(f"Price to Sales: {price_to_sales:.2f}")
    st.write(f"Enterprise Value: ${enterprise_value:,.2f}")
    st.write(f"EV Multiple: {ev_multiple:.2f}")

# Financial Metrics Section
st.subheader('Financial Metrics')
col5, col6 = st.columns(2)

with col5:
    st.write(f"Beta: {beta:.2f}")
    st.write(f"WACC: {wacc:.2%}")
    st.write(f"Cost of Equity: {short_re:.2%}")

with col6:
    st.write(f"Cost of Debt: {rd:.2%}")
    st.write(f"CAGR: {mkr_cagr:.2%}")
    st.write(f"Average Excess Return: {mkr_avg_excess_return:.2%}")