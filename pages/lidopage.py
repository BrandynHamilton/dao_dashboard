import streamlit as st
import pandas as pd
import numpy as np

from Lido import * 

st.title('LidoDAO (LDO)')

st.write(f"${ldo_current_price:,.2f}")
st.line_chart(ldo_history['price'])

latest_health_score = metrics_standard_scaled['financial_health_category'].iloc[1]

color_map = {'bad': 'red', 'okay': 'yellow', 'good': 'green'}
score_color = color_map.get(latest_health_score, 'black')
st.markdown(f'<h3 style="color: white;">Financial Health: <span style="color: {score_color};">{latest_health_score.capitalize()}</span></h3>', unsafe_allow_html=True)
# Get the latest financial health category

st.text("""

Executive Summary:
LidoDAO, commanding a presence in the DeFi ecosystem with its LDO token priced at $2.77, oversees liquid staking protocols. The organization faces a net loss of $40,586,965.90 but holds a strong asset base with assets of over 21 billion dollars and an equity value of $91,616,907.0751. Its financial health is marked as stable despite current profitability challenges.

Financial Overview:
The balanced liquidity position, as shown by the current ratio, is contrasted by negative profitability ratios and a high debt to equity ratio of 229.89. This leverage indicates a considerable financial risk that requires strategic attention. Nevertheless, the market has maintained a positive outlook on LidoDAO, as evidenced by a market to book ratio of 26.91 and an average excess return of 61.87%.

Operational Insights and Strategic Recommendations:
LidoDAO's operational efforts in managing liquid staking protocols have been commendable, yet financial setbacks suggest the need for strategic adjustments.

Cost Optimization: Reviewing and reducing operating expenses is paramount to turning the net loss into profit without compromising operational integrity.

Revenue Enhancement: Innovating and expanding revenue streams, including reevaluating service fees and staking strategies, can increase net revenue.

Leverage Management: Restructuring debt and considering equity options to manage the high leverage will be vital for financial health.

Governance and Community Engagement: Engaging LDO holders in governance can harness community-driven innovation and enhance protocol decisions.

Research and Development: Investing in R&D and protocol upgrades is essential to stay ahead in the competitive DeFi market and to offer cutting-edge solutions.

Investor Perspective:
Investors should weigh LidoDAO's potential and its role in the DeFi sector against the backdrop of negative profitability and high leverage. The organization's substantial asset base and growth indicators may appeal to investors with a long-term perspective and tolerance for current financial volatility.

Management Perspective:
For LidoDAO's management, the focus should be on financial restructuring, strategic investment in growth areas, and leveraging its unique DAO governance for operational and strategic agility. This approach is critical to navigating the current financial complexities and solidifying LidoDAO's market position.

Conclusion:
LidoDAO's potential in the DeFi space is significant, with a strong asset foundation and an active community governance model. By addressing the highlighted financial and operational challenges, both investors and management can work towards ensuring LidoDAO's sustainable growth and continued innovation in the liquid staking protocol market.

""")



balancesheet_data = {
    'Assets': assets.iloc[0],
    'Liabilities': abs(liabilities.iloc[0]),
    'Equity': abs(equity.iloc[0])
}

balancesheet = pd.DataFrame.from_dict(balancesheet_data, orient='index', columns=['Amount'])
balancesheet.index.name = 'Item'


col1, col2 = st.columns(2)

with col1:
    st.subheader('TTM Consolidated Income Statement')
    st.write(consolidated_income_statement)

with col2:
    st.subheader('Live Balance Sheet')
    st.dataframe(balancesheet)

st.subheader('Liquidity Ratios')
st.write('Current Ratio:', current_ratio.iloc[0])
#st.write('Cash Ratio:', cash_ratio)

st.subheader('Leverage Ratios')
st.write('Debt Ratio:', debt_ratio.iloc[0])
st.write('Debt to Equity Ratio:', debt_to_equity.iloc[0])
             
st.subheader('Profitability Ratios')

st.write('Net Profit Margin:', net_profit_margin.iloc[0])
st.write('Return on Assets:', roa.iloc[0])
st.write('Return on Equity:', roe.iloc[0])

st.subheader('Market Value Ratios')

st.write('Earnings per Share:', eps)
st.write('Price to Earnings:', price_to_earnings)
st.write('Market to Book:', market_to_book.iloc[0])
#st.write('Price to Sales:', price_to_sales)
#st.write('Enterprise Value:', enterprise_value)
#st.write('EV Multiple:', ev_multiple)

st.subheader('Financial Metrics')

st.write('Beta:', beta)
st.write('WACC:', wacc.iloc[0] )
st.write('Cost of Debt:', cost_of_debt)
st.write('Cost of Equity:', cost_equity)
st.write(f"CAGR: {lido_cagr:.2%}")
st.write(f"Average Excess Return: {ldo_avg_excess_return:.2%}")
