import streamlit as st
import pandas as pd
import numpy as np

from Lido import * 

from sidebar import create_sidebar

create_sidebar()

st.title('LidoDAO (LDO)')

st.write(f"${ldo_current_price:,.2f}")
st.line_chart(ldo_history['price'])

latest_health_score = metrics_standard_scaled['financial_health_category'].iloc[1]

color_map = {'bad': 'red', 'okay': 'yellow', 'good': 'green'}
score_color = color_map.get(latest_health_score, 'black')
st.markdown(f'<h3 style="color: white;">Financial Health: <span style="color: {score_color};">{latest_health_score.capitalize()}</span></h3>', unsafe_allow_html=True)
# Get the latest financial health category

import streamlit as st

with st.container():
    st.markdown("""  
LidoDAO, with its governance token LDO priced at \$2.77, operates within the decentralized finance (DeFi) ecosystem, managing liquid staking protocols. Despite a challenging period with a net loss of \$40,586,965.90, the organization maintains a strong asset base. The financial health of LidoDAO is currently stable, with assets exceeding liabilities and an equity of \$91,616,907.0751.
""")


with st.expander("Financial Health and Analysis"):
    st.markdown("""  
    LidoDAO exhibits a balanced liquidity stance, but this is overshadowed by negative returns and a high debt to equity ratio of 229.89. Such financial leverage points to potential risks requiring astute management. Nonetheless, market sentiment remains optimistic about LidoDAO's prospects, as reflected in a solid market to book ratio of 26.91 and an impressive average excess return of 61.87%.
    """)

with st.expander("Operational Insights and Strategic Recommendations"):
    st.markdown("""  
    With earnings per share at $84.05 and a market to book ratio of 27.17, investor confidence is palpable. The company's CAGR of 42.64% and an average excess return of 111.15% underscore its robust market standing. The SML analysis, in light of MakerDAO's beta of 0.78, reveals a lower systemic risk relative to the market, indicating resilience against market volatility.
    """)

with st.expander("Operational Insights and Strategic Recommendations"):
    st.markdown("""  
    The diligent management of LidoDAO's liquid staking protocols is noteworthy. However, recent financial shortfalls signal a need for tactical readjustment. Cost reduction strategies are crucial for converting the net loss to profit while maintaining operational integrity. Revenue growth can be spurred through innovative streams and revisiting service fees and staking mechanisms. Addressing the high leverage through debt restructuring and equity solutions is essential for the DAO's fiscal well-being. Fostering active participation in governance through LDO token holders could drive innovation and bolster decision-making processes.
    """)

with st.expander("Investor Recommendations"):
    st.markdown("""
    Investors considering LidoDAO should assess its potential role in the DeFi ecosystem against the backdrop of the current negative profitability and high leverage. The DAO's considerable asset base and positive growth indicators may attract those with a long-term outlook and a tolerance for the existing financial fluctuations.
    """)

with st.expander("Management Recommendations"):
    st.markdown("""  
    For LidoDAO's leadership, the priority lies in fiscal reformation, strategic growth investments, and leveraging its distinctive DAO governance to enhance operational flexibility and strategy. This tactical approach is pivotal for navigating the existing financial intricacies and reinforcing LidoDAO's market presence.
    """)

with st.expander("Conclusion"):
    st.markdown("""  
    LidoDAO's potential in the DeFi domain is noteworthy, bolstered by a solid asset foundation and an engaged community governance framework. Addressing the outlined financial and operational challenges is key for both investors and the management team, aiming for LidoDAO's enduring growth and innovation in the liquid staking protocol landscape.
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

col3, col4 = st.columns(2)

with col3:
    st.subheader('Liquidity Ratios')
    st.write(f"Current Ratio: {current_ratio.iloc[0]:.2f}")
with col4: 
    st.subheader('Leverage Ratios')
    st.write(f'Debt Ratio: {debt_ratio.iloc[0]:.2f}')
    st.write(f'Debt to Equity Ratio: {debt_to_equity.iloc[0]:.2f}')

col5, col6 = st.columns(2)

with col5:
    st.subheader('Profitability Ratios')
    
    st.write(f'Net Profit Margin: {net_profit_margin.iloc[0]:.2%}')
    st.write(f'Return on Assets: {roa.iloc[0]:.2%}')
    st.write(f'Return on Equity: {roe.iloc[0]:.2%}')

with col6:
    st.subheader('Market Value Ratios')
    
    st.write(f'Earnings per Share: {eps:.2f}')
    st.write(f'Price to Earnings: {price_to_earnings:.2f}')
    st.write(f'Market to Book: {market_to_book.iloc[0]:.2f}')
    st.write(f"Enterprise Value: {enterprise_value:,.2f}")
    st.write(f"Enterprise Value to Revenue: {ev_to_rev.iloc[0]:.2f}")

st.subheader('Financial Metrics')

col7, col8 = st.columns(2)

with col7:
    st.write(f'Beta: {beta:.2f}')
    st.write(f'Cost of Debt: {cost_of_debt:.2%}')
    st.write(f'Cost of Equity: {cost_equity:.2%}')

with col8:
    st.write(f'WACC: {wacc.iloc[0]:.2%}')
    st.write(f"CAGR: {lido_cagr:.2%}")
    st.write(f"Average Excess Return: {ldo_avg_excess_return:.2%}")
    

st.markdown("""
---
Data and insights sourced from [Steakhouse's Lido SAFU dashboard on Dune Analytics](https://dune.com/steakhouse/lido-safu).
""", unsafe_allow_html=True)

