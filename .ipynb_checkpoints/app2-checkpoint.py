import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import requests
import seaborn as sns
import yfinance as yf
from makerdao import *
from makerdao import beta as makerdao_beta
from Lido import *
from Lido import beta as lidodao_beta

# Define a function for the home page
def home_page():
    st.title("DAO Dashboard")

    def create_interactive_sml(risk_free_rate, market_risk_premium, makerdao_beta, lidodao_beta, makerdao_return, lidodao_return, term):
        betas = np.linspace(0, 1.5, 100)
        expected_returns = risk_free_rate + betas * market_risk_premium
        
        # Create the SML line
        sml_line = go.Scatter(x=betas, y=expected_returns, mode='lines', name=f'{term} SML')
        
        # Add MakerDAO token as points for expected (based on average risk premium) and actual returns (CAGR or avg excess return)
        makerdao_expected = go.Scatter(
        x=[makerdao_beta],
        y=[risk_free_rate + makerdao_beta * average_yearly_risk_premium] if term == 'Short-term' else [risk_free_rate + makerdao_beta * cumulative_risk_premium],
        mode='markers', 
        marker=dict(color='red'),
        name=f'MakerDAO {term} Expected'
        )
        
        makerdao_actual = go.Scatter(
        x=[makerdao_beta],
        y=[makerdao_return],
        mode='markers', 
        marker=dict(color='pink'),
        name=f'MakerDAO {term} Actual'
        )
        
        # Add LidoDAO token as points for expected (based on average risk premium) and actual returns (CAGR or avg excess return)
        lidodao_expected = go.Scatter(
        x=[lidodao_beta],
        y=[risk_free_rate + lidodao_beta * average_yearly_risk_premium] if term == 'Short-term' else [risk_free_rate + lidodao_beta * cumulative_risk_premium],
        mode='markers', 
        marker=dict(color='blue'),
        name=f'LidoDAO {term} Expected'
        )
        
        lidodao_actual = go.Scatter(
        x=[lidodao_beta],
        y=[lidodao_return],
        mode='markers', 
        marker=dict(color='lightblue'),
        name=f'LidoDAO {term} Actual'
        )
        
        # Add Risk-Free Rate line
        risk_free_line = go.Scatter(
        x=[0, max(betas)], 
        y=[risk_free_rate, risk_free_rate], 
        mode='lines', 
        line=dict(dash='dash', color='green'),
        name='Risk-Free Rate'
        )
        
        # Layout settings
        layout = go.Layout(
        title=f'Security Market Line (SML) - {term}',
        xaxis=dict(title='Beta (Systematic Risk)'),
        yaxis=dict(title='Return'),
        showlegend=True
        )
        
    # Combine all the plots
        fig = go.Figure(data=[sml_line, makerdao_expected, makerdao_actual, lidodao_expected, lidodao_actual, risk_free_line], layout=layout)
        return fig
    
    # Dropdown selection for time frame
    time_frame = st.selectbox(
    "Select Time Frame - Short-term is <12 mo, Long-term is >12",
    ("Short-term", "Long-term")
    )
    
    # Determine which returns to use based on selected time frame
    if time_frame == "Short-term":
        makerdao_return = mkr_avg_excess_return
        lidodao_return = ldo_avg_excess_return
    else:  # Long-term
        makerdao_return = mkr_cagr
        lidodao_return = lido_cagr
    
    # Create and display the plot with the selected time frame
    fig = create_interactive_sml(risk_free_rate, average_yearly_risk_premium if time_frame == "Short-term" else cumulative_risk_premium, makerdao_beta, lidodao_beta, makerdao_return, lidodao_return, time_frame)
    st.plotly_chart(fig)
    
    percent_risk = average_yearly_risk_premium * 100
    st.subheader('Benchmark: DefiPulse Index (DPI)')
    
    st.line_chart(dpi_history['price'])
    st.write(f"DPI CAGR is {cagr_percentage:.2f}%")
    st.write(f"DPI Average Excess Return is {percent_risk:.2f}%")
    
    
    # Define a function for DAO 1 Analysis page
def dao1_page():
    st.title('MakerDAO (MKR)')
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
        # Content specific to DAO 1

# Define a function for DAO 2 Analysis page
def dao2_page():
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

    

# Initialize session state if not already done
if 'page' not in st.session_state:
    st.session_state.page = "home_page"

# Layout the navigation
nav1, nav2, nav3 = st.columns(3)
with nav1:
    if st.button("Home"):
        st.session_state.page = "home_page"
with nav2:
    if st.button("DAO 1 Analysis"):
        st.session_state.page = "dao1"
with nav3:
    if st.button("DAO 2 Analysis"):
        st.session_state.page = "dao2"

# Call the function based on the session state
if st.session_state.page == "home_page":
    home_page()
elif st.session_state.page == "dao1":
    dao1_page()
elif st.session_state.page == "dao2":
    dao2_page()
