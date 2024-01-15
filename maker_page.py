import streamlit as st
from makerdao import *

def show_makerpage():
        
        
    expenses = ttm_data['expenses']
    lending_income = ttm_data['lending_income']
    liquidation_income = ttm_data['liquidation_income']
    trading_income = ttm_data['trading_income']
    net_income = ttm_data['net_income']
    
    
    
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
    
    
    
    
    
    # Accessing the data by label
    
    
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
    
    liquidity_df = pd.DataFrame({
    'Metric': ['Current Ratio', 'Cash Ratio'],
    'Value': [f"{current_ratio.iloc[-1]:.2f}", f"{cash_ratio:.2f}"]
    })
    
    leverage_df = pd.DataFrame({
        'Metric': ['Debt Ratio', 'Debt to Equity Ratio'],
        'Value': [f"{debt_ratio:.2f}", f"{debt_to_equity:.2f}"]
    })
    
    profitability_df = pd.DataFrame({
        'Metric': ['Net Profit Margin', 'Return on Assets', 'Return on Equity'],
        'Value': [f"{net_profit_margin:.2%}", f"{ROA:.2%}", f"{ROE:.2%}"]
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
        'Value': [f"{beta:.2f}", f"{wacc:.2%}", f"{short_re:.2%}", f"{rd:.2%}*", f"{mkr_cagr:.2%}", f"{mkr_avg_excess_return:.2%}"]
    })
    
    # Displaying tables in Streamlit
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader('Liquidity Ratios')
        st.table(liquidity_df.set_index('Metric'))
    
        st.subheader('Leverage Ratios')
        st.table(leverage_df.set_index('Metric'))
    
    with col4:
        st.subheader('Profitability Ratios')
        st.table(profitability_df.set_index('Metric'))
    
    st.subheader('Market Value Ratios')
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