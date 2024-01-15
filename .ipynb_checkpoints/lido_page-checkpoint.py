import streamlit as st
from Lido import * 


def show_lidopage():
    
   
    
    st.title('LidoDAO (LDO)')
    
    st.write(f"${ldo_current_price:,.2f}")
    st.line_chart(ldo_history['price'])
    
    latest_health_score = metrics_standard_scaled['financial_health_category'].iloc[1]
    
    color_map = {'bad': 'red', 'okay': 'yellow', 'good': 'green'}
    score_color = color_map.get(latest_health_score, 'black')
    st.markdown(f'<h3 style="color: white;">Financial Health: <span style="color: {score_color};">{latest_health_score.capitalize()}</span></h3>', unsafe_allow_html=True)
    # Get the latest financial health category
    
    def generate_dynamic_summary():
    
        # Constructing the summary
        summary = (
            f"LidoDAO (LDO) is currently priced at ${ldo_current_price}. "
            f"The financial health is rated as '{latest_health_score.capitalize()}' with a net profit margin of {net_profit_margin.iloc[0]:.2%}. "
            f"The current ratio of {current_ratio.iloc[0]:.2f} and a debt to equity ratio of {debt_to_equity.iloc[0]:.2f} reflect its financial stability."
        )
        return summary
    
    
    with st.container():
        st.write(""" ### Executive Summary
        """)
        st.write(generate_dynamic_summary())
    
    with st.expander("Financial Health and Analysis"):
        st.write(f"""
        **Financial Overview:**  
        LidoDAO, with its governance token LDO priced at \${ldo_current_price:,.2f}, operates within the decentralized finance (DeFi) ecosystem, managing liquid staking protocols. Despite a challenging period with a net loss of \${ttm_metrics['($) >>Total Protocol Income'].iloc[0]:,.2f}, the organization maintains a strong asset base. The financial health of LidoDAO is currently stable, with assets exceeding liabilities and an equity of \${abs(equity.iloc[0]):,.2f}.
    
        **Market Position and Business Operations:**
    
        LidoDAO exhibits a balanced liquidity stance, but this is overshadowed by negative returns and a high debt to equity ratio of {debt_ratio.iloc[0]:.2f}. Such financial leverage points to potential risks requiring astute management. Nonetheless, market sentiment remains optimistic about LidoDAO's prospects, as reflected in a solid market to book ratio of {market_to_book.iloc[0]:.2f} and an impressive average excess return of {ldo_avg_excess_return:.2%}.
        """)
    
    with st.expander("Investor Outlook"):
        st.markdown("""
        Investors considering LidoDAO should assess its potential role in the DeFi ecosystem against the backdrop of the current negative profitability and high leverage. The DAO's considerable asset base and positive growth indicators may attract those with a long-term outlook and a tolerance for the existing financial fluctuations.
        """)
    
    with st.expander("Management Outlook"):
        st.markdown("""  
        For LidoDAO's leadership, the priority lies in fiscal reformation, strategic growth investments, and leveraging its distinctive DAO governance to enhance operational flexibility and strategy. This tactical approach is pivotal for navigating the existing financial intricacies and reinforcing LidoDAO's market presence.
        """)
    
    balancesheet_data = {
        'Assets': assets.iloc[0],
        'Liabilities': abs(liabilities.iloc[0]),
        'Equity': abs(equity.iloc[0])
    }
    
    balancesheet = pd.DataFrame.from_dict(balancesheet_data, orient='index', columns=['Amount'])
    #balancesheet.index.name = 'Item'
    
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader('TTM Consolidated Income Statement')
        st.table(consolidated_income_statement)
    
    with col2:
        st.subheader('Live Balance Sheet')
        st.table(balancesheet.style.format({"Amount": "${:,.2f}"}))
    
    col3, col4 = st.columns(2)
    
    cash_percentage = cash_ratio * 100
    
    liquidity_df = pd.DataFrame({
    'Metric': ['Current Ratio', 'Cash Ratio'],
    'Value': [f"{current_ratio.iloc[0]:.2f}", f"{cash_ratio.iloc[0]:.4f}"]
    })

    leverage_df = pd.DataFrame({
        'Metric': ['Debt Ratio', 'Debt to Equity Ratio'],
        'Value': [f"{debt_ratio.iloc[0]:.2f}", f"{debt_to_equity.iloc[0]:.2f}"]
    })
    
    profitability_df = pd.DataFrame({
        'Metric': ['Net Profit Margin', 'Return on Assets', 'Return on Equity'],
        'Value': [f"{net_profit_margin.iloc[0]:.2%}", f"{roa.iloc[0]:.2%}", f"{roe.iloc[0]:.2%}"]
    })
    
    market_value_df = pd.DataFrame({
        'Metric': ['Earnings per Share', 'Price to Earnings', 'Market to Book', 'Enterprise Value', 'EV to Revenue'],
        'Value': [f"{eps:.2f}", f"{price_to_earnings:.2f}", f"{market_to_book.iloc[0]:.2f}", f"${enterprise_value:,.2f}", f"{ev_to_rev.iloc[0]:.2f}"]
    })
    
    financial_metrics_df = pd.DataFrame({
        'Metric': ['Beta', 'Cost of Debt', 'Cost of Equity', 'WACC', 'CAGR', 'Average Excess Return'],
        'Value': [f"{beta:.2f}", f"{cost_of_debt:.2%}", f"{cost_equity:.2%}", f"{wacc.iloc[0]:.2%}", f"{lido_cagr:.2%}", f"{ldo_avg_excess_return:.2%}"]
    })
    
    # Displaying tables in Streamlit
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader('Liquidity Ratios')
        st.table(liquidity_df.set_index('Metric'))
    
    with col4:
        st.subheader('Leverage Ratios')
        st.table(leverage_df.set_index('Metric'))
    
    col5, col6 = st.columns(2)
    
    with col5:
        st.subheader('Profitability Ratios')
        st.table(profitability_df.set_index('Metric'))
    
    with col6:
        st.subheader('Market Value Ratios')
        st.table(market_value_df.set_index('Metric'))
    
    st.subheader('Financial Metrics')
    
    col7, col8 = st.columns(2)
    
    with col7:
        st.table(financial_metrics_df.iloc[:3].set_index('Metric'))
    
    with col8:
        st.table(financial_metrics_df.iloc[3:].set_index('Metric'))
    
        
    
    st.markdown("""
    ---
    Data and insights sourced from [Steakhouse's Lido SAFU dashboard on Dune Analytics](https://dune.com/steakhouse/lido-safu).
    """, unsafe_allow_html=True)