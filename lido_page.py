import streamlit as st
from data.Lido import * 
from data.rocketpool import eth_history
from data.formulas import *
from data.Lido import wacc as short_dpi_wacc, d as ldo_liabilities
from data.makerdao import cumulative_risk_premium

dpi_cumulative_risk_premium = cumulative_risk_premium


ldo_mk = e
ldo_liabilities = ldo_liabilities.iloc[0]
ldo_rd = rd
dpi_market_premium = average_yearly_risk_premium


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

data_df = merged_df.merge(eth_history['daily_returns_eth'], left_index=True, right_index=True)

x_eth = data_df['daily_returns_eth'].values.reshape(-1, 1)
x_dpi = data_df['daily_returns_dpi'].values.reshape(-1, 1)
y = data_df['daily_returns_ldo'].values

eth_ldo_beta = calculate_beta(x_eth,y)
dpi_ldo_beta = calculate_beta(x_dpi,y)


eth_yearly_risk_premium = eth_annual_returns.to_frame('annual_return').merge(tbilldf_yearly, left_index=True, right_index=True )
eth_yearly_risk_premium.drop(columns = ['decimal'], inplace=True)

eth_yearly_risk_premium = eth_yearly_risk_premium['annual_return'] - eth_yearly_risk_premium['value']

eth_market_premium = eth_yearly_risk_premium.mean()



eth_ldo_re = calculate_rd(current_risk_free, eth_ldo_beta, eth_market_premium)

eth_ldo_wacc = calculate_wacc(ldo_mk, ldo_liabilities, eth_ldo_re, ldo_rd)



eth_cagr = calculate_historical_returns(eth_history)

eth_cumulative_risk_premium = eth_cagr - current_risk_free

#dpi_cumulative_risk_premium = cumulative_risk_premium



eth_ldo_long_re = calculate_rd(current_risk_free, eth_ldo_beta, eth_cumulative_risk_premium)

eth_ldo_long_wacc = calculate_wacc(ldo_mk, ldo_liabilities, eth_ldo_long_re, ldo_rd)



def show_lidopage():

    st.title('LidoDAO (LDO)')
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
        selected_beta = eth_ldo_beta
        market_premium = eth_market_premium if time_frame_selection == 'Short Term' else eth_cumulative_risk_premium
        re = calculate_rd(current_risk_free, selected_beta, market_premium)
    elif benchmark_selection == 'DPI':
        selected_beta = dpi_ldo_beta
        market_premium = dpi_market_premium if time_frame_selection == 'Short Term' else dpi_cumulative_risk_premium
        re = calculate_rd(current_risk_free, selected_beta, market_premium)

    selected_wacc = calculate_wacc(ldo_mk, ldo_liabilities, re, ldo_rd)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Selected WACC", f"{selected_wacc:.3%}")
    with col2:
        st.metric("Selected Beta", f"{selected_beta:.2f}")
    with col3:
        st.metric("Selected Cost of Equity", f"{re:.2%}")
    
    
   
    
    
    
    st.metric("Price", f"${ldo_current_price:,.2f}")
    st.line_chart(ldo_history['price'])
    
    latest_health_score = metrics_standard_scaled['financial_health_category'].iloc[1]
    
    color_map = {'bad': 'red', 'okay': 'yellow', 'good': 'green'}
    score_color = color_map.get(latest_health_score, 'black')
    st.markdown(f'<h3 style="color: white;">Financial Health: <span style="color: {score_color};">{latest_health_score.capitalize()}</span></h3>', unsafe_allow_html=True)
    
    
    def generate_dynamic_summary():
    
        
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
    'Value': [f"{current_ratio.iloc[0]:.3f}", f"{cash_ratio.iloc[0]:.4f}"]
    })

    leverage_df = pd.DataFrame({
        'Metric': ['Debt Ratio', 'Debt to Equity Ratio'],
        'Value': [f"{debt_ratio.iloc[0]:.3f}", f"{debt_to_equity.iloc[0]:.2f}"]
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
        'Value': [f"{selected_beta:.2f}", f"{cost_of_debt:.2%}", f"{re:.2%}", f"{selected_wacc:.3%}", f"{lido_cagr:.2%}", f"{ldo_avg_excess_return:.2%}"]
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
        st.subheader('Market Value Metrics')
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