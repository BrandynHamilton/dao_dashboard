import streamlit as st
import pandas as pd
from lido_page import ldo_market_value, lido_cagr, current_ratio as ldo_current_ratio,  enterprise_value as lido_ev, ldo_avg_excess_return, ev_to_rev as lido_ev_rev, ldo_dpi_long_wacc, ldo_dpi_short_wacc, ldo_eth_long_wacc, ldo_eth_short_wacc
from maker_page import market_value as mkr_market_value, mkr_cagr, current_ratio as mkr_current_ratio, enterprise_value as mkr_ev, mkr_avg_excess_return, ev_to_rev as mkr_ev_rev, mkr_dpi_long_wacc, mkr_dpi_short_wacc, mkr_eth_long_wacc, mkr_eth_short_wacc
from data.makerdao import wacc as mkr_wacc, market_to_book as mkr_mtb
from data.Lido import wacc as lido_wacc, market_to_book as ldo_mtb
from data.rocketpool import market_to_book as rpl_mtb
from rocketpool_page import rpl_market_value, rpl_cagr, current_ratio as rpl_current_ratio, ev_df, rpl_avg_excess_return, ttm_ev_rev as rpl_ev_rev, wacc as rpl_wacc, rpl_dpi_long_wacc, rpl_dpi_short_wacc, rpl_eth_long_wacc, rpl_eth_short_wacc
import plotly.express as px



rpl_ev = ev_df['ev_historical'].iloc[-1]

data = {
    "DAO": ["MakerDAO", "LidoDAO", "Rocketpool"],
    "Market Cap": [mkr_market_value, ldo_market_value, rpl_market_value],
    "CAGR": [mkr_cagr, lido_cagr, rpl_cagr],
    "Current Ratio": [mkr_current_ratio.iloc[-1], ldo_current_ratio.iloc[0], rpl_current_ratio.iloc[-1]],
    "Market to Book": [mkr_mtb, ldo_mtb, rpl_mtb],
    "Short DPI WACC": [mkr_wacc, lido_wacc, rpl_wacc[0]],
    "Long DPI WACC": [mkr_dpi_long_wacc, ldo_dpi_long_wacc, rpl_dpi_long_wacc],
    "Short ETH WACC": [mkr_eth_short_wacc, ldo_eth_short_wacc, rpl_eth_short_wacc],
    "Long ETH WACC": [mkr_eth_long_wacc, ldo_eth_long_wacc, rpl_eth_long_wacc],
    "Enterprise Value": [mkr_ev, lido_ev, rpl_ev],
    "Excess Returns": [mkr_avg_excess_return, ldo_avg_excess_return, rpl_avg_excess_return],
    "EV/R Ratio": [mkr_ev_rev, lido_ev_rev.iloc[0], rpl_ev_rev]
}


dao_list = pd.DataFrame(data)

#dao_list.set_index('DAO', inplace=True)

def calculate_total_market_cap(df):
    # Convert market cap values to numeric and sum them
    df['Market Cap'] = df['Market Cap'].replace('[\$,]', '', regex=True).astype(float)
    return df['Market Cap'].sum()

def calculate_weighted_average_cagr(df):
    total_market_cap = df['Market Cap'].sum()
    df['Weighted CAGR'] = (df['Market Cap'] / total_market_cap) * df['CAGR']
    return df['Weighted CAGR'].sum()

def calculate_weighted_average_excess(df):
    total_market_cap = df['Market Cap'].sum()
    df['Weighted Excess Returns'] = (df['Market Cap'] / total_market_cap) * df['Excess Returns']
    return df['Weighted Excess Returns'].sum()
    
def calculate_aggregate_liquidity_ratio(df):
    # Weighted average based on market cap
    total_market_cap = df['Market Cap'].sum()
    df['Weighted Liquidity'] = (df['Market Cap'] / total_market_cap) * df['Current Ratio']
    return df['Weighted Liquidity'].sum()

def calculate_short_ETH_weighted_average_wacc(df):
    # Ensure Market Cap and WACC are numeric
    df['Market Cap'] = pd.to_numeric(df['Market Cap'], errors='coerce')
    df['Short ETH WACC'] = pd.to_numeric(df['Short ETH WACC'], errors='coerce')

    # Calculate total market cap
    total_market_cap = df['Market Cap'].sum()

    # Calculate weighted WACC
    df['Weighted WACC'] = (df['Market Cap'] / total_market_cap) * df['Short ETH WACC']
    return df['Weighted WACC'].sum()

def calculate_long_ETH_weighted_average_wacc(df):
    # Ensure Market Cap and WACC are numeric
    df['Market Cap'] = pd.to_numeric(df['Market Cap'], errors='coerce')
    df['Long ETH WACC'] = pd.to_numeric(df['Long ETH WACC'], errors='coerce')

    # Calculate total market cap
    total_market_cap = df['Market Cap'].sum()

    # Calculate weighted WACC
    df['Weighted WACC'] = (df['Market Cap'] / total_market_cap) * df['Long ETH WACC']
    return df['Weighted WACC'].sum()

def calculate_short_dpi_weighted_average_wacc(df):
    # Ensure Market Cap and WACC are numeric
    df['Market Cap'] = pd.to_numeric(df['Market Cap'], errors='coerce')
    df['Short DPI WACC'] = pd.to_numeric(df['Short DPI WACC'], errors='coerce')

    # Calculate total market cap
    total_market_cap = df['Market Cap'].sum()

    # Calculate weighted WACC
    df['Weighted WACC'] = (df['Market Cap'] / total_market_cap) * df['Short DPI WACC']
    return df['Weighted WACC'].sum()

def calculate_long_dpi_weighted_average_wacc(df):
    # Ensure Market Cap and WACC are numeric
    df['Market Cap'] = pd.to_numeric(df['Market Cap'], errors='coerce')
    df['Long DPI WACC'] = pd.to_numeric(df['Long DPI WACC'], errors='coerce')

    # Calculate total market cap
    total_market_cap = df['Market Cap'].sum()

    # Calculate weighted WACC
    df['Weighted WACC'] = (df['Market Cap'] / total_market_cap) * df['Long DPI WACC']
    return df['Weighted WACC'].sum()


def get_enterprise_values(df):
    return df[['DAO', 'Enterprise Value']]

def get_market_cap_evr_data(df):
    # Returns a DataFrame suitable for plotting a bubble chart
    return df[['DAO', 'Market Cap', 'EV/R Ratio']]

def calculate_weighted_average_market_to_book(df):
    # Convert Market Cap and M/Book Ratio to numeric
    df['Market Cap'] = pd.to_numeric(df['Market Cap'], errors='coerce')
    df['Market to Book'] = pd.to_numeric(df['Market to Book'], errors='coerce')

    # Calculate total market cap
    total_market_cap = df['Market Cap'].sum()

    # Calculate weighted M/Book Ratio
    df['Weighted M/Book'] = (df['Market Cap'] / total_market_cap) * df['Market to Book']
    return df['Weighted M/Book'].sum()


import matplotlib.pyplot as plt

def plot_enterprise_values(df):
    df['Enterprise Value'] = df['Enterprise Value'].apply(lambda x: x / 1e6)  # Convert to millions
    fig = px.bar(df, x='DAO', y='Enterprise Value', text='Enterprise Value')
    fig.update_traces(texttemplate='%{text:.2s}M', textposition='outside')
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
    fig.update_yaxes(tickprefix="$", ticksuffix="M")
    fig.update_layout(title_text='Enterprise Values of DAOs')
    return fig

import plotly.express as px

def create_bubble_chart(df):
    color_discrete_map = {
        'MakerDAO': 'red',  # Red color for MakerDAO
        'LidoDAO': 'blue',  # Blue color for LidoDAO
        'Rocketpool': 'yellow'  # Yellow color for Rocketpool
    }
    
    # Create the bubble chart with the color mapping
    fig = px.scatter(df, x='DAO', y='EV/R Ratio', size='Market Cap', color='DAO',
                     color_discrete_map=color_discrete_map)
    
    # You can customize your figure further if needed
    return fig


def show_homepage():

    
    
    
    st.title('DAO Dashboard: DeFi Investment & Analysis Hub')
    st.markdown("""
    Welcome to DAO Dashboard, your platform for in-depth financial analysis and market insights into Decentralized Autonomous Organizations (DAOs). Navigate through the intricate financial landscapes of MakerDAO, LidoDAO, and other leading DAOs with ease and precision via the sidebar.
    """)

    benchmark_options = ['DPI', 'ETH']
    benchmark_selection = st.radio("Select the Benchmark:", benchmark_options, index=0)

    if benchmark_selection == 'ETH':
        weighted_avg_short_wacc = calculate_short_ETH_weighted_average_wacc(dao_list)
        weighted_avg_long_wacc = calculate_long_ETH_weighted_average_wacc(dao_list)
    elif benchmark_selection == 'DPI':
        weighted_avg_short_wacc = calculate_short_dpi_weighted_average_wacc(dao_list)
        weighted_avg_long_wacc = calculate_long_dpi_weighted_average_wacc(dao_list)

    
    
    # Aggregate Statistics
    st.markdown("## Aggregate Statistics")
     #Assuming you have a function to calculate these metrics
    #def benchmark_calc(dao_list)
    weighted_avg_market_to_book = calculate_weighted_average_market_to_book(dao_list)
    total_market_cap = calculate_total_market_cap(dao_list)
    average_cagr = calculate_weighted_average_cagr(dao_list)
    aggregate_current_ratio = calculate_aggregate_liquidity_ratio(dao_list)
    weighted_avg_short_wacc = weighted_avg_short_wacc
    weighted_avg_long_wacc = weighted_avg_long_wacc
    average_excess = calculate_weighted_average_excess(dao_list)
    
        #st.metric("Total Market Cap", f"${total_market_cap:,.2f}")
    col1, col2, col3 = st.columns(3)
        
    with col1:
        st.metric("Average Excess Return", f"{average_excess:.2%}")
        st.metric("Average CAGR", f"{average_cagr:.2%}")
    with col2:
        st.metric("Aggregate Current Ratio", f"{aggregate_current_ratio:.2f}")
        st.metric('Weighted Avg. Market to Book', f"{weighted_avg_market_to_book:.2f}")
    with col3:
        st.metric("Weighted Avg. Short WACC", f"{weighted_avg_short_wacc:.2%}")
        st.metric("Weighted Avg. Long WACC", f"{weighted_avg_long_wacc:.2%}")
        
    #for debugging:
    #st.table(dao_list)
    # Benchmarking Visuals
    st.markdown("""
    ## Benchmarking Visuals
    Compare and contrast individual DAOs against each other and industry benchmarks for a nuanced investment perspective.
    """)

    # Bar Chart for Enterprise Values
    st.subheader('Enterprise Values Comparison')
    enterprise_values = get_enterprise_values(dao_list)
    st.plotly_chart(plot_enterprise_values(enterprise_values), use_container_width=True)
# you'll need to implement this
    

    # Line Graph for Average Excess Return
    #st.subheader('Average Excess Return Over Time')
    #excess_returns = get_excess_returns_over_time(dao_list)  # you'll need to implement this
    #st.line_chart(excess_returns)

    # Bubble Chart for Market Cap vs EV/R Ratio
    st.subheader('Market Cap vs EV/R Ratio')
    market_cap_evr_data = get_market_cap_evr_data(dao_list)
    bubble_chart = create_bubble_chart(market_cap_evr_data)
    st.plotly_chart(bubble_chart)