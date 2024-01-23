import streamlit as st
import pandas as pd
from lido_page import ldo_market_value, lido_cagr, current_ratio as ldo_current_ratio,  enterprise_value as lido_ev, ldo_avg_excess_return, ev_to_rev as lido_ev_rev
from maker_page import market_value as mkr_market_value, mkr_cagr, current_ratio as mkr_current_ratio, enterprise_value as mkr_ev, mkr_avg_excess_return, ev_to_rev as mkr_ev_rev
from data.makerdao import wacc as mkr_wacc
from data.Lido import wacc as lido_wacc
from rocketpool_page import rpl_market_value, rpl_cagr, current_ratio as rpl_current_ratio, ev_df, rpl_avg_excess_return, ttm_ev_rev as rpl_ev_rev, wacc as rpl_wacc


rpl_ev = ev_df['ev_historical'].iloc[-1]

data = {
    "DAO": ["MakerDAO", "LidoDAO", "Rocketpool"],
    "Market Cap": [mkr_market_value, ldo_market_value, rpl_market_value],
    "CAGR": [mkr_cagr, lido_cagr, rpl_cagr],
    "Current Ratio": [mkr_current_ratio.iloc[-1], ldo_current_ratio.iloc[0], rpl_current_ratio.iloc[-1]],
    "WACC": [mkr_wacc, lido_wacc, rpl_wacc[0]],
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

def calculate_weighted_average_wacc(df):
    # Ensure Market Cap and WACC are numeric
    df['Market Cap'] = pd.to_numeric(df['Market Cap'], errors='coerce')
    df['WACC'] = pd.to_numeric(df['WACC'], errors='coerce')

    # Calculate total market cap
    total_market_cap = df['Market Cap'].sum()

    # Calculate weighted WACC
    df['Weighted WACC'] = (df['Market Cap'] / total_market_cap) * df['WACC']
    return df['Weighted WACC'].sum()


def get_enterprise_values(df):
    return df[['DAO', 'Enterprise Value']]

def get_market_cap_evr_data(df):
    # Returns a DataFrame suitable for plotting a bubble chart
    return df[['DAO', 'Market Cap', 'EV/R Ratio']]


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
    fig = px.scatter(df, x='DAO', y='EV/R Ratio', size='Market Cap', color='DAO')
    return fig


def show_homepage():

    st.table(dao_list)
    
    
    st.title('DAO Dashboard: DeFi Investment & Analysis Hub')
    st.markdown("""
    Welcome to DAO Dashboard, your platform for in-depth financial analysis and market insights into Decentralized Autonomous Organizations (DAOs). Navigate through the intricate financial landscapes of MakerDAO, LidoDAO, and other leading DAOs with ease and precision via the sidebar.
    """)
    
    # Aggregate Statistics
    st.markdown("## Aggregate Statistics")
     #Assuming you have a function to calculate these metrics
    total_market_cap = calculate_total_market_cap(dao_list)
    average_cagr = calculate_weighted_average_cagr(dao_list)
    aggregate_liquidity_ratio = calculate_aggregate_liquidity_ratio(dao_list)
    weighted_avg_wacc = calculate_weighted_average_wacc(dao_list)
    average_excess = calculate_weighted_average_excess(dao_list)
    
    #st.metric("Total Market Cap", f"${total_market_cap:,.2f}")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Average Excess Return", f"{average_excess:.2%}")
    with col2:
        st.metric("Average CAGR", f"{average_cagr:.2%}")
    with col3:
        st.metric("Aggregate Current Ratio", f"{aggregate_liquidity_ratio:.2f}")
    with col4:
        st.metric("Weighted Avg. WACC", f"{weighted_avg_wacc:.2%}")
    

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