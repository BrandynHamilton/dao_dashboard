import streamlit as st
import pandas as pd
from data.makerdao import aligned_data, short_re as dpi_mkr_re, e as mkr_mk, d as mkr_liabilities, rd as mkr_rd, current_risk_free, average_yearly_risk_premium as dpi_market_premium, tbilldf, long_wacc as dpi_long_wacc, balance_sheet_time as mkr_bs, quarterly_df as mkr_quarterly_df, wacc, dividend_ps as mkr_income
from data.rocketpool import eth_history 
from data.formulas import *
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from data.Lido import quarterly_stats as ldo_income



def dividend_widget():
   
    st.title('Dividend Widget')

    dao_options = ['MKR', 'LDO']
    dao_selection = st.radio("Select the DAO:", dao_options, index=0)

    if dao_selection == 'MKR':
        quarterly_df = mkr_income
    elif dao_selection == 'LDO':
        quarterly_df = ldo_income
        

    
    def dividend_model(df, percentage_to_distribute):
        df2 = df.copy()
        df2['dividend'] = df2['net income'] * percentage_to_distribute
        df2['dividend_per_share'] = df2['dividend'] / df2['supply']
        df2['dividend'] = df2['dividend'].apply(lambda x: 0 if x <= 0 else x)
        df2['dividend_per_share'] = df2['dividend_per_share'].apply(lambda x: 0 if x <= 0 else x)
        return df2[['net income','dividend','dividend_per_share']]
    
    
    
    percentage_to_distribute = st.slider(
        'Select the Percentage to Distribute',
        min_value=0.0, 
        max_value=1.0, 
        value=0.90,  # Default value
        step=0.01,
        format="%.2f"
    )
    
    # Display the current slider value
    st.write(f"Percentage to Distribute: {percentage_to_distribute:.2f}")
    
    # Step 2: Update data based on slider input
    dividend_data = dividend_model(quarterly_df, percentage_to_distribute)
    
    
    
    
    # Step 3: Create a figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add Dividend bars to the figure - primary y-axis
    fig.add_trace(
        go.Bar(x=dividend_data.index, y=dividend_data['dividend'], name='Dividends'),
        secondary_y=False,
    )
    
    # Add Dividend per Share line to the figure - secondary y-axis
    fig.add_trace(
        go.Scatter(x=dividend_data.index, y=dividend_data['dividend_per_share'], mode='lines', name='Dividend per Share'),
        secondary_y=True,
    )
    
    # Update the layout
    fig.update_layout(
        title='Dividends and Dividend per Share Over Time',
        xaxis_title='Time',
        yaxis_title='Dividends',
        yaxis2_title='Dividend per Share',
        legend_title='Metrics',
        barmode='group'
    )
    
    st.plotly_chart(fig)
    st.write("Time Series")
    st.dataframe(dividend_data.iloc[::-1], use_container_width=True)
    
