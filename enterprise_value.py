import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from data.lidoev import filtered_ev_metrics as lido_metrics
from data.makerev import historical_ev as maker_metrics, ev_metrics as monthly_metrics
from data.rocketpool import beta_num as rpl_beta, rpl_cagr, rpl_avg_excess_return, ev_df, enterprise_value as rpl_enterprise_value

def enterprise_metrics():
    st.title("Enterprise Value (EV) Metrics")
    fig2 = go.Figure()

    # Add EV to Revenue (Filtered and Truncated) - primary y-axis
    fig2.add_trace(go.Scatter(
        x=monthly_metrics.index, 
        y=monthly_metrics['ev_to_rev_truncated'], 
        mode='lines', 
        name='MakerDao EV to Revenue', 
        line=dict(color='red')
    ))
    fig2.add_trace(go.Scatter(
        x=lido_metrics.index, 
        y=lido_metrics['ev_to_rev'].iloc[:-1], 
        mode='lines', 
        name='LidoDao EV to Revenue', 
        line=dict(color='blue')
    ))
    fig2.add_trace(go.Scatter(
        x=ev_df.index, 
        y=ev_df['ev_to_rev'], 
        mode='lines', 
        name='Rocketpool EV to Revenue', 
        line=dict(color='yellow')
    ))

    fig3 = go.Figure()

    # Add Historical EV (Filtered and Unfiltered) - secondary y-axis
    fig3.add_trace(go.Scatter(
        x=monthly_metrics.index, 
        y=monthly_metrics['historical_ev'], 
        mode='lines', 
        name='MakerDao EV', 
        line=dict(color='red')
    ))
    fig3.add_trace(go.Scatter(
        x=lido_metrics.index, 
        y=lido_metrics['historical_ev'].iloc[:-1], 
        mode='lines', 
        name='LidoDao EV', 
        line=dict(color='blue')
    ))
    fig3.add_trace(go.Scatter(
        x=ev_df.index, 
        y=ev_df['ev_historical'], 
        mode='lines', 
        name='Rocketpool EV', 
        line=dict(color='yellow')
    ))

    # Create a layout with a secondary y-axis
    fig2.update_layout(
        title='Monthly EV/Rev Over Time',
        xaxis_title='Date',
        yaxis_title='EV to Revenue',
    )

    fig3.update_layout(
        title='EV Over Time',
        xaxis_title='Date',
        yaxis_title='Enterprise Value',
    )

    
    with st.expander('Summary'):
        st.write("""Enterprise Value (EV) and Enterprise Value to Revenue (EV/R) ratios are key financial metrics that can offer insights into the valuation and performance of decentralized autonomous organizations (DAOs), similar to their use in evaluating traditional companies.""")
        st.write(""" ***Enterprise Value (EV):*** 
                This is a measure of a company's total value, often used as a more comprehensive alternative to market capitalization. In the context of DAOs, it can indicate the overall economic size and value of the DAO.
                    """)
        st.write(""" ***Enterprise Value to Revenue (EV/R):*** This ratio compares the company's enterprise value to its revenue.
                    It provides an indication of how the market values every dollar of the DAO's revenues.
                    A higher ratio might suggest that the market expects higher growth or has a higher valuation of the DAO's future potential.
                    For DAOs, this could be influenced by factors like the adoption rate of their platforms, the growth of their assets under management, or their income from protocol fees.""")

        st.write("""Historical analysis of these metrics can reveal trends and changes in the market's valuation of these organizations over time. For instance, a rising EV/R might indicate increasing market optimism about the DAO's future growth prospects. Conversely, a decreasing ratio might suggest a market reevaluation of the DAO's potential.""")

    st.plotly_chart(fig3, use_container_width=True)
    st.plotly_chart(fig2, use_container_width=True)

# Remember to call the function where needed
