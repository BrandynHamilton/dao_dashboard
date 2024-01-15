import streamlit as st
import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from makerdao import beta as makerdao_beta, average_yearly_risk_premium, current_risk_free as risk_free_rate, mkr_cagr, mkr_avg_excess_return, cumulative_risk_premium, cagr_percentage, dpi_history
from Lido import beta as lidodao_beta, lido_cagr, ldo_avg_excess_return
from rocketpool import beta as rpl_beta, rpl_cagr, rpl_avg_excess_return
from sidebar import create_sidebar
from lidoev import filtered_ev_metrics as lido_metrics
from makerev import ev_metrics as maker_metrics
from rocketpooldata import enterprise_value as rpl_enterprise_value, balance_sheet as rpl_bs, ev_df
from home_page import *
from maker_page import *
from lido_page import *
#from rocketpool_page import *

def show_homepage():
    st.title('DAO Dashboard')
    
    st.subheader('Security Market Line (SML)')
    st.write(""" 
    
    The graphical representation of the Security Market Line (SML), where returns on DAO equity tokens are analyzed using the DeFi Pulse Index as a benchmark, is a valuable tool for both investors and DAO financial managers.
    
    Security Market Line (SML) Representation:
    
    This graph plots expected returns against the risk (measured by beta) of DAO equity tokens. It allows for the comparison of actual and expected returns relative to the market risk.
    
    For Investors:
    
    Risk-Return Analysis: Investors can assess if DAO tokens are yielding adequate returns for their risk level. Points above the SML indicate above-market performance, while below-SML points suggest underperformance.
    
    Investment Decision-Making: The graph helps in identifying potentially undervalued or overvalued tokens, guiding investment choices based on individual risk preferences.
    
    For DAO Financial Managers:
    
    Performance Insight: The SML graph provides a clear visual of how the market values the DAO's tokens relative to their risk, offering insights for strategic financial planning and token management.
    
    Benchmarking and Strategy: Using the DeFi Pulse Index as a benchmark, DAO managers can compare their performance to the broader DeFi sector, which is crucial for competitive positioning and strategy formulation.
    Time Horizon Considerations:
    
    Short-Term Analysis (Average Excess Return and Average Yearly Risk Premium): Useful for understanding immediate market dynamics and for short-term financial planning.
    
    Long-Term Analysis (CAGR and Cumulative Risk Premium): Crucial for evaluating the long-term growth and sustainability of DAO tokens, aiding in strategic decisions like long-term investments or asset allocation.
    
    """)
    def create_interactive_sml(risk_free_rate, market_risk_premium, makerdao_beta, lidodao_beta, rpl_beta, makerdao_return, lidodao_return, rpl_return, term):
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
    
        # Convert rpl_beta from array to scalar if necessary
        rpl_beta_value = rpl_beta[0] if isinstance(rpl_beta, np.ndarray) else rpl_beta
    
        # Calculate expected return for Rocket Pool
        rpl_expected_return = risk_free_rate + rpl_beta_value * market_risk_premium
    
        # Add Rocket Pool token data points
        rpl_expected = go.Scatter(
            x=[rpl_beta_value],
            y=[rpl_expected_return],
            mode='markers',
            marker=dict(color='orange'),
            name='Rocket Pool Expected'
        )
        
        rpl_actual = go.Scatter(
            x=[rpl_beta_value],
            y=[rpl_return],
            mode='markers',
            marker=dict(color='yellow'),
            name='Rocket Pool Actual'
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
            title=f'SML - {term}',
            xaxis=dict(title='Beta (Systematic Risk)'),
            yaxis=dict(title='Return'),
            showlegend=True
        )
    
        # Combine all the plots
        fig = go.Figure(data=[sml_line, makerdao_expected, makerdao_actual, lidodao_expected, lidodao_actual, rpl_expected, rpl_actual, risk_free_line], layout=layout)
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
        rpl_return = rpl_avg_excess_return
    else:  # Long-term
        makerdao_return = mkr_cagr
        lidodao_return = lido_cagr
        rpl_return = rpl_cagr
    
    # Create and display the plot with the selected time frame
    fig = create_interactive_sml(risk_free_rate, average_yearly_risk_premium if time_frame == "Short-term" else cumulative_risk_premium, makerdao_beta, lidodao_beta, rpl_beta, makerdao_return, lidodao_return, rpl_return, time_frame)
    
    
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("")
        
    
    with col2:
        percent_risk = average_yearly_risk_premium * 100
        with st.expander("Benchmark: DefiPulse Index (DPI)"):
            st.line_chart(dpi_history['price'])
            st.write(f"DPI CAGR is {cagr_percentage:.2f}%")
            st.write(f"DPI Average Excess Return is {percent_risk:.2f}%")
    
    st.plotly_chart(fig)
    
    
    
    
    fig2 = go.Figure()

    # Add EV to Revenue (Filtered and Truncated) - primary y-axis
    
    fig2.add_trace(go.Scatter(x=maker_metrics.index, y=maker_metrics['ev_to_rev_truncated'], mode='lines', name='MakerDao EV to Revenue', line=dict(color='red')))
    fig2.add_trace(go.Scatter(x=lido_metrics.index, y=lido_metrics['ev_to_rev'], mode='lines', name='LidoDao EV to Revenue', line=dict(color='blue')))
    fig2.add_trace(go.Scatter(x=ev_df.index, y=ev_df['ev_to_rev'], mode='lines', name='Rocketpool EV to Revenue', line=dict(color='yellow')))
    
    fig3 = go.Figure()
    
    # Add Historical EV (Filtered and Unfiltered) - secondary y-axis
    fig3.add_trace(go.Scatter(x=maker_metrics.index, y=maker_metrics['historical_ev'], mode='lines', name='MakerDao EV', line=dict(color='red')))
    fig3.add_trace(go.Scatter(x=lido_metrics.index, y=lido_metrics['historical_ev'], mode='lines', name='LidoDao EV', line=dict(color='blue')))
    fig3.add_trace(go.Scatter(x=rpl_enterprise_value.index, y=rpl_enterprise_value['ev_historical'], mode='lines', name='Rocketpool EV', line=dict(color='yellow')))

    

        
        # Create a layout with a secondary y-axis
    fig2.update_layout(
        title='EV/Rev Over Time',
        xaxis_title='Date',
        yaxis_title='EV to Revenue',
        )
    
    
    fig3.update_layout(
        title='EV Over Time',
        xaxis_title='Date',
        yaxis_title='Enterprise Value',
        )
    
    with st.container():
        st.subheader('Enterprise Value (EV) Metrics')
        st.write("""Enterprise Value (EV) and Enterprise Value to Revenue (EV/R) ratios are key financial metrics that can offer insights into the valuation and performance of decentralized autonomous organizations (DAOs) like MakerDAO and LidoDAO, similar to their use in evaluating traditional companies.""")
        st.write(""" Enterprise Value (EV): This is a measure of a company's total value, often used as a more comprehensive alternative to market capitalization. In the context of DAOs:
            It can indicate the overall economic size and value of the DAO.
            For DAOs like MakerDAO and LidoDAO, which are involved in cryptocurrency and blockchain operations, EV might include the market value of their governance tokens, cash reserves, and other assets minus debts.
            Enterprise Value to Revenue (EV/R): This ratio compares the company's enterprise value to its revenue.
            It provides an indication of how the market values every dollar of the DAO's revenues.
            A higher ratio might suggest that the market expects higher growth or has a higher valuation of the DAO's future potential.
            For DAOs, this could be influenced by factors like the adoption rate of their platforms, the growth of their assets under management, or their income from protocol fees.
            Historical analysis of these metrics can reveal trends and changes in the market's valuation of these organizations over time. For instance, a rising EV/R might indicate increasing market optimism about the DAO's future growth prospects. Conversely, a decreasing ratio might suggest a market reevaluation of the DAO's potential.
        
        """)
        
        st.plotly_chart(fig3, use_container_width=True)
        st.plotly_chart(fig2, use_container_width=True)
        