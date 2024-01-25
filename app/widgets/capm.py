import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from ..data.makerdao import beta as makerdao_beta, average_yearly_risk_premium, current_risk_free as risk_free_rate, mkr_cagr, mkr_avg_excess_return, cumulative_risk_premium, cagr_percentage, dpi_history
from ..data.Lido import beta as lidodao_beta, lido_cagr, ldo_avg_excess_return
from ..data.rocketpool import beta_num as rpl_beta, rpl_cagr, rpl_avg_excess_return, ev_df, enterprise_value as rpl_enterprise_value
from ..sidebar import create_sidebar
from ..data.lidoev import filtered_ev_metrics as lido_metrics
from ..data.makerev import historical_ev as maker_metrics, ev_metrics as monthly_metrics
from ..maker_page import dpi_market_premium, dpi_cumulative_risk_premium, eth_market_premium, eth_cumulative_risk_premium, eth_mkr_beta, dpi_mkr_beta
from ..lido_page import eth_ldo_beta, dpi_ldo_beta
from ..rocketpool_page import eth_rpl_beta, dpi_rpl_beta
from ..data.formulas import calculate_beta

def create_interactive_sml(risk_free_rate, market_risk_premium, makerdao_beta, lidodao_beta, rpl_beta, makerdao_return, lidodao_return, rpl_return, term):
        betas = np.linspace(0, 1.5, 100)
        expected_returns = risk_free_rate + betas * market_risk_premium
    
        # Create the SML line
        sml_line = go.Scatter(x=betas, y=expected_returns, mode='lines', name=f'{term} SML')
    
        # Add MakerDAO token as points for expected (based on selected market risk premium) and actual returns
        makerdao_expected = go.Scatter(
            x=[makerdao_beta],
            y=[risk_free_rate + makerdao_beta * market_risk_premium],  # Use the selected market risk premium
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
    
        # Add LidoDAO token as points for expected and actual returns
        lidodao_expected = go.Scatter(
            x=[lidodao_beta],
            y=[risk_free_rate + lidodao_beta * market_risk_premium],  # Use the selected market risk premium
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
    
        # Add Rocket Pool token as points for expected and actual returns
        rpl_expected = go.Scatter(
            x=[rpl_beta_value],
            y=[risk_free_rate + rpl_beta_value * market_risk_premium],  # Use the selected market risk premium
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
        return fig, market_risk_premium

def show_industry_metrics():
    
    st.title('CAPM')

    st.subheader('Security Market Line (SML) Analysis')

    st.write(""" 
    This graph plots expected returns against the risk (measured by beta) of DAO equity tokens. It facilitates the comparison of actual and expected returns relative to market risk, offering valuable insights for investors and DAO financial managers.""")

    with st.expander("For Investors"):
        st.write("""
        - **Risk-Return Analysis:** Evaluate whether DAO tokens are providing adequate returns for their assumed level of risk. Points above the SML suggest better-than-market performance, while points below indicate potential underperformance.
    - **Investment Decision-Making:** Use the graph to pinpoint potentially undervalued or overvalued tokens, guiding investment decisions aligned with risk preferences.
        
        
        """)
    with st.expander("For DAO Financial Managers"):
        st.write("""
        - **Performance Insight:** The SML graph depicts how the market appraises the DAO's tokens vis-à-vis their risk, offering essential insights for strategic financial planning and token management.
    - **Benchmarking and Strategy:** By using ETH or DPI as a benchmark, DAO managers can contrast their performance with the broader DeFi sector, informing strategic positioning and long-term planning.
        
        """)

    with st.expander("Benchmark Considerations"):
        st.write("""
        - **ETH:** When using Ethereum as a benchmark, the SML reflects the token's performance in relation to the overall Ethereum market. This is relevant for investors focusing on the systemic risk associated with the Ethereum blockchain and its wider ecosystem.
    - **DPI:** Choosing the DeFi Pulse Index as a benchmark provides a view of the token's performance against a basket of DeFi tokens. This perspective is useful for assessing performance relative to a diversified DeFi portfolio.
        """)

    with st.expander("Time Horizon Considerations"):
        st.write("""
        - **Short-Term Analysis (Average Excess Return and Average Yearly Risk Premium):** Understand short-term market trends and dynamics for near-term financial decision-making.
    - **Long-Term Analysis (CAGR and Cumulative Risk Premium):** Assess the long-term viability and growth potential of DAO tokens, aiding in strategic investment or resource allocation.
        """)
    
    
    
    
   
    
    
    # ... [other code] ...

    # Dropdown selection for benchmark
    benchmark = st.selectbox(
        "Select Benchmark",
        ["DPI", "ETH"]  # Replace with actual benchmark names
    )
    
    # Dropdown selection for time frame
    time_frame = st.selectbox(
        "Select Time Frame",
        ("Short-term", "Long-term")
    )
    
    # Recalculate betas based on selected benchmark
    makerdao_beta = eth_mkr_beta if benchmark == "ETH" else dpi_mkr_beta
    lidodao_beta = eth_ldo_beta if benchmark == "ETH" else dpi_ldo_beta
    rpl_beta = eth_rpl_beta if benchmark == "ETH" else dpi_rpl_beta
    
  
    
    # Determine which returns to use based on selected time frame
    if time_frame == "Short-term":
        if benchmark == "ETH":
            market_risk_premium = eth_market_premium
        else:
            market_risk_premium = dpi_market_premium
    else:  # Long-term
        if benchmark == "ETH":
            market_risk_premium = eth_cumulative_risk_premium
        else:
            market_risk_premium = dpi_cumulative_risk_premium
    if time_frame == "Short-term":
        makerdao_return = mkr_avg_excess_return
        lidodao_return = ldo_avg_excess_return
        rpl_return = rpl_avg_excess_return
    else:  # Long-term
        makerdao_return = mkr_cagr
        lidodao_return = lido_cagr
        rpl_return = rpl_cagr
     
            
    
    
    # Create and display the SML plot with user-selected benchmarks and time frames
    fig, market_risk_premium = create_interactive_sml(
    risk_free_rate,
    market_risk_premium,  # Determined by the benchmark (ETH or DPI) and time frame (short or long-term)
    makerdao_beta,
    lidodao_beta,
    rpl_beta,
    makerdao_return,  # Make sure this is defined before calling the function
    lidodao_return,   # Make sure this is defined before calling the function
    rpl_return,       # Make sure this is defined before calling the function
    time_frame
    )
    
    st.plotly_chart(fig)

    
    
    
    
    
        