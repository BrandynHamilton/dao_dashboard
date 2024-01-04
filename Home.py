import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from makerdao import beta as makerdao_beta, average_yearly_risk_premium, current_risk_free as risk_free_rate, mkr_cagr, mkr_avg_excess_return, cumulative_risk_premium, cagr_percentage, dpi_history
from Lido import beta as lidodao_beta, lido_cagr, ldo_avg_excess_return

st.title('DAO Dashboard')

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

with st.expander('Benchmark: DefiPulse Index (DPI)'):
    st.line_chart(dpi_history['price'])
    st.write(f"DPI CAGR is {cagr_percentage:.2f}%")
    st.write(f"DPI Average Excess Return is {percent_risk:.2f}%")

import streamlit as st
from PIL import Image
import requests
from io import BytesIO

# Get the CoinGecko logo image from a URL
response = requests.get('https://static.coingecko.com/s/coingecko-logo.png')
coingecko_logo = Image.open(BytesIO(response.content))

# Display the logo and attribution at the bottom of the sidebar or home page
st.sidebar.image(coingecko_logo, width=100)
st.sidebar.markdown(
    'Crypto market data provided by [CoinGecko](https://www.coingecko.com)'
)
