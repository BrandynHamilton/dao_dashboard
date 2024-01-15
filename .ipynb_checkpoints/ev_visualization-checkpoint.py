import streamlit as st
import numpy as np
from lidoev import filtered_ev_metrics as lido_metrics
from makerev import ev_metrics as maker_metrics
import pandas as pd
import plotly.graph_objs as go

# Create a figure
fig2 = go.Figure()

# Add EV to Revenue (Filtered and Truncated) - primary y-axis
fig.add_trace(go.Scatter(x=lido_metrics.index, y=lido_metrics['ev_to_rev'], mode='lines', name='LidoDao Ev to Revenue'))
fig.add_trace(go.Scatter(x=maker_metrics.index, y=maker_metrics['ev_to_rev_truncated'], mode='lines', name='MakerDao EV to Revenue'))

# Add Historical EV (Filtered and Unfiltered) - secondary y-axis
fig.add_trace(go.Scatter(x=maker_metrics.index, y=maker_metrics['historical_ev'], mode='lines', name='MakerDao EV', yaxis='y2'))
fig.add_trace(go.Scatter(x=lido_metrics.index, y=lido_metrics['historical_ev'], mode='lines', name='LidoDao EV', yaxis='y2'))

# Create a layout with a secondary y-axis
fig.update_layout(
    title='Enterprise Value and Ev to Revenue Over Time',
    xaxis_title='Date',
    yaxis_title='EV to Revenue',
    yaxis2=dict(
        title='Enterprise Value',
        overlaying='y',  # Align the secondary y-axis with the primary
        side='right'  # Position the secondary y-axis on the right
    )
)

# Display the Plotly figure in Streamlit
st.plotly_chart(fig2, use_container_width=True)
