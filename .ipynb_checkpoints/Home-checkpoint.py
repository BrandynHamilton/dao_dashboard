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
from rocketpooldata import enterprise_value as rpl_enterprise_value, balance_sheet as rpl_bs
from home_page import *
from maker_page import *
from lido_page import *
#from rocketpool_page import *


# Call the new function to display the logo at the top center
create_sidebar()




if 'page' not in st.session_state:
    st.session_state['page'] = 'home'  # default page

if st.session_state['page'] == 'home':
    show_homepage()

elif st.session_state['page'] == 'lidodao':
    show_lidopage()
    
elif st.session_state['page'] == 'makerdao':

   show_makerpage()
        
#elif st.session_state['page'] == 'rocketpool':
    #show_rocketpoolpage()
    
    
    
    
    
