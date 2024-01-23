import streamlit as st
from sidebar import create_sidebar
from home_page import show_homepage
from maker_page import show_makerpage
from lido_page import show_lidopage
from financial_analysis import show_industry_metrics
from rocketpool_page import show_rocketpoolpage

create_sidebar()

if 'page' not in st.session_state:
    st.session_state['page'] = 'home'  # default page

if st.session_state['page'] == 'home':
    show_homepage()

elif st.session_state['page'] == 'financial_analysis':
    show_industry_metrics()

elif st.session_state['page'] == 'lidodao':
    show_lidopage()

elif st.session_state['page'] == 'makerdao':
    show_makerpage()

elif st.session_state['page'] == 'rocketpool':
    show_rocketpoolpage()
    
    
    
    
    
