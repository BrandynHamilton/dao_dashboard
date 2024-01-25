import streamlit as st
from sidebar import create_sidebar
from home_page import show_homepage
from maker_page import show_makerpage
from lido_page import show_lidopage
from widgets.capm import show_industry_metrics
from widgets.enterprise_value import enterprise_metrics
from rocketpool_page import show_rocketpoolpage
from widgets.npv_calculator import wacc_calculator_page
from widgets.dividend_widget import dividend_widget
from widgets.proforma import pro_forma_statements


create_sidebar()


if 'page' not in st.session_state:
    st.session_state['page'] = 'home'  # default page

if st.session_state['page'] == 'home':
    show_homepage()

elif st.session_state['page'] == 'capm':
    show_industry_metrics()  # This function needs to be defined by you

elif st.session_state['page'] == 'enterprise_value':
    enterprise_metrics()  # This function needs to be defined by you

elif st.session_state['page'] == 'lidodao':
    show_lidopage()

elif st.session_state['page'] == 'makerdao':
    show_makerpage()

elif st.session_state['page'] == 'rocketpool':
    show_rocketpoolpage()
    
elif st.session_state['page'] == 'npv_calculator':
    wacc_calculator_page()
elif st.session_state['page'] == 'dividend_widget':
    dividend_widget()
elif st.session_state['page'] == 'proforma':
    pro_forma_statements()
    
    
    
