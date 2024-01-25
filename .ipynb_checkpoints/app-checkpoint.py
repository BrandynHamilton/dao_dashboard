import streamlit as st
from app.sidebar import create_sidebar
from app.home_page import show_homepage
from app.maker_page import show_makerpage
from app.lido_page import show_lidopage
from app.widgets.capm import show_industry_metrics
from app.widgets.enterprise_value import enterprise_metrics
from app.rocketpool_page import show_rocketpoolpage
from app.widgets.npv_calculator import wacc_calculator_page
from app.widgets.dividend_widget import dividend_widget
from app.widgets.proforma import pro_forma_statements


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
    
    
    
