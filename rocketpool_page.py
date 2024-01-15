import streamlit as st

def show_rocketpoolpage():
    from rocketpool import *
    from rocketpooldata import *

    #balance_sheet = balance_sheet.transpose()
    st.line_chart(rpl_history['price'])
    st.table(balance_sheet.iloc[-1])
        
            # display LidoDao content
        # ... and so on for each page ...