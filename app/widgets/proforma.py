import streamlit as st
from .data.makerdao import incomestmt as mkr_incomestmt, balancesheet as mkr_balancesheet
from .data.Lido import income_statement as lido_incomestmt, balancesheet as ldo_balancesheet, lidoincome_df, income_statement_data

# Fetch data
def pro_forma_statements():
    st.title('Pro Forma Financial Statements')

    dao_options = ['MKR', 'LDO']
    dao_selection = st.radio("Select the DAO:", dao_options, index=0)

    if dao_selection == 'MKR':
        incomestmt = mkr_incomestmt
        balancesheet = mkr_balancesheet
        col1, col2 = st.columns(2)
        with col1:
            st.write('TTM Income Statement')
            st.write(incomestmt)
        with col2:
            st.write('Balance Sheet')
            st.write(balancesheet)
    elif dao_selection == 'LDO':
        incomestmt = lido_incomestmt
        balancesheet = ldo_balancesheet
        col1, col2 = st.columns(2)
        with col1:
            st.write('TTM Income Statement')
            st.write(incomestmt)
        with col2:
            st.write('Balance Sheet')
            st.write(balancesheet)

    sales_growth_rate = st.number_input("Sales Growth Rate (%)", min_value=0.0, format="%.2f")
    expense_growth_rate = st.number_input("Expense Growth Rate (%)", min_value=0.0, format="%.2f")
    
    calculate_button = st.button("Calculate Pro Forma Statements")

    if calculate_button:
    # Pro Forma Calculations
        if dao_selection == 'MKR':
            current_revenue = incomestmt.at['Lending Income', 'Amount'] + \
                              incomestmt.at['Liquidation Income', 'Amount'] + \
                              incomestmt.at['Trading Income', 'Amount']
            current_expenses = abs(incomestmt.at['Expenses', 'Amount'])
            
        elif dao_selection == 'LDO':
            current_revenue = incomestmt.at['Net Revenue', 'Amount'] 
            current_expenses = abs(incomestmt.at['Operating Expenses', 'Amount']) + \
                               abs(incomestmt.at['Liquidity Expenses', 'Amount']) + \
                               abs(incomestmt.at['Cost of Revenue', 'Amount'])
    
        
    
        # Pro Forma Income Statement
        projected_revenue = current_revenue * (1 + sales_growth_rate / 100)
        projected_expenses = current_expenses * (1 + expense_growth_rate / 100)
        projected_net_income = projected_revenue - projected_expenses
    
        # Pro Forma Balance Sheet
        current_assets = balancesheet.at['Assets', 'Amount']
        projected_assets = current_assets * (1 + sales_growth_rate / 100)
    
        st.write("Pro Forma Net Income: ", f"${projected_net_income:,.2f}")
        st.write("Pro Forma Total Assets: ", f"${projected_assets:,.2f}")
        st.write(current_revenue)
        st.write(current_expenses)
        
