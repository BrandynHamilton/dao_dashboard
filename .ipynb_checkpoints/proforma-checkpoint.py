import streamlit as st
from data.makerdao import *

# Fetch data
income_statement = incomestmt
balance_sheet = balancesheet

sales_growth_rate = st.number_input("Sales Growth Rate (%)", min_value=0.0, format="%.2f")
expense_growth_rate = st.number_input("Expense Growth Rate (%)", min_value=0.0, format="%.2f")

# Pro Forma Calculations
current_revenue = incomestmt.at['Lending Income', 'Amount'] + \
                  incomestmt.at['Liquidation Income', 'Amount'] + \
                  incomestmt.at['Trading Income', 'Amount']
current_expenses = abs(incomestmt.at['Expenses', 'Amount'])

# Pro Forma Income Statement
projected_revenue = current_revenue * (1 + sales_growth_rate / 100)
projected_expenses = current_expenses * (1 + expense_growth_rate / 100)
projected_net_income = projected_revenue - projected_expenses

# Pro Forma Balance Sheet
current_assets = balancesheet.at['Assets', 'Amount']
projected_assets = current_assets * (1 + sales_growth_rate / 100)

st.write("Pro Forma Net Income: ", f"${projected_net_income:,.2f}")
st.write("Pro Forma Total Assets: ", f"${projected_assets:,.2f}")

