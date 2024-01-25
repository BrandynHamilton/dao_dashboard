import pandas as pd
import streamlit as st
from .data.makerdao import *

# Part 1: Debt Pool Investment Simulation
def calculate_apy(apr, compounding_periods=12):
    return (1 + apr / compounding_periods) ** compounding_periods - 1

st.title("DAO Debt Pool Investment Simulation")

# Inputs for Debt Pool Simulation
deposited_amount = st.number_input("Amount Deposited by Investor", value=50000.0)
investment_period = st.number_input("Investment Period (in years)", value=1.0)


# Calculation
apr = wacc  # APR is equivalent to the interest rate (WACC)
apy = calculate_apy(wacc)  # Compounding assumed monthly

# Display APR and APY
st.write(f"APR (DAO's perspective): {apr * 100:.2f}%")
st.write(f"APY (Investor's perspective): {apy * 100:.2f}%")
# Part 3: Calculate Total Return for Investor

# Assuming interest is paid annually for simplicity
total_interest = deposited_amount * apr * investment_period

# Total return is the sum of interest payments and the principal amount
# Assuming the principal is returned in full
total_return = total_interest + deposited_amount

# Display Total Return
st.write(f"Total Return for Investor: ${total_return:,.2f}")

# You can also show the total profit (interest payments only)
total_profit = total_interest
st.write(f"Total Profit from Interest Payments: ${total_profit:,.2f}")


# Part 2: Updating Balance Sheet and Income Statement
# Assuming existing data is already fetched into 'incomestmt' and 'balancesheet'

# Update Balance Sheet
balancesheet.at['Liabilities', 'Amount'] += deposited_amount

# Update Income Statement
incomestmt.at['Interest Expense', 'Amount'] = -deposited_amount * apr * investment_period

# Display updated financial statements
st.subheader("Updated Balance Sheet")
st.table(balancesheet)

st.subheader("Updated Income Statement")
st.table(incomestmt)