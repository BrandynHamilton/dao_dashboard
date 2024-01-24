import streamlit as st
from maker_page import dpi_market_premium, dpi_cumulative_risk_premium, eth_market_premium, eth_cumulative_risk_premium, calculate_wacc, calculate_rd, current_risk_free, calculate_npv_and_total_cash_flows, calculate_irr, calculate_payback_period, calculate_discounted_payback_period, calculate_profitability_index 
from data.formulas import calculate_beta 
import importlib


def wacc_calculator_page():
    
    

        # DAO Selection
        

    def dynamic_wacc_calculation(dao_selection, benchmark_selection, time_frame_selection):
        # Import required data based on DAO selection
        if dao_selection == 'MKR':
            from data.makerdao import e as dao_e, d as dao_d, rd as dao_rd 
            from maker_page import x_eth, x_dpi, y
        elif dao_selection == 'LDO':
            from data.Lido import e as dao_e, ldo_liabilities as dao_d, rd as dao_rd 
            from lido_page import x_eth, x_dpi, y
        elif dao_selection == 'RPL':
            from data.rocketpool import e as dao_e, d as dao_d, rd as dao_rd
            from rocketpool_page import x_eth, x_dpi, y

        # Calculate beta based on benchmark selection
        if benchmark_selection == 'ETH':
            beta = calculate_beta(x_eth, y)  # Use ETH data for beta calculation
        elif benchmark_selection == 'DPI':
            beta = calculate_beta(x_dpi, y)  # Use DPI data for beta calculation

        # Determine market premium based on time frame and benchmark selection
        if time_frame_selection == 'Short Term':
            market_premium = eth_market_premium if benchmark_selection == 'ETH' else dpi_market_premium
        elif time_frame_selection == 'Long Term':
            market_premium = eth_cumulative_risk_premium if benchmark_selection == 'ETH' else dpi_cumulative_risk_premium

        # Calculate re and wacc
        selected_beta = beta
        re = calculate_rd(current_risk_free, beta, market_premium)
        wacc_value = calculate_wacc(dao_e, dao_d, re, dao_rd)

        return selected_beta, re, wacc_value, dao_e, dao_d, dao_rd
    
    st.title("DAO WACC Calculator")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        dao_options = ['MKR', 'LDO', 'RPL']
        dao_selection = st.radio("Select the DAO:", dao_options, index=0)
        
    with col2:
        benchmark_options = ['ETH', 'DPI']
        benchmark_selection = st.radio("Select the Benchmark:", benchmark_options, index=0)
        
    with col3:
        time_frame_options = ['Short Term', 'Long Term']
        time_frame_selection = st.radio("Select the Time Frame:", time_frame_options, index=0)
        
        
        
    

    

    

    # Call the calculation function with the current selections
    selected_beta, re, selected_wacc, dao_e, dao_d, dao_rd = dynamic_wacc_calculation(dao_selection, benchmark_selection, time_frame_selection)
    
    # Display the results
    col4, col5, col6 = st.columns(3)
    with col4:
        st.metric("Selected WACC", f"{selected_wacc:.3%}")
    with col5:
        st.metric("Selected Beta", f"{selected_beta:.2f}")
    with col6:
        st.metric("Selected Cost of Equity", f"{re:.2%}")
        

    # Inputs for NPV calculation
    st.subheader("NPV Calculation")
    initial_investment_input = st.number_input('Initial Investment', value=100000.0)
    # Ensure the initial investment is negative
    initial_investment = -abs(initial_investment_input)

    # Assuming selected_wacc is defined earlier in your script
    discount_rate = selected_wacc

    st.write('Enter Cash Flows for Each Period:')
    
    
    
    # Initialize the number of periods in session state if not already set
    if 'num_periods' not in st.session_state or 'update' in st.session_state:
        st.session_state.num_periods = 5
        st.session_state.update = False  # Reset the update flag

    # Input for number of periods, updating the session state
    num_periods = st.number_input('Number of Periods', value= st.session_state.num_periods, min_value=1, max_value=10, key='num_periods')

    # Update session state when number of periods changes
    if num_periods != st.session_state.num_periods:
        st.session_state.num_periods = num_periods
        st.session_state.cash_flows = [0.0 for _ in range(num_periods)]
        st.session_state.cash_flows = [0.0 for _ in range(num_periods)]
        st.session_state.update = True

    # Create inputs for cash flows using session state
    for i in range(st.session_state.num_periods):
        st.session_state.cash_flows[i] = st.number_input(f'Cash Flow for Period {i+1}', 
                                                         value=st.session_state.cash_flows[i],
                                                         key=f'cash_flow_{i}')  # Unique key for each input

    # Calculate and display NPV
    if st.button('Calculate Financial Metrics'):
        npv, total_cash_flows = calculate_npv_and_total_cash_flows(discount_rate, initial_investment, st.session_state.cash_flows)
        
        non_zero_cash_flows = [cf for cf in st.session_state.cash_flows if cf != 0]
        total_cf = [initial_investment] + non_zero_cash_flows
        
        irr = calculate_irr(initial_investment, st.session_state.cash_flows)
        payback_period = calculate_payback_period(initial_investment, st.session_state.cash_flows)
        discounted_payback_period = calculate_discounted_payback_period(discount_rate, initial_investment, st.session_state.cash_flows)
        pi = calculate_profitability_index(discount_rate, initial_investment, st.session_state.cash_flows)

        st.write(f'Net Present Value (NPV): ${npv:,.2f}')
        st.write(f'Total Cashflows: {total_cf}')
        if irr is not None:
            st.write(f'Internal Rate of Return (IRR): {irr * 100:.2f}%')
        else:
            st.write('Internal Rate of Return (IRR): Not calculable')

        st.write(f'Payback Period: {payback_period} periods' if payback_period else 'Payback Period: More than the number of periods')
        st.write(f'Discounted Payback Period: {discounted_payback_period} periods' if discounted_payback_period else 'Discounted Payback Period: More than the number of periods')
        st.write(f'Profitability Index (PI): {pi:.2f}')
       

    