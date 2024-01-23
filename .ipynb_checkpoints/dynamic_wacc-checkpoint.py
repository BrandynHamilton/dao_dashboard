import streamlit as st
from maker_page import dpi_market_premium, dpi_cumulative_risk_premium, eth_market_premium, eth_cumulative_risk_premium, calculate_wacc, calculate_rd, current_risk_free 
from data.formulas import calculate_beta 


def dynamic_wacc_calculation(dao_selection, benchmark_selection, time_frame_selection):
    # Import required data based on DAO selection
    if dao_selection == 'MKR':
        from data.makerdao import d as dao_d, rd as dao_rd 
        from maker_page import market_value as e, x_eth, x_dpi, y
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

def main():
    # Streamlit UI for user input
    st.title("DAO WACC Calculator")

    # DAO Selection
    dao_options = ['MKR', 'LDO', 'RPL']
    dao_selection = st.radio("Select the DAO:", dao_options, index=0)  # Default to first option

    # Benchmark Selection
    benchmark_options = ['ETH', 'DPI']
    benchmark_selection = st.radio("Select the Benchmark:", benchmark_options, index=0)  # Default to first option

    # Time Frame Selection
    time_frame_options = ['Short Term', 'Long Term']
    time_frame_selection = st.radio("Select the Time Frame:", time_frame_options, index=0)  # Default to first option

    # Calculate WACC dynamically based on user input
    if st.button("Calculate WACC"):
        selected_beta, re, selected_wacc, dao_e, dao_d, dao_rd = dynamic_wacc_calculation(dao_selection, benchmark_selection, time_frame_selection)
        st.metric("Selected WACC", f"{selected_wacc:.3%}")
        st.metric("Selected Beta", f"{selected_beta:.2f}")
        st.metric("Selected Cost of Equity", f"{re:.2%}")
        st.metric("marketcap", f"${dao_e:,.2f}")
        st.metric("liabilities", f"${dao_d:,.2f}")
        st.metric("cost of debt", f"{dao_rd:.2%}")

if __name__ == "__main__":
    main()