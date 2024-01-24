import streamlit as st
import csv
from PIL import Image

def save_email_to_csv(email, file_path='subscribers.csv'):
    with open(file_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([email])

def create_sidebar():
    logo = Image.open('Images/transparentlogo.png') 
    st.sidebar.image(logo, use_column_width=True) 

    # Navigation select box
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Home", "Financial Analysis", "DAOs", "Management Tools"],
        key='page_select'  # Unique key for this selectbox
    )
    
    # Update the session state based on selection
    if page == "DAOs":
        dao_page = st.sidebar.radio(
            "Choose a DAO",
            ["LidoDao", "MakerDao", "Rocketpool"],
            key='dao_select'  # Unique key for this radio button group
        )
        st.session_state['page'] = dao_page.lower()
    elif page == "Management Tools":
        tool_page = st.sidebar.radio(
            "Choose a tool",
            ["WACC Calculator"],
            key='tool_select'  # Unique key for this radio button group
        )
        st.session_state['page'] = tool_page.lower().replace(" ", "_")
    elif page == "Financial Analysis":
        fa_page = st.sidebar.radio(
            "Choose an analysis tool",
            ["CAPM", "Enterprise Value"],
            key='fa_select'  # Unique key for this radio button group
        )
        st.session_state['page'] = fa_page.lower().replace(" ", "_")
    
    # Subscription section
    st.sidebar.markdown("---")
    st.sidebar.header("Subscribe to Our Newsletter")
    email = st.sidebar.text_input("Enter your email", key="email")

    if st.sidebar.button("Subscribe"):
        if email and "@" in email:  # Basic check for a valid email
            save_email_to_csv(email)
            st.sidebar.success("You have successfully subscribed!")
        else:
            st.sidebar.error("Please enter a valid email address.")
    
    # Link for DAO suggestions
    st.sidebar.markdown("Have a DAO to suggest? [Tell us more](YOUR_INQUIRY_LINK)", unsafe_allow_html=True)
    
    # Displaying the CoinGecko logo
    coingecko_logo = Image.open(r'Images/coingeckologo.png')
    st.sidebar.image(coingecko_logo, width=50)
    st.sidebar.markdown('<p style="font-size:smaller;">Crypto market data provided by <a href="https://www.coingecko.com" target="_blank">CoinGecko</a></p>', unsafe_allow_html=True)


