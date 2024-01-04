# sidebar.py
import streamlit as st
import csv
from PIL import Image

def save_email_to_csv(email, file_path='subscribers.csv'):
    with open(file_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([email])

def create_sidebar():
    # Email Subscription Section
    st.sidebar.header("Subscribe to Our Newsletter")
    email = st.sidebar.text_input("Enter your email", key="email")

    if st.sidebar.button("Subscribe"):
        if email and "@" in email:  # Basic check for a valid email
            save_email_to_csv(email)
            st.sidebar.success("You have successfully subscribed!")
        else:
            st.sidebar.error("Please enter a valid email address.")

    # CoinGecko Logo and Attribution
    coingecko_logo = Image.open('Images/coingecko_logo.png')  # Update path as needed
    st.sidebar.image(coingecko_logo, width=50)
    st.sidebar.markdown(
        'Crypto market data provided by [CoinGecko](https://www.coingecko.com)',
        unsafe_allow_html=True
    )

    # Additional Sidebar Elements (if any)
    # ...

# Example use of additional functions or configurations
# You can add more functions or configurations here if needed for your sidebar
