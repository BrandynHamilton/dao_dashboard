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
            st.sidebar.write(" ")  # Add multiple of these if needed to create space
            st.sidebar.markdown("Have a DAO to suggest? [Tell us more](YOUR_INQUIRY_LINK)", unsafe_allow_html=True)
    st.sidebar.write(" ")     
    st.sidebar.write(" ")
    st.sidebar.write(" ")
    st.sidebar.write(" ")
    st.sidebar.write(" ")
    st.sidebar.write(" ")
    st.sidebar.write(" ")

    coingecko_logo = Image.open(r'Images/coingeckologo.png')  # Update path as needed
    st.sidebar.image(coingecko_logo, width=50)
    st.sidebar.markdown('<p style="font-size:smaller;">Crypto market data provided by <a href="https://www.coingecko.com" target="_blank">CoinGecko</a></p>', unsafe_allow_html=True)

   
