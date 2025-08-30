import streamlit as st 
import requests 

st.title("Live currency converter")
amount = st.number_input("Enter the amount in NPR", min_value=1)

target_currency = st.selectbox("convert to:", ['JPY', 'USD', 'EUR', 'GBP'])

if st.button("convert"):
    url = "https://api.exchangerate-api.com/v4/latest/NPR"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        rate = data['rates'][target_currency]
        converted = rate * amount
        st.success(f"NPR to {target_currency}, {amount} = {converted}")

    else:
        st.error("Failed to fetch conversion rate api")