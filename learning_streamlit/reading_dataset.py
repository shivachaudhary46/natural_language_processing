import streamlit as st
import pandas as pd 

st.title("Reading Dataset")
file = st.file_uploader("Upload your csv file", type=["csv"])

if file:
    df = pd.read_csv(file, index_col=0)
    st.subheader("Data Preview")
    st.dataframe(df)

st.sidebar.text_input("Enter Your Name !")

if file:
    st.subheader('Summary Stats')
    st.write(df.describe())

if file:
    cities = df['country'].unique()
    st.subheader("Filtering by Countries")
    select_country = st.selectbox("filter by country", cities)
    filtered_df = df[df["country"] == select_country] 
    st.dataframe(filtered_df)
