import streamlit as st 


st.title("Favourite Programming Language")
st.subheader("Created with Streamlit app")
st.text("welcome to streamlit app")
st.write("choose your Favourite Programming Language")

lang = st.selectbox("languages : ", ['Python', 'Javascript', 'Web Development', "Fast Api", "Pytorch", "C++"])

st.success(f"You choose {lang} as fav language")

