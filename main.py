import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Multipage App",
    page_icon="👋",
)

st.title("Main Page")
st.sidebar.success("Select a page above.")




if "data" not in st.session_state:
    st.session_state["data"] = ""
    st.session_state["output"]=""
    st.session_state["date"]=""
    st.session_state["media"]=""
    st.session_state["organic"]=""

data = st.file_uploader("Choose a CSV file", type=["csv"])
if data is not None:
    data = pd.read_csv(data)
    st.dataframe(data)
    output=st.sidebar.selectbox("Select the output variable", data.columns)
    date=st.sidebar.selectbox("Select the data variable", data.columns)
    media = st.sidebar.multiselect("Select Media Variable", data.columns)   
    organic = st.sidebar.multiselect("Select organic Variable", data.columns) 
    submit = st.button("Submit")
    if submit:
        st.session_state["data"] = data
        st.session_state["output"]=output
        st.session_state["date"]=date
        st.session_state["media"]=media
        st.session_state["organic"]=organic

