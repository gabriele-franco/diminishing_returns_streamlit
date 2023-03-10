import streamlit as st
import pandas as pd
import csv

st.set_page_config(
    page_title="Cassandra MMM App",
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

datas = st.file_uploader("Choose a CSV file", type=["csv"])


if datas is not None:
    data = pd.read_csv(datas)
    st.dataframe(data)
    if data.isnull().values.any():
        st.write("DataFrame contains NA values")
    else:
        st.write("DataFrame does not contain NA values")
    media_auto = [col for col in data.columns if 'spend' in col]
    #output_auto= [col for col in data.columns if 'revenue' in col]
    #date_auto=[col for col in data.columns if 'date' in col]
    output=st.sidebar.selectbox("Select the output variable", data.columns, 1)
    date=st.sidebar.selectbox("Select the data variable", data.columns,0)
    media = st.sidebar.multiselect("Select Media Variable", data.columns, media_auto)   
    organic = st.sidebar.multiselect("Select organic Variable", data.columns)
    submit = st.button("Submit")
    if submit:
        st.session_state["data"] = data
        st.session_state["output"]=output
        st.session_state["date"]=date
        st.session_state["media"]=media
        st.session_state["organic"]=organic
        data.to_csv('dataset.csv')

