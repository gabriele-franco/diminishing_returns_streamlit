
import streamlit as st
from func import saturation_hill, saturation_robyn, create_number_list
import pandas as pd

import plotly.express as px




st.set_page_config(page_title="CSV Uploader", page_icon=":chart_with_upwards_trend:", layout="wide")

st.title("Upload your CSV file")

uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])


# Upload CSV file

# Read CSV file
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.dataframe(data)
    
    # Create sidebar
    st.sidebar.title("Data Exploration")
    
    # Single select
    select_col = st.sidebar.selectbox("Select a column to explore", data.columns)
    
    # Slider for gamma
    gamma = st.sidebar.slider("Gamma", 0.0, 1.0, step=0.1)
    
    # Slider for alpha
    alpha = st.sidebar.slider("Alpha", 0.0, 3.0, step=0.1)
    
    coeff = st.sidebar.text_input("Enter coeff value")
    coefficiente=float(coeff)
    # Apply saturation_robyn function to selected column
    if st.button("Apply transformation"):
        df=create_number_list(data, select_col)
        df['dim'] = saturation_hill(df['spent'], alpha, gamma)        
        fig = px.line(
            df,
            x="spent",
            y="dim")
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)
