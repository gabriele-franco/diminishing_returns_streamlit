
import streamlit as st
from func import saturation_hill, create_number_list, adstock
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

    # Slider for shape
    shape=st.sidebar.slider("Shape", 0.1, 10.0, step=0.1)


    # Slider for scale
    scale=st.sidebar.slider("Scale", 0.0001, 0.5, step=0.0001)
    
    coeff = st.sidebar.text_input("Enter coeff value")
    coefficiente=float(coeff)
    # Apply saturation_robyn function to selected column
    if st.sidebar.button("Apply transformation"):
        df=create_number_list(data, select_col)
        df['dim'] = saturation_hill(df['spent'], alpha, gamma)        
        fig = px.line(
            df,
            x="spent",
            y="dim")
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)

        adstock_df= adstock( shape, scale, windlen=None, type="pdf")
        adstock_tt=pd.DataFrame(adstock_df)
        fig2 = px.line(
            adstock_df,
            x="day",
            y="theta_vec_cum")
        st.plotly_chart(fig2, theme="streamlit", use_container_width=True) 
