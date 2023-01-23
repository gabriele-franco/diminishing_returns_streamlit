
import streamlit as st
from func import saturation_hill, create_number_list, adstock,transform_json
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px




st.set_page_config(page_title="CSV Uploader", page_icon=":chart_with_upwards_trend:", layout="wide")

st.title("Upload your CSV file")

uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
uploaded_json=st.file_uploader("Choose a JSON file", type=["json"])


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
        # Create a subplot with 2 columns and 1 row
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
        # Plot the first graph on the first column
        ax1.plot(df['spent'], df['dim'])
        ax1.set_xlabel("Original values")
        ax1.set_ylabel("Transformed values")
        ax1.set_title("Saturation Hill")
        
        adstock_df= adstock( shape, scale, windlen=None, type="pdf")
        adstock_tt=pd.DataFrame(adstock_df)
        # Plot the second graph on the second column
        ax2.plot(adstock_df['day'], adstock_df['theta_vec_cum'])
        ax2.set_xlabel("Days")
        ax2.set_ylabel("Theta vec cum")
        ax2.set_title("Adstock")
        # Show the subplot
        st.pyplot()


if uploaded_json is not None:
    data = pd.read_json(uploaded_json)
    transformed_dict=transform_json(data)
    st.code(transformed_dict, language='python')