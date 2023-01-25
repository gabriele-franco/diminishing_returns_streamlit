import streamlit as st
import pandas as pd
from func import saturation_hill, create_number_list, adstock
import matplotlib.pyplot as plt

st.set_page_config(page_title="CSV Uploader", page_icon=":chart_with_upwards_trend:", layout="wide")

st.title("Upload your CSV file")

uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])


# Upload CSV file
button_clicked = False
# Read CSV file
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.dataframe(data)
    output=st.sidebar.selectbox("Select the output variable", data.columns)
    date=st.sidebar.selectbox("Select the data variable", data.columns)
    media = st.sidebar.multiselect("Select Media Variable", data.columns)   
    organic = st.sidebar.multiselect("Select organic Variable", data.columns) 
    button_clicked = st.cache(lambda: False)
    # Create variable to keep track of button state


    # Check if the button is clicked
    if st.sidebar.button("Save"):
        button_clicked = True
        

    if button_clicked:
        st.title("Select Date Range")
        data[date]=pd.to_datetime(data[date])

        start_date = st.date_input(label='Start Date', value=data[date].min())
        end_date = st.date_input(label='End Date', value=data[date].max())
        iterations = st.text_input("Enter iterations")
        column_values = {}
       # Display the selected columns
        if media:
            for i, col in enumerate(media):
                st.write(col)
                gamma = st.slider(label=f'{col}_gamma', min_value=0.1, max_value=1.1, step=0.1)
                alpha = st.slider(label=f'{col}_alpha', min_value=0.1, max_value=3.1, step=0.1)
                shape = st.slider(label=f'{col}_shape', min_value=0.1, max_value=10.1, step=0.1)
                scale = st.slider(label=f'{col}_scale', min_value=0.0001, max_value=0.5, step=0.0001)
                column_values[col] = {'gamma': gamma, 'alpha': alpha, 'shape': shape, 'scale': scale}

                # Create a new section in the sidebar
                ad=st.selectbox("Select the media", media)
                st.subheader(f"{ad} Graph")
                df=create_number_list(data, ad)
                df['dim'] = saturation_hill(df['spent'], column_values[ad]['alpha'], column_values[ad]['gamma'])
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
                st.pyplot(fig)

        else:
            st.write("Please select at least one column.")



    

