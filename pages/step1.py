import streamlit as st
import pandas as pd
from func import create_number_list, saturation_hill, adstock,get_average_last_15_days
from func import measure_delayed_effect,measure_dim_effect
import matplotlib.pyplot as plt



#Save all the variable in the session_state
data=st.session_state["data"]
output=st.session_state["output"]
date=st.session_state["date"]
media=st.session_state["media"]
organic=st.session_state["organic"]

if "column_values" not in st.session_state:
    st.session_state['column_values']=0.1

optimal_values=False
#st.dataframe(data)
st.title("Select Date Range")
data[date]=pd.to_datetime(data[date])
lagged={}
correlations={}

for i in media:
    lagged[i]={'shape':0.1,'scale':0.1,'alpha':0.3, 'gamma': 1.01}

if st.button('Find Optimal Values'):
    optimal_values=True
    for i in media:
        optimal_dim,corr_dim=measure_dim_effect(data, output, i)
        optimal_params, corr = measure_delayed_effect(data, output, i)
        lagged[i]={'shape':optimal_params[0],'scale':optimal_params[1],
        'alpha':optimal_dim[0], 'gamma': optimal_dim[1]}
        correlations[i]={'correlation_adstock':abs(corr),'correlation_dim':abs(corr_dim)}
        st.session_state['lagged']=lagged

            
"""else:
    for i in media:
        lagged[i]={'shape':0.1,'scale':0.1,'alpha':0.3, 'gamma': 1.01}
        column_values={}"""


lagged=st.session_state['lagged']


start_date = st.date_input(label='Start Date', value=data[date].min())
end_date = st.date_input(label='End Date', value=data[date].max())
iterations = st.text_input("Enter iterations")
column_values = {}
variance={}
# Display the selected columns
if media:
    for i, col in enumerate(media):
        gamma = st.sidebar.slider(label=f'{col}_gamma', min_value=0.1, max_value=1.1,value=float(lagged[col]['gamma']), step=0.1)
        var_gamma=st.sidebar.number_input(label=f'{col}_gamma variance', min_value=0, max_value=100, value=70)
        alpha = st.sidebar.slider(label=f'{col}_alpha', min_value=0.1, max_value=3.1,value=float(lagged[col]['alpha']), step=0.1)
        var_alpha=st.sidebar.number_input(label=f'{col}_alpha variance', min_value=0, max_value=100, value=70)
        shape = st.sidebar.slider(label=f'{col}_shape', min_value=0.1, max_value=10.1,value=float(lagged[col]['shape']), step=0.01)
        var_shape=st.sidebar.number_input(label=f'{col}_shape variance', min_value=0, max_value=100, value=70)
        scale = st.sidebar.slider(label=f'{col}_scale', min_value=0.0001, max_value=0.5,value=float(lagged[col]['scale']), step=0.0001)
        var_scale=st.sidebar.number_input(label=f'{col}_scale variance', min_value=0, max_value=100, value=70)
        variance[col]={'gamma':var_gamma, 'alpha':var_alpha, 'shape':var_shape, 'scale':var_scale}
        column_values[col] = {'gamma': gamma, 'alpha': alpha, 'shape': shape, 'scale': scale}

coef=st.number_input(label='coefficient', value=50)

spent=get_average_last_15_days(data, media)

# Create a new section in the sidebar
ad=st.selectbox("Select the media", media)
st.subheader(f"{ad} Graph")
df=create_number_list(data, ad)
df['dim'] = saturation_hill(df['spent'], column_values[ad]['alpha'], column_values[ad]['gamma'])
# Create a subplot with 2 columns and 1 row
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
# Plot the first graph on the first column
ax1.plot(df['spent'], df['dim']*coef)
# Add a vertical line at x-coordinate 15
ax1.axvline(x=spent[ad], color='red', linestyle='--', linewidth=2)
st.write(f'average spent {spent[ad]}')
ax1.set_xlabel("Original values")
ax1.set_ylabel("Transformed values")
ax1.set_title("Saturation Hill")

adstock_df= adstock( column_values[ad]['shape'], column_values[ad]['scale'], windlen=None, type="pdf")
adstock_tt=pd.DataFrame(adstock_df)
# Plot the second graph on the second column
ax2.plot(adstock_df['day'], adstock_df['theta_vec_cum'])
ax2.set_xlabel("Days")
ax2.set_ylabel("Theta vec cum")
ax2.set_title("Adstock")
# Show the subplot
st.pyplot(fig)

if optimal_values==True:
    column_values=lagged
else:
    column_values=column_values.copy()



st.session_state['column_values']=column_values
st.session_state['start_date']=start_date
st.session_state['end_date']=end_date
st.session_state['iterations']=iterations
st.session_state['variance']=variance




