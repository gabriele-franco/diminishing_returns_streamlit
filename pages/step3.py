import streamlit as st
import pandas as pd 
from func import robyn_cpo
import plotly.express as px
df= pd.read_csv('pages/models/pareto_clusters.csv')
st.dataframe(df.head(10))

solid=st.text_input('enter solID')

def robyn_cpo(json):
    cpo={}
    spend={}
    for i in json['ExportedModel']['summary']:
        if 'S' in i['variable']:
            var=i['variable']
            spent=i['mean_spend']
            response=i['mean_response']
            spend[var]={'mean_spend': spent}
            if response == 0:
                cpo[var]={'cpo':0}
            else:
                cpo[var]={'cpo':spent/(response+1)}
        else:
            continue
    return cpo, spend

def get_json(solid):
    v=f'pages/models/RobynModel-{solid}.json'
    robyn_json=pd.read_json(v)
    return robyn_json


def create_cpo_graph(cpo, spend):
    cpo_df = pd.DataFrame.from_dict(cpo, orient='index')
    spend_df = pd.DataFrame.from_dict(spend, orient='index')

    # Merge the two dataframes
    df = cpo_df.merge(spend_df, left_index=True, right_index=True)
    df.columns = ['cpo', 'mean_spend']

    # Plot the bar chart
    fig = px.bar(df, x=df.index, y='mean_spend', color='cpo', text='cpo', 
                labels={'mean_spend': 'Spend', 'cpo': 'CPO'})
    st.plotly_chart(fig)





if solid:
    json=get_json(solid)
    cpo,spend=robyn_cpo(json)
    create_cpo_graph(cpo, spend)



