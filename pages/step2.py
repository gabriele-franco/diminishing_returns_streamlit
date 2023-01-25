import streamlit as st
from func import display_dict,generate_robyn_inputs

column_values=st.session_state['column_values']
start_date = st.session_state['start_date']
end_date= st.session_state['end_date']
iterations=   st.session_state['iterations']
data=st.session_state["data"]
output=st.session_state["output"]
date=st.session_state["date"]
media=st.session_state["media"]
organic=st.session_state["organic"]
variance=st.session_state['variance']

init=generate_robyn_inputs(date, output, media, organic, start_date, end_date, iterations,column_values, variance )
st.code(init)
