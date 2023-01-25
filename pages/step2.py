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


def display_dict(data):
    result = ""
    for key, value in data.items():
        for sub_key, sub_value in value.items():
            original_value = sub_value
            lower_value = round(original_value * 0.8, 4)
            higher_value = round(original_value * 1.2, 4)
            result += f"{key}_{sub_key} = c{lower_value,higher_value},\n"
    return result

init=generate_robyn_inputs(date, output, media, organic, start_date, end_date)

p=display_dict(column_values)

st.code(init)
st.code(p)