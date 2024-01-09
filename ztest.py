import pandas as pd
import streamlit as st

@st.cache_data
def load_data():
    df = pd.read_csv("https://github.com/selva86/datasets/raw/master/Auto.csv")
    return df

# callback to remove column from dataframe and store result in session state
def persist_dataframe(delete_col):
    # drop column from dataframe
    if delete_col in st.session_state["updated_df"]:
        st.session_state["updated_df"] = st.session_state["updated_df"].drop(
            columns=[delete_col]
        )
    else:
        st.sidebar.warning("Column previously deleted. Select another column.")
    
    st.write("Updated dataframe")
    st.dataframe(st.session_state["updated_df"])
    st.write(st.session_state["updated_df"].columns.tolist())

df = load_data()

# initialize session state variable
if "updated_df" not in st.session_state:
    st.session_state.updated_df = df

# display original df in col1 and updated df in col2
col1, col2 = st.columns(2)

with col1:
    st.write("Original dataframe")
    st.dataframe(df)
    st.write(df.columns.tolist())

with st.sidebar.form("my_form"):
    index = df.columns.tolist().index(
        st.session_state["updated_df"].columns.tolist()[0]
    )
    delete_col = st.selectbox(
        "Select column to delete", options=df.columns, index=index, key="delete_col"
    )
    delete = st.form_submit_button(label="Delete", on_click=lambda: persist_dataframe(delete_col))
