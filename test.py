import streamlit as st
import pandas as pd

def create_data():
    st.subheader("Create Data")

    # Use the form container to prevent form submission on Enter
    with st.form("data_creation_form"):
        num_rows = st.number_input("Number of Rows:", min_value=1, value=5)
        num_cols = st.number_input("Number of Columns:", min_value=1, value=3)

        columns = []
        for i in range(num_cols):
            column_name = st.text_input(f"Column {i + 1} Name:", key=f"column_{i}")
            columns.append(column_name)

        data = []
        for row_index in range(num_rows):
            row_data = []
            for col_index in range(num_cols):
                cell_value = st.text_input(f"Enter value ({row_index + 1}, {col_index + 1}):", key=f"value_{row_index}_{col_index}")
                row_data.append(cell_value)
            data.append(row_data)

        if st.form_submit_button("Create DataFrame"):
            df = pd.DataFrame(data, columns=columns)
            st.dataframe(df)

# Sample usage in your main script
if st.sidebar.button("Data"):
    create_data()
