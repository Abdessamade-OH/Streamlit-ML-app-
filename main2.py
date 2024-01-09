import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Fonction pour importer fichier csv
@st.cache_data
def cache_csv_data(file_uploader):
    uploaded_file = file_uploader

    if uploaded_file is not None:
        st.success("Your data has been successfully imported.")
        return pd.read_csv(uploaded_file)


# Function to check if data is uploaded
def check_data_uploaded():
    # Attempt to get data from cache
    cached_data = cache_csv_data(None)  # Pass None to bypass actual file_uploader

    if cached_data is None:
        st.warning("Veuillez d'abord télécharger des données.")
        st.stop()
    else:
        st.success("Your data has been successfully imported.")

# Fonction pour afficher les onglets
def display_tabs():
    tab1, tab2 = st.tabs(["Data", "Visualise"])

    # onglet importation des données
    with tab1:
        st.header("Data")

        # Streamlit elements
        uploaded_file = st.file_uploader("Import a CSV file", type=["csv"])

        # Use the function to import CSV data
        data = cache_csv_data(uploaded_file)

        # Display the data
        if data is not None:
            st.write("Imported Data:")
            st.dataframe(data)

        with st.form(key="Exit", border=False):
            st.form_submit_button("Exit", on_click=lambda: st.session_state.update({"page": 0}))  # Revenir à la landing page

    
    # onglet visualisation des données
    with tab2:
        st.header("Visualise")

        check_data_uploaded()

        


    

# Fonction principale
def main():
    display_tabs()

if __name__ == "__main__":
    main()