import streamlit as st
import pandas as pd

# Fonction pour initialiser la session
def init_session():
    if "page" not in st.session_state:
        st.session_state.page = "landing"
        st.session_state.data = None

# Fonction pour afficher la landing page
def landing_page():
    st.title("Bienvenue dans notre application de Machine Learning.")
    st.write("Cliquez sur le bouton ci-dessous pour commencer.")
    
    if st.button("Get Started"):
        st.session_state.page = "importation"

# Fonction pour afficher la page d'importation des données
def importation_page():
    st.title("Importation des données")
    uploaded_file = st.file_uploader("Importer un fichier CSV", type=["csv"])

    if uploaded_file is not None:
        st.session_state.data = pd.read_csv(uploaded_file)
        st.write("Aperçu des données importées:")
        st.write(st.session_state.data.head())

    col1, col2 = st.columns(2)
    if col2.button("Précédent"):
        st.session_state.page = "landing"
    if col1.button("Suivant"):
        st.session_state.page = "nettoyage"

# Fonction pour afficher la page de nettoyage des données
def nettoyage_page():
    st.title("Nettoyage des données")
    # Ajoutez ici le code pour nettoyer les données
    st.write("Nettoyage des données ici.")

    col1, col2 = st.columns(2)
    if col2.button("Précédent"):
        st.session_state.page = "importation"

# Fonction principale
def main():
    init_session()

    if st.session_state.page == "landing":
        landing_page()
    elif st.session_state.page == "importation":
        importation_page()
    elif st.session_state.page == "nettoyage":
        nettoyage_page()

if __name__ == "__main__":
    main()
