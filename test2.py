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

    # Liste des onglets
    tabs = ["Importation des données", "Nettoyage des données"]
    
    # Afficher les onglets sur le côté gauche
    selected_tab = st.sidebar.radio("Étapes", tabs)

    # Mettre en surbrillance l'onglet actuel avec une couleur différente
    for tab in tabs:
        if selected_tab == tab:
            st.sidebar.markdown(f"**{tab}**", unsafe_allow_html=True)
        else:
            st.sidebar.markdown(tab)

    if st.button("Suivant"):
        st.session_state.page = "nettoyage"

# Fonction pour afficher la page de nettoyage des données
def nettoyage_page():
    st.title("Nettoyage des données")
    # Ajoutez ici le code pour nettoyer les données
    st.write("Nettoyage des données ici.")

    # Liste des onglets
    tabs = ["Importation des données", "Nettoyage des données"]
    
    # Afficher les onglets sur le côté gauche
    selected_tab = st.sidebar.radio("Étapes", tabs)

    # Mettre en surbrillance l'onglet actuel avec une couleur différente
    for tab in tabs:
        if selected_tab == tab:
            st.sidebar.markdown(f"**{tab}**", unsafe_allow_html=True)
        else:
            st.sidebar.markdown(tab)

    if st.button("Précédent"):
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
