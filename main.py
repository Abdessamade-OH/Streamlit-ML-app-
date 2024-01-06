import streamlit as st
import pandas as pd

def init_session():
    if "page" not in st.session_state:
        st.session_state.page = "landing"
        st.session_state.data = None
        st.session_state.current_tab_index = 0
        st.session_state.next_tab_enabled = False

def landing_page():
    st.title("Bienvenue dans notre application de Machine Learning.")
    st.write("Cliquez sur le bouton ci-dessous pour commencer.")
    
    if st.button("Get Started"):
        st.session_state.page = "get_started_page"

def get_started_page():
    st.title("Get Started Page")

    # Onglets pour importer et nettoyer les données
    tabs = ["Importer les données", "Visualiser et Nettoyer les données"]

    # Afficher les onglets horizontalement
    col1, col2 = st.columns(len(tabs))
    for i, tab in enumerate(tabs):
        if i == st.session_state.current_tab_index:
            col1.button(tab, key=f"tab_button_{i}", disabled=True)
        else:
            if col1.button(tab, key=f"tab_button_{i}"):
                st.session_state.current_tab_index = i
                st.session_state.next_tab_enabled = False

    # Afficher le contenu de l'onglet actuel
    if st.session_state.current_tab_index == 0:
        importation_tab()
    elif st.session_state.current_tab_index == 1:
        visualisation_tab()

    # Bouton Suivant à la fin de chaque onglet
    if st.button("Suivant") and st.session_state.next_tab_enabled:
        st.session_state.current_tab_index += 1
        st.session_state.next_tab_enabled = False

def importation_tab():
    st.subheader("Importation des données")

    # Ajouter le code pour importer les données
    uploaded_file = st.file_uploader("Importer un fichier CSV", type=["csv"])

    if uploaded_file is not None:
        st.session_state.data = pd.read_csv(uploaded_file)
        st.write("Aperçu des données importées:")
        st.write(st.session_state.data.head())

def visualisation_tab():
    st.subheader("Visualisation et Nettoyage des données")

    # Visualiser les données et ajouter les étapes de nettoyage ici
    if st.session_state.data is not None:
        st.write("Aperçu des données:")
        st.write(st.session_state.data)

        # Ajouter les étapes de nettoyage des données

def main():
    init_session()

    if st.session_state.page == "landing":
        landing_page()
    elif st.session_state.page == "get_started_page":
        get_started_page()

if __name__ == "__main__":
    main()
