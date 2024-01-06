import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Fonction pour initialiser la session
def init_session():
    if "page" not in st.session_state:
        st.session_state.page = 0  # Utiliser 0 comme numéro initial d'onglet
        st.session_state.data = None

# Fonction pour afficher la landing page
def landing_page():
    st.title("Bienvenue dans notre application de Machine Learning.")
    st.write("Cliquez sur le bouton ci-dessous pour commencer.")
    
    st.text("")  # Crée une séparation visuelle sans bordure
    with st.form(key="landing_form", border=False):
        st.form_submit_button("Get Started", on_click=lambda: st.session_state.update({"page": 1}))

# Fonction pour afficher la page d'importation des données
def importation_page():
    st.title("Importation des données")
    uploaded_file = st.file_uploader("Importer un fichier CSV", type=["csv"])

    if uploaded_file is not None:
        st.session_state.data = pd.read_csv(uploaded_file)
        st.write("Aperçu des données importées:")
        st.write(st.session_state.data.head())

    # Liste des onglets
    tabs = ["Importation des données", "Nettoyage des données", "Visualisation des données"]
    
    # Afficher les onglets avec la tab actuelle en couleur
    for i, tab in enumerate(tabs):
        st.sidebar.markdown(f'<div style="{get_tab_style(i)}">{tab}</div>', unsafe_allow_html=True)

    st.text("")  # Crée une séparation visuelle sans bordure
    with st.form(key="importation_form", border=False):
        st.form_submit_button("Exit", on_click=lambda: st.session_state.update({"page": 0}))
        st.form_submit_button("Suivant", on_click=lambda: st.session_state.update({"page": 2}), disabled=True)

# Fonction pour afficher la page de nettoyage des données
def nettoyage_page():
    st.title("Nettoyage des données")
    # Ajoutez ici le code pour nettoyer les données
    st.write("Nettoyage des données ici.")

    # Liste des onglets
    tabs = ["Importation des données", "Nettoyage des données", "Visualisation des données"]
    
    # Afficher les onglets avec la tab actuelle en couleur
    for i, tab in enumerate(tabs):
        st.sidebar.markdown(f'<div style="{get_tab_style(i)}">{tab}</div>', unsafe_allow_html=True)

    st.text("")  # Crée une séparation visuelle sans bordure
    with st.form(key="nettoyage_form", border=False):
        st.form_submit_button("Précédent", on_click=lambda: st.session_state.update({"page": 1}))
        st.form_submit_button("Suivant", on_click=lambda: st.session_state.update({"page": 3}))

# Fonction pour afficher la page de visualisation des données
def visualisation_page():
    st.title("Visualisation des données")
    if st.session_state.data is not None:
        st.subheader("Visualisation des données:")
        # Ajoutez ici le code pour visualiser les données, par exemple avec Seaborn ou Matplotlib
        plt.figure(figsize=(8, 6))
        sns.heatmap(st.session_state.data.corr(), annot=True, cmap="coolwarm")
        st.pyplot()
    else:
        st.warning("Veuillez importer des données d'abord.")

    # Liste des onglets
    tabs = ["Importation des données", "Nettoyage des données", "Visualisation des données"]
    
    # Afficher les onglets avec la tab actuelle en couleur
    for i, tab in enumerate(tabs):
        st.sidebar.markdown(f'<div style="{get_tab_style(i)}">{tab}</div>', unsafe_allow_html=True)

    st.text("")  # Crée une séparation visuelle sans bordure
    with st.form(key="visualisation_form", border=False):
        st.form_submit_button("Précédent", on_click=lambda: st.session_state.update({"page": 2}))

# Fonction pour obtenir le style de l'onglet en fonction de son numéro
def get_tab_style(tab_number):
    if st.session_state.page == tab_number+1:
        return "color:red; text-align: left;"
    else:
        return "text-align: left;"

# Fonction principale
def main():
    init_session()

    if st.session_state.page == 0:
        landing_page()
    elif st.session_state.page == 1:
        importation_page()
    elif st.session_state.page == 2:
        nettoyage_page()
    elif st.session_state.page == 3:
        visualisation_page()

if __name__ == "__main__":
    main()
