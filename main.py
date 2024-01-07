import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Définir la largeur de la page
st.set_page_config(layout="wide")

# Fonction pour initialiser la session
def init_session():
    if "page" not in st.session_state:
        st.session_state.page = 0  # Utiliser 0 comme numéro initial de page
        st.session_state.data = None  # Initialiser la variable "data" à None
        st.session_state.columns_to_drop = []  # Initialiser la liste des colonnes à supprimer

# Fonction pour afficher la landing page
def landing_page():

    with open('style.css') as f:
        css = f.read()

    st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)
    st.markdown('<p class="font">Welcome !</p>', unsafe_allow_html=True)

    st.title("Bienvenue dans notre application de Machine Learning.")
    st.write("Cliquez sur le bouton ci-dessous pour commencer.")
    
    st.text("")  # Crée une séparation visuelle sans bordure
    with st.form(key="landing_form", border=False):
        st.form_submit_button("Get Started", on_click=lambda: st.session_state.update({"page": 1}))

# Fonction pour afficher les onglets
def display_tabs():
    tab1, tab2, tab3, tab4 = st.tabs(["Data", "Visualise", "Clean", "Split"])

    # onglet importation des données
    with tab1:
        st.header("Data")

        # Ajout de la fonctionnalité pour importer un fichier CSV
        uploaded_file = st.file_uploader("Importer un fichier CSV", type=["csv"])
        if uploaded_file is not None:
            st.session_state.data = pd.read_csv(uploaded_file)
            st.success("Vos données on été importées avec succès.")

        with st.form(key="Exit", border=False):
            st.form_submit_button("Exit", on_click=lambda: st.session_state.update({"page": 0}))  # Revenir à la landing page

    
    # onglet visualisation des données
    with tab2:
        st.header("Visualise")

        # Vérifier si des données sont disponibles avant de procéder à la visualisation
        if st.session_state.data is not None:
            # Sélection des colonnes pour la visualisation
            st.subheader("Sélectionnez deux colonnes pour la visualisation:")
            st.session_state.selected_columns = st.multiselect("Sélectionnez deux colonnes", st.session_state.data.columns, key="select_columns")

            # Sélection du type de graphe
            chart_type = st.selectbox("Sélectionnez le type de graphe", ["Scatter Plot", "Line Plot", "Bar Plot"])

            # Affichage du graphe en fonction du type choisi
            if st.button("Afficher le graphe"):
                if len(st.session_state.selected_columns) == 2:
                    if chart_type == "Scatter Plot":
                        fig, ax = plt.subplots()
                        sns.scatterplot(x=st.session_state.selected_columns[0], y=st.session_state.selected_columns[1], data=st.session_state.data, ax=ax)
                        st.pyplot(fig)
                    elif chart_type == "Line Plot":
                        fig, ax = plt.subplots()
                        sns.lineplot(x=st.session_state.selected_columns[0], y=st.session_state.selected_columns[1], data=st.session_state.data, ax=ax)
                        st.pyplot(fig)
                    elif chart_type == "Bar Plot":
                        fig, ax = plt.subplots()
                        sns.barplot(x=st.session_state.selected_columns[0], y=st.session_state.selected_columns[1], data=st.session_state.data, ax=ax)
                        st.pyplot(fig)
                    else:
                        st.warning("Veuillez sélectionner un type de graphe valide.")
                else:
                    st.warning("Veuillez sélectionner exactement deux colonnes pour la visualisation.")
        else:
            st.warning("Veuillez importer des données d'abord.")


    # onglet netoyage des données  
    with tab3:
        st.header("Clean")

        # Affichage du nombre de valeurs manquantes
        if st.session_state.data is not None:
            st.subheader("Analyse des données:")
            st.write("Nombre de valeurs manquantes par colonne:")
            missing_values = st.session_state.data.isnull().sum()
            st.write(missing_values)

            # Sélection des colonnes à supprimer
            st.subheader("Supprimer des colonnes:")
            selected_columns_to_drop = st.multiselect("Sélectionnez les colonnes à supprimer", st.session_state.data.columns)
            if st.button("Supprimer les colonnes sélectionnées"):
                if selected_columns_to_drop:
                    st.session_state.data = st.session_state.data.drop(columns=selected_columns_to_drop)
                    st.success("Les colonnes sélectionnées ont été supprimées avec succès.")
                    # Afficher l'aperçu des données après les remplacements, l'encodage et la suppression des colonnes
                    st.write("Aperçu des données après la suppression des colonnes:")
                    st.write(st.session_state.data.head())
                else:
                    st.warning("Veuillez sélectionner au moins une colonne.")

            # Bouton pour remplacer NaN par 0
            if st.button("Remplacer NaN par 0"):
                st.session_state.data = st.session_state.data.fillna(0)
                st.success("Les NaN ont été remplacés par 0 avec succès.")

            # Bouton pour remplacer les valeurs manquantes
            st.subheader("Remplacer les valeurs manquantes:")
            replace_option = st.selectbox("Choisissez une option de remplacement :", ["0", "Moyenne", "Médiane"])
            if st.button("Appliquer le remplacement"):
                if replace_option == "0":
                    st.session_state.data = st.session_state.data.fillna(0)
                    st.success("Les valeurs manquantes ont été remplacées par 0 avec succès.")
                elif replace_option == "Moyenne":
                    st.session_state.data = st.session_state.data.fillna(st.session_state.data.mean())
                    st.success("Les valeurs manquantes ont été remplacées par la moyenne avec succès.")
                elif replace_option == "Médiane":
                    st.session_state.data = st.session_state.data.fillna(st.session_state.data.median())
                    st.success("Les valeurs manquantes ont été remplacées par la médiane avec succès.")
                else:
                    st.warning("Veuillez sélectionner une option de remplacement valide.")


            # Encodage des variables catégorielles
            categorical_cols = st.session_state.data.select_dtypes(include=['object']).columns.tolist()
            if categorical_cols:
                st.subheader("Encodage des variables catégorielles:")
                encoding_option = st.selectbox("Choisissez une option d'encodage :", ["One-Hot", "Ordinal"])
                if st.button("Appliquer l'encodage"):
                    if encoding_option == "One-Hot":
                        st.session_state.data = pd.get_dummies(st.session_state.data, columns=categorical_cols, drop_first=True)
                        st.success("Encodage One-Hot appliqué avec succès.")
                    elif encoding_option == "Ordinal":
                        # Implémentez ici l'encodage ordinal si nécessaire
                        st.warning("L'encodage ordinal n'est pas encore implémenté.")
                    else:
                        st.warning("Veuillez sélectionner une option d'encodage valide.")
            else:
                st.warning("Aucune variable catégorielle à encoder.")

            # Bouton pour normaliser les données
            if st.button("Normaliser les données"):
                # Sélectionner uniquement les colonnes numériques
                numeric_columns = st.session_state.data.select_dtypes(include=['number']).columns

                # Vérifier s'il y a des colonnes numériques pour éviter l'erreur
                if not numeric_columns.empty:
                    st.session_state.data[numeric_columns] = (st.session_state.data[numeric_columns] - st.session_state.data[numeric_columns].min()) / (st.session_state.data[numeric_columns].max() - st.session_state.data[numeric_columns].min())
                    st.success("Les données ont été normalisées avec succès.")
                else:
                    st.warning("Aucune colonne numérique pour normaliser.")




            # Affichage de l'aperçu des données après les remplacements
            st.write("Aperçu des données après les remplacements:")
            st.write(st.session_state.data.head())

        else:
            st.warning("Veuillez importer des données d'abord.")


    # onglet division des données  
    with tab4:
        st.header("Split")

# Fonction principale
def main():
    init_session()

    if st.session_state.page == 0:
        landing_page()
    elif st.session_state.page == 1:
        display_tabs()

if __name__ == "__main__":
    main()