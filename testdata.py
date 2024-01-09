import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split

# Définir la largeur de la page
st.set_page_config(layout="wide")

# Fonction pour initialiser la session
def init_session():
    if "page" not in st.session_state:
        st.session_state.page = 0  # Utiliser 0 comme numéro initial de page
        st.session_state.data = None  # Initialiser la variable "data" à None
        st.session_state.columns_to_drop = []  # Initialiser la liste des colonnes à supprimer
        st.session_state.modified_data = None  # Initialiser la variable pour le DataFrame modifié

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

# Fonction pour importer fichier csv
def import_csv(): 
    uploaded_file = st.file_uploader("Importer un fichier CSV", type=["csv"])
    if uploaded_file is not None:
        st.success("Vos données ont été importées avec succès.")
        st.session_state.data = pd.read_csv(uploaded_file)  # Variable pour stocker les données importées
        return st.session_state.data

# Fonction pour vérifier si des données existent dans st.session_state.data
def check_data_exists():
    return st.session_state.data is not None

# Fonction pour supprimer des colonnes
def supprimer_col():
    st.subheader("Supprimer des colonnes:")

    if st.session_state.modified_data is not None:
        data_to_modify = st.session_state.modified_data
    elif st.session_state.data is not None:
        data_to_modify = st.session_state.data
    else:
        st.warning("Aucune donnée n'est disponible. Veuillez importer un fichier CSV dans l'onglet 'Data' avant de supprimer des colonnes.")
        return

    selected_columns_to_drop = st.multiselect("Sélectionnez les colonnes à supprimer", data_to_modify.columns)
    if st.button("Supprimer les colonnes sélectionnées"):
        if selected_columns_to_drop:
            data_to_modify = data_to_modify.drop(columns=selected_columns_to_drop)
            st.session_state.modified_data = data_to_modify
            st.success("Les colonnes sélectionnées ont été supprimées avec succès.")
            st.write("Aperçu des données après suppression :")
            st.write(data_to_modify.head())
        else:
            st.warning("Veuillez sélectionner au moins une colonne à supprimer.")


# Fonction pour encodage des variables catégorielles
def encodage():
    st.subheader("Encodage des variables catégorielles:")

    if st.session_state.modified_data is not None:
        data_to_modify = st.session_state.modified_data
    elif st.session_state.data is not None:
        data_to_modify = st.session_state.data
    else:
        st.warning("Aucune donnée n'est disponible. Veuillez importer un fichier CSV dans l'onglet 'Data' avant de faire l'encodage.")
        return

    categorical_cols = data_to_modify.select_dtypes(include=['object']).columns.tolist()

    if categorical_cols:
        encoding_option = st.selectbox("Choisissez une option d'encodage :", ["One-Hot", "Ordinal"])
        if st.button("Appliquer l'encodage"):
            if encoding_option == "One-Hot":
                data_to_modify = pd.get_dummies(data_to_modify, columns=categorical_cols, drop_first=True)
                st.session_state.modified_data = data_to_modify
                st.success("Encodage One-Hot appliqué avec succès.")
                st.write("Aperçu des données après l'encodage:")
                st.write(data_to_modify.head())
            elif encoding_option == "Ordinal":
                # Implémentez ici l'encodage ordinal si nécessaire
                st.warning("L'encodage ordinal n'est pas encore implémenté.")
            else:
                st.warning("Veuillez sélectionner une option d'encodage valide.")
    else:
        st.warning("Aucune variable catégorielle à encoder.")


# Fonction pour normaliser les variables numériques
def normaliser():
    st.subheader("Normalisation des variables numériques:")

    if st.session_state.modified_data is not None:
        data_to_modify = st.session_state.modified_data
    elif st.session_state.data is not None:
        data_to_modify = st.session_state.data
    else:
        st.warning("Aucune donnée n'est disponible. Veuillez importer un fichier CSV dans l'onglet 'Data' avant de normaliser.")
        return

    data_copy = data_to_modify.copy()  # Create a copy of the data before modification
    numerical_cols = data_copy.select_dtypes(include=['number']).columns.tolist()

    if numerical_cols:
        if st.button("Appliquer la normalisation"):
            data_copy[numerical_cols] = (data_copy[numerical_cols] - data_copy[numerical_cols].min()) / (data_copy[numerical_cols].max() - data_copy[numerical_cols].min())
            st.session_state.modified_data = data_copy
            st.success("Normalisation appliquée avec succès.")
            st.write("Aperçu des données après la normalisation:")
            st.write(data_copy.head())
    else:
        st.warning("Aucune variable numérique à normaliser.")



# Fonction pour diviser les données en ensemble d'entraînement et de test
def split_data():
    
    if st.session_state.modified_data is None:
        st.warning("Aucune donnée nettoyée n'est disponible. Veuillez nettoyer et encoder vos données dans l'onglet 'Clean' avant de diviser.")
        return
    
    target_variable = st.selectbox("Sélectionnez la variable cible :", st.session_state.modified_data.columns)
    random_state = st.number_input("Sélectionnez la valeur pour 'random_state' :", min_value=0, step=1, value=42)
    test_size_percentage = st.slider("Sélectionnez la proportion d'entraînement :", min_value=10, max_value=90, step=10, value=80)
    
    if st.button("Diviser les données"):
        test_size = test_size_percentage / 100.0  # Convert percentage to fraction
        X_train, X_test = train_test_split(st.session_state.modified_data, test_size=test_size, random_state=random_state)
        st.session_state.split_data = {
            "X_train": X_train.drop(columns=[target_variable]),
            "y_train": X_train[target_variable],
            "X_test": X_test.drop(columns=[target_variable]),
            "y_test": X_test[target_variable]
        }
        st.success("Les données ont été divisées avec succès.")
        
        st.write("Aperçu de l'ensemble d'entraînement:")
        st.write(X_train.head())
        
        st.write("Aperçu de l'ensemble de test:")
        st.write(X_test.head())




# Fonction pour afficher les onglets
def display_tabs():
    tab1, tab2, tab3, tab4 = st.tabs(["Data", "Visualise", "Clean", "Split"])

    # onglet importation des données
    with tab1:
        st.header("Data")

        st.session_state.data = import_csv()  # Assign the imported data to st.session_state.data

        with st.form(key="Exit", border=False):
            st.form_submit_button("Exit", on_click=lambda: st.session_state.update({"page": 0}))  # Revenir à la landing page
    
    # onglet netoyage des données  
    with tab3:
        st.header("Clean")

        # Appel de la fonction d'importation des données avant d'exécuter les autres fonctions
        if not check_data_exists():
            st.warning("Veuillez importer un fichier CSV dans l'onglet 'Data' avant de nettoyer.")
        else:
            # Exécution de la fonction supprimer_col()
            supprimer_col()

            # Exécution de la fonction encodage() seulement si des données existent
            encodage()

            # Exécution de la fonction normaliser() seulement si des données existent
            normaliser()

    # onglet division des données  
    with tab4:
        st.header("Split")

        # Exécution de la fonction split_data()
        split_data()



# Fonction principale
def main():
    init_session()

    if st.session_state.page == 0:
        landing_page()
    elif st.session_state.page == 1:
        display_tabs()

if __name__ == "__main__":
    main()
