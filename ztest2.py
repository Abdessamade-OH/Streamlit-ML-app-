import streamlit as st
import pandas as pd

# Fonction pour importer fichier CSV
def import_csv(): 
    uploaded_file = st.file_uploader("Importer un fichier CSV", type=["csv"])
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)

# Fonction pour vérifier si des données existent dans st.session_state.data
def check_data_exists():
    return 'data' in st.session_state and st.session_state.data is not None

# Fonction pour supprimer des colonnes
def supprimer_col():
    st.subheader("Supprimer des colonnes:")
    
    if not check_data_exists():
        st.warning("Aucune donnée valide n'a été importée. Veuillez importer un fichier CSV.")
        return
    
    selected_columns_to_drop = st.multiselect("Sélectionnez les colonnes à supprimer", st.session_state.data.columns)
    if st.button("Supprimer les colonnes sélectionnées"):
        if selected_columns_to_drop:
            st.session_state.data = st.session_state.data.drop(columns=selected_columns_to_drop)
            st.success("Les colonnes sélectionnées ont été supprimées avec succès.")
            st.write("Aperçu des données après suppression :")
            st.write(st.session_state.data.head())
        else:
            st.warning("Veuillez sélectionner au moins une colonne à supprimer.")

# Fonction pour encodage des variables catégorielles
def encodage():
    st.subheader("Encodage des variables catégorielles:")
    
    if not check_data_exists():
        st.warning("Aucune donnée valide n'a été importée. Veuillez importer un fichier CSV.")
        return
    
    categorical_cols = st.session_state.data.select_dtypes(include=['object']).columns.tolist()
    
    if categorical_cols:
        encoding_option = st.selectbox("Choisissez une option d'encodage :", ["One-Hot", "Ordinal"])
        if st.button("Appliquer l'encodage"):
            if encoding_option == "One-Hot":
                st.session_state.data = pd.get_dummies(st.session_state.data, columns=categorical_cols, drop_first=True)
                st.success("Encodage One-Hot appliqué avec succès.")
                st.write("Aperçu des données après l'encodage:")
                st.write(st.session_state.data.head())
            elif encoding_option == "Ordinal":
                # Implémentez ici l'encodage ordinal si nécessaire
                st.warning("L'encodage ordinal n'est pas encore implémenté.")
            else:
                st.warning("Veuillez sélectionner une option d'encodage valide.")
    else:
        st.warning("Aucune variable catégorielle à encoder.")

# Appel de la fonction d'importation des données avant d'exécuter les autres fonctions
if not check_data_exists():
    st.session_state.data = import_csv()

# Exécution de la fonction supprimer_col()
supprimer_col()

# Exécution de la fonction encodage() seulement si des données existent
encodage()
