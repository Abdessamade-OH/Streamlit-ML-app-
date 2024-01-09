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


# Fonction pour importer fichier csv
def import_csv(tab_index): 
    uploaded_file = st.file_uploader("Importer un fichier CSV", type=["csv"], key=f"file_uploader_{tab_index}")
    if uploaded_file is not None:
        st.success("Vos données on été importées avec succès.")
        # Ajout de la ligne pour afficher st.session_state.data
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






# Fonction pour afficher les onglets
def display_tabs():
    tab1, tab2, tab3, tab4 = st.tabs(["Data", "Visualise", "Clean", "Split"])

    # onglet importation des données
    with tab1:
        st.header("Data")

        import_csv(1) # Pass tab index to generate a unique key

        with st.form(key="Exit", border=False):
            st.form_submit_button("Exit", on_click=lambda: st.session_state.update({"page": 0}))  # Revenir à la landing page

    
    # onglet visualisation des données
    with tab2:
        st.header("Visualise")



    # onglet netoyage des données  
    with tab3:
        st.header("Clean")

        # Appel de la fonction d'importation des données avant d'exécuter les autres fonctions
        if not check_data_exists():
            st.session_state.data = import_csv(3)

        # Exécution de la fonction supprimer_col()
        supprimer_col()

        # Exécution de la fonction encodage() seulement si des données existent
        encodage()


        
        

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