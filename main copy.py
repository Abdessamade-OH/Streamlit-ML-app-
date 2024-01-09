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
def import_csv(): 
    uploaded_file = st.file_uploader("Importer un fichier CSV", type=["csv"], key="file_uploader")
    if uploaded_file is not None:
        st.success("Vos données on été importées avec succès.")
        # Ajout de la ligne pour afficher st.session_state.data
        return pd.read_csv(uploaded_file)
    
# Fonction pour vérifier si des données existent dans st.session_state.data
def check_data_exists():
    return 'data' in st.session_state and st.session_state.data is not None


# Fonction pour visualiser les données
def visualise_tab():
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

def analyse_data():
    st.subheader("Analyse des données:")
    st.write("Nombre de valeurs manquantes par colonne:")
    missing_values = st.session_state.data.isnull().sum()
    st.write(missing_values)

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


def remplacer_nan():
    st.subheader("Remplacer NaN par 0:")
    if st.button("Remplacer NaN par 0"):
        original_data = original_data.fillna(0)
        st.session_state.data = original_data.copy()  # Mettez à jour la session_data avec les modifications
        st.success("Les NaN ont été remplacés par 0 avec succès.")


def remplacer_val():
    st.subheader("Remplacer les valeurs manquantes:")
    replace_option = st.selectbox("Choisissez une option de remplacement :", ["0", "Moyenne", "Médiane"])
    if st.button("Appliquer le remplacement"):
        if replace_option == "0":
            original_data = original_data.fillna(0)
            st.session_state.data = original_data.copy()  # Mettez à jour la session_data avec les modifications
            st.success("Les valeurs manquantes ont été remplacées par 0 avec succès.")
        elif replace_option == "Moyenne":
            original_data = original_data.fillna(original_data.mean())
            st.session_state.data = original_data.copy()  # Mettez à jour la session_data avec les modifications
            st.success("Les valeurs manquantes ont été remplacées par la moyenne avec succès.")
        elif replace_option == "Médiane":
            original_data = original_data.fillna(original_data.median())
            st.session_state.data = original_data.copy()  # Mettez à jour la session_data avec les modifications
            st.success("Les valeurs manquantes ont été remplacées par la médiane avec succès.")



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


def normaliser():
    st.subheader("Normaliser les données:")
    numeric_columns = original_data.select_dtypes(include=['number']).columns
    
    # Bouton pour appliquer la normalisation
    if st.button("Normaliser les données"):
        if not numeric_columns.empty:
            original_data[numeric_columns] = (original_data[numeric_columns] - original_data[numeric_columns].min()) / (original_data[numeric_columns].max() - original_data[numeric_columns].min())
            st.session_state.data = original_data.copy()  # Mettez à jour la session_data avec les modifications
            st.success("Les données ont été normalisées avec succès.")
            st.write("Aperçu des données après la normalisation:")
            st.write(original_data.head())
        else:
            st.warning("Aucune colonne numérique pour normaliser.")



# Fonction pour afficher la tab "Split"
def split_tab():
    # Vérifier si des données sont disponibles avant de procéder à la division
    if st.session_state.data is not None:
        # Sélection de la cible pour la prédiction
        st.subheader("Sélectionnez la colonne cible:")
        target_column = st.selectbox("Sélectionnez la colonne cible", st.session_state.data.columns, key="select_target_column")

        # Pourcentage de données pour l'ensemble d'entraînement
        st.subheader("Pourcentage pour l'ensemble d'entraînement:")
        train_percentage = st.slider("Pourcentage d'entraînement", 0, 100, 80, key="train_percentage")

        # Graine aléatoire
        st.subheader("Graine aléatoire (Random State):")
        random_state = st.number_input("Entrez la graine aléatoire", value=42, key="random_state")

        # Bouton pour diviser les données
        if st.button("Diviser les données"):
            # Sélectionner uniquement les colonnes numériques
            numeric_columns = st.session_state.data.select_dtypes(include=['number']).columns

            # Vérifier s'il y a des colonnes numériques pour éviter l'erreur
            if not numeric_columns.empty:
                # Diviser les données
                from sklearn.model_selection import train_test_split

                X_train, X_test, y_train, y_test = train_test_split(
                    st.session_state.data.drop(columns=[target_column]),
                    st.session_state.data[target_column],
                    test_size=train_percentage / 100,
                    random_state=random_state  # Utiliser la graine aléatoire spécifiée par l'utilisateur
                )

                # Afficher des informations sur les ensembles
                st.write("Ensemble d'entraînement:")
                st.write(X_train.head())
                st.write("Ensemble de test:")
                st.write(X_test.head())

                st.success("Les données ont été divisées avec succès.")
            else:
                st.warning("Aucune colonne numérique pour diviser.")
        else:
            st.warning("Veuillez sélectionner une colonne cible.")

    else:
        st.warning("Veuillez importer des données d'abord.")



# Fonction pour afficher les onglets
def display_tabs():
    tab1, tab2, tab3, tab4 = st.tabs(["Data", "Visualise", "Clean", "Split"])

    # onglet importation des données
    with tab1:
        st.header("Data")

        import_csv()

        with st.form(key="Exit", border=False):
            st.form_submit_button("Exit", on_click=lambda: st.session_state.update({"page": 0}))  # Revenir à la landing page

    
    # onglet visualisation des données
    with tab2:
        st.header("Visualise")

        visualise_tab()


    # onglet netoyage des données  
    with tab3:
        st.header("Clean")

        # Appel de la fonction d'importation des données avant d'exécuter les autres fonctions
        if not check_data_exists():
            st.session_state.data = import_csv()

        # Exécution de la fonction supprimer_col()
        supprimer_col()

        # Exécution de la fonction encodage() seulement si des données existent
        encodage()


        
        

    # onglet division des données  
    with tab4:
        st.header("Split")

        split_tab()

# Fonction principale
def main():
    init_session()

    if st.session_state.page == 0:
        landing_page()
    elif st.session_state.page == 1:
        display_tabs()

if __name__ == "__main__":
    main()