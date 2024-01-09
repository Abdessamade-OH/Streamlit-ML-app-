import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
        st.session_state.split_data = None  # Initialiser la variable "split_data" à None

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


# Fonction pour visualiser les données
def visualize_data():
    # Vérifier si des données nettoyées sont disponibles
    if st.session_state.modified_data is not None:
        data_to_visualize = st.session_state.modified_data
    elif st.session_state.data is not None:
        data_to_visualize = st.session_state.data
    else:
        st.warning("Aucune donnée n'est disponible. Veuillez importer un fichier CSV dans l'onglet 'Data' avant de visualiser.")
        return

    # Sélection des colonnes pour la visualisation
    st.subheader("Sélectionnez deux colonnes pour la visualisation:")
    st.session_state.selected_columns = st.multiselect("Sélectionnez deux colonnes", data_to_visualize.columns, key="select_columns")

    # Sélection du type de graphe
    chart_type = st.selectbox("Sélectionnez le type de graphe", ["Scatter Plot", "Line Plot", "Bar Plot"])

    # Affichage du graphe en fonction du type choisi
    if st.button("Afficher le graphe"):
        if len(st.session_state.selected_columns) == 2:
            if chart_type == "Scatter Plot":
                fig, ax = plt.subplots()
                sns.scatterplot(x=st.session_state.selected_columns[0], y=st.session_state.selected_columns[1], data=data_to_visualize, ax=ax)
                st.pyplot(fig)
            elif chart_type == "Line Plot":
                fig, ax = plt.subplots()
                sns.lineplot(x=st.session_state.selected_columns[0], y=st.session_state.selected_columns[1], data=data_to_visualize, ax=ax)
                st.pyplot(fig)
            elif chart_type == "Bar Plot":
                fig, ax = plt.subplots()
                sns.barplot(x=st.session_state.selected_columns[0], y=st.session_state.selected_columns[1], data=data_to_visualize, ax=ax)
                st.pyplot(fig)
            else:
                st.warning("Veuillez sélectionner un type de graphe valide.")
        else:
            st.warning("Veuillez sélectionner exactement deux colonnes pour la visualisation.")



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


# Fonction pour remplacer les valeurs manquantes
def remplacer_valeurs_manquantes():
    st.subheader("Remplacer les valeurs manquantes:")

    if st.session_state.modified_data is not None:
        data_to_modify = st.session_state.modified_data
    elif st.session_state.data is not None:
        data_to_modify = st.session_state.data
    else:
        st.warning("Aucune donnée n'est disponible. Veuillez importer un fichier CSV dans l'onglet 'Data' avant de remplacer les valeurs manquantes.")
        return

    replace_option = st.selectbox("Choisissez une option de remplacement :", ["0", "Moyenne", "Médiane"])
    selected_columns = st.multiselect("Sélectionnez les colonnes à modifier", data_to_modify.columns)

    if st.button("Appliquer le remplacement"):
        if selected_columns:
            if replace_option == "0":
                data_to_modify[selected_columns] = data_to_modify[selected_columns].fillna(0)
                st.session_state.modified_data = data_to_modify
                st.success("Les valeurs manquantes ont été remplacées par 0 avec succès.")
            elif replace_option == "Moyenne":
                data_to_modify[selected_columns] = data_to_modify[selected_columns].fillna(data_to_modify[selected_columns].mean())
                st.session_state.modified_data = data_to_modify
                st.success("Les valeurs manquantes ont été remplacées par la moyenne avec succès.")
            elif replace_option == "Médiane":
                data_to_modify[selected_columns] = data_to_modify[selected_columns].fillna(data_to_modify[selected_columns].median())
                st.session_state.modified_data = data_to_modify
                st.success("Les valeurs manquantes ont été remplacées par la médiane avec succès.")
        else:
            st.warning("Veuillez sélectionner au moins une colonne à modifier.")



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
        selected_columns = st.multiselect("Sélectionnez les colonnes à encoder", categorical_cols)
        
        if selected_columns:
            encoding_option = st.selectbox("Choisissez une option d'encodage :", ["One-Hot", "Ordinal"])
            if st.button("Appliquer l'encodage"):
                if encoding_option == "One-Hot":
                    data_to_modify = pd.get_dummies(data_to_modify, columns=selected_columns, drop_first=True)
                    st.session_state.modified_data = data_to_modify
                    st.success("Encodage One-Hot appliqué avec succès.")
                    st.write("Aperçu des données après l'encodage:")
                    st.write(data_to_modify.head())
                elif encoding_option == "Ordinal":
                    # Implement ordinal encoding here if needed
                    st.warning("L'encodage ordinal n'est pas encore implémenté.")
                else:
                    st.warning("Veuillez sélectionner une option d'encodage valide.")
        else:
            st.warning("Veuillez sélectionner au moins une colonne à encoder.")
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
    
    if st.session_state.modified_data is not None:
        data_to_split = st.session_state.modified_data
    elif st.session_state.data is not None:
        data_to_split = st.session_state.data
        st.warning("Vous pouvez nettoyer vos données dans l'onglet 'Clean' avant de les diviser.")
    else:
        st.warning("Aucune donnée n'est disponible. Veuillez importer un fichier CSV dans l'onglet 'Data' avant de diviser.")
        return
    
    target_variable = st.selectbox("Sélectionnez la variable cible :", data_to_split.columns)
    random_state = st.number_input("Sélectionnez la valeur pour 'random_state' :", min_value=0, step=1, value=42)
    test_size_percentage = st.slider("Sélectionnez la proportion d'entraînement :", min_value=10, max_value=90, step=10, value=80)
    
    if st.button("Diviser les données"):
        test_size = test_size_percentage / 100.0  # Convert percentage to fraction
        X_train, X_test, y_train, y_test = train_test_split(
            data_to_split.drop(columns=[target_variable]),
            data_to_split[target_variable],
            test_size=test_size,
            random_state=random_state
        )
        st.session_state.split_data = {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test
        }
        st.success("Les données ont été divisées avec succès.")
        
        st.write("Taille de l'ensemble d'entraînement:", len(X_train))
        st.write("Taille de l'ensemble de test:", len(X_test))
        
        st.write("Aperçu de l'ensemble d'entraînement:")
        st.write(X_train.head())
        st.write(y_train.head())
        
        st.write("Aperçu de l'ensemble de test:")
        st.write(X_test.head())
        st.write(y_test.head())


# Fonction pour choisir le type de problème
def choix_du_probleme():
    if st.session_state.split_data is None:
        st.warning("Veuillez diviser les données dans l'onglet 'Split' avant de continuer.")
        return

    st.subheader("Choisissez le type de problème:")
    supervised_option = st.radio("Supervisé ou non supervisé ?", ["Supervisé", "Non Supervisé"])

    if supervised_option == "Supervisé":
        probleme_type = st.radio("Classification ou Régression ?", ["Classification", "Régression"])
        if st.button("Continuer"):
            st.session_state.problem_type = {"Supervisé": True, "Type": probleme_type}
            st.session_state.classification_or_regression = probleme_type  # Save the classification or regression type
            st.success(f"Vous avez choisi un problème de {probleme_type.lower()} supervisée.")
    elif supervised_option == "Non Supervisé":
        if st.button("Continuer"):
            st.session_state.problem_type = {"Supervisé": False, "Type": "Non Supervisé"}
            st.session_state.classification_or_regression = None  # Clear the classification or regression type
            st.success("Vous avez choisi un problème non supervisé.")
    else:
        st.warning("Veuillez sélectionner une option valide.")








# Fonction pour afficher les onglets
def display_tabs():
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Data", "Visualise", "Clean", "Split", "Choix du problème", "PCA & SMOTE"])

    # onglet importation des données
    with tab1:
        st.header("Data")

        st.session_state.data = import_csv()  # Assign the imported data to st.session_state.data

        with st.form(key="Exit", border=False):
            st.form_submit_button("Exit", on_click=lambda: st.session_state.update({"page": 0}))  # Revenir à la landing page

    # Onglet visualisation des données
    with tab2:
        st.header("Visualize")

        visualize_data()

    
    # onglet netoyage des données  
    with tab3:
        st.header("Clean")

        # Appel de la fonction d'importation des données avant d'exécuter les autres fonctions
        if not check_data_exists():
            st.warning("Veuillez importer un fichier CSV dans l'onglet 'Data' avant de nettoyer.")
        else:
            # Exécution de la fonction supprimer_col()
            supprimer_col()

            # Exécution de la fonction remplacer_valeurs_manquantes()
            remplacer_valeurs_manquantes()

            # Exécution de la fonction encodage() seulement si des données existent
            encodage()

            # Exécution de la fonction normaliser() seulement si des données existent
            normaliser()

    # onglet division des données  
    with tab4:
        st.header("Split")

        # Exécution de la fonction split_data()
        split_data()

    # Onglet choix du problème
    with tab5:
        st.header("Choix du problème")

        choix_du_probleme()

    with tab6:
        st.write(st.session_state.split_data)



# Fonction principale
def main():
    init_session()

    if st.session_state.page == 0:
        landing_page()
    elif st.session_state.page == 1:
        display_tabs()

if __name__ == "__main__":
    main()
