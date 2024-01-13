import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA


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
        st.session_state.user_choice = None # Initialiser la variable "user_choice" à None
        st.session_state.resampled_data = None # Initialiser la variable "resampled_data" à None

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


# Fonction pour choisir le type de problème
def choix_du_probleme():
    # Vérifier si des données sont disponibles
    if st.session_state.data is not None:
        # Affichage de la partie pour choisir le type de problème
        st.subheader("Choisissez le type de problème:")
        
        supervised_options = ["Classification Supervisé", "Regression Supervisé", "Classification Non Supervisé"]
        user_choice = st.selectbox("Supervisé ou non supervisé ?", supervised_options)

        if st.button("Continuer"):
            # Enregistrement du choix dans une variable de session
            st.session_state.user_choice = user_choice
            st.success(f"Vous avez choisi un problème de {st.session_state.user_choice} .")
    else:
        pass




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
    selection_column, display_column = st.columns(2)

    # Dans la colonne de sélection, permettez à l'utilisateur de choisir les colonnes
    st.session_state.selected_columns = selection_column.multiselect("Sélectionnez deux colonnes", data_to_visualize.columns, key="select_columns")

    # Dans la colonne de sélection, permettez à l'utilisateur de choisir le graphique
    chart_type = selection_column.selectbox("Sélectionnez le type de graphe", ["Scatter Plot", "Line Plot", "Bar Plot"])

    # Dans la colonne de sélection, affichez le bouton pour afficher le graphique
    if selection_column.button("Afficher le graphe"):
        if len(st.session_state.selected_columns) == 2:
            # Créer la figure et les axes avec la taille spécifiée
            fig, ax = plt.subplots(figsize=(10, 6))
            
            if chart_type == "Scatter Plot":
                sns.scatterplot(x=st.session_state.selected_columns[0], y=st.session_state.selected_columns[1], data=data_to_visualize, ax=ax)
            elif chart_type == "Line Plot":
                sns.lineplot(x=st.session_state.selected_columns[0], y=st.session_state.selected_columns[1], data=data_to_visualize, ax=ax)
            elif chart_type == "Bar Plot":
                sns.barplot(x=st.session_state.selected_columns[0], y=st.session_state.selected_columns[1], data=data_to_visualize, ax=ax)
            else:
                st.warning("Veuillez sélectionner un type de graphe valide.")
            
            # Afficher le graphique dans la colonne d'affichage
            display_column.pyplot(fig)
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
        else:
            st.warning("Veuillez sélectionner au moins une colonne à supprimer.")


# Fonction pour supprimer les lignes dupliquées
def supprimer_lignes_dupliquees():
    st.subheader("Supprimer les lignes dupliquées:")

    if st.session_state.modified_data is not None:
        data_to_modify = st.session_state.modified_data
    elif st.session_state.data is not None:
        data_to_modify = st.session_state.data
    else:
        st.warning("Aucune donnée n'est disponible. Veuillez importer un fichier CSV dans l'onglet 'Data' avant de supprimer les lignes dupliquées.")
        return

    if st.button("Supprimer les lignes dupliquées"):
        data_to_modify = data_to_modify.drop_duplicates()
        st.session_state.modified_data = data_to_modify
        st.success("Les lignes dupliquées ont été supprimées avec succès.")
        st.write("Aperçu des données après suppression des lignes dupliquées :")





# Fonction pour remplacer les valeurs manquantes, NaN ou valeurs uniques
def remplacer_valeurs():
    st.subheader("Remplacer les valeurs manquantes, NaN ou valeurs uniques:")

    if st.session_state.modified_data is not None:
        data_to_modify = st.session_state.modified_data
    elif st.session_state.data is not None:
        data_to_modify = st.session_state.data
    else:
        st.warning("Aucune donnée n'est disponible. Veuillez importer un fichier CSV dans l'onglet 'Data' avant de remplacer les valeurs manquantes, NaN ou valeurs uniques.")
        return

    # Choix entre remplacer les valeurs manquantes, NaN ou valeurs uniques
    value_type = st.radio("Choisissez le type de valeurs à remplacer :", ["Valeurs Manquantes", "NaN", "Valeurs Uniques"])
    selected_columns = st.multiselect("Sélectionnez les colonnes à modifier", data_to_modify.columns)

    # Choix entre 0, moyenne, medianne, mode pour le remplacement
    replace_option = st.selectbox("Choisissez une option de remplacement :", ["0", "Moyenne", "Médiane", "Mode"])

    # Générer une clé unique dynamiquement
    button_key = "remplacer_button_" + str(hash(str(selected_columns) + value_type + replace_option))

    # Afficher la zone de saisie uniquement lorsque l'utilisateur sélectionne "Valeurs Uniques"
    if value_type == "Valeurs Uniques":
        unique_value_to_replace = st.text_input("Entrez la valeur unique à remplacer :")

        # Vérifier si l'utilisateur a saisi une valeur
        if not unique_value_to_replace:
            st.warning("Veuillez entrer une valeur unique à remplacer.")
            return
    else:
        unique_value_to_replace = None

    if st.button("Appliquer le remplacement", key=button_key):
        if selected_columns:
            remplacer_valeurs_selectionnees(data_to_modify, selected_columns, value_type, replace_option, unique_value_to_replace)
        else:
            st.warning("Veuillez sélectionner au moins une colonne à modifier.")

# Fonction pour remplacer les valeurs manquantes, NaN ou valeurs uniques dans les colonnes sélectionnées
def remplacer_valeurs_selectionnees(data, columns, value_type, replace_option, unique_value_to_replace=None):
    if replace_option == "Mode":
        mode_value = data[columns].mode().iloc[0]
        if pd.isna(mode_value):  # Gérer le cas où le mode n'est pas disponible (par exemple, toutes les valeurs sont uniques)
            st.warning("Le mode n'est pas disponible. Veuillez choisir une autre option de remplacement.")
            return

    if value_type == "Valeurs Manquantes":
        if replace_option == "0":
            data[columns] = data[columns].fillna(0)
        elif replace_option == "Moyenne":
            data[columns] = data[columns].fillna(data[columns].mean())
        elif replace_option == "Médiane":
            data[columns] = data[columns].fillna(data[columns].median())
        elif replace_option == "Mode":
            data[columns] = data[columns].fillna(mode_value)
    elif value_type == "NaN":
        if replace_option == "0":
            data[columns] = data[columns].replace(0, np.nan)
        elif replace_option == "Moyenne":
            data[columns] = data[columns].replace(data[columns].mean(), np.nan)
        elif replace_option == "Médiane":
            data[columns] = data[columns].replace(data[columns].median(), np.nan)
        elif replace_option == "Mode":
            data[columns] = data[columns].replace(mode_value, np.nan)
    elif value_type == "Valeurs Uniques":
        if not data[columns].isin([unique_value_to_replace]).any().any():
            st.warning("La valeur unique que vous avez entrée n'est pas présente dans les données. Veuillez entrer une valeur existante.")
            return
        
        # Remplacer la valeur unique par l'option choisie
        data[columns] = data[columns].replace(unique_value_to_replace, replace_option)

    st.session_state.modified_data = data
    st.success("Le remplacement a été effectué avec succès.")






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
            encoding_option = st.selectbox("Choisissez une option d'encodage :", ["Label" ,"One-Hot", "Ordinal"])
            if st.button("Appliquer l'encodage"):
                if encoding_option == "Label":
                    label_encoder = LabelEncoder()
                    for column in selected_columns:
                        data_to_modify[column] = label_encoder.fit_transform(data_to_modify[column])
                    st.session_state.modified_data = data_to_modify
                    st.success("Label encoding appliqué avec succès.")
                elif encoding_option == "One-Hot":
                    data_to_modify = pd.get_dummies(data_to_modify, columns=selected_columns, drop_first=True)
                    st.session_state.modified_data = data_to_modify
                    st.success("Encodage One-Hot appliqué avec succès.")
                elif encoding_option == "Ordinal":
                    ordinal_encoder = OrdinalEncoder()
                    data_to_modify[selected_columns] = ordinal_encoder.fit_transform(data_to_modify[selected_columns])
                    st.session_state.modified_data = data_to_modify
                    st.success("Encodage ordinal appliqué avec succès.")
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
    else:
        st.warning("Aucune variable numérique à normaliser.")




# Fonction pour diviser les données en ensemble d'entraînement et de test
def split_data():
    # Vérifier si des données sont disponibles
    if st.session_state.data is None:
        st.warning("Veuillez d'abord importer des données dans l'onglet 'Data'.")
        return
    
    # Vérifier si l'utilisateur a choisi le type de problème
    if st.session_state.user_choice is None:
        st.warning("Veuillez d'abord choisir le type de problème dans l'onglet 'Data'.")
        return


    # Vérifier si des données sont disponibles
    if st.session_state.modified_data is not None:
        data_to_split = st.session_state.modified_data
    elif st.session_state.data is not None:
        data_to_split = st.session_state.data


    # Check if there are categorical columns
    categorical_cols = data_to_split.select_dtypes(include=['object']).columns.tolist()

    if categorical_cols:
        st.warning("Votre jeu de données contient des colonnes catégorielles. Veuillez les encoder avant de diviser les données.")
        return
    

    # Si le problème est supervisé, permettre à l'utilisateur de choisir la variable cible
    if st.session_state.user_choice in ["Classification Supervisé", "Regression Supervisé"]:

        target_variable = st.selectbox("Sélectionnez la variable cible :", data_to_split.columns)
        random_state = st.number_input("Sélectionnez la valeur pour 'random_state' :", min_value=0, step=1, value=42)
        test_size_percentage = st.slider("Sélectionnez la proportion d'entraînement :", min_value=10, max_value=90, step=1, value=80)

        if st.button("Diviser les données"):
            
            test_size = test_size_percentage / 100.0  # Convertir le pourcentage en fraction

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

    elif st.session_state.user_choice == "Classification Non Supervisé":

        random_state = st.number_input("Sélectionnez la valeur pour 'random_state' :", min_value=0, step=1, value=42)
        test_size_percentage = st.slider("Sélectionnez la proportion d'entraînement :", min_value=10, max_value=90, step=1, value=80)

        if st.button("Diviser les données"):
            
            test_size = test_size_percentage / 100.0  # Convert percentage to fraction

            X_train, X_test = train_test_split(
                data_to_split,
                test_size=test_size,
                random_state=random_state
            )

            y_train, y_test = None, None

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

            st.write("Aperçu de l'ensemble de test:")
            st.write(X_test.head())



# Fonction pour appliquer la technique SMOTE
def smote_function():
    # Vérifier si l'utilisateur a déjà divisé les données
    if st.session_state.split_data is not None:
        data_to_smote = st.session_state.split_data

    # Vérifier si le problème est de classification supervisée
    if st.session_state.user_choice == "Classification Supervisé" and st.session_state.split_data is not None:
        st.header("SMOTE")

        # Button to trigger SMOTE
        if st.button("Appliquer SMOTE"):
            # Get the target variable from the split data
            target_variable = data_to_smote["y_train"].name

            # Separate features and target variable
            X_train, y_train = data_to_smote["X_train"], data_to_smote["y_train"]


            # Apply SMOTE only on training data
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

            # Combine features and target variable
            data_resampled = pd.concat([pd.DataFrame(X_resampled, columns=X_train.columns), pd.Series(y_resampled, name=target_variable)], axis=1)

            st.session_state.resampled_data = data_resampled
            st.success("SMOTE appliqué avec succès.")
            
            st.write("Aperçu des données après l'application de SMOTE:")
            st.write(data_resampled.head())
    else:
        pass



# Fonction pour appliquer l'analyse en composantes principales (PCA)
def apply_pca():
    # Vérifier si des données sont disponibles
    if st.session_state.data is None:
        st.warning("Veuillez d'abord importer des données dans l'onglet 'Data'.")
        return

    # Vérifier si l'utilisateur a choisi le type de problème
    if st.session_state.user_choice is None:
        st.warning("Veuillez d'abord choisir le type de problème dans l'onglet 'Data'.")
        return
    
    # Vérifier si l'utilisateur a déjà divisé les données
    if st.session_state.split_data is None:
        st.warning("Veuillez d'abord diviser les données dans l'onglet 'Split'.")
    else:
        data_for_pca = st.session_state.split_data

    # Vérifier si le problème est de classification supervisée
    if st.session_state.user_choice in ["Classification Supervisé", "Regression Supervisé"] and st.session_state.split_data is not None:
        # Button to apply PCA
        if st.button("Appliquer PCA"):
            # Separate features and target variable
            X_train, y_train = data_for_pca["X_train"], data_for_pca["y_train"]

            # Apply PCA only on training data
            pca = PCA()
            X_pca = pca.fit_transform(X_train)

            # Combine transformed features and target variable
            data_pca = pd.concat([pd.DataFrame(X_pca, columns=[f"PC{i}" for i in range(1, X_pca.shape[1] + 1)]), y_train.reset_index(drop=True)], axis=1)

            st.session_state.resampled_data = data_pca
            st.success("PCA appliqué avec succès.")

            st.write("Aperçu des données après l'application de PCA:")
            st.write(data_pca.head())
    elif st.session_state.user_choice in ["Classification Non Supervisé"] and st.session_state.split_data is not None:
        # Button to apply PCA
        if st.button("Appliquer PCA"):
            # Unsupervised PCA for Classification Non Supervisé
            X = data_for_pca["X_train"]

            # Apply PCA to the entire dataset
            pca = PCA()
            X_pca = pca.fit_transform(X)

            # Combine transformed features
            data_pca = pd.DataFrame(X_pca, columns=[f"PC{i}" for i in range(1, X_pca.shape[1] + 1)])

            st.session_state.resampled_data = data_pca
            st.success("PCA appliqué avec succès.")

            st.write("Aperçu des données après l'application de PCA:")
            st.write(data_pca.head())
    else:
        pass





# Fonction pour afficher les onglets
def display_tabs():
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Data", "Visualise", "Clean", "Split", "Resampling Data"])

    # onglet importation des données
    with tab1:
        st.header("Data")

        st.session_state.data = import_csv()  # Assign the imported data to st.session_state.data

        choix_du_probleme()

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
            # Création de deux colonnes
            left_column, right_column = st.columns(2)

            # Dans la colonne de gauche
            with left_column:
                st.subheader("Actions de Nettoyage:")
                
                # Exécution de la fonction supprimer_col()
                supprimer_col()

                # Exécution de la fonction supprimer_lignes_dupliquees()
                supprimer_lignes_dupliquees()

                # Exécution de la fonction remplacer_valeurs() seulement si des données existent
                remplacer_valeurs()

                # Exécution de la fonction encodage() seulement si des données existent
                encodage()

                # Exécution de la fonction normaliser() seulement si des données existent
                normaliser()

            # Dans la colonne de droite
            with right_column:
                st.subheader("Aperçu des données:")
                
                # Affichage du tableau des données
                if st.session_state.modified_data is not None:
                    st.write("Aperçu des données après nettoyage:")
                    st.write(st.session_state.modified_data)
                    
                    # Affichage de la taille des données
                    st.write(f"Taille des données : {st.session_state.modified_data.shape}")
                    # Affichage du nombre de valeurs manquantes par colonne
                    st.write("Nombre de valeurs manquantes par colonne:")
                    st.write(st.session_state.modified_data.isnull().sum())
                    # Affichage du nombre de NaN values par colonne
                    st.write("Nombre de NaN values par colonne:")
                    st.write(st.session_state.modified_data.isna().sum())
                elif st.session_state.data is not None:
                    st.warning("Aucune modification n'a été effectuée. Voici l'aperçu des données importées.")
                    st.write("Aperçu des données importées:")
                    st.write(st.session_state.data)
                    # Affichage de la taille des données
                    st.write(f"Taille des données : {st.session_state.data.shape}")
                    # Affichage du nombre de valeurs manquantes par colonne
                    st.write("Nombre de valeurs manquantes par colonne:")
                    st.write(st.session_state.data.isnull().sum())
                    # Affichage du nombre de NaN values par colonne
                    st.write("Nombre de NaN values par colonne:")
                    st.write(st.session_state.data.isna().sum())


    # onglet division des données  
    with tab4:
        st.header("Split")

        # Exécution de la fonction split_data()
        split_data()

    with tab5:
        st.header("Resampling Data")

        apply_pca()

        smote_function()



# Fonction principale
def main():
    init_session()

    if st.session_state.page == 0:
        landing_page()
    elif st.session_state.page == 1:
        display_tabs()

if __name__ == "__main__":
    main()
