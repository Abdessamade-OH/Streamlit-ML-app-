import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from itertools import cycle
from sklearn.metrics import r2_score
from sklearn.metrics import auc, classification_report, confusion_matrix, mean_absolute_error, mean_squared_error, precision_recall_curve, roc_curve, silhouette_score
import joblib
import os
import tkinter as tk
from tkinter import filedialog

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
        st.session_state.algorithm_choice = None # Initialiser la variable "algorithm_choice" à None
        st.session_state.model_hyperparameters = None # Initialiser la variable "model_hyperparameters" à None
        st.session_state.model = None # Initialiser la variable "model" à None
        st.session_state.folder_path = None # Initialiser la variable "folder_path" à None

# Fonction pour afficher la landing page
def landing_page():
    with open('style.css') as f:
        css = f.read()

    st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

    st.title("OptiML")
    
    st.subheader("Découvrez une plateforme intuitive d'exploration de données et d'apprentissage automatique qui donne vie à votre parcours data.")
    st.subheader("Que vous soyez un passionné de données, un analyste ou un amateur d'apprentissage automatique, OptiML offre une interface conviviale pour importer, explorer et analyser vos ensembles de données en toute simplicité.")
   
    st.subheader("Cliquez sur le bouton ci-dessous pour commencer :")

    with st.form(key="landing_form", border=False):
        st.form_submit_button("Get Started", on_click=lambda: st.session_state.update({"page": 1}))

    st.text("")  # Crée une séparation visuelle sans bordure
    st.text("")  # Crée une séparation visuelle sans bordure

    st.subheader("Guide des Onglets : Découvrez Chaque Étape de Votre Projet d'Apprentissage Automatique")
    st.write("Importation des Données : Importez vos ensembles de données dans l'application pour démarrer votre projet.")
    st.write("Visualisation : Explorez visuellement vos données avec des graphiques et des tableaux pour en comprendre la structure.")
    st.write("Nettoyage des Données : Effectuez des opérations de nettoyage, telles que le traitement des valeurs manquantes ou la suppression des valeurs aberrantes.")
    st.write("Préparation des Données : Divisez vos données en ensembles d'entraînement et de test pour préparer le modèle.")
    st.write("Transformation des Données : Appliquez des transformations, notamment l'analyse en composantes principales (PCA) et la suréchantillonnage des données avec SMOTE, pour optimiser la préparation de votre modèle.")
    st.write("Entraînement du Modèle : Choisissez et entraînez votre modèle d'apprentissage automatique avec les données préparées.")
    st.write("Évaluation du Modèle : Évaluez les performances de votre modèle avec des métriques adaptées au problème.")
    st.write("Exportation du Modèle : Exportez votre modèle entraîné pour une utilisation future.")

    

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
    chart_type = selection_column.selectbox("Sélectionnez le type de graphe", ["Scatter Plot", "Line Plot", "Bar Plot", "Box Plot", "Heatmap"])

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
            elif chart_type == "Box Plot":
                sns.boxplot(x=st.session_state.selected_columns[0], y=st.session_state.selected_columns[1], data=data_to_visualize, ax=ax)
            elif chart_type == "Heatmap":
                heatmap_data = data_to_visualize[st.session_state.selected_columns].corr()
                sns.heatmap(heatmap_data, annot=True, cmap="coolwarm", ax=ax)
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
    
    # Add an option to select all columns
    all_columns_option = "Toutes les colonnes"
    columns_with_all_option = [all_columns_option] + data_to_modify.columns.tolist()
    selected_columns = st.multiselect("Sélectionnez les colonnes à modifier", columns_with_all_option)

    # Check if the user selected "Toutes les colonnes" and replace with all columns
    if all_columns_option in selected_columns:
        selected_columns = data_to_modify.columns.tolist()

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


def convert_object_columns_to_float():
    st.subheader("Conversion des colonnes d'objet en float:")

    if st.session_state.modified_data is not None:
        data_to_modify = st.session_state.modified_data
    elif st.session_state.data is not None:
        data_to_modify = st.session_state.data
    else:
        st.warning("Aucune donnée n'est disponible. Veuillez importer un fichier CSV dans l'onglet 'Data' avant de faire la conversion.")
        return

    object_cols = data_to_modify.select_dtypes(include=['object']).columns.tolist()

    # Add an option to select all columns
    all_columns_option = "Toutes les colonnes"
    object_cols_dropdown = [all_columns_option] + object_cols
    selected_columns = st.multiselect("Sélectionnez les colonnes à convertir", object_cols_dropdown)

    if st.button("Appliquer la conversion"):
        if all_columns_option in selected_columns:
            # Convert all object columns
            for column in object_cols:
                if all(pd.to_numeric(data_to_modify[column], errors='coerce').notnull()):
                    data_to_modify[column] = pd.to_numeric(data_to_modify[column], errors='coerce')
                    st.success(f"Conversion réussie pour la colonne {column}.")
                else:
                    st.warning(f"La colonne {column} contient des valeurs non numériques.")
        elif selected_columns:
            # Convert selected columns
            for column in selected_columns:
                if column != all_columns_option:
                    if all(pd.to_numeric(data_to_modify[column], errors='coerce').notnull()):
                        data_to_modify[column] = pd.to_numeric(data_to_modify[column], errors='coerce')
                        st.success(f"Conversion réussie pour la colonne {column}.")
                    else:
                        st.warning(f"La colonne {column} contient des valeurs non numériques.")
        else:
            st.warning("Veuillez sélectionner au moins une colonne à convertir.")
        


    


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
        # Add an option to select all columns
        all_columns_option = "Toutes les colonnes"
        columns_with_all_option = [all_columns_option] + categorical_cols
        selected_columns = st.multiselect("Sélectionnez les colonnes à encoder", columns_with_all_option)
        
        if all_columns_option in selected_columns:
            selected_columns = categorical_cols
        
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
        test_size_percentage = st.slider("Sélectionnez la proportion de test :", min_value=10, max_value=90, step=1, value=20)

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
        test_size_percentage = st.slider("Sélectionnez la proportion de test :", min_value=10, max_value=90, step=1, value=20)

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
        st.warning("Veuillez d'abord importer des données dans l'onglet 'Importation des Données'.")
        return

    # Vérifier si l'utilisateur a choisi le type de problème
    if st.session_state.user_choice is None:
        st.warning("Veuillez d'abord choisir le type de problème dans l'onglet 'Importation des Données'.")
        return
    
    # Vérifier si l'utilisateur a déjà divisé les données
    if st.session_state.split_data is None:
        st.warning("Veuillez d'abord diviser les données dans l'onglet 'Préparation des Données'.")
        return
    else:
        data_for_pca = st.session_state.split_data

    # Vérifier si le problème est de classification supervisée
    if st.session_state.user_choice in ["Classification Supervisé", "Regression Supervisé"]:
        # Separate features and target variable
        X_train, y_train = data_for_pca["X_train"], data_for_pca["y_train"]

        num_components = X_train.shape[1]

        # User input for the number of components
        num_components_user = st.number_input("Choisissez le nombre de composantes pour PCA:", min_value=1, max_value=num_components, value=num_components, step=1)

        # Apply PCA with the chosen number of components
        if st.button("Appliquer PCA"):
            pca = PCA(n_components=num_components_user)
            X_pca = pca.fit_transform(X_train)

            # Combine transformed features and target variable
            data_pca = pd.concat([pd.DataFrame(X_pca, columns=[f"PC{i}" for i in range(1, X_pca.shape[1] + 1)]), y_train.reset_index(drop=True)], axis=1)

            st.session_state.resampled_data = data_pca
            st.success("PCA appliqué avec succès.")

            st.write("Aperçu des données après l'application de PCA:")
            st.write(data_pca.head())

    elif st.session_state.user_choice in ["Classification Non Supervisé"]:
        # Unsupervised PCA for Classification Non Supervisé
        X_train = data_for_pca["X_train"]

        num_components = X_train.shape[1]

        # User input for the number of components
        num_components_user = st.number_input("Choisissez le nombre de composantes pour PCA:", min_value=1, max_value=num_components, value=num_components, step=1)

        # Apply PCA with the chosen number of components
        if st.button("Appliquer PCA"):
            pca = PCA(n_components=num_components_user)
            X_pca = pca.fit_transform(data_for_pca["X_train"])

            # Combine transformed features
            data_pca = pd.DataFrame(X_pca, columns=[f"PC{i}" for i in range(1, X_pca.shape[1] + 1)])

            st.session_state.resampled_data = data_pca
            st.success("PCA appliqué avec succès.")

            st.write("Aperçu des données après l'application de PCA:")
            st.write(data_pca.head())

    else:
        pass


# Fonction pour afficher la méthode du coude
def plot_elbow_method():
    # Vérifier si des données sont disponibles
    if st.session_state.data is None:
        return

    # Vérifier si l'utilisateur a choisi le type de problème
    if st.session_state.user_choice is None:
        return
    
    # Vérifier si l'utilisateur a déjà divisé les données
    if st.session_state.split_data is None:
        return
    else:
        data_for_pca = st.session_state.split_data
        

    # Vérifier si le problème est de classification supervisée
    if st.session_state.user_choice in ["Classification Supervisé", "Regression Supervisé"]:
        # Separate features and target variable
        X_train = data_for_pca["X_train"]

        # Perform the elbow method to determine the optimal number of components
        st.subheader("Méthode du coude pour déterminer le nombre optimal de composantes:")

        # Automatically set num_components to be the minimum between the number of rows and columns in X_train
        num_components = X_train.shape[1]
        pca = PCA(n_components=num_components)
        X_pca = pca.fit_transform(X_train)

        # Plot the explained variance ratio
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance_ratio = explained_variance_ratio.cumsum()

        fig, ax = plt.subplots()
        ax.plot(range(1, num_components + 1), cumulative_variance_ratio, marker='o', linestyle='-', color='b')
        ax.set_xlabel('Nombre de composantes')
        ax.set_ylabel('Variance cumulée expliquée')
        ax.set_title('Méthode du coude pour PCA')

        # Display the plot
        st.pyplot(fig)

    elif st.session_state.user_choice in ["Classification Non Supervisé"]:
        # Unsupervised PCA for Classification Non Supervisé
        X_train = data_for_pca["X_train"]

        # Perform the elbow method to determine the optimal number of components for Unsupervised PCA
        st.subheader("Méthode du coude pour déterminer le nombre optimal de composantes:")

        # Automatically set num_components to be the minimum between the number of rows and columns in X_train
        num_components = X_train.shape[1]
        pca = PCA(n_components=num_components)
        X_pca = pca.fit_transform(X_train)

        # Plot the explained variance ratio
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance_ratio = explained_variance_ratio.cumsum()

        fig, ax = plt.subplots()
        ax.plot(range(1, num_components + 1), cumulative_variance_ratio, marker='o', linestyle='-', color='b')
        ax.set_xlabel('Nombre de composantes')
        ax.set_ylabel('Variance cumulée expliquée')
        ax.set_title('Méthode du coude pour PCA')

        # Display the plot
        st.pyplot(fig)

    else:
        pass



def choix_algorithme():
    # Vérifier si des données sont disponibles
    if st.session_state.data is None:
        st.warning("Veuillez d'abord importer des données dans l'onglet 'Importation des Données'.")
        return

    # Vérifier si l'utilisateur a choisi le type de problème
    if st.session_state.user_choice is None:
        st.warning("Veuillez d'abord choisir le type de problème dans l'onglet 'Importation des Données'.")
        return
    
    # Vérifier si l'utilisateur a déjà divisé les données
    if st.session_state.split_data is None:
        st.warning("Veuillez d'abord diviser les données dans l'onglet 'Préparation des Données'.")
        return

    algorithms_classification = ["Regression logistique", "Arbre de décision CART", "Naif bayes", "SVM", "KNN", "Random forest"]
    algorithms_regression = ["Regression linéaire", "Arbre de décision CART", "SVM", "KNN", "Random forest"]
    algorithms_clustering = ["K-means"]

    if st.session_state.user_choice == "Classification Supervisé":
        selected_algorithm = st.selectbox("Choisissez l'algorithme :", algorithms_classification)
    elif st.session_state.user_choice == "Regression Supervisé":
        selected_algorithm = st.selectbox("Choisissez l'algorithme :", algorithms_regression)
    elif st.session_state.user_choice == "Classification Non Supervisé":
        selected_algorithm = st.selectbox("Choisissez l'algorithme :", algorithms_clustering)
    else:
        st.warning("Type de problème non pris en charge.")
        return
    
    # Ajouter un bouton "Confirmer"
    if st.button("Confirmer"):
        st.session_state.algorithm_choice = selected_algorithm
        st.success(f"Algorithme sélectionné : {selected_algorithm}")


def choisir_hyperparametres():
    # Vérifier si l'algorithme a été choisi
    if st.session_state.algorithm_choice is not None:
        st.subheader("Choix des Hyperparamètres:")
        algorithm_choice = st.session_state.algorithm_choice

        hyperparameters = {}

        if algorithm_choice in ["Regression logistique"]:
            C = st.number_input("Paramètre C :", min_value=0.1, max_value=10.0, step=0.1, value=1.0)
            hyperparameters["C"] = C
            penalty = st.radio("Choix de la pénalité :", ["l2", "none"])
            hyperparameters["penalty"] = penalty

        elif algorithm_choice == "Regression linéaire":
            st.warning("Pas d'hyperparamètres pour la Regression linéaire.")

        elif algorithm_choice in ["Arbre de décision CART"]:
            max_depth = st.number_input("Profondeur maximale de l'arbre :", min_value=1, max_value=20, step=1, value=3)
            hyperparameters["max_depth"] = max_depth
            if st.session_state.user_choice == "Classification Supervisé":
                criterion_options = st.radio("Critère de division :", ["gini", "entropy"])
            elif st.session_state.user_choice == "Regression Supervisé":
                criterion_options = st.radio("Critère de division :", ["absolute_error", "poisson", "squared_error", "friedman_mse"])
            else:
                st.warning("Type de problème non pris en charge.")
            hyperparameters["criterion"] = criterion_options

        elif algorithm_choice == "Naif bayes":
            nb_type = st.radio("Type de Naive Bayes :", ["Gaussian", "Multinomial", "Bernoulli"])
            hyperparameters["nb_type"] = nb_type

        elif algorithm_choice == "SVM":
            kernel = st.radio("Type de noyau :", ["linear", "poly", "rbf", "sigmoid"])
            hyperparameters["kernel"] = kernel
            C_svm = st.number_input("Paramètre C pour SVM :", min_value=0.1, max_value=10.0, step=0.1, value=1.0)
            hyperparameters["C_svm"] = C_svm

        elif algorithm_choice == "KNN":
            n_neighbors = st.number_input("Nombre de voisins :", min_value=1, max_value=20, step=1, value=5)
            hyperparameters["n_neighbors"] = n_neighbors
            algorithm_knn = st.radio("Algorithme KNN :", ["auto", "ball_tree", "kd_tree", "brute"])
            hyperparameters["algorithm_knn"] = algorithm_knn

        elif algorithm_choice == "Random forest":
            n_estimators = st.number_input("Nombre d'estimateurs :", min_value=1, max_value=100, step=1, value=10)
            hyperparameters["n_estimators"] = n_estimators
            max_depth_rf = st.number_input("Profondeur maximale de l'arbre :", min_value=1, max_value=20, step=1, value=3)
            hyperparameters["max_depth_rf"] = max_depth_rf

        elif algorithm_choice == "K-means":
            n_clusters = st.number_input("Nombre de clusters :", min_value=2, max_value=20, step=1, value=8)
            hyperparameters["n_clusters"] = n_clusters

        # Afficher le bouton pour confirmer les hyperparamètres
        if st.button("Confirmer les Hyperparamètres"):
            st.session_state.model_hyperparameters = hyperparameters
            st.success("Hyperparamètres confirmés avec succès.")
    else:
        pass


def execute_algorithm():
    if st.session_state.model_hyperparameters is not None:
        # Afficher le bouton pour Entraîner le modèle
        if st.button("Entraîner le modèle"):
            
            algorithm_choice = st.session_state.algorithm_choice
            hyperparameters = st.session_state.model_hyperparameters

            if st.session_state.split_data is not None :
                model_data = st.session_state.split_data
            elif st.session_state.resampled_data is not None:
                model_data = st.session_state.resampled_data

            # Exécuter l'algorithme en fonction du choix de l'utilisateur
            if algorithm_choice == "Regression logistique":
                model = execute_logistic_regression(hyperparameters, model_data)
                st.session_state.model = model

            elif algorithm_choice == "Arbre de décision CART":
                if st.session_state.user_choice == "Classification Supervisé":
                    model = execute_decision_tree_classifier(hyperparameters, model_data)
                    st.session_state.model = model
                elif st.session_state.user_choice == "Regression Supervisé":
                    model = execute_decision_tree_regressor(hyperparameters, model_data)
                    st.session_state.model = model

            elif algorithm_choice == "Naif bayes":
                model = execute_naive_bayes(hyperparameters, model_data)
                st.session_state.model = model

            elif algorithm_choice == "SVM":
                if st.session_state.user_choice == "Classification Supervisé":
                    model = execute_svm_classifier(hyperparameters, model_data)
                    st.session_state.model = model
                elif st.session_state.user_choice == "Regression Supervisé":
                    model = execute_svm_regressor(hyperparameters, model_data)
                    st.session_state.model = model

            elif algorithm_choice == "KNN":
                if st.session_state.user_choice == "Classification Supervisé":
                    model = execute_knn_classifier(hyperparameters, model_data)
                    st.session_state.model = model
                elif st.session_state.user_choice == "Regression Supervisé":
                    model = execute_knn_regressor(hyperparameters, model_data)
                    st.session_state.model = model

            elif algorithm_choice == "Random forest":
                if st.session_state.user_choice == "Classification Supervisé":
                    model = execute_random_forest_classifier(hyperparameters, model_data)
                    st.session_state.model = model
                elif st.session_state.user_choice == "Regression Supervisé":
                    model = execute_random_forest_regressor(hyperparameters, model_data)
                    st.session_state.model = model

            elif algorithm_choice == "Regression linéaire":
                model = execute_linear_regression(model_data)
                st.session_state.model = model

            elif algorithm_choice == "K-means":
                model = execute_kmeans(hyperparameters, model_data)
                st.session_state.model = model
    else:
        pass


def execute_logistic_regression(hyperparameters, model_data):
    model = LogisticRegression(**hyperparameters)
    # Extracting data from the split_data dictionary
    X_train, y_train = model_data["X_train"], model_data["y_train"]

    try:
        model.fit(X_train, y_train)
        st.success("Régression logistique exécutée avec succès.")
        return model  # Return the trained model
    except Exception as e:
        st.error(f"Erreur lors de l'entraînement de la régression logistique : {str(e)}")
        return None  # Return None or handle the error appropriately

def execute_decision_tree_classifier(hyperparameters, model_data):
    max_depth = hyperparameters.get("max_depth", 3)  # Default to 3 if not specified
    criterion = hyperparameters.get("criterion", "gini")  # Default to "gini" if not specified

    try:
        model = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion)
        # Extracting data from the model_data dictionary
        X_train, y_train = model_data["X_train"], model_data["y_train"]

        model.fit(X_train, y_train)
        st.success("Decision Tree (Classifier) executed successfully.")
        return model  # Return the trained model
    except Exception as e:
        st.error(f"Error during Decision Tree (Classifier) training: {str(e)}")
        return None  # Return None or handle the error appropriately

def execute_decision_tree_regressor(hyperparameters, model_data):
    max_depth = hyperparameters.get("max_depth", 3)  # Default to 3 if not specified
    criterion = hyperparameters.get("criterion", "squared_error")  # Default to "squared_error" if not specified

    try:
        model = DecisionTreeRegressor(max_depth=max_depth, criterion=criterion)
        # Extracting data from the model_data dictionary
        X_train, y_train = model_data["X_train"], model_data["y_train"]

        model.fit(X_train, y_train)
        st.success("Decision Tree (Regressor) executed successfully.")
        return model  # Return the trained model
    except Exception as e:
        st.error(f"Error during Decision Tree (Regressor) training: {str(e)}")
        return None  # Return None or handle the error appropriately
        
def execute_linear_regression(model_data):
    model = LinearRegression()
    # Extracting data from the split_data dictionary
    X_train, y_train = model_data["X_train"], model_data["y_train"]

    try:
        model.fit(X_train, y_train)
        st.success("Régression linéaire exécutée avec succès.")
        return model  # Return the trained model
    except Exception as e:
        st.error(f"Erreur lors de l'entraînement de la régression linéaire : {str(e)}")
        return None  # Return None or handle the error appropriately

def execute_naive_bayes(hyperparameters, model_data):
    nb_type = hyperparameters.get("nb_type", "Gaussian")  # Default to Gaussian if not specified

    try:
        if nb_type == "Gaussian":
            model = GaussianNB()
        elif nb_type == "Multinomial":
            model = MultinomialNB()
        elif nb_type == "Bernoulli":
            model = BernoulliNB()
        else:
            raise ValueError("Invalid Naive Bayes type selected.")
        
        # Extracting data from the model_data dictionary
        X_train, y_train = model_data["X_train"], model_data["y_train"]

        model.fit(X_train, y_train)
        st.success("Naive Bayes executed successfully.")
        return model  # Return the trained model
    except Exception as e:
        st.error(f"Error during Naive Bayes training: {str(e)}")
        return None  # Return None or handle the error appropriately

def execute_svm_classifier(hyperparameters, model_data):
    kernel = hyperparameters.get("kernel", "rbf")  # Default to "rbf" if not specified
    C_svm = hyperparameters.get("C_svm", 1.0)  # Default to 1.0 if not specified

    try:
        model = SVC(kernel=kernel, C=C_svm)
        # Extracting data from the model_data dictionary
        X_train, y_train = model_data["X_train"], model_data["y_train"]

        model.fit(X_train, y_train)
        st.success("SVM (Classifier) executed successfully.")
        return model  # Return the trained model
    except Exception as e:
        st.error(f"Error during SVM (Classifier) training: {str(e)}")
        return None  # Return None or handle the error appropriately

def execute_svm_regressor(hyperparameters, model_data):
    kernel = hyperparameters.get("kernel", "rbf")  # Default to "rbf" if not specified
    C_svm = hyperparameters.get("C_svm", 1.0)  # Default to 1.0 if not specified

    try:
        model = SVR(kernel=kernel, C=C_svm)
        # Extracting data from the model_data dictionary
        X_train, y_train = model_data["X_train"], model_data["y_train"]

        model.fit(X_train, y_train)
        st.success("SVM (Regressor) executed successfully.")
        return model  # Return the trained model
    except Exception as e:
        st.error(f"Error during SVM (Regressor) training: {str(e)}")
        return None  # Return None or handle the error appropriately
        
def execute_knn_classifier(hyperparameters, model_data):
    n_neighbors = hyperparameters.get("n_neighbors", 5)  # Default to 5 if not specified
    algorithm_knn = hyperparameters.get("algorithm_knn", "auto")  # Default to "auto" if not specified

    try:
        model = KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=algorithm_knn)
        # Extracting data from the model_data dictionary
        X_train, y_train = model_data["X_train"], model_data["y_train"]

        model.fit(X_train, y_train)
        st.success("K-Nearest Neighbors (Classifier) executed successfully.")
        return model  # Return the trained model
    except Exception as e:
        st.error(f"Error during K-Nearest Neighbors (Classifier) training: {str(e)}")
        return None  # Return None or handle the error appropriately

def execute_knn_regressor(hyperparameters, model_data):
    n_neighbors = hyperparameters.get("n_neighbors", 5)  # Default to 5 if not specified
    algorithm_knn = hyperparameters.get("algorithm_knn", "auto")  # Default to "auto" if not specified

    try:
        model = KNeighborsRegressor(n_neighbors=n_neighbors, algorithm=algorithm_knn)
        # Extracting data from the model_data dictionary
        X_train, y_train = model_data["X_train"], model_data["y_train"]

        model.fit(X_train, y_train)
        st.success("K-Nearest Neighbors (Regressor) executed successfully.")
        return model  # Return the trained model
    except Exception as e:
        st.error(f"Error during K-Nearest Neighbors (Regressor) training: {str(e)}")
        return None  # Return None or handle the error appropriately

def execute_random_forest_classifier(hyperparameters, model_data):
    n_estimators = hyperparameters.get("n_estimators", 10)  # Default to 10 if not specified
    max_depth_rf = hyperparameters.get("max_depth_rf", 3)  # Default to 3 if not specified

    try:
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth_rf)
        # Extracting data from the model_data dictionary
        X_train, y_train = model_data["X_train"], model_data["y_train"]

        model.fit(X_train, y_train)
        st.success("Random Forest (Classifier) executed successfully.")
        return model  # Return the trained model
    except Exception as e:
        st.error(f"Error during Random Forest (Classifier) training: {str(e)}")
        return None  # Return None or handle the error appropriately

def execute_random_forest_regressor(hyperparameters, model_data):
    n_estimators = hyperparameters.get("n_estimators", 10)  # Default to 10 if not specified
    max_depth_rf = hyperparameters.get("max_depth_rf", 3)  # Default to 3 if not specified

    try:
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth_rf)
        # Extracting data from the model_data dictionary
        X_train, y_train = model_data["X_train"], model_data["y_train"]

        model.fit(X_train, y_train)
        st.success("Random Forest (Regressor) executed successfully.")
        return model  # Return the trained model
    except Exception as e:
        st.error(f"Error during Random Forest (Regressor) training: {str(e)}")
        return None  # Return None or handle the error appropriately
        
def execute_kmeans(hyperparameters, model_data):
    n_clusters = hyperparameters.get("n_clusters", 8)  # Default to 8 clusters if not specified

    try:
        model = KMeans(n_clusters=n_clusters)
        # Extracting data from the model_data dictionary
        X_train = model_data["X_train"]

        model.fit(X_train)
        st.success("K-means executed successfully.")
        return model  # Return the trained model
    except Exception as e:
        st.error(f"Error during K-means clustering: {str(e)}")
        return None  # Return None or handle the error appropriately
        

def evaluate_model():
    if st.session_state.model is not None:
        model = st.session_state.model
        X_test = st.session_state.split_data["X_test"]
        y_test = st.session_state.split_data["y_test"]
        problem_type = st.session_state.user_choice

        st.subheader("Récapitulatif du Modèle:")
        st.write(f"Type de problème: {problem_type}")
        st.write(f"Algorithme: {st.session_state.algorithm_choice}")
        st.write(f"Hyperparamètres: {st.session_state.model_hyperparameters}")

        # Afficher le bouton pour Evaluer le modèle
        if st.button("Evaluer le modèle"):
            st.subheader("Évaluation du Modèle:")
            if problem_type == "Classification Supervisé":
                y_pred = model.predict(X_test)
                display_classification_metrics(y_test, y_pred)
            elif problem_type == "Regression Supervisé":
                y_pred = model.predict(X_test)
                display_regression_metrics(y_test, y_pred)
            elif problem_type == "Classification Non Supervisé":
                display_clustering_metrics(model, X_test)
            else:
                st.warning("Type de problème non pris en charge.")
            
    else:
        st.warning("Veuillez d'abord entraîner le modèle dans l'onglet approprié.")


def display_classification_metrics(y_true, y_pred):
    # Création de deux colonnes
    left_column, right_column = st.columns(2)

    with left_column:
        st.subheader("Rapport de Classification:")
        report = classification_report(y_true, y_pred)
        st.text(report)



        st.subheader("Matrice de Confusion:")
        cm = confusion_matrix(y_true, y_pred)

        # Créer une heatmap avec seaborn
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel('Valeurs prédites')
        plt.ylabel('Valeurs réelles')
        plt.title('Matrice de Confusion')

        # Afficher l'image dans Streamlit
        st.pyplot(plt.gcf())  # Pass the current figure to st.pyplot

    
    with right_column:
        # Plot ROC curve for each class
        st.subheader("Courbe ROC (One-vs-One):")
        # Compute ROC curve and ROC area for each class
        n_classes = len(np.unique(y_true))
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(n_classes):
            y_true_i = np.where(y_true == i, 1, 0)
            y_pred_i = np.where(y_pred == i, 1, 0)

            fpr[i], tpr[i], _ = roc_curve(y_true_i, y_pred_i)
            roc_auc[i] = auc(fpr[i], tpr[i])

        fig_roc = plt.figure()
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'ROC curve (area = {roc_auc[i]:.2f})')

        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve (One-vs-One)')
        plt.legend(loc="lower right")
        st.pyplot(fig_roc)

        st.subheader("Courbe Précision-Recall (One-vs-All):")
        # Compute Precision-Recall curve and area for each class
        precision = dict()
        recall = dict()
        pr_auc = dict()

        # Plot combined Precision-Recall curve for all classes
        fig_pr_curve_combined = plt.figure()

        for i in range(n_classes):
            y_true_i = np.where(y_true == i, 1, 0)
            y_pred_i = np.where(y_pred == i, 1, 0)

            precision[i], recall[i], _ = precision_recall_curve(y_true_i, y_pred_i)
            pr_auc[i] = auc(recall[i], precision[i])

            # Plot Precision-Recall curve for each class
            plt.plot(recall[i], precision[i], lw=2, label=f'Class {i} (AUC = {pr_auc[i]:.2f})')

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve (One-vs-All)')
        plt.legend(loc="upper right")
        st.pyplot(fig_pr_curve_combined)


def display_regression_metrics(y_true, y_pred):
    # Création de deux colonnes
    left_column, right_column = st.columns(2)

    with left_column:
        st.subheader("Mean Absolute Error (MAE):")
        mae = mean_absolute_error(y_true, y_pred)
        st.write(f"MAE: {mae}")

        st.subheader("Mean Squared Error (MSE):")
        mse = mean_squared_error(y_true, y_pred)
        st.write(f"MSE: {mse}")

        st.subheader("R-squared (R2):")
        r2 = r2_score(y_true, y_pred)
        st.write(f"R2: {r2}")

    with right_column:
        st.subheader("Residuals Plot:")
        residuals = y_true - y_pred

        fig_res = plt.figure()
        plt.scatter(y_pred, residuals)
        plt.axhline(0, color='red', linestyle='--', linewidth=2)
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residuals Plot')
        st.pyplot(fig_res)


def display_clustering_metrics(model, X_test):
    # Création de deux colonnes
    left_column, right_column = st.columns(2)

    with left_column:
        st.subheader("Silhouette Score:")
        silhouette_avg = silhouette_score(X_test, model.predict(X_test))
        st.write(f"Silhouette Score: {silhouette_avg}")

        st.subheader("Inertia:")
        st.write(f"Inertia: {model.inertia_}")
    
    with right_column:
        st.subheader("Cluster Visualization:")
        visualize_clusters(X_test, model)

def visualize_clusters(X, model):
    # Convert DataFrame to NumPy array
    if isinstance(X, pd.DataFrame):
        X_array = X.values
    else:
        X_array = X

    # Print the shape and content of X_array and model.predict(X_array)
    print("X_array shape:", X_array.shape)
    print("model.predict(X_array) shape:", model.predict(X_array).shape)
    print("model.predict(X_array):", model.predict(X_array))

    # Implement your own cluster visualization method
    # This could include a scatter plot, 3D plot, or any other suitable visualization
    # For simplicity, here's a basic scatter plot with the first two features
    fig_cluster = plt.figure()
    sns.scatterplot(x=X_array[:, 0], y=X_array[:, 1], hue=model.predict(X_array), palette='viridis')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Cluster Visualization')
    st.pyplot(fig_cluster)



def select_folder():
   root = tk.Tk()
   root.withdraw()
   folder_path = filedialog.askdirectory(master=root)
   root.destroy()
   return folder_path



def import_model():
    if st.session_state.model is not None:
        trained_model = st.session_state.model

        # Get the model name from the user
        model_name = st.text_input("Entrez le nom du modèle:")


        if st.button("Select Folder"):
            selected_folder_path = st.session_state.get("folder_path", None)
            selected_folder_path = select_folder()
            st.session_state.folder_path = selected_folder_path

        if model_name and st.session_state.folder_path:
            # Combine the model name with the chosen directory to get the full path
            model_file_path = os.path.join(st.session_state.folder_path, f"{model_name}.joblib")

            # Save the trained model to the specified file
            joblib.dump(trained_model, model_file_path)

            # Display success message
            st.success(f"Modèle enregistré avec succès dans {model_file_path}")
            
    else:
        st.warning("Veuillez d'abord entraîner le modèle dans l'onglet approprié.")

def create_dataframe():
    st.subheader('Les colonnes (Metadata)')
    if st.session_state.data is not None:
        st.warning('Si vous modifier les colonnes, Tous les données vont être actualisés')
    # Get the number of columns from the user
    num_cols = st.number_input("Entrer le nombre des colonnes", min_value=1, max_value=10000, value=3)

    # Create a list of column names and types
    col_names = []
    col_types = []
    for i in range(num_cols):
        col_name = st.text_input(f"Enter the name of column {i+1}", value=f"col{i+1}")
        col_type = st.selectbox(f"Select the type of column {i+1}", options=["int", "float", "bool", "str"], index=0)
        col_names.append(col_name)
        col_types.append(col_type)

    # Create an empty dataframe with the specified column names and types
    df = pd.DataFrame(columns=col_names)
    for col_name, col_type in zip(col_names, col_types):
        if col_type == "int":
            df[col_name] = df[col_name].astype(int)
        elif col_type == "float":
            df[col_name] = df[col_name].astype(float)
        elif col_type == "bool":
            df[col_name] = df[col_name].astype(bool)
        elif col_type == "str":
            df[col_name] = df[col_name].astype(str)

    st.subheader('Les lignes (Data)')
    # Use the data editor widget to edit the dataframe
    edited_df = st.data_editor(df, num_rows='dynamic')

    # Return the edited dataframe
    return edited_df

# Fonction pour afficher les onglets
def display_tabs():
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8= st.tabs(["Importation des Données", "Visualisation", "Nettoyage des Données", "Préparation des Données", "Transformation des Données", "Entraînement du modèle", "Evaluation du modèle", "Exportation du modèle"])

    # onglet importation des données
    with tab1:
        st.header("Importation des Données")
        import_option = st.radio("Choisissez une option:", ("Importer un fichier CSV", "Créer vos propres données"))

        if import_option == "Importer un fichier CSV":
            import_csv()
        else:
            result = create_dataframe()
            if not result.empty:
                st.session_state.data = result
            else:
                st.session_state.data = None

        if check_data_exists():
            # Continue with your code that uses the data
            st.write("Les données sont prêtes à être utilisées!")

        choix_du_probleme()
        

        with st.form(key="Exit", border=False):
            st.form_submit_button("Exit", on_click=lambda: st.session_state.update({"page": 0}))  # Revenir à la landing page

    # Onglet visualisation des données
    with tab2:
        st.header("Visualisation")

        visualize_data()

    
    # onglet netoyage des données  
    with tab3:
        st.header("Nettoyage des Données")

        
        if st.button("Réinitialiser les données 1"):
            st.session_state.modified_data = st.session_state.data

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

                # Exécution de la fonction convert_object_columns_to_float() seulement si des données existent
                convert_object_columns_to_float()

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
                    column1, column2 = st.columns(2)
                    with column1:
                        # Affichage du nombre de valeurs manquantes par colonne
                        st.write("Nombre de valeurs manquantes par colonne:")
                        st.write(st.session_state.modified_data.isnull().sum())
                        # Display the modified_data types of each column
                        st.write("Types de chaque colonne:")
                        st.write(st.session_state.modified_data.dtypes)
                    with column2:
                        # Affichage du nombre de NaN values par colonne
                        st.write("Nombre de NaN values par colonne:")
                        st.write(st.session_state.data.isna().sum())
                elif st.session_state.data is not None:
                    st.warning("Aucune modification n'a été effectuée. Voici l'aperçu des données importées.")
                    st.write("Aperçu des données importées:")
                    st.write(st.session_state.data)
                    # Affichage de la taille des données
                    st.write(f"Taille des données : {st.session_state.data.shape}")
                    column1, column2 = st.columns(2)
                    with column1:
                        # Affichage du nombre de valeurs manquantes par colonne
                        st.write("Nombre de valeurs manquantes par colonne:")
                        st.write(st.session_state.data.isnull().sum())
                        # Display the data types of each column
                        st.write("Types de chaque colonne:")
                        st.write(st.session_state.data.dtypes)
                    with column2:
                        # Affichage du nombre de NaN values par colonne
                        st.write("Nombre de NaN values par colonne:")
                        st.write(st.session_state.data.isna().sum())
                    


    # onglet division des données  
    with tab4:
        st.header("Préparation des Données")

        # Exécution de la fonction split_data()
        split_data()

    with tab5:
        st.header("Transformation des Données")

        # Création de deux colonnes
        left_column, right_column = st.columns(2)

        # Dans la colonne de gauche
        with left_column:
            apply_pca()

            smote_function()
            

        # Dans la colonne de droite
        with right_column:
            plot_elbow_method()

        


    with tab6: 
        st.header("Choix de l'algorithme")

        choix_algorithme()

        choisir_hyperparametres()

        execute_algorithm()


    with tab7:
        st.header("Evaluation du modèle")

        evaluate_model()


    with tab8:
        st.header("Exportation du modèle")

        import_model()



# Fonction principale
def main():
    init_session()

    if st.session_state.page == 0:
        landing_page()
    elif st.session_state.page == 1:
        display_tabs()

if __name__ == "__main__":
    main()
