"""Page d'accueil de l'application Streamlit."""
import streamlit as st

from utils import get_metrics, get_model

st.set_page_config(page_title="Risque santé mentale étudiants", layout="wide")

st.title("Prédiction du risque de santé mentale chez les étudiants")

st.markdown(
"""
    Prototype de classification supervisée qui estime un niveau de risque
    (**faible**, **modéré** ou **élevé**) à partir de facteurs de mode de vie,
    académiques et psychosociaux.

    - **Performances** : métriques sur le jeu de test, matrice de confusion,
    courbes ROC, comparaison des modèles et importance des variables.
    - **Prédiction** : tester le modèle sur un profil saisi ou sur un fichier CSV.

    Le pipeline (préparation, entraînement, évaluation) est orchestré par **Airflow**.
"""
)

model, metadata, error = get_model()

metrics = get_metrics()

c1, c2, c3, c4 = st.columns(4)

if error:
    c1.error(error)
else:
    c1.metric("Modèle retenu", metadata["model_name"].replace("_", " "))
    c2.metric("Entraîné le", metadata["trained_at"][:10])
    
if metrics:
    c3.metric("F1 macro (test)", f"{metrics['f1_macro']:.3f}")
    c4.metric("Rappel risque élevé", f"{metrics['positive_class_recall']:.3f}")