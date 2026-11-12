"""Fonctions et textes partagés par les pages Streamlit."""

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Permet "import src" avec : streamlit run app/streamlit_app.py
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
    
from src.config import cfg
from src.predict import load_artifacts

# Libellés lisibles (français) et unités, issus du dictionnaire de données.
FEATURE_INFO = {
    "Age": ("Âge", "années"),
    "Gender": ("Genre", None),
    "Year_of_Study": ("Année d'études", None),
    "Sleep_Hours": ("Sommeil", "heures / nuit"),
    "Study_Hours_Per_Week": ("Travail personnel", "heures / semaine"),
    "GPA": ("Moyenne (GPA)", "sur 4.0"),
    "Social_Media_Hours": ("Réseaux sociaux", "heures / jour"),
    "Screen_Time_Hours": ("Temps d'écran total", "heures / jour"),
    "Physical_Activity": ("Activité physique", None),
    "Financial_Stress": ("Stress financier", "score 1-10"),
    "Family_Support": ("Soutien familial", "score 1-10"),
    "Attendance_Rate": ("Taux de présence", "%"),
    "Part_Time_Job": ("Emploi à temps partiel", None),
    "Diet_Quality": ("Qualité de l'alimentation", None),
    "Previous_Mental_Health_History": ("Antécédent de santé mentale", None),
}

RISK_LABELS = {"Low": "Faible", "Medium": "Modéré", "High": "Élevé"}

def label(feature: str) -> str:
    name, unit = FEATURE_INFO.get(feature, (feature, None))
    return f"{name} ({unit})" if unit else name

@st.cache_resource(show_spinner="Chargement du modèle...")
def get_model():
    """Retourne (model, metadata) ou (None, None) si aucun modèle n'est entraîné."""
    try:
        model, metadata = load_artifacts()
        return model, metadata, None
    except FileNotFoundError:
        return None, None, "Aucun modèle entraîné. Lancez le DAG Airflow ou `python -m src.train`."
    except Exception as exc:
        return None, None, f"Impossible de charger le modèle ({type(exc).__name__} : {exc}). Réentraînez-le."
    
def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def get_metrics() -> dict | None:
    return _read_json(cfg.METRICS_PATH)

@st.cache_data
def get_tuning() -> dict | None:
    return _read_json(cfg.TUNING_RESULTS_PATH)

@st.cache_data
def get_csv(name: str) -> pd.DataFrame | None:
    path = {"cv": cfg.CV_RESULTS_PATH, "importance": cfg.FEATURE_IMPORTANCE_PATH}[name]
    return pd.read_csv(path) if path.exists() else None

def require_model():
    """Arrête la page proprement si le modèle n'est pas disponible."""
    model, metadata, error = get_model()
    if error:
        st.error(error)
        st.stop()
    return model, metadata