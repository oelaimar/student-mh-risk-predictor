"""Chargement du pipeline sauvegardé et prédiction sur de nouvelles données."""
import json

import joblib
import pandas as pd

from src.config import cfg
from src.evaluate import ordered_proba
from src.validation import validate_input

def load_artifacts():
    """Retourne (pipeline, metadata). Lève FileNotFoundError si le modèle manque."""
    if not cfg.MODEL_PATH.exists() or not cfg.METADATA_PATH.exists():
        raise FileNotFoundError(
            "Modèle introuvable. Lancez l'entraînement (DAG Airflow)."
        )
    model = joblib.load(cfg.MODEL_PATH)
    with open(cfg.METADATA_PATH, encoding="utf-8") as f:
        metadata = json.load(f)
    return metadata, model

def predict(df: pd.DataFrame, model, metadata: dict) -> tuple[pd.DataFrame, list[str]]:
    """Retourne (données + prédiction + probabilités, avertissements)."""
    X, warnings = validate_input(df, metadata)
    classes = metadata["classes"]
    
    out = df.copy()
    out["prediction"] = model.predict(X)
    proba = ordered_proba(model, X, classes)
    for i, cls in enumerate(classes):
        out[f"proba_{cls}"] = proba[:, i].round(3)
    return out, warnings