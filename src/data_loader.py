import logging
from pathlib import Path

import pandas as pd
import numpy as np
from scipy import stats

from src.config import cfg

log = logging.getLogger(__name__)

def load_data(path: str | Path | None = None):
    """Charge le dataset brut (CSV) avec des messages d'erreur explicites."""
    path = Path(path or cfg.DATA_ROW / "student_mental_health.csv")
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset introuvable : {path}. Téléchargez-le depuis Kaggle dans data/raw/."
        )
    if path.suffix.lower() != ".csv":
        raise ValueError(f"Format non supporté : {path.suffix} (CSV attendu).")
    
    try:
        df = pd.read_csv(path)
    
    except pd.error.EmptyDataError as exc:
        raise ValueError(f"Le fichier {path} est vide.") from exc
    
    log.info("Données chargées : %s lignes x %s colonnes", *df.shape)
    
    return df

def cleaning(df: pd.DataFrame) -> pd.DataFrame:
    
    df = df.copy()
    # drop the Student_ID there is no impact
    df = df.drop(columns=["Student_ID"])
    
    # delete outliers with zscore
    z_scores = np.abs(stats.zscore(df[cfg.NUM_COLS], nan_policy="omit"))
    mask = (z_scores <= 3).all(axis=1)
    df = df[mask].copy()
               
    
    return df.reset_index(drop=True)

