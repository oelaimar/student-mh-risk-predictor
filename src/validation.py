import pandas as pd

from src.config import cfg

class DataValidationError(ValueError):
    """Données invalides : le pipeline ou l'application doit s'arrêter proprement."""

def validate_raw(df: pd.DataFrame, min_rows: int = 30) -> None: 
    """Vérifie que le dataset est exploitable pour l'entraînement."""
    
    errors: list[str] = []
    target = cfg.TARGET_COL
    
    if df.empty:
        errors.append("Le dataset est vide.")
    elif len(df) < min_rows:
        errors.append(f"Trop peu de lignes ({len(df)} < {min_rows}).")
    
    if target not in df.columns:
        errors.append(f"Colonne cible '{target}' absente. Colonnes : {list(df.columns)}")
    else:
        unknown = set(df[target].dropna().unique()) - set(cfg.TARGET_CLASSES)
        if unknown:
            errors.append(f"Classes inattendues dans la cible : {sorted(map(str, unknown))}")
        if df[target].nunique(dropna=True) < 2:
            errors.append("La cible contient moins de 2 classes.")
    
    if errors:
        raise DataValidationError(" | ".join(errors))