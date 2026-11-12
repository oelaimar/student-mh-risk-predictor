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
    
def validate_input(df: pd.DataFrame, metadata: dict) -> tuple[pd.DataFrame, list[str]]:
    """Prépare de nouvelles données pour la prédiction.

    Retourne (X, avertissements) :
    - erreur bloquante si des colonnes attendues manquent ;
    - conversion des colonnes numériques (texte invalide -> NaN, imputé par le pipeline) ;
    - avertissements pour les valeurs hors de la plage d'entraînement
      ou les modalités inconnues (gérées par le pipeline, mais signalées).
    """
    if df is None or df.empty:
        raise DataValidationError("Aucune donnée à prédire.")
    
    expected = metadata["features"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise DataValidationError(f"Colonnes manquantes : {missing}")
    
    X = df[expected].copy()
    warnings: list[str] = []
    
    for col in metadata["numeric_features"]:
        raw = X[col]
        X[col] = pd.to_numeric(raw, errors="coerce")
        n_bad = int(X[col].isna().sum() - raw.isna().sum())
        if n_bad:
            warnings.append(f"{col} : {n_bad} valeur(s) non numérique(s) remplacée(s) par la médiane.")
        stats = metadata["numeric_stats"][col]
        n_out = int(((X[col] < stats["min"]) | (X[col] > stats["max"])).sum())
        if n_out:
            warnings.append(
                f"{col} : {n_out} valeur(s) hors plage [{stats['min']} ; {stats['max']}], "
                "ramenée(s) aux bornes apprises."
            )
    allowed = {**metadata["categories"], **metadata["ordinal_categories"]}
    for col, values in allowed.items():
        unknown = set(X[col].dropna().astype(str)) - set(map(str, values))
        if unknown:
            warnings.append(f"{col} : modalité(s) inconnue(s) {sorted(unknown)}.")

    n_na = int(X.isna().sum().sum())
    if n_na:
        warnings.append(f"{n_na} valeur(s) manquante(s) imputée(s) par le pipeline.")

    return X, warnings
