import numpy as np
import pandas as pd
import logging

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

from src.config import cfg

log = logging.getLogger(__name__)
       

# --- Feature engineering

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crée des variables métier (calculs ligne par ligne, donc sans leakage).
    
    - Sleep_Deficit : heures manquantes sous le seuil de 7 h de sommeil
      (le manque de sommeil est le facteur le plus lié au risque à l'EDA).
    - Social_Media_Share : part du temps d'écran passée sur les réseaux sociaux.
    - Support_Stress_Balance : soutien familial moins stress financier
      (facteur protecteur vs facteur de risque).

    Les NaN se propagent et sont ensuite imputés par le pipeline.
    """
    df = df.copy()
    cols = set(df.columns)
    if "Sleep_Hours" in cols:
        df["Sleep_Deficit"] = (7 - df["Sleep_Hours"]).clip(lower=0)
    if {"Social_Media_Hours", "Screen_Time_Hours"} <= cols:
        screen = df["Screen_Time_Hours"].where(df["Screen_Time_Hours"] > 0)
        df["Social_Media_Share"] = df["Social_Media_Hours"] / screen
    if {"Family_Support", "Financial_Stress"} <= cols:
        df["Support_Stress_Balance"] = df["Family_Support"] - df["Financial_Stress"]
    return df
        
# --- Split

def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df.drop(columns=[cfg.TARGET_COL]), df[cfg.TARGET_COL]

def split_data(df: pd.DataFrame):
    """Split 80/20 stratifié : même proportion de classes dans le train et le test."""
    X, y = split_xy(df)
    return train_test_split(X, y, test_size=cfg.TEST_SIZE, stratify=y, random_state=cfg.RANDOM_STATE)

def infer_column_types(X: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    """Retourne (numériques, nominales, ordinales) après feature engineering."""
    X = add_features(X)
    ordinal = [c for c in cfg.ORDINAL_COLS if c in X.columns]
    numeric = [c for c in X.select_dtypes(include="number").columns if c not in ordinal]
    nominal = [c for c in X.columns if c not in numeric and c not in ordinal]
    return numeric, nominal, ordinal


# --- Pipeline
def build_preprocessor(numeric: list[str], nominal: list[str], ordinal: list[str]) -> ColumnTransformer:
    """Imputation, outliers, encodage et standardisation selon le type de variable."""
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")), # median : robuste aux valeurs extrêmes
        ("scaler", StandardScaler())
    ])
    
    nominal_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse_output=False))
    ])
    
    ordinal_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
                    categories=[cfg.ORDINAL_COLS[c] for c in ordinal],
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                )),
        ("scalar", StandardScaler())
    ])
    
    transformers = [("num", numeric_pipe, numeric), ("nom", nominal_pipe, nominal)]
    
    if ordinal:
        transformers.append(("ord", ordinal_pipe, ordinal))
            
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def build_pipeline(model, numeric, nominal, ordinal) -> Pipeline:
    """Pipeline complet : feature engineering -> preprocessing -> modèle."""
    steps = [
        ("features", FunctionTransformer(add_features, validate=False)),
        ("preprocess", build_preprocessor(numeric, nominal, ordinal)),
        ("model", model),
    ]
    return Pipeline(steps)