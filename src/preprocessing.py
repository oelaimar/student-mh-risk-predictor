import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

from src.config import cfg

def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df.drop(columns=[cfg.TARGET_COL]), df[cfg.TARGET_COL]

def split_data(df: pd.DataFrame):
    """Split 80/20 stratifié : même proportion de classes dans le train et le test."""
    X, y = split_xy(df)
    return train_test_split(X, y, test_size=cfg.TEST_SIZE, stratify=y, random_state=cfg.RANDOM_STATE)

def step_train(train_path: str | None = None) -> str:
    """Compare, optimise et sauvegarde le meilleur pipeline complet (preprocessing + modèle)."""
    cfg.ensure_dirs()
    