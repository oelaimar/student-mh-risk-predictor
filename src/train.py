import logging
import time
from datetime import datetime

import joblib
import pandas as pd
import sklearn
import numpy as np
from scipy.stats import loguniform
from src.config import cfg
from src.data_loader import load_data, cleaning
from src.validation import validate_raw
from src.preprocessing import split_data, split_xy, infer_column_types, build_pipeline
from src.evaluate import save_json

from sklearn.metrics import make_scorer, precision_score
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate


log = logging.getLogger(__name__)
RS = cfg.RANDOM_STATE

def get_models() -> dict:
    """Modèles comparés avec les mêmes données et le même preprocessing."""
    return {
        "dummy_baseline" : DummyClassifier(strategy="most_frequent"),
        "logistic_regression" : LogisticRegression(max_iter=2000, random_state=RS),
        "random_forest" : RandomForestClassifier(n_estimators=300, random_state=RS, n_jobs=-1),
        "decision_tree" : DecisionTreeClassifier(random_state=RS)
    }

PARAM_DISTRIBUTIONS = {
    "logistic_regression": {
        "model__C": loguniform(1e-3, 1e2),
        "model__class_weight": [None, "balanced"],
    },
    "random_forest": {
        "model__n_estimators": [200, 300, 500],
        "model__max_depth": [None, 5, 10, 20],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", "log2"],
    },
    "decision_tree": {
        "model__max_depth": [3, 5, 7, 10, 15, None],
        "model__min_samples_split": [2, 5, 10, 20],
        "model__min_samples_leaf": [1, 2, 5, 10],
        "model__criterion": ["gini", "entropy"],
    },
}

def _cv() -> StratifiedKFold:
    return StratifiedKFold(n_splits=cfg.CV_FOLDS, shuffle=True, random_state=RS)

# --- préparation

def step_prepare(raw_path: str | None = None) -> dict:
    """load, clean and split data (80/20 stratify)"""
    cfg.ensure_dirs()
    df = cleaning(load_data(raw_path))
    validate_raw(df)
    
    X_train, X_test, y_train, y_test = split_data(df)
    pd.concat([X_train, y_train], axis=1).to_csv(cfg.TRAIN_FILE, index=False)
    pd.concat([X_test, y_test], axis=1).to_csv(cfg.TEST_FILE, index=False)
    log.info("Train : %s lignes | Test : %s lignes", len(X_train), len(X_test))
    return {"train": str(cfg.TRAIN_FILE), "test": str(cfg.TEST_FILE)}

# --- omparaison, tuning, sauvegarde 
def compare_models(X, y, numeric, nominal, ordinal) -> pd.DataFrame:
    """Validation croisée de chaque modèle avec le même pipeline."""
    scoring = {
        "accuracy": "accuracy",
        "precision_macro": make_scorer(precision_score, average="macro", zero_division=0),
        "recall_macro" : "recall_macro",
        "f1_macro" : "f1_macro",
    }
    rows = []
    for name, model in get_models().items():
        pipe = build_pipeline(model, numeric, nominal, ordinal)
        start = time.perf_counter()
        scores = cross_validate(pipe, X, y, cv=_cv(), scoring=scoring, n_jobs=-1, error_score="raise")
        row = {"model": name, "fit_time_s": round(time.perf_counter() - start, 2)}
        for s in scoring:
            row[f"{s}_mean"] = scores[f"test_{s}"].mean()
            row[f"{s}_std"] = scores[f"test_{s}"].std()
        rows.append(row)
        log.info("CV %-20s %s = %.3f", name, cfg.SCORING, row[f"{cfg.SCORING}_mean"])
    results = pd.DataFrame(rows).sort_values(f"{cfg.SCORING}_mean", ascending=False)
    results.to_csv(cfg.CV_RESULTS_PATH, index=False)
    return results

def tune_models(X, y, numeric, nominal, ordinal, candidates : list[str]):
    """RandomizedSearchCV sur les modèles les plus prometteurs."""
    models = get_models()
    best_name, best_search, summary = None, None, {}
    for name in candidates:
        params = dict(PARAM_DISTRIBUTIONS[name])
        search = RandomizedSearchCV(
            build_pipeline(models[name], numeric, nominal, ordinal),
            param_distributions=params,
            n_iter=cfg.N_ITER_SEARCH,
            scoring=cfg.SCORING,
            cv=_cv(),
            random_state=RS,
            n_jobs=-1,
        )
        start = time.perf_counter()
        search.fit(X, y)
        summary[name] = {
            "best_score": search.best_score_,
            "best_params": search.best_params_,
            "search_time_s": round(time.perf_counter() - start, 2),
        }
        log.info("Tuning %-20s best %s = %.3f", name, cfg.SCORING, search.best_score_)
        if best_search is None or search.best_score_ > best_search.best_score_:
            best_name, best_search = name, search
    save_json(summary, cfg.TUNING_RESULTS_PATH)
    return best_name, best_search
        

def step_train(train_path: str | None = None) -> str:
    """Compare, optimise et sauvegarde le meilleur pipeline complet (preprocessing + modèle)."""
    cfg.ensure_dirs()
    df = pd.read_csv(train_path or cfg.TRAIN_FILE)
    X, y = split_xy(df)
    numeric, nominal, ordinal = infer_column_types(X)
    
    results = compare_models(X, y, numeric, nominal, ordinal)
    candidates = [m for m in results["model"] if m in PARAM_DISTRIBUTIONS][: cfg.TOP_K_MODELS]
    best_name, search = tune_models(X, y, numeric, nominal, ordinal, candidates)
    
    model = search.best_estimator_
    joblib.dump(model, cfg.MODEL_PATH)
    
    raw_numeric = [c for c in numeric if c in X.columns]
    metadata = {
        "model_name": best_name,
        "best_params": search.best_params_,
        "cv_score": search.best_score_,
        "scoring": cfg.SCORING,
        "target": cfg.TARGET_COL,
        "classes": cfg.TARGET_CLASSES,
        "features": list(X.columns),
        "numeric_features": raw_numeric,
        "engineered_features": [c for c in numeric if c not in X.columns],
        "nominal_features": nominal,
        "ordinal_features": ordinal,
        "numeric_stats": {
            c: {
                "min": float(X[c].min()),
                "max": float(X[c].max()),
                "median": float(X[c].median()),
                "is_integer": bool((X[c].dropna() % 1 == 0).all()),
            }
            for c in raw_numeric
        },
        "categories": {c: sorted(X[c].dropna().astype(str).unique().tolist()) for c in nominal},
        "ordinal_categories": {c: cfg.ORDINAL_COLS[c] for c in ordinal},
        "n_train": len(X),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
    }
    save_json(metadata, cfg.METADATA_PATH)
    log.info("Modèle sauvegardé : %s (%s)", cfg.MODEL_PATH, best_name)
    return str(cfg.MODEL_PATH)