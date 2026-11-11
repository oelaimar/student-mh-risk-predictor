"""Métriques et graphiques d'évaluation (classification multiclasse)."""

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance 
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

log = logging.getLogger(__name__)

def ordered_proba(model, X, classes: list) -> np.ndarray:
    """Probabilités avec les colonnes dans l'ordre de `classes` (et non l'ordre alphabétique)."""
    proba = model.predict_proba(X)
    idx = [list(model.classes_).index(c) for c in classes]
    return proba[:, idx]

def compute_metrics(y_true, y_pred, y_proba, classes: list, positive_class: str | None = None) -> dict:
    """Calcule les métriques"""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "labels": [str(c) for c in classes],
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=classes).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, labels=classes, output_dict=True, zero_division=0
        ),
    }
    
    try:
        # roc_auc_score exige des labels triés : on réordonne les colonnes de probabilités
        sorted_classes = sorted(classes)
        idx = [classes.index(c) for c in sorted_classes]
        metrics["roc_auc_ovr"] = roc_auc_score(
            y_true, y_proba[:, idx], multi_class="ovr", labels=sorted_classes
        )
    except ValueError as exc:
        log.warning("ROC-AIC non calculabe : %s", exc)
        
    if positive_class in classes:
        y_true_arr, y_pred_arr = np.asarray(y_true), np.asarray(y_pred)
        is_pos = y_true_arr == positive_class
        metrics["positive_class"] = positive_class
        metrics["positive_class_recall"] = recall_score(
            y_true, y_pred, labels=[positive_class], average="macro", zero_division=0
        )
        # Erreur la plus coûteuse : un étudiant à risque élevé prédit à risque faible.
        metrics["high_predicted_as_low"] = int((is_pos & (y_pred_arr == classes[0])).sum())
    return {k: (float(v) if isinstance(v, np.floating) else v) for k, v in metrics.items()}
    
def plot_confusion_matrix(cm, labels, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(np.array(cm), display_labels=labels).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Matrice de confusion (jeu de test)")
    ax.set_xlabel("Classe prédite")
    ax.set_ylabel("Classe réelle")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)

def plot_roc_curves(y_true, y_proba, classes: list, path: Path) -> None:
    """Courbes ROC one-vs-rest : une courbe par classe."""
    fig, ax = plt.subplots(figsize=(6, 5))
    y_true = np.asarray(y_true)
    for i, cls in enumerate(classes):
        RocCurveDisplay.from_predictions(y_true == cls, y_proba[:, i], name=str(cls), ax=ax)
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Hasard")
    ax.set_title("Courbes ROC (one-vs-rest)")
    ax.set_xlabel("Taux de faux positifs")
    ax.set_ylabel("Taux de vrais positifs")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    
def compute_feature_importance(model, X, y, scoring: str, random_state: int) -> pd.DataFrame:
    """Importance par permutation sur les variables d'origine (lisible pour le métier)."""
    result = permutation_importance(
        model, X, y, scoring=scoring, n_repeats=10, random_state=random_state, n_jobs=-1
    )
    return (
        pd.DataFrame({
            "feature": X.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        })
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def plot_feature_importance(importance: pd.DataFrame, path: Path) -> None:
    data = importance.sort_values("importance_mean")
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(data["feature"], data["importance_mean"], xerr=data["importance_std"], color="#4C72B0")
    ax.set_title("Importance des variables (permutation, jeu de test)")
    ax.set_xlabel("Baisse du score quand la variable est mélangée")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_json(obj, path : Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)