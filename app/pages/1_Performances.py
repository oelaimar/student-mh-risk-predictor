"""Page : performances du modèle sur le jeu de test."""
import pandas as pd
import streamlit as st

from utils import RISK_LABELS, cfg, get_csv, get_metrics, get_tuning, label, require_model

st.set_page_config(page_title="Performances", layout="wide")
st.title("Performances du modèle")

model, metadata = require_model()
metrics = get_metrics()
if metrics is None:
    st.error("Aucune métrique trouvée (reports/metrics.json). Lancez l'évaluation.")
    st.stop()

st.caption(
    f"Modèle : **{metadata['model_name']}** · jeu de test : {metrics.get('n_test', '-')} étudiants · "
    f"évalué le {metrics.get('evaluated_at', '-').replace('T', ' ')}"
)

# --- Métriques principales -----------------------------------------------------
cols = st.columns(6)
items = [
    ("Accuracy", "accuracy"), ("Précision", "precision_macro"), ("Rappel", "recall_macro"),
    ("F1-score", "f1_macro"), ("ROC-AUC", "roc_auc_ovr"), ("Rappel risque élevé", "positive_class_recall"),
]
for col, (name, key) in zip(cols, items):
    value = metrics.get(key)
    col.metric(name, f"{value:.3f}" if value is not None else "-")
st.caption(
    "Précision, rappel et F1 en moyenne macro (chaque classe compte autant). "
    "ROC-AUC en one-vs-rest. Le rappel du risque élevé est suivi en priorité : "
    "ne pas repérer un étudiant à risque est l'erreur la plus coûteuse."
)
if metrics.get("high_predicted_as_low") is not None:
    st.info(
        f"Étudiants à risque **élevé** prédits à risque **faible** (erreur critique) : "
        f"**{metrics['high_predicted_as_low']}**"
    )

# --- Graphiques ------------------------------------------------------------------
left, right = st.columns(2)
with left:
    st.subheader("Matrice de confusion")
    path = cfg.FIGURES_DIR / "confusion_matrix.png"
    if path.exists():
        st.image(str(path))
    else:
        st.dataframe(pd.DataFrame(metrics["confusion_matrix"], index=metrics["labels"], columns=metrics["labels"]))
with right:
    st.subheader("Courbes ROC")
    path = cfg.FIGURES_DIR / "roc_curves.png"
    if path.exists():
        st.image(str(path))

# --- Rapport par classe ----------------------------------------------------------
st.subheader("Rapport par classe")
report = pd.DataFrame(metrics["classification_report"]).T
report = report.rename(index=RISK_LABELS)
st.dataframe(report.style.format("{:.3f}"), use_container_width=True)

# --- Comparaison des modèles -----------------------------------------------------
st.subheader("Comparaison des modèles (validation croisée 5 folds, jeu d'entraînement)")
cv = get_csv("cv")
if cv is not None:
    cv = cv.set_index("model")
    st.dataframe(cv.style.format("{:.3f}"), use_container_width=True)
    st.bar_chart(cv[[f"{cfg.SCORING}_mean"]])

tuning = get_tuning()
if tuning:
    st.subheader("Optimisation des hyperparamètres (RandomizedSearchCV)")
    rows = [{"modèle": k, "meilleur score CV": round(v["best_score"], 4),
             "durée (s)": v.get("search_time_s")} for k, v in tuning.items()]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    with st.expander("Hyperparamètres du modèle retenu"):
        st.json(metadata["best_params"])

# --- Importance des variables ----------------------------------------------------
importance = get_csv("importance")
if importance is not None:
    st.subheader("Importance des variables (permutation sur le jeu de test)")
    importance["variable"] = importance["feature"].map(label)
    st.bar_chart(importance.set_index("variable")["importance_mean"].sort_values(), horizontal=True)
    st.caption("Baisse du score quand une variable est mélangée aléatoirement : plus elle est grande, "
               "plus le modèle s'appuie sur cette variable. Association statistique, pas causalité.")
