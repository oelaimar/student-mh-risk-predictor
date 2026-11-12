"""Page : tester le modèle sur de nouvelles données."""
import pandas as pd
import streamlit as st

from utils import RISK_LABELS, cfg, label, require_model
from src.predict import predict
from src.validation import DataValidationError

st.set_page_config(page_title="Prédiction", layout="wide")
st.title("Tester le modèle")

model, metadata = require_model()
tab_single, tab_batch = st.tabs(["Profil individuel", "Fichier CSV"])


def show_warnings(warnings: list[str]) -> None:
    for w in warnings:
        st.warning(w)


# --- Prédiction individuelle --------------------------------------------------
with tab_single:
    with st.form("single_prediction"):
        values = {}
        cols = st.columns(3)
        for i, feature in enumerate(metadata["features"]):
            col = cols[i % 3]
            if feature in metadata["numeric_stats"]:
                s = metadata["numeric_stats"][feature]
                if s["is_integer"]:
                    values[feature] = col.number_input(
                        label(feature), min_value=int(s["min"]), max_value=int(s["max"]),
                        value=int(round(s["median"])), step=1,
                    )
                else:
                    values[feature] = col.number_input(
                        label(feature), min_value=float(s["min"]), max_value=float(s["max"]),
                        value=float(s["median"]), step=0.1, format="%.1f",
                    )
            elif feature in metadata["ordinal_categories"]:
                options = metadata["ordinal_categories"][feature]
                values[feature] = col.selectbox(label(feature), options, index=len(options) // 2)
            elif feature in metadata["categories"]:
                values[feature] = col.selectbox(label(feature), metadata["categories"][feature])
        submitted = st.form_submit_button("Estimer le niveau de risque", type="primary")

    if submitted:
        try:
            result, warnings = predict(pd.DataFrame([values]), model, metadata)
            show_warnings(warnings)
            pred = result["prediction"].iloc[0]
            text = f"Niveau de risque estimé : **{RISK_LABELS.get(pred, pred)}** (indicatif)"
            {"Low": st.success, "Medium": st.warning, "High": st.error}.get(pred, st.info)(text)

            proba = pd.DataFrame({
                "niveau": [RISK_LABELS.get(c, c) for c in metadata["classes"]],
                "probabilité": [result[f"proba_{c}"].iloc[0] for c in metadata["classes"]],
            }).set_index("niveau")
            st.bar_chart(proba)
        except (DataValidationError, ValueError) as exc:
            st.error(f"Prédiction impossible : {exc}")

# --- Prédiction par lot -------------------------------------------------------
with tab_batch:
    st.write("Colonnes attendues :", ", ".join(f"`{c}`" for c in metadata["features"]))
    sample = cfg.ROOT / "data" / "sample" / "sample_students.csv"
    if sample.exists():
        st.download_button("Télécharger un fichier d'exemple", sample.read_bytes(),
                           file_name="sample_students.csv", mime="text/csv")

    uploaded = st.file_uploader("Fichier CSV", type="csv")
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            result, warnings = predict(df, model, metadata)
            show_warnings(warnings)
            st.success(f"{len(result)} ligne(s) prédite(s).")

            counts = result["prediction"].map(RISK_LABELS).value_counts()
            st.bar_chart(counts)
            st.dataframe(result, use_container_width=True)
            st.download_button(
                "Télécharger les résultats", result.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv", mime="text/csv",
            )
        except pd.errors.EmptyDataError:
            st.error("Le fichier est vide.")
        except (DataValidationError, ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
            st.error(f"Fichier invalide : {exc}")
