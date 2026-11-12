import logging
from datetime import datetime, timedelta

from airflow.decorators import dag, task

log = logging.getLogger(__name__)


def notify_failure(context) -> None:
    """Callback d'échec (visible dans les logs de la tâche)."""
    ti = context["task_instance"]
    log.error(
        "ÉCHEC | dag=%s | tâche=%s | run=%s | essai=%s | erreur=%s | logs=%s",
        ti.dag_id, ti.task_id, context.get("run_id"), ti.try_number,
        context.get("exception"), ti.log_url,
    )
    
default_args = {
    "retries" : 1,
    "retry_delay" : timedelta(minutes=2),
    "on_failure_callback" : notify_failure,
}

@dag(
    dag_id="mental_health_ml_pipeline",
    description="Préparation, entraînement et évaluation du modèle de risque",
    schedule=None,                     # déclenchement manuel uniquement
    start_date=datetime(2026, 11, 1),  # requis par Airflow 2, sans effet ici
    max_active_runs=1,
    default_args=default_args,
    tags=["ml", "sante-mentale", "jury-blanc"],
    doc_md=__doc__,
)
def mental_health_ml_pipeline():
    # Imports DANS les tâches : le scheduler analyse ce fichier très souvent,
    # il reste rapide sans charger pandas / scikit-learn.
    @task
    def check_raw_data() -> str:
        from src.config import cfg
        from src.data_loader import cleaning, load_data
        from src.validation import validate_raw
        
        df = cleaning(load_data())
        validate_raw(df)
        log.info("Données valides : %s lignes, répartition %s", len(df),
                 df[cfg.TARGET_COL].value_counts().to_dict())
        return str(cfg.RAW_FILE)
    
    @task
    def prepare_data(raw_path: str) -> dict:
        from src.train import step_prepare

        return step_prepare(raw_path)
   
    @task(execution_timeout=timedelta(hours=1))
    def train_model(paths: dict) -> str:
        from src.train import step_train

        return step_train(paths["train"])
    
    @task
    def evaluate_model(model_path: str, paths: dict) -> dict:
        from src.train import step_evaluate

        m = step_evaluate(model_path, paths["test"])
        keys = ("accuracy", "precision_macro", "recall_macro", "f1_macro", "positive_class_recall")
        return {k: round(m[k], 4) for k in keys}

    @task
    def quality_gate(metrics: dict) -> None:
        from src.config import cfg

        if metrics["f1_macro"] < cfg.MIN_F1:
            raise ValueError(
                f"F1 macro {metrics['f1_macro']:.3f} < seuil {cfg.MIN_F1} : modèle à revoir."
            )
        log.info("Contrôle qualité OK : %s", metrics)

    raw = check_raw_data()
    paths = prepare_data(raw)
    model_path = train_model(paths)
    metrics = evaluate_model(model_path, paths)
    quality_gate(metrics)


mental_health_ml_pipeline()