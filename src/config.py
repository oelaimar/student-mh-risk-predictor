"""Configuration centrale du projet (chemins, cible, hyperparamètres)."""

import os
from pathlib import Path

class Config():
    
    _instance: "Config | None "= None
    
    def __new__(cls) -> "Config":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load()
        return cls._instance

    def _load(self) -> None:
        """Charge la configuration une seule fois."""
        # --- Chemins ---
        self.ROOT = Path(os.getenv("PROJECT_ROOT") or Path(__file__).resolve().parents[1]).resolve()

        self.DATA_ROW = self.ROOT / "data" / "raw"
        self.DATA_PROCESSED = self.ROOT / "data" / "processed"
        self.MODELS_DIR = self.ROOT / "models"
        self.REPORTS_DIR = self.ROOT / "reports"
        self.FIGURES_DIR = self.ROOT / self.REPORTS_DIR / "figures"

        self.RAW_FILES = self.REPORTS_DIR / os.getenv("RAW_FILENAME", "students_mental_health.csv")
        self.TRAIN_FILE = self.DATA_PROCESSED / "train.csv"
        self.TEST_FILE = self.DATA_PROCESSED / "test.csv"

        self.MODEL_PATH = self.MODELS_DIR / "model_pipeline.joblib"
        self.METADATA_PATH = self.MODELS_DIR / "metadata.json"
        self.METRICS_PATH = self.REPORTS_DIR / "metrics.json"
        self.CV_RESULTS_PATH = self.REPORTS_DIR / "cv_results.csv"
        self.TUNING_RESULTS_PATH = self.REPORTS_DIR / "tuning_results.json"

        # --- Données ---
        self.TARGET_COL = os.getenv("TARGET_COL", "Mental_Health_Risk")
        self.DROP_COLS: list[str] = ["Student_ID"]
        self.ORDINAL_COLS: dict[str, list] = {
            "Physical_Activity": ["Low", "Moderate", "High"],
            "Diet_Quality": ["Poor", "Average", "Good"],
        }

        # --- Modélisation ---
        self.RANDOM_STATE = 42
        self.TEST_SIZE = 0.2
        self.CV_FOLDS = 5
        self.SCORING = os.getenv("SCORING", "f1_macro")
        self.TOP_K_MODELS = 2
        self.N_ITER_SEARCH = int(os.getenv("N_ITER_SEARCH", "20"))
        self.USE_SMOTE = os.getenv("USE_SMOTE", "true").lower() == "true"
        self.MIN_F1 = float(os.getenv("MIN_F1", "0.5"))

    def ensure_dirs(self) -> None:
        """Crée les dossiers de sortie s'ils n'existent pas."""
        for d in (self.DATA_PROCESSED, self.MODELS_DIR, self.REPORTS_DIR, self.FIGURES_DIR):
            d.mkdir(parents=True, exist_ok=True)
            
            
cfg = Config()