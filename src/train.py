import logging 

import pandas as pd
import sklearn
import numpy as np

from src.config import cfg
from src.data_loader import load_data, cleaning
from src.validation import validate_raw
from src.preprocessing import split_data

log = logging.getLogger(__name__)

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