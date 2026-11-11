"""Data loading and preprocessing."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config


def load_data(path: Path = config.DATA_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'.\n"
            "Download it from Mendeley Data (rz75w3fkxy) and place it in the data/ folder. "
            "See data/README.md for details."
        )
    return pd.read_csv(path)


def preprocess(df: pd.DataFrame):
    """Drop rows without a target, median-impute numeric columns, split X / y."""
    df = df.dropna(subset=[config.TARGET_COLUMN]).copy()
    numeric = df.select_dtypes(include=[np.number]).columns
    df[numeric] = df[numeric].fillna(df[numeric].median())

    X = df.drop(columns=[config.TARGET_COLUMN] + config.DROP_COLUMNS, errors="ignore")
    X = X.select_dtypes(include=[np.number])
    y = df[config.TARGET_COLUMN]
    return X, y


def split(X, y):
    return train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )
