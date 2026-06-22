from pathlib import Path

import pandas as pd

from src.config import DATA_PATH, FEATURE_COLUMNS


def load_data(path: Path = DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    """Load Iris dataset and return feature matrix and species labels."""
    df = pd.read_csv(path)

    if "Id" in df.columns:
        df = df.drop(columns=["Id"])

    X = df[FEATURE_COLUMNS]
    y = df["Species"]
    return X, y
