from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

COLUMNS = [
    "unit",
    "cycle",
    "setting_1",
    "setting_2",
    "setting_3",
    "sensor_1",
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_5",
    "sensor_6",
    "sensor_7",
    "sensor_8",
    "sensor_9",
    "sensor_10",
    "sensor_11",
    "sensor_12",
    "sensor_13",
    "sensor_14",
    "sensor_15",
    "sensor_16",
    "sensor_17",
    "sensor_18",
    "sensor_19",
    "sensor_20",
    "sensor_21",
]

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
DATASETS = ["FD001", "FD002", "FD003", "FD004"]


def load_train(dataset: str = "FD001") -> pd.DataFrame:
    path = DATA_DIR / f"train_{dataset}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Train file not found: {path}")
    return pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS)


def load_test(dataset: str = "FD001") -> pd.DataFrame:
    path = DATA_DIR / f"test_{dataset}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Test file not found: {path}")
    return pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS)


def load_rul(dataset: str = "FD001") -> pd.DataFrame:
    path = DATA_DIR / f"RUL_{dataset}.txt"
    if not path.exists():
        raise FileNotFoundError(f"RUL file not found: {path}")
    return pd.read_csv(path, sep=r"\s+", header=None, names=["RUL"])


def load_dataset(dataset: str = "FD001", include_rul: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """Load one dataset's train/test files and, optionally, its RUL labels.

    Always returns a 3-tuple: (train, test, rul_or_none). When include_rul is False,
    the third element will be None. This makes the return shape stable for static type
    checkers and callers that unpack into three variables.
    """
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset '{dataset}'. Valid options: {DATASETS}")
    train = load_train(dataset)
    test = load_test(dataset)
    if include_rul:
        rul = load_rul(dataset)
        return train, test, rul
    return train, test, None


def load_all_datasets():
    """Load FD001–FD004 into a consistent train/test/RUL dictionary."""
    all_datasets = {}

    for dataset_id in DATASETS:
        train, test, rul = load_dataset(dataset_id, include_rul=True)
        all_datasets[dataset_id] = {
            "train": train,
            "test": test,
            "rul": rul,
        }

    return all_datasets
