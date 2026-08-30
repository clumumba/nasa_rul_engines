from __future__ import annotations

from typing import Iterable

import pandas as pd


def validate_columns(df: pd.DataFrame, expected_columns: Iterable[str]) -> None:
    expected = list(expected_columns)
    missing = [col for col in expected if col not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")


def validate_dataframe(df: pd.DataFrame, expected_columns: Iterable[str]) -> None:
    if df.empty:
        raise ValueError("DataFrame is empty.")

    validate_columns(df, expected_columns)

    if df.isnull().any().any():
        null_columns = df.columns[df.isnull().any()].tolist()
        raise ValueError(f"DataFrame contains null values in: {null_columns}")

    if "unit" not in df.columns:
        raise ValueError("DataFrame must contain a 'unit' column.")

    if df["unit"].nunique() == 0:
        raise ValueError("No engine units found in dataset.")

    if "cycle" not in df.columns:
        raise ValueError("DataFrame must contain a 'cycle' column.")
