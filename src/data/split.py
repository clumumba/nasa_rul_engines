from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split


def split_by_unit(df: pd.DataFrame, target_col: str = "RUL", test_size: float = 0.2, random_state: int = 42):
    units = df["unit"].unique()
    train_units, val_units = train_test_split(units, test_size=test_size, random_state=random_state)

    train_df = df[df["unit"].isin(train_units)].copy()
    val_df = df[df["unit"].isin(val_units)].copy()

    if train_df.empty or val_df.empty:
        raise ValueError("Split by unit produced an empty training or validation set.")

    return train_df, val_df
