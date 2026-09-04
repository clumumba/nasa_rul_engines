from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

SENSOR_COLUMNS = [f"sensor_{idx}" for idx in range(1, 22)]

#compute remaining useful life (RUL) labels for each engine unit
def compute_rul_labels(df: pd.DataFrame, max_rul: int = 130) -> pd.Series:
    cycle_max = df.groupby("unit")["cycle"].transform("max")
    rul = cycle_max - df["cycle"]
    return rul.clip(lower=0, upper=max_rul)

#add rolling and lag features for each engine unit
def add_engine_features(
    df: pd.DataFrame,
    window: int = 5,
    lag_fill_values: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    output = df.sort_values(["unit", "cycle"]).copy()

    for sensor in SENSOR_COLUMNS:
        lag_fill = (
            float(lag_fill_values[sensor])
            if lag_fill_values is not None
            else float(output[sensor].median())
        )
        output[f"{sensor}_lag1"] = output.groupby("unit")[sensor].shift(1).fillna(lag_fill)
        output[f"{sensor}_delta"] = output.groupby("unit")[sensor].diff().fillna(0.0)
        output[f"{sensor}_rolling_mean_{window}"] = (
            output.groupby("unit")[sensor]
            .transform(lambda series: series.rolling(window=window, min_periods=1).mean())
        )
        output[f"{sensor}_rolling_std_{window}"] = (
            output.groupby("unit")[sensor]
            .transform(lambda series: series.rolling(window=window, min_periods=1).std().fillna(0.0))
        )

    return output
