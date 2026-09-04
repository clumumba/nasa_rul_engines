import pandas as pd

from src.data.load import load_train
from src.features.build_features import (
    SENSOR_COLUMNS,
    add_engine_features,
    compute_rul_labels,
)


def test_feature_pipeline_creates_expected_fields():
    df = load_train("FD001").head(50)
    df = add_engine_features(df, window=5)
    rul = compute_rul_labels(df, max_rul=130)

    assert "sensor_1_rolling_mean_5" in df.columns
    assert "sensor_1_rolling_std_5" in df.columns
    assert len(rul) == len(df)
    assert rul.min() >= 0


def test_feature_pipeline_uses_training_lag_fill_values_for_new_engines():
    rows = []
    for unit, first_value in [(10, 100.0), (20, 200.0)]:
        for cycle in [1, 2]:
            row = {"unit": unit, "cycle": cycle}
            row.update({sensor: first_value + cycle for sensor in SENSOR_COLUMNS})
            rows.append(row)

    validation_df = pd.DataFrame(rows)
    training_fill_values = {sensor: 42.0 for sensor in SENSOR_COLUMNS}

    featured = add_engine_features(
        validation_df,
        window=2,
        lag_fill_values=training_fill_values,
    )

    first_cycles = featured[featured["cycle"] == 1]
    assert (first_cycles["sensor_1_lag1"] == 42.0).all()
    second_cycle_lags = featured.loc[
        featured["cycle"] == 2, "sensor_1_lag1"
    ].tolist()
    assert second_cycle_lags == [101.0, 201.0]
