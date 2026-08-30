from src.data.load import load_train
from src.features.build_features import add_engine_features, compute_rul_labels


def test_feature_pipeline_creates_expected_fields():
    df = load_train("FD001").head(50)
    df = add_engine_features(df, window=5)
    rul = compute_rul_labels(df, max_rul=130)

    assert "sensor_1_rolling_mean_5" in df.columns
    assert "sensor_1_rolling_std_5" in df.columns
    assert len(rul) == len(df)
    assert rul.min() >= 0
