import numpy as np
import pandas as pd
import pytest

from src.models.evaluate import compute_nasa_score, evaluate_last_cycle


class SensorModel:
    feature_names_in_ = np.array(["sensor_1"])

    def predict(self, features):
        return features["sensor_1"].to_numpy()


def test_nasa_score_penalizes_early_and_late_predictions():
    score = compute_nasa_score(
        y_true=[100.0, 100.0, 100.0],
        y_pred=[90.0, 100.0, 110.0],
    )

    expected = (np.exp(10.0 / 13.0) - 1.0) + (np.exp(1.0) - 1.0)
    assert score == pytest.approx(expected)


def test_official_evaluation_uses_only_each_engines_last_cycle():
    featured_test = pd.DataFrame(
        {
            "unit": [2, 1, 2, 1],
            "cycle": [1, 1, 2, 2],
            "sensor_1": [999.0, 999.0, 21.0, 12.0],
        }
    )
    rul = pd.DataFrame({"RUL": [10.0, 20.0]})

    metrics = evaluate_last_cycle(SensorModel(), featured_test, rul)

    assert metrics["MAE"] == pytest.approx(1.5)
    assert metrics["RMSE"] == pytest.approx(np.sqrt(2.5))
    assert metrics["NASA_SCORE"] == pytest.approx(
        (np.exp(2.0 / 10.0) - 1.0) + (np.exp(1.0 / 10.0) - 1.0)
    )


def test_official_evaluation_rejects_mismatched_label_count():
    featured_test = pd.DataFrame(
        {"unit": [1, 2], "cycle": [1, 1], "sensor_1": [10.0, 20.0]}
    )
    rul = pd.DataFrame({"RUL": [10.0]})

    with pytest.raises(ValueError, match="engine count"):
        evaluate_last_cycle(SensorModel(), featured_test, rul)
