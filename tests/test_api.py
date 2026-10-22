import joblib
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from sklearn.linear_model import LinearRegression

from src.api.app import app
from src.features.build_features import SENSOR_COLUMNS, add_engine_features


def test_health_endpoint(monkeypatch, tmp_path):
    model = LinearRegression().fit(pd.DataFrame([[1.0]], columns=["sensor_1"]), [12.5])
    model_path = tmp_path / "health_model.joblib"
    joblib.dump(model, model_path)
    monkeypatch.setenv("MODEL_PATH", str(model_path))

    with TestClient(app) as test_client:
        response = test_client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["model_loaded"] is True
    assert "model_path" in payload


def test_metadata_endpoint_returns_model_schema(monkeypatch, tmp_path):
    model = LinearRegression().fit(
        pd.DataFrame([[1.0, 2.0]], columns=["setting_1", "sensor_1"]), [12.5]
    )
    model_path = tmp_path / "metadata_model.joblib"
    joblib.dump(model, model_path)
    monkeypatch.setenv("MODEL_PATH", str(model_path))

    with TestClient(app) as test_client:
        response = test_client.get("/metadata")

    assert response.status_code == 200
    assert response.json() == {
        "feature_count": 2,
        "feature_names": ["setting_1", "sensor_1"],
    }


def test_predict_endpoint_uses_selected_model(monkeypatch, tmp_path):
    model = LinearRegression()
    feature_names = [
        "sensor_1",
        "sensor_1_lag1",
        "sensor_1_delta",
        "sensor_1_rolling_mean_5",
    ]
    X = pd.DataFrame([[1.0, 2.0, 1.0, 1.5]], columns=feature_names)
    y = np.array([12.5], dtype=float)
    model.fit(X, y)

    model_path = tmp_path / "demo_model.joblib"
    joblib.dump(model, model_path)
    monkeypatch.setenv("MODEL_PATH", str(model_path))

    payload = {name: float(index + 1) for index, name in enumerate(feature_names)}
    with TestClient(app) as test_client:
        response = test_client.post("/predict", json={"features": payload})

    assert response.status_code == 200
    assert isinstance(response.json()["predicted_rul"], float)


def test_predict_endpoint_rejects_features_that_do_not_match_model_schema(monkeypatch, tmp_path):
    model = LinearRegression()
    feature_names = ["sensor_1", "sensor_1_lag1"]
    model.fit(pd.DataFrame([[1.0, 2.0]], columns=feature_names), [12.5])

    model_path = tmp_path / "demo_model.joblib"
    joblib.dump(model, model_path)
    monkeypatch.setenv("MODEL_PATH", str(model_path))

    with TestClient(app) as test_client:
        response = test_client.post("/predict", json={"features": {"sensor_1": 1.0}})

    assert response.status_code == 422
    assert response.json()["detail"]["missing_features"] == ["sensor_1_lag1"]


def test_predict_endpoint_rejects_non_finite_features(monkeypatch, tmp_path):
    model = LinearRegression().fit(pd.DataFrame([[1.0]], columns=["sensor_1"]), [12.5])
    model_path = tmp_path / "demo_model.joblib"
    joblib.dump(model, model_path)
    monkeypatch.setenv("MODEL_PATH", str(model_path))

    with TestClient(app) as test_client:
        response = test_client.post("/predict", json={"features": {"sensor_1": "NaN"}})

    assert response.status_code == 422


def test_predict_history_endpoint_engineers_features_using_training_metadata(
    monkeypatch, tmp_path
):
    history = []
    for cycle in range(1, 4):
        reading = {
            "cycle": cycle,
            "setting_1": 0.1 * cycle,
            "setting_2": 0.2 * cycle,
            "setting_3": 0.3 * cycle,
        }
        reading.update(
            {sensor: float(index + cycle) for index, sensor in enumerate(SENSOR_COLUMNS)}
        )
        history.append(reading)

    training_history = pd.DataFrame(history)
    training_history["unit"] = 1
    training_features = add_engine_features(
        training_history,
        window=5,
        lag_fill_values={sensor: 0.0 for sensor in SENSOR_COLUMNS},
    )
    feature_names = [
        column for column in training_features.columns if column not in {"unit", "cycle"}
    ]
    training_matrix = training_features[feature_names]
    model = LinearRegression().fit(training_matrix, [20.0, 19.0, 18.0])
    model.feature_window_ = 5
    model.lag_fill_values_ = {sensor: 0.0 for sensor in SENSOR_COLUMNS}

    model_path = tmp_path / "history_model.joblib"
    joblib.dump(model, model_path)
    monkeypatch.setenv("MODEL_PATH", str(model_path))

    with TestClient(app) as test_client:
        response = test_client.post("/predict/history", json={"history": history})

    assert response.status_code == 200
    assert response.json()["cycle"] == 3
    assert response.json()["history_cycles"] == 3
    assert isinstance(response.json()["predicted_rul"], float)


def test_predict_history_endpoint_rejects_nonconsecutive_cycles(monkeypatch, tmp_path):
    model = LinearRegression().fit(pd.DataFrame([[1.0]], columns=["sensor_1"]), [12.5])
    model_path = tmp_path / "history_validation_model.joblib"
    joblib.dump(model, model_path)
    monkeypatch.setenv("MODEL_PATH", str(model_path))

    reading = {
        "cycle": 1,
        "setting_1": 0.1,
        "setting_2": 0.2,
        "setting_3": 0.3,
    }
    reading.update({sensor: 1.0 for sensor in SENSOR_COLUMNS})
    second_reading = {**reading, "cycle": 3}

    with TestClient(app) as test_client:
        response = test_client.post(
            "/predict/history",
            json={"history": [reading, second_reading]},
        )

    assert response.status_code == 422
