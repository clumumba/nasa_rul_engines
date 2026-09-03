import joblib
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from sklearn.linear_model import LinearRegression

from src.api.app import app


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
