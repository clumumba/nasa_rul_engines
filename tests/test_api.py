import joblib
import numpy as np
from fastapi.testclient import TestClient
from sklearn.linear_model import LinearRegression

from src.api.app import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert "model_path" in payload


def test_predict_endpoint_uses_selected_model(monkeypatch, tmp_path):
    model = LinearRegression()
    feature_names = [f"sensor_{idx}" for idx in range(1, 22)]
    X = np.array([[float(idx + 1) for idx in range(len(feature_names))]], dtype=float)
    y = np.array([12.5], dtype=float)
    model.fit(X, y)

    model_path = tmp_path / "demo_model.joblib"
    joblib.dump(model, model_path)
    monkeypatch.setenv("MODEL_PATH", str(model_path))

    payload = {name: float(index + 1) for index, name in enumerate(feature_names)}
    response = client.post("/predict", json={"features": payload})

    assert response.status_code == 200
    assert isinstance(response.json()["predicted_rul"], float)
