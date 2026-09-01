from __future__ import annotations

import os
from pathlib import Path

import joblib
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

APP_DIR = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = APP_DIR / "artifacts" / "fd001_model.joblib"
CONFIG_PATH = APP_DIR / "configs" / "inference_config.yaml"


def resolve_model_path() -> Path:
    env_path = os.getenv("MODEL_PATH")
    if env_path:
        return Path(env_path)

    if CONFIG_PATH.exists():
        with CONFIG_PATH.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        configured = config.get("model_path")
        if configured:
            path = APP_DIR / configured if not Path(configured).is_absolute() else Path(configured)
            if path.exists():
                return path

    return DEFAULT_MODEL_PATH

# FastAPI app instance
app = FastAPI(
    title="NASA Engine RUL API",
    description="Production-ready inference service for the NASA engine remaining useful life model.",
    version="0.1.0",
)


class PredictionRequest(BaseModel):
    features: dict[str, float]


@app.get("/")
def root() -> dict:
    return {"service": "nasa-engine-rul", "status": "ok"}

# check the health of the service
@app.get("/health")
def health() -> dict:
    model_path = resolve_model_path()
    return {"status": "ok", "model_loaded": model_path.exists(), "model_path": str(model_path)}


@app.post("/predict")
def predict(payload: PredictionRequest) -> dict:
    model_path = resolve_model_path()
    if not model_path.exists():
        raise HTTPException(status_code=503, detail="Model artifact not found. Train the model first.")

    model = joblib.load(model_path)
    row = payload.features
    prediction = model.predict([[float(row.get(key, 0.0)) for key in sorted(row.keys())]])[0]
    return {"predicted_rul": float(prediction)}
