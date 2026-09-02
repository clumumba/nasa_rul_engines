from __future__ import annotations

import os
from contextlib import asynccontextmanager
from math import isfinite
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, field_validator

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


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Load the model before accepting traffic and keep it in memory."""
    model_path = resolve_model_path()
    if not model_path.is_file():
        raise RuntimeError(f"Model artifact not found: {model_path}")

    try:
        model = joblib.load(model_path)
        feature_names = get_model_feature_names(model)
    except Exception as exc:
        raise RuntimeError(f"Unable to load model artifact '{model_path}': {exc}") from exc

    application.state.model = model
    application.state.feature_names = feature_names
    application.state.model_path = model_path
    yield


# FastAPI app instance
app = FastAPI(
    title="NASA Engine RUL API",
    description="Production-ready inference service for the NASA engine remaining useful life model.",
    version="0.1.0",
    lifespan=lifespan,
)


class PredictionRequest(BaseModel):
    features: dict[str, float]

    @field_validator("features")
    @classmethod
    def validate_feature_values(cls, features: dict[str, float]) -> dict[str, float]:
        invalid = sorted(name for name, value in features.items() if not isfinite(value))
        if invalid:
            raise ValueError(f"Feature values must be finite numbers: {invalid}")
        return features


def get_model_feature_names(model: Any) -> list[str]:
    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is not None:
        return [str(name) for name in feature_names]

    booster = getattr(model, "get_booster", lambda: None)()
    booster_names = getattr(booster, "feature_names", None)
    if booster_names is not None:
        return [str(name) for name in booster_names]

    feature_count = getattr(model, "n_features_in_", None)
    if feature_count is not None:
        return [f"feature_{index}" for index in range(feature_count)]

    raise HTTPException(status_code=500, detail="Model does not expose its feature schema.")


@app.get("/")
def root() -> dict:
    return {"service": "nasa-engine-rul", "status": "ok"}

# check the health of the service
@app.get("/health")
def health(request: Request) -> dict:
    model_path = getattr(request.app.state, "model_path", None)
    if model_path is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    return {"status": "ok", "model_loaded": True, "model_path": str(model_path)}


@app.post("/predict")
def predict(payload: PredictionRequest, request: Request) -> dict:
    model = getattr(request.app.state, "model", None)
    feature_names = getattr(request.app.state, "feature_names", None)
    if model is None or feature_names is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    supplied_names = set(payload.features)
    expected_names = set(feature_names)
    missing = sorted(expected_names - supplied_names)
    unexpected = sorted(supplied_names - expected_names)
    if missing or unexpected:
        detail = {}
        if missing:
            detail["missing_features"] = missing
        if unexpected:
            detail["unexpected_features"] = unexpected
        raise HTTPException(status_code=422, detail=detail)

    row = {name: float(payload.features[name]) for name in feature_names}
    prediction = model.predict(pd.DataFrame([row], columns=feature_names))[0]
    return {"predicted_rul": float(prediction)}
