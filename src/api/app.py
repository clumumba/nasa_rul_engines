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
from pydantic import BaseModel, Field, field_validator

from src.features.build_features import add_engine_features

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


class CycleReading(BaseModel):
    cycle: int = Field(ge=1)
    setting_1: float
    setting_2: float
    setting_3: float
    sensor_1: float
    sensor_2: float
    sensor_3: float
    sensor_4: float
    sensor_5: float
    sensor_6: float
    sensor_7: float
    sensor_8: float
    sensor_9: float
    sensor_10: float
    sensor_11: float
    sensor_12: float
    sensor_13: float
    sensor_14: float
    sensor_15: float
    sensor_16: float
    sensor_17: float
    sensor_18: float
    sensor_19: float
    sensor_20: float
    sensor_21: float

    @field_validator("*")
    @classmethod
    def validate_finite_readings(cls, value: Any) -> Any:
        if isinstance(value, float) and not isfinite(value):
            raise ValueError("Cycle readings must be finite numbers.")
        return value


class HistoryPredictionRequest(BaseModel):
    history: list[CycleReading] = Field(min_length=1)

    @field_validator("history")
    @classmethod
    def validate_cycle_sequence(cls, history: list[CycleReading]) -> list[CycleReading]:
        cycles = sorted(reading.cycle for reading in history)
        if len(cycles) != len(set(cycles)):
            raise ValueError("History must contain each cycle only once.")
        if any(current != previous + 1 for previous, current in zip(cycles, cycles[1:])):
            raise ValueError("History cycles must be consecutive.")
        return history


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


@app.get("/metadata")
def metadata(request: Request) -> dict:
    """Return the exact input contract embedded in the loaded model."""
    feature_names = getattr(request.app.state, "feature_names", None)
    if feature_names is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    return {"feature_count": len(feature_names), "feature_names": feature_names}


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


@app.post("/predict/history")
def predict_from_history(payload: HistoryPredictionRequest, request: Request) -> dict:
    model = getattr(request.app.state, "model", None)
    feature_names = getattr(request.app.state, "feature_names", None)
    if model is None or feature_names is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    lag_fill_values = getattr(model, "lag_fill_values_", None)
    feature_window = getattr(model, "feature_window_", None)
    if lag_fill_values is None or feature_window is None:
        raise HTTPException(
            status_code=503,
            detail="Model is missing training preprocessing metadata. Retrain and redeploy it.",
        )

    history_rows = [reading.model_dump() for reading in payload.history]
    history = pd.DataFrame(history_rows)
    history["unit"] = 1
    featured = add_engine_features(
        history,
        window=int(feature_window),
        lag_fill_values=lag_fill_values,
    )
    latest = featured.sort_values("cycle").iloc[-1]
    missing = sorted(set(feature_names) - set(featured.columns))
    if missing:
        raise HTTPException(
            status_code=500,
            detail={"model_features_not_generated": missing},
        )

    row = latest[feature_names].astype(float).to_frame().T
    prediction = model.predict(row)[0]
    return {
        "predicted_rul": float(prediction),
        "cycle": int(latest["cycle"]),
        "history_cycles": len(history_rows),
    }
