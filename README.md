# NASA Engine RUL MLOps Project

This project implements a production-grade MLOps setup around the NASA turbofan engine remaining useful life (RUL) prediction problem. It includes a model training pipeline, automated validation, Docker packaging, MLflow experiment tracking, and a DVC pipeline definition.

## Project structure

- `src/data` — dataset loaders and validation utilities
- `src/features` — feature engineering for sensor and degradation signals
- `src/models` — training and evaluation logic
- `src/api` — FastAPI prediction service
- `configs/` — training and inference configuration
- `artifacts/` — generated model and metrics outputs
- `.github/workflows/ci.yml` — CI validation workflow
- `dvc.yaml` — DVC stage definition for reproducible training
- `docker-compose.yml` — local stack for API and MLflow

## Local setup

1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .nasa
   . .venv/bin/activate  # Windows: .venv\Scripts\activate or Source .nasa/Scripts/activate
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

2. Train the model:
   ```bash
   python -m src.pipelines.train_pipeline --dataset FD001 --config configs/train_config.yaml
   ```

3. Run the API locally:
   ```bash
   uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
   ```

4. Open the health endpoint:
   ```bash
   curl http://localhost:8000/health
   ```

## MLflow

MLflow tracking is configured in `src/pipelines/train_pipeline.py` with a local file-backed store by default.

```bash
export MLFLOW_TRACKING_URI=file://$PWD/mlruns
python -m src.pipelines.train_pipeline --dataset FD001 --config configs/train_config.yaml
mlflow ui --backend-store-uri file://$PWD/mlruns --host 0.0.0.0 --port 5000
```

## DVC

The repository includes a DVC pipeline definition in `dvc.yaml` that covers the full ML lifecycle from raw data ingestion to model serving:

```bash
pip install dvc
# if the project is initialized as a git repo:
dvc init
dvc repro
```

The pipeline stages are:

- `ingest_data` — validates and materializes the raw FD001 training dataset into `data/processed/`
- `train_model` — trains the XGBoost model and writes model and metrics artifacts
- `serve_model` — validates the model artifact and writes a serving manifest/config used by the FastAPI API

## Docker

Build and run the inference API and MLflow server locally:

```bash
docker compose up --build
```

The API is exposed on `http://localhost:8000` and MLflow on `http://localhost:5000`.

## CI/CD

GitHub Actions automatically installs dependencies and runs unit tests for every push and pull request. The workflow is defined in `.github/workflows/ci.yml`.

## Example prediction payload

```json
{
  "features": {
    "sensor_1": 41.0,
    "sensor_2": 0.82,
    "sensor_3": 100.0,
    "sensor_4": 52.0,
    "sensor_5": 445.0,
    "sensor_6": 1.0,
    "sensor_7": 0.0,
    "sensor_8": 0.0,
    "sensor_9": 0.0,
    "sensor_10": 0.0,
    "sensor_11": 0.0,
    "sensor_12": 0.0,
    "sensor_13": 0.0,
    "sensor_14": 0.0,
    "sensor_15": 0.0,
    "sensor_16": 0.0,
    "sensor_17": 0.0,
    "sensor_18": 0.0,
    "sensor_19": 0.0,
    "sensor_20": 0.0,
    "sensor_21": 0.0
  }
}
```
