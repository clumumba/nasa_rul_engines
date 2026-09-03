# NASA Engine RUL MLOps Project

This project implements an MLOps workflow for predicting remaining useful life (RUL) for NASA turbofan engines. It includes data validation, feature engineering, XGBoost training, MLflow tracking, DVC reproducibility, a FastAPI inference service, Docker packaging, and Kubernetes deployment manifests.

## Quick start

```bash
python -m pip install -r requirements.txt
python -m dvc repro
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

The API is available at `http://localhost:8000`. Open `http://localhost:8000/docs` for interactive API documentation.

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

See [PRODUCTION_CHANGES.md](PRODUCTION_CHANGES.md) for the inference startup,
health-check, dependency, and deployment changes required for production.

Build and run the inference API and MLflow server locally:

```bash
docker compose up --build
```

The API is exposed on `http://localhost:8000` and MLflow on `http://localhost:5000`.

## CI/CD

GitHub Actions automatically installs dependencies, runs unit tests, reproduces the DVC pipeline, and validates the pipeline definition for every push and pull request. The workflows also build and publish the API image to Docker Hub on pushes, validate Kubernetes manifests, and deploy tagged releases when the `KUBE_CONFIG` repository secret is configured.

- `.github/workflows/ci.yml` — tests and `dvc repro`
- `.github/workflows/docker-build.yml` — DVC reproduction and Docker build/publish
- `.github/workflows/deploy_svc.yml` — Kubernetes manifest validation
- `.github/workflows/kubectl-deploy.yml` — Kubernetes deployment
- `deploy.yml` and `deploy_svc.yml` — API deployment and service manifests

## Kubernetes deployment

The root-level manifests deploy the Docker Hub image `clumumba62/nasa_rul:v1`:

- `deploy.yml` — two API replicas with CPU and memory requests/limits
- `deploy_svc.yml` — `NodePort` service on port `30080`

Confirm that `kubectl` is installed and connected to the target cluster:

```bash
kubectl config current-context
kubectl cluster-info
```

Apply the deployment and service:

```bash
kubectl apply -f deploy.yml
kubectl apply -f deploy_svc.yml
```

Verify the rollout and resources:

```bash
kubectl rollout status deployment/nasa-rul-deployment
kubectl get deployments,pods,services -l app=nasa_rul
kubectl describe service nasa-rul-service
```

The service uses `NodePort` `30080`. For Minikube, get the service URL with:

```bash
minikube service nasa-rul-service --url
```

For another cluster, call the health endpoint at `<node-ip>:30080`:

```bash
curl http://<node-ip>:30080/health
```

Remove the resources when finished:

```bash
kubectl delete -f deploy_svc.yml
kubectl delete -f deploy.yml
```

### Kubernetes deployment through GitHub Actions

The Docker workflow publishes a commit-tagged image to Docker Hub on every push to `main`. After that workflow succeeds, the deployment workflow runs on the self-hosted Docker Desktop runner, replaces the image in the checked-out `deploy.yml` with `clumumba62/nasa_rul:<commit-sha>`, and applies the deployment and service. Add a repository secret named `KUBE_CONFIG` only if using a remote runner/cluster; the Docker Desktop self-hosted runner uses its local `docker-desktop` context.

You can also start the deployment manually through `workflow_dispatch`:

```bash
kubectl config use-context docker-desktop
kubectl apply -f deploy.yml
kubectl apply -f deploy_svc.yml
```

The Docker workflow requires the `DOCKER_USERNAME` and `DOCKER_LOGIN` repository secrets. It publishes both `clumumba62/nasa_rul:<commit-sha>` and `clumumba62/nasa_rul:v1`; the deployment uses the immutable commit tag.

## Prediction payload

The model is trained on the engineered feature vector, not only the 21 raw sensor
values. The `/predict` endpoint requires every trained feature name, including
`_lag1`, `_delta`, `_rolling_mean_5`, and `_rolling_std_5` fields for each sensor.
Feature names and ordering are read from the saved model, so missing or unknown
features are rejected with HTTP 422 rather than silently filled with zeros.

Example using a reduced test model:

```json
{
  "features": {
    "sensor_1": 41.0,
    "sensor_1_lag1": 40.5,
    "sensor_1_delta": 0.5,
    "sensor_1_rolling_mean_5": 40.8,
    "sensor_1_rolling_std_5": 0.2
  }
}
```

The current production FD001 model expects **108** values: `setting_1` through
`setting_3`, the 21 raw sensor values, and four derived values for each sensor.
Get the exact ordered contract from `GET /metadata`; this avoids relying on a
hard-coded list when a model is replaced.

## How an end user sends a prediction

`POST /predict` intentionally accepts an already-engineered feature vector. A
single raw sensor reading is not sufficient: lag, delta, and rolling values need
the earlier readings for the same engine unit. An application integrating this
model should retain each engine's recent history, use the same feature code as
training, and submit the latest engineered row.

```python
import requests
from src.features.build_features import add_engine_features

# history contains chronologically ordered readings for one unit: unit, cycle,
# setting_1..setting_3, and sensor_1..sensor_21. Keep at least five cycles.
engineered = add_engine_features(history)
required = requests.get("http://localhost:8001/metadata").json()["feature_names"]
features = engineered.iloc[-1][required].astype(float).to_dict()

response = requests.post("http://localhost:8001/predict", json={"features": features})
response.raise_for_status()
print(response.json()["predicted_rul"])
```

For the first cycles of a new engine, the integration must use the same
initial-value policy as training. Do not invent missing lag or rolling values;
send the feature values produced by the versioned feature pipeline.
