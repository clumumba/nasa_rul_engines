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

The deployment workflow runs for version tags such as `v1.0.0` or manually through `workflow_dispatch`. Add a repository secret named `KUBE_CONFIG` containing the target cluster kubeconfig:

```bash
git tag v1.0.0
git push origin v1.0.0
```

The Docker workflow publishes `clumumba62/nasa_rul:v1`. The `docker-login` GitHub secret must contain a Docker Hub access token or password.

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
