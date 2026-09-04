from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import mlflow
import mlflow.xgboost
import yaml

from src.data.load import COLUMNS, load_train
from src.data.split import split_by_unit
from src.data.validate import validate_dataframe
from src.features.build_features import SENSOR_COLUMNS, add_engine_features, compute_rul_labels
from src.models.baseline_xgb import save_model, train_xgb_model
from src.models.evaluate import compute_metrics


def configure_mlflow() -> None:
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    root_dir = Path(__file__).resolve().parents[2]
    mlruns_dir = root_dir / "mlruns"
    mlruns_dir.mkdir(parents=True, exist_ok=True)
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", mlruns_dir.resolve().as_uri())
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("nasa-engine-rul")


def run_training(dataset: str = "FD001", config_path: str = "configs/train_config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    configure_mlflow()

    with mlflow.start_run(run_name=f"train-{dataset}") as run:
        mlflow.log_params({
            "dataset": dataset,
            "max_rul": config.get("max_rul", 130),
            "test_size": config.get("test_size", 0.2),
            "random_state": config.get("random_state", 42),
            "feature_window": config.get("feature_window", 5),
        })

        train_df = load_train(dataset)
        validate_dataframe(train_df, COLUMNS)

        train_raw, val_raw = split_by_unit(
            train_df,
            test_size=config.get("test_size", 0.2),
            random_state=config.get("random_state", 42),
        )

        # Fit preprocessing statistics on training engines only. Reusing those
        # values for validation prevents information from held-out engines from
        # leaking into their first-cycle lag features.
        lag_fill_values = {
            sensor: float(train_raw[sensor].median()) for sensor in SENSOR_COLUMNS
        }
        feature_window = config.get("feature_window", 5)
        train_split = add_engine_features(
            train_raw,
            window=feature_window,
            lag_fill_values=lag_fill_values,
        )
        val_split = add_engine_features(
            val_raw,
            window=feature_window,
            lag_fill_values=lag_fill_values,
        )
        max_rul = config.get("max_rul", 130)
        train_split["RUL"] = compute_rul_labels(train_split, max_rul=max_rul)
        val_split["RUL"] = compute_rul_labels(val_split, max_rul=max_rul)

        model, predictions, y_val = train_xgb_model(train_split, val_split, target_col="RUL")
        metrics = compute_metrics(y_val, predictions)

        output_dir = Path(config.get("output_dir", "artifacts"))
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = save_model(model, output_dir / f"{dataset.lower()}_model.joblib")

        metrics_path = output_dir / f"{dataset.lower()}_metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        mlflow.log_metrics({key: float(value) for key, value in metrics.items()})
        mlflow.log_artifact(str(metrics_path))
        mlflow.xgboost.log_model(model, artifact_path="model")

        return {"run_id": run.info.run_id, "model_path": str(model_path), "metrics": metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the NASA engine RUL model with MLflow tracking.")
    parser.add_argument("--dataset", default="FD001", help="Dataset name to train on")
    parser.add_argument("--config", default="configs/train_config.yaml", help="Training configuration file")
    args = parser.parse_args()
    print(run_training(dataset=args.dataset, config_path=args.config))


if __name__ == "__main__":
    main()
