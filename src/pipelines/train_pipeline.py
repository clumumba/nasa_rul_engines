from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import mlflow
import mlflow.xgboost
import yaml

from src.data.load import COLUMNS, DATASETS, load_rul, load_test, load_train
from src.data.split import split_by_unit
from src.data.validate import validate_dataframe
from src.features.build_features import SENSOR_COLUMNS, add_engine_features, compute_rul_labels
from src.models.baseline_xgb import fit_xgb_model, save_model, train_xgb_model
from src.models.evaluate import compute_metrics, evaluate_last_cycle


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

        holdout_model, predictions, y_val = train_xgb_model(
            train_split,
            val_split,
            target_col="RUL",
        )
        holdout_metrics = {
            f"holdout_{name}": value
            for name, value in compute_metrics(y_val, predictions).items()
        }

        # Once the internal holdout has measured the chosen configuration, fit
        # the deployable model on every available training engine and evaluate
        # one final-cycle prediction per engine against NASA's official labels.
        full_lag_fill_values = {
            sensor: float(train_df[sensor].median()) for sensor in SENSOR_COLUMNS
        }
        full_train = add_engine_features(
            train_df,
            window=feature_window,
            lag_fill_values=full_lag_fill_values,
        )
        full_train["RUL"] = compute_rul_labels(full_train, max_rul=max_rul)
        model = fit_xgb_model(full_train, target_col="RUL")

        test_df = load_test(dataset)
        validate_dataframe(test_df, COLUMNS)
        featured_test = add_engine_features(
            test_df,
            window=feature_window,
            lag_fill_values=full_lag_fill_values,
        )
        official_metrics = {
            f"official_test_{name}": value
            for name, value in evaluate_last_cycle(
                model,
                featured_test,
                load_rul(dataset),
            ).items()
        }
        metrics = {**holdout_metrics, **official_metrics}

        output_dir = Path(config.get("output_dir", "artifacts"))
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = save_model(model, output_dir / f"{dataset.lower()}_model.joblib")

        metrics_path = output_dir / f"{dataset.lower()}_metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        mlflow.log_metrics({key: float(value) for key, value in metrics.items()})
        mlflow.log_artifact(str(metrics_path))
        mlflow.xgboost.log_model(model, artifact_path="model")

        return {"run_id": run.info.run_id, "model_path": str(model_path), "metrics": metrics}


def run_all_training(config_path: str = "configs/train_config.yaml") -> dict[str, dict]:
    """Train and evaluate every NASA C-MAPSS subset."""
    return {
        dataset: run_training(dataset=dataset, config_path=config_path)
        for dataset in DATASETS
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the NASA engine RUL model with MLflow tracking.")
    parser.add_argument(
        "--dataset",
        default="FD001",
        type=str.upper,
        choices=[*DATASETS, "ALL"],
        help="Dataset to train on, or ALL for FD001 through FD004",
    )
    parser.add_argument("--config", default="configs/train_config.yaml", help="Training configuration file")
    args = parser.parse_args()
    if args.dataset == "ALL":
        result = run_all_training(config_path=args.config)
    else:
        result = run_training(dataset=args.dataset, config_path=args.config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
