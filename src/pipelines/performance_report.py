from __future__ import annotations

import json
from pathlib import Path
import shutil

import joblib
import matplotlib.pyplot as plt
import yaml

from src.data.load import DATASETS, load_rul, load_test, load_train
from src.features.build_features import add_engine_features, compute_rul_labels
from src.models.evaluate import compute_metrics, compute_nasa_score

ROOT_DIR = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = ROOT_DIR / "artifacts"
PLOT_DIR = ROOT_DIR / "docs" / "assets"


def evaluate_saved_models() -> dict[str, dict]:
    config = yaml.safe_load(
        (ROOT_DIR / "configs" / "train_config.yaml").read_text(encoding="utf-8")
    )
    results: dict[str, dict] = {}
    predictions_by_dataset: dict[str, tuple[list[float], list[float]]] = {}

    for dataset in DATASETS:
        model_path = ARTIFACT_DIR / f"{dataset.lower()}_model.joblib"
        model = joblib.load(model_path)
        train = load_train(dataset)
        test = load_test(dataset)
        true_rul = load_rul(dataset)["RUL"].to_numpy(dtype=float)

        lag_fill_values = getattr(
            model,
            "lag_fill_values_",
            {sensor: float(train[sensor].median()) for sensor in train.columns if sensor.startswith("sensor_")},
        )
        feature_window = int(getattr(model, "feature_window_", config["feature_window"]))
        featured_test = add_engine_features(
            test,
            window=feature_window,
            lag_fill_values=lag_fill_values,
        )
        final_rows = (
            featured_test.sort_values(["unit", "cycle"])
            .groupby("unit", sort=True)
            .tail(1)
            .sort_values("unit")
        )
        feature_names = [str(name) for name in model.feature_names_in_]
        predictions = model.predict(final_rows[feature_names]).astype(float)
        if len(predictions) != len(true_rul):
            raise ValueError(
                f"{dataset}: prediction count does not match official RUL label count."
            )

        train_labels = compute_rul_labels(
            train,
            max_rul=int(config["max_rul"]),
        )
        baseline_predictions = [float(train_labels.mean())] * len(true_rul)
        model_metrics = compute_metrics(true_rul, predictions)
        model_metrics["NASA_SCORE"] = compute_nasa_score(true_rul, predictions)
        baseline_metrics = compute_metrics(true_rul, baseline_predictions)
        baseline_metrics["NASA_SCORE"] = compute_nasa_score(
            true_rul,
            baseline_predictions,
        )
        results[dataset] = {
            "evaluation": "official NASA test set, final cycle per engine",
            "engine_count": len(true_rul),
            "model": model_metrics,
            "mean_training_rul_baseline": baseline_metrics,
        }
        predictions_by_dataset[dataset] = (true_rul.tolist(), predictions.tolist())

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "performance_metrics.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )
    (ROOT_DIR / "docs").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(
        ARTIFACT_DIR / "performance_metrics.json",
        ROOT_DIR / "docs" / "performance_metrics.json",
    )
    write_performance_plots(results, predictions_by_dataset)
    return results


def write_performance_plots(
    results: dict[str, dict],
    predictions_by_dataset: dict[str, tuple[list[float], list[float]]],
) -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    datasets = list(results)
    positions = range(len(datasets))
    width = 0.35

    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for axis, metric, title in zip(
        axes,
        ("MAE", "RMSE"),
        ("Official-test MAE", "Official-test RMSE"),
    ):
        model_values = [results[name]["model"][metric] for name in datasets]
        baseline_values = [
            results[name]["mean_training_rul_baseline"][metric] for name in datasets
        ]
        axis.bar(
            [position - width / 2 for position in positions],
            model_values,
            width,
            label="XGBoost",
        )
        axis.bar(
            [position + width / 2 for position in positions],
            baseline_values,
            width,
            label="Training-label mean baseline",
        )
        axis.set_title(title)
        axis.set_ylabel("RUL cycles (lower is better)")
        axis.set_xticks(list(positions), datasets)
        axis.legend()
    figure.suptitle("C-MAPSS official-test performance")
    performance_plot = ARTIFACT_DIR / "model-performance.png"
    figure.savefig(performance_plot, dpi=160)
    shutil.copyfile(performance_plot, PLOT_DIR / performance_plot.name)
    plt.close(figure)

    true_rul, predictions = predictions_by_dataset["FD001"]
    figure, axis = plt.subplots(figsize=(6, 6), constrained_layout=True)
    axis.scatter(true_rul, predictions, alpha=0.75, label="FD001 engine predictions")
    minimum = min(min(true_rul), min(predictions))
    maximum = max(max(true_rul), max(predictions))
    axis.plot([minimum, maximum], [minimum, maximum], linestyle="--", color="black", label="Ideal")
    axis.set(
        title="FD001 official-test predictions",
        xlabel="True RUL (cycles)",
        ylabel="Predicted RUL (cycles)",
    )
    axis.legend()
    predictions_plot = ARTIFACT_DIR / "fd001-predictions.png"
    figure.savefig(predictions_plot, dpi=160)
    shutil.copyfile(predictions_plot, PLOT_DIR / predictions_plot.name)
    plt.close(figure)


if __name__ == "__main__":
    print(json.dumps(evaluate_saved_models(), indent=2))
