from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def compute_nasa_score(y_true, y_pred) -> float:
    """Return the asymmetric C-MAPSS score, which penalizes late alerts more."""
    true_values = np.asarray(y_true, dtype=float).reshape(-1)
    predicted_values = np.asarray(y_pred, dtype=float).reshape(-1)
    if true_values.shape != predicted_values.shape:
        raise ValueError("NASA score inputs must contain the same number of values.")

    errors = predicted_values - true_values
    penalties = np.where(
        errors < 0,
        np.exp(-errors / 13.0) - 1.0,
        np.exp(errors / 10.0) - 1.0,
    )
    return float(penalties.sum())


def evaluate_last_cycle(model, featured_test_df, rul_df):
    """Evaluate one prediction per test engine against the official RUL labels."""
    final_rows = (
        featured_test_df.sort_values(["unit", "cycle"])
        .groupby("unit", sort=True)
        .tail(1)
        .sort_values("unit")
    )
    true_rul = rul_df["RUL"].to_numpy(dtype=float)
    if len(final_rows) != len(true_rul):
        raise ValueError(
            "Official test engine count does not match the number of RUL labels."
        )

    feature_names = [str(name) for name in model.feature_names_in_]
    predictions = model.predict(final_rows[feature_names])
    metrics = compute_metrics(true_rul, predictions)
    metrics["NASA_SCORE"] = compute_nasa_score(true_rul, predictions)
    return metrics
