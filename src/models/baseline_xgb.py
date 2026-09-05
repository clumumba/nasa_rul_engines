from __future__ import annotations

from pathlib import Path

import joblib
import xgboost as xgb


def select_feature_columns(df, target_col: str = "RUL"):
    excluded = {"unit", "cycle", target_col}
    return [col for col in df.columns if col not in excluded]


def fit_xgb_model(train_df, target_col: str = "RUL"):
    feature_cols = select_feature_columns(train_df, target_col)

    x_train = train_df[feature_cols]
    y_train = train_df[target_col]

    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42,
    )

    model.fit(x_train, y_train)
    return model


def train_xgb_model(train_df, val_df, target_col: str = "RUL"):
    model = fit_xgb_model(train_df, target_col)
    feature_cols = select_feature_columns(train_df, target_col)
    x_val = val_df[feature_cols]
    y_val = val_df[target_col]
    predictions = model.predict(x_val)

    return model, predictions, y_val


def save_model(model, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    return output_path
