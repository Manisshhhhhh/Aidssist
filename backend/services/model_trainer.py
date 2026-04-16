from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from backend.services.target_detector import coerce_datetime_series, infer_target_type


RANDOM_STATE = 42


def _build_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _prepare_feature_frame(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    prepared = df[features].copy()
    for column_name in list(prepared.columns):
        series = prepared[column_name]
        datetime_values = coerce_datetime_series(series, column_name=column_name)
        if datetime_values is not None:
            prepared[column_name] = datetime_values.map(
                lambda value: float(value.toordinal()) if pd.notna(value) else np.nan
            )
    return prepared


def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_columns = [
        str(column)
        for column in X.columns
        if pd.api.types.is_numeric_dtype(X[column]) or pd.api.types.is_bool_dtype(X[column])
    ]
    categorical_columns = [str(column) for column in X.columns if str(column) not in numeric_columns]

    transformers = []
    if numeric_columns:
        transformers.append(
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                numeric_columns,
            )
        )
    if categorical_columns:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", _build_one_hot_encoder()),
                    ]
                ),
                categorical_columns,
            )
        )

    return ColumnTransformer(transformers=transformers, remainder="drop")


def _evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": None,
    }
    if len(y_true) >= 2:
        try:
            metrics["r2"] = float(r2_score(y_true, y_pred))
        except Exception:
            metrics["r2"] = None
    return metrics


def _evaluate_classification(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | None]:
    accuracy = float(accuracy_score(y_true, y_pred))
    return {
        # The ml_intelligence contract requires unified mae/r2 fields.
        # For classification we expose error rate and accuracy-driven quality.
        "mae": float(1.0 - accuracy),
        "r2": accuracy,
        "accuracy": accuracy,
    }


def _regression_score(metrics: dict[str, float | None]) -> tuple[float, float]:
    r2_value = float(metrics.get("r2")) if metrics.get("r2") is not None else float("-inf")
    mae_value = float(metrics.get("mae")) if metrics.get("mae") is not None else float("inf")
    return (r2_value, -mae_value)


def _classification_score(metrics: dict[str, float | None]) -> float:
    if metrics.get("accuracy") is None:
        return float("-inf")
    return float(metrics["accuracy"])


def _build_models(task_type: str) -> dict[str, Any]:
    if task_type == "classification":
        return {
            "logistic_regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            "random_forest": RandomForestClassifier(
                n_estimators=200,
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
        }
    return {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
    }


def train_model(df: pd.DataFrame, target: str, features: list[str]) -> dict[str, Any]:
    if df is None or df.empty or target not in df.columns or not features:
        return {
            "model": None,
            "metrics": {"mae": None, "r2": None},
            "predictions": [],
            "model_name": None,
            "task_type": None,
        }

    target_series = df[target].copy()
    task_type = infer_target_type(target_series)
    X = _prepare_feature_frame(df, features)
    working = X.copy()
    working["_target"] = target_series
    working = working.dropna(subset=["_target"]).reset_index(drop=True)
    if working.shape[0] < 8:
        return {
            "model": None,
            "metrics": {"mae": None, "r2": None},
            "predictions": [],
            "model_name": None,
            "task_type": task_type,
        }

    y_raw = working.pop("_target")
    X = working
    label_encoder: LabelEncoder | None = None

    if task_type == "classification":
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_raw.astype(str))
        class_counts = pd.Series(y).value_counts()
        stratify = y if int(class_counts.min()) >= 2 and len(class_counts) > 1 else None
    else:
        y = pd.to_numeric(y_raw, errors="coerce")
        valid_mask = y.notna()
        X = X.loc[valid_mask].reset_index(drop=True)
        y = y.loc[valid_mask].to_numpy(dtype=float)
        stratify = None

    if len(y) < 8:
        return {
            "model": None,
            "metrics": {"mae": None, "r2": None},
            "predictions": [],
            "model_name": None,
            "task_type": task_type,
        }

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )
    preprocessor = _build_preprocessor(X)
    candidate_models = _build_models(task_type)

    best_name: str | None = None
    best_pipeline: Pipeline | None = None
    best_predictions: np.ndarray | None = None
    best_metrics: dict[str, float | None] | None = None

    for model_name, estimator in candidate_models.items():
        pipeline = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("model", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        if task_type == "classification":
            metrics = _evaluate_classification(np.asarray(y_test), np.asarray(predictions))
            is_better = (
                best_metrics is None
                or _classification_score(metrics) > _classification_score(best_metrics)
                or (
                    _classification_score(metrics) == _classification_score(best_metrics)
                    and str(model_name) < str(best_name)
                )
            )
        else:
            metrics = _evaluate_regression(np.asarray(y_test, dtype=float), np.asarray(predictions, dtype=float))
            is_better = (
                best_metrics is None
                or _regression_score(metrics) > _regression_score(best_metrics)
                or (
                    _regression_score(metrics) == _regression_score(best_metrics)
                    and str(model_name) < str(best_name)
                )
            )

        if is_better:
            best_name = model_name
            best_pipeline = pipeline
            best_predictions = np.asarray(predictions)
            best_metrics = metrics

    if best_pipeline is None or best_metrics is None or best_predictions is None:
        return {
            "model": None,
            "metrics": {"mae": None, "r2": None},
            "predictions": [],
            "model_name": None,
            "task_type": task_type,
        }

    if task_type == "classification" and label_encoder is not None:
        decoded_predictions = label_encoder.inverse_transform(best_predictions.astype(int))
        predictions_payload = decoded_predictions.tolist()
    else:
        predictions_payload = best_predictions.tolist()

    return {
        "model": best_pipeline,
        "metrics": best_metrics,
        "predictions": predictions_payload,
        "model_name": best_name,
        "task_type": task_type,
    }


__all__ = ["train_model"]
