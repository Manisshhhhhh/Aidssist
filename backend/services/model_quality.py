from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


def _coerce_numeric_array(values: Any) -> np.ndarray:
    if values is None:
        return np.asarray([], dtype=float)
    if isinstance(values, pd.Series):
        series = pd.to_numeric(values, errors="coerce")
        return series.to_numpy(dtype=float)
    if isinstance(values, pd.DataFrame):
        if values.empty:
            return np.asarray([], dtype=float)
        flattened = values.to_numpy().reshape(-1)
        return pd.to_numeric(pd.Series(flattened), errors="coerce").to_numpy(dtype=float)
    if isinstance(values, np.ndarray):
        flattened = values.reshape(-1)
        return pd.to_numeric(pd.Series(flattened), errors="coerce").to_numpy(dtype=float)
    if isinstance(values, (list, tuple, set)):
        return pd.to_numeric(pd.Series(list(values)), errors="coerce").to_numpy(dtype=float)
    return pd.to_numeric(pd.Series([values]), errors="coerce").to_numpy(dtype=float)


def _align_numeric_pairs(y_true: Any, y_pred: Any) -> tuple[np.ndarray, np.ndarray, list[str]]:
    warnings: list[str] = []
    true_array = _coerce_numeric_array(y_true)
    pred_array = _coerce_numeric_array(y_pred)

    if true_array.size == 0 or pred_array.size == 0:
        warnings.append("Model evaluation is unavailable because true or predicted values are empty.")
        return np.asarray([], dtype=float), np.asarray([], dtype=float), warnings

    pair_count = min(true_array.size, pred_array.size)
    if true_array.size != pred_array.size:
        warnings.append("Model evaluation aligned mismatched true/predicted lengths to the overlapping rows.")
    true_array = true_array[:pair_count]
    pred_array = pred_array[:pair_count]

    valid_mask = np.isfinite(true_array) & np.isfinite(pred_array)
    if not valid_mask.any():
        warnings.append("Model evaluation is unavailable because all paired rows are invalid.")
        return np.asarray([], dtype=float), np.asarray([], dtype=float), warnings

    dropped_count = int(pair_count - int(valid_mask.sum()))
    if dropped_count:
        warnings.append(f"Model evaluation dropped {dropped_count} invalid paired rows.")

    return true_array[valid_mask], pred_array[valid_mask], warnings


def evaluate_model(y_true, y_pred):
    metrics, _ = evaluate_model_with_warnings(y_true, y_pred)
    return metrics


def evaluate_model_with_warnings(y_true: Any, y_pred: Any) -> tuple[dict[str, float | None], list[str]]:
    aligned_true, aligned_pred, warnings = _align_numeric_pairs(y_true, y_pred)
    if aligned_true.size == 0:
        return {"mae": None, "r2": None}, warnings

    metrics: dict[str, float | None] = {
        "mae": float(mean_absolute_error(aligned_true, aligned_pred)),
        "r2": None,
    }
    if aligned_true.size < 2:
        warnings.append("R2 requires at least two aligned rows, so it is unavailable for this model.")
        return metrics, warnings

    try:
        metrics["r2"] = float(r2_score(aligned_true, aligned_pred))
    except Exception:
        warnings.append("R2 could not be computed for the current model output.")
    return metrics, warnings


def interpret_model_quality(mae: float | None, r2: float | None) -> str:
    del mae
    if r2 is None:
        return "weak"
    if float(r2) > 0.8:
        return "strong"
    if float(r2) > 0.5:
        return "moderate"
    return "weak"


def build_explanation(
    *,
    model: Any | None = None,
    feature_names: Iterable[str] | None = None,
    raw_importance: Iterable[float] | None = None,
    top_n: int = 5,
) -> dict[str, list[Any]]:
    importance_values: np.ndarray
    if raw_importance is not None:
        importance_values = np.asarray(list(raw_importance), dtype=float).reshape(-1)
    elif model is not None and hasattr(model, "coef_"):
        coefficients = np.asarray(getattr(model, "coef_"), dtype=float)
        if coefficients.ndim > 1:
            importance_values = np.mean(np.abs(coefficients), axis=0)
        else:
            importance_values = np.abs(coefficients.reshape(-1))
    elif model is not None and hasattr(model, "feature_importances_"):
        importance_values = np.abs(np.asarray(getattr(model, "feature_importances_"), dtype=float).reshape(-1))
    else:
        return {"top_features": [], "impact": []}

    if importance_values.size == 0:
        return {"top_features": [], "impact": []}

    resolved_feature_names = [str(name) for name in list(feature_names or [])]
    if len(resolved_feature_names) != importance_values.size:
        resolved_feature_names = [f"feature_{index + 1}" for index in range(importance_values.size)]

    magnitude = np.abs(importance_values)
    total = float(magnitude.sum())
    if total <= 0:
        normalized = np.zeros_like(magnitude)
    else:
        normalized = magnitude / total

    ranked_indexes = list(np.argsort(-normalized))[: max(1, int(top_n))]
    top_features = [resolved_feature_names[index] for index in ranked_indexes]
    impact = [round(float(normalized[index]), 6) for index in ranked_indexes]
    return {"top_features": top_features, "impact": impact}


def build_simple_prediction_diagnostics(
    df: pd.DataFrame,
    *,
    target_column: str | None,
    datetime_column: str | None = None,
) -> dict[str, Any]:
    warnings: list[str] = []
    if not target_column or target_column not in df.columns:
        warnings.append("Prediction diagnostics could not identify a valid target column.")
        return {
            "model_metrics": {"mae": None, "r2": None},
            "explanation": {"top_features": [], "impact": []},
            "warnings": warnings,
            "model_name": None,
        }

    working = df.copy()
    working[target_column] = pd.to_numeric(working[target_column], errors="coerce")
    working = working.dropna(subset=[target_column]).reset_index(drop=True)
    if len(working) < 3:
        warnings.append("Prediction diagnostics need at least three usable rows.")
        return {
            "model_metrics": {"mae": None, "r2": None},
            "explanation": {"top_features": [], "impact": []},
            "warnings": warnings,
            "model_name": None,
        }

    feature_name = "row_index"
    if datetime_column and datetime_column in working.columns:
        parsed_dates = pd.to_datetime(working[datetime_column], errors="coerce", format="mixed")
        valid_mask = parsed_dates.notna()
        if int(valid_mask.sum()) >= 3:
            working = working.loc[valid_mask].copy()
            working["_aidssist_feature"] = parsed_dates.loc[valid_mask].map(pd.Timestamp.toordinal).astype(float)
            working = working.sort_values("_aidssist_feature").reset_index(drop=True)
            feature_name = str(datetime_column)
        else:
            warnings.append("The detected datetime column could not be parsed reliably, so diagnostics used row order instead.")
            working["_aidssist_feature"] = np.arange(len(working), dtype=float)
    else:
        working["_aidssist_feature"] = np.arange(len(working), dtype=float)

    if len(working) < 3:
        warnings.append("Prediction diagnostics lost too many rows after feature preparation.")
        return {
            "model_metrics": {"mae": None, "r2": None},
            "explanation": {"top_features": [], "impact": []},
            "warnings": warnings,
            "model_name": None,
        }

    model = LinearRegression()
    X = working[["_aidssist_feature"]].to_numpy(dtype=float)
    y = working[target_column].to_numpy(dtype=float)
    model.fit(X, y)
    predictions = model.predict(X)
    metrics, metric_warnings = evaluate_model_with_warnings(y, predictions)
    warnings.extend(metric_warnings)

    return {
        "model_metrics": metrics,
        "explanation": build_explanation(model=model, feature_names=[feature_name], top_n=1),
        "warnings": list(dict.fromkeys(warnings)),
        "model_name": "linear_regression",
        "model_quality": interpret_model_quality(metrics.get("mae"), metrics.get("r2")),
    }
