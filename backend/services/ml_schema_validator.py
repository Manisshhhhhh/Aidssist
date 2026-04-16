from __future__ import annotations

import math
from typing import Any


_REQUIRED_KEYS = {
    "target",
    "problem_type",
    "features",
    "metrics",
    "top_features",
    "feature_importance",
    "predictions_sample",
    "data_quality_score",
    "confidence",
    "warnings",
    "recommendations",
}
_ALLOWED_PROBLEM_TYPES = {"regression", "classification"}


def _is_finite_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate_string_list(value: Any, field_name: str, *, allow_empty: bool = True) -> list[str]:
    _require(isinstance(value, list), f"'{field_name}' must be a list.")
    normalized = [str(item).strip() for item in value]
    _require(all(item for item in normalized), f"'{field_name}' must contain only non-empty strings.")
    if not allow_empty:
        _require(bool(normalized), f"'{field_name}' must not be empty.")
    return normalized


def validate_ml_output(output):
    """
    Validates structure of ml_intelligence output.
    Raises error if invalid.
    """

    _require(isinstance(output, dict), "ML output must be a dictionary.")

    keys = set(output.keys())
    missing = sorted(_REQUIRED_KEYS - keys)
    unexpected = sorted(keys - _REQUIRED_KEYS)
    _require(not missing, f"ML output is missing required keys: {', '.join(missing)}.")
    _require(not unexpected, f"ML output contains unexpected keys: {', '.join(unexpected)}.")

    target = str(output.get("target") or "").strip()
    _require(bool(target), "'target' must be a non-empty string.")

    problem_type = str(output.get("problem_type") or "").strip().lower()
    _require(problem_type in _ALLOWED_PROBLEM_TYPES, "'problem_type' must be 'regression' or 'classification'.")

    features = _validate_string_list(output.get("features"), "features", allow_empty=False)

    metrics = output.get("metrics")
    _require(isinstance(metrics, dict), "'metrics' must be a dictionary.")
    _require(set(metrics.keys()) == {"mae", "r2"}, "'metrics' must contain exactly 'mae' and 'r2'.")
    _require(_is_finite_number(metrics.get("mae")), "'metrics.mae' must be a finite float.")
    _require(_is_finite_number(metrics.get("r2")), "'metrics.r2' must be a finite float.")

    top_features = _validate_string_list(output.get("top_features"), "top_features", allow_empty=False)

    feature_importance = output.get("feature_importance")
    _require(isinstance(feature_importance, dict), "'feature_importance' must be a dictionary.")
    _require(bool(feature_importance), "'feature_importance' must not be empty.")
    for feature_name, score in feature_importance.items():
        resolved_name = str(feature_name).strip()
        _require(bool(resolved_name), "'feature_importance' keys must be non-empty strings.")
        _require(_is_finite_number(score), f"'feature_importance.{resolved_name}' must be a finite float.")
    _require(all(feature in feature_importance for feature in top_features), "'top_features' must exist in 'feature_importance'.")
    _require(all(feature in features for feature in feature_importance), "'feature_importance' keys must be part of 'features'.")

    predictions_sample = output.get("predictions_sample")
    _require(isinstance(predictions_sample, list), "'predictions_sample' must be a list.")
    _require(bool(predictions_sample), "'predictions_sample' must not be empty.")
    _require(len(predictions_sample) <= 5, "'predictions_sample' must contain at most 5 values.")
    for prediction in predictions_sample:
        _require(prediction is not None, "'predictions_sample' must not contain None values.")
        if isinstance(prediction, str):
            _require(bool(prediction.strip()), "'predictions_sample' string values must be non-empty.")
        else:
            _require(_is_finite_number(prediction), "'predictions_sample' numeric values must be finite.")

    data_quality_score = output.get("data_quality_score")
    _require(_is_finite_number(data_quality_score), "'data_quality_score' must be a finite float.")
    _require(0.0 <= float(data_quality_score) <= 1.0, "'data_quality_score' must be between 0.0 and 1.0.")

    confidence = output.get("confidence")
    _require(_is_finite_number(confidence), "'confidence' must be a finite float.")
    _require(0.0 <= float(confidence) <= 1.0, "'confidence' must be between 0.0 and 1.0.")

    _validate_string_list(output.get("warnings"), "warnings", allow_empty=True)
    _validate_string_list(output.get("recommendations"), "recommendations", allow_empty=False)

