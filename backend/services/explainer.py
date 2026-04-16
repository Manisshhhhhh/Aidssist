from __future__ import annotations

from typing import Any

import numpy as np


def _resolve_estimator(model: Any) -> Any:
    if model is None:
        return None
    if hasattr(model, "named_steps") and "model" in getattr(model, "named_steps", {}):
        return model.named_steps["model"]
    return model


def _resolve_feature_names(model: Any, features: list[str]) -> list[str]:
    if (
        model is not None
        and hasattr(model, "named_steps")
        and "preprocessor" in getattr(model, "named_steps", {})
    ):
        preprocessor = model.named_steps["preprocessor"]
        if hasattr(preprocessor, "get_feature_names_out"):
            try:
                return [str(name) for name in preprocessor.get_feature_names_out()]
            except Exception:
                pass
    return [str(feature) for feature in features]


def _map_transformed_feature(transformed_name: str, original_features: list[str]) -> str:
    raw_name = str(transformed_name)
    if "__" in raw_name:
        raw_name = raw_name.split("__", 1)[1]

    for original in sorted((str(item) for item in original_features), key=len, reverse=True):
        if raw_name == original or raw_name.startswith(f"{original}_"):
            return original
    return raw_name


def explain_model(model: Any, features: list[str], *, top_n: int = 5) -> dict[str, list[Any]]:
    estimator = _resolve_estimator(model)
    if estimator is None:
        return {"top_features": [], "impact": []}

    raw_importance: np.ndarray
    if hasattr(estimator, "coef_"):
        coefficients = np.asarray(getattr(estimator, "coef_"), dtype=float)
        if coefficients.ndim > 1:
            raw_importance = coefficients.mean(axis=0)
        else:
            raw_importance = coefficients.reshape(-1)
    elif hasattr(estimator, "feature_importances_"):
        raw_importance = np.asarray(getattr(estimator, "feature_importances_"), dtype=float).reshape(-1)
    else:
        return {"top_features": [], "impact": []}

    if raw_importance.size == 0:
        return {"top_features": [], "impact": []}

    transformed_features = _resolve_feature_names(model, features)
    if len(transformed_features) != raw_importance.size:
        transformed_features = [f"feature_{index + 1}" for index in range(raw_importance.size)]

    grouped_importance: dict[str, float] = {}
    for transformed_name, score in zip(transformed_features, raw_importance, strict=False):
        original_name = _map_transformed_feature(transformed_name, features)
        grouped_importance[original_name] = grouped_importance.get(original_name, 0.0) + float(score)

    total_magnitude = float(sum(abs(value) for value in grouped_importance.values()))
    if total_magnitude <= 0:
        return {"top_features": [], "impact": []}

    ranked = sorted(
        grouped_importance.items(),
        key=lambda item: (-abs(item[1]), item[0]),
    )[: max(1, int(top_n))]
    return {
        "top_features": [feature for feature, _ in ranked],
        "impact": [round(float(score) / total_magnitude, 6) for _, score in ranked],
    }


__all__ = ["explain_model"]
