from __future__ import annotations

import math
from typing import Any

import pandas as pd

from backend.data_quality import build_data_quality_report


def _safe_float(value: Any, *, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(parsed):
        return default
    return parsed


def _round_float(value: Any, digits: int = 2) -> float:
    return round(_safe_float(value), digits)


def _clip_unit_interval(value: Any) -> float:
    return round(min(1.0, max(0.0, _safe_float(value))), 2)


def _normalize_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for item in value:
        resolved = str(item).strip()
        if not resolved or resolved in seen:
            continue
        seen.add(resolved)
        normalized.append(resolved)
    return normalized


def _normalize_features(value: Any) -> list[str]:
    return _normalize_string_list(value)


def _normalize_feature_importance(value: Any, features: list[str]) -> dict[str, float]:
    raw = dict(value or {})
    normalized: dict[str, float] = {}
    for feature in features:
        if feature not in raw:
            continue
        score = abs(_safe_float(raw.get(feature)))
        if score <= 0:
            continue
        normalized[feature] = score

    total = sum(normalized.values())
    if total <= 0:
        return {}

    return {
        feature: round(score / total, 2)
        for feature, score in sorted(normalized.items(), key=lambda item: (-item[1], item[0]))
    }


def _normalize_predictions(value: Any) -> list[Any]:
    if not isinstance(value, list):
        return []

    normalized: list[Any] = []
    for prediction in value[:5]:
        if prediction is None:
            continue
        if isinstance(prediction, str):
            resolved = prediction.strip()
            if resolved:
                normalized.append(resolved)
            continue
        normalized.append(round(_safe_float(prediction), 2))
    return normalized


def _build_missing_data_warnings(df: pd.DataFrame) -> list[str]:
    if df is None or df.empty:
        return []

    warnings: list[str] = []
    for column_name in df.columns:
        missing_ratio = float(df[column_name].isna().mean()) if len(df) else 0.0
        if missing_ratio >= 0.4:
            warnings.append(f"High missing values in {column_name}")
        elif missing_ratio >= 0.2:
            warnings.append(f"Moderate missing values in {column_name}")
    return warnings


def _build_default_recommendations(target: str, top_features: list[str]) -> list[str]:
    recommendations: list[str] = []
    for feature in top_features[:2]:
        recommendations.append(f"Prioritize {feature} because it is one of the strongest drivers of {target}.")
    if not recommendations and target:
        recommendations.append(f"Collect more signal around {target} before using this model for decisions.")
    return recommendations


def postprocess_ml_output(raw_output, df):
    """
    Cleans and enriches ML output.
    """

    payload = dict(raw_output or {})
    features = _normalize_features(payload.get("features"))
    feature_importance = _normalize_feature_importance(
        payload.get("feature_importance") or payload.get("importance_scores"),
        features,
    )
    top_features = [feature for feature, _ in sorted(feature_importance.items(), key=lambda item: (-item[1], item[0]))[:5]]

    problem_type = str(
        payload.get("problem_type")
        or payload.get("target_type")
        or payload.get("type")
        or ""
    ).strip().lower()

    raw_metrics = dict(payload.get("metrics") or {})
    accuracy = _safe_float(raw_metrics.get("accuracy"))
    raw_r2 = raw_metrics.get("r2")
    raw_mae = raw_metrics.get("mae")
    if problem_type == "classification":
        if raw_r2 is None and "accuracy" in raw_metrics:
            raw_r2 = accuracy
        if raw_mae is None and "accuracy" in raw_metrics:
            raw_mae = 1.0 - accuracy

    metrics = {
        "mae": _round_float(raw_mae),
        "r2": _round_float(raw_r2),
    }

    data_quality_report = build_data_quality_report(df)
    data_quality_score = _clip_unit_interval(_safe_float(data_quality_report.get("score")) / 10.0)
    confidence = round(min(data_quality_score, max(0.0, min(1.0, _safe_float(metrics.get("r2"))))), 2)

    warnings = _normalize_string_list(payload.get("warnings"))
    warnings.extend(_build_missing_data_warnings(df))
    if float(metrics["r2"]) < 0.3:
        warnings.append("Low R2 score; review feature quality and model suitability.")
    warnings = _normalize_string_list(warnings)

    recommendations = _normalize_string_list(payload.get("recommendations"))
    if not recommendations:
        recommendations = _build_default_recommendations(
            str(payload.get("target") or "").strip(),
            top_features,
        )

    predictions = _normalize_predictions(payload.get("predictions_sample") or payload.get("predictions"))

    return {
        "target": str(payload.get("target") or "").strip(),
        "problem_type": problem_type,
        "features": features,
        "metrics": metrics,
        "top_features": top_features,
        "feature_importance": feature_importance,
        "predictions_sample": predictions,
        "data_quality_score": data_quality_score,
        "confidence": confidence,
        "warnings": warnings,
        "recommendations": recommendations,
    }
