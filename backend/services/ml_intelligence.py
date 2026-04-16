from __future__ import annotations

from typing import Any

import pandas as pd

from backend.services.feature_selector import select_features
from backend.services.explainer import explain_model
from backend.services.model_trainer import train_model
from backend.services.recommendation_engine import generate_recommendations
from backend.services.target_detector import (
    detect_target_column,
    infer_target_type,
    is_datetime_like,
    is_id_like,
)


def _empty_ml_intelligence() -> dict[str, Any]:
    return {
        "target": "",
        "problem_type": "",
        "features": [],
        "metrics": {"mae": 0.0, "r2": 0.0},
        "top_features": [],
        "feature_importance": {},
        "predictions": [],
        "data_quality_score": 0.0,
        "confidence": 0.0,
        "warnings": [],
        "recommendations": [],
    }


def _valid_target(df: pd.DataFrame, target: str | None) -> bool:
    if not target or target not in df.columns:
        return False
    series = df[target]
    if is_id_like(series, column_name=target):
        return False
    if is_datetime_like(series, column_name=target):
        return False
    return True

def build_ml_intelligence(
    df: pd.DataFrame,
    *,
    user_query: str | None = None,
    target_hint: str | None = None,
    insights: list[str] | None = None,
) -> dict[str, Any]:
    payload = _empty_ml_intelligence()
    if df is None or df.empty or len(df.columns) < 2:
        return payload

    detection = detect_target_column(df, user_query)
    target = str(target_hint or "").strip() if _valid_target(df, target_hint) else ""
    if not target and _valid_target(df, detection.get("target")):
        target = str(detection.get("target") or "")
    if not target:
        return payload

    target_type = str(detection.get("type") or infer_target_type(df[target]))
    feature_payload = select_features(df, target)
    selected_features = list(feature_payload.get("selected_features") or [])
    training_payload = train_model(df, target, selected_features)
    explanation = explain_model(
        training_payload.get("model"),
        selected_features,
        top_n=len(selected_features) or 5,
    )

    explanation_scores = {
        feature: score
        for feature, score in zip(
            explanation.get("top_features", []),
            explanation.get("impact", []),
            strict=False,
        )
    }
    recommendation_payload = generate_recommendations(
        df,
        target,
        list(insights or []),
        explanation_scores or feature_payload.get("importance_scores") or {},
    )
    metrics = dict(training_payload.get("metrics") or {"mae": 0.0, "r2": 0.0})
    payload.update(
        {
            "target": target,
            "problem_type": target_type,
            "features": selected_features,
            "metrics": metrics,
            "top_features": list(explanation.get("top_features") or [])[:5],
            "feature_importance": dict(explanation_scores or feature_payload.get("importance_scores") or {}),
            "predictions": list(training_payload.get("predictions") or []),
            "data_quality_score": 0.0,
            "confidence": round(float(detection.get("confidence") or 0.0), 4),
            "warnings": [],
            "recommendations": list(recommendation_payload.get("recommendations") or []),
        }
    )
    return payload


__all__ = ["build_ml_intelligence"]
