from __future__ import annotations

from typing import Any


def _resolve_data_quality_score(data_quality: dict[str, Any] | float | int | None) -> float:
    if isinstance(data_quality, dict):
        raw_score = data_quality.get("score")
    else:
        raw_score = data_quality
    try:
        return float(raw_score)
    except (TypeError, ValueError):
        return 0.0


def assess_risk(data_quality: dict[str, Any] | float | int | None, model_quality: str | None) -> str:
    quality_score = _resolve_data_quality_score(data_quality)
    resolved_model_quality = str(model_quality or "weak").strip().lower()

    if quality_score < 5.0 or resolved_model_quality == "weak":
        return "high"
    if resolved_model_quality == "not_applicable":
        if quality_score >= 8.0:
            return "low"
        return "medium"
    if quality_score >= 8.0 and resolved_model_quality == "strong":
        return "low"
    return "medium"


def build_risk_statement(data_quality: dict[str, Any] | float | int | None, model_quality: str | None) -> str:
    quality_score = _resolve_data_quality_score(data_quality)
    resolved_model_quality = str(model_quality or "weak").strip().lower()
    risk_level = assess_risk(data_quality, resolved_model_quality)

    reasons: list[str] = []
    if quality_score < 5.0:
        reasons.append("poor data quality")
    elif quality_score < 8.0:
        reasons.append("moderate data quality")

    if resolved_model_quality == "weak":
        reasons.append("weak model quality")
    elif resolved_model_quality == "moderate" and risk_level != "low":
        reasons.append("moderate model quality")

    if not reasons:
        return risk_level
    return f"{risk_level} due to {', '.join(reasons)}"
