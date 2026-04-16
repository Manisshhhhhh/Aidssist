from __future__ import annotations

from typing import Any

import pandas as pd

from backend.dataset_understanding import analyze_dataset
from backend.suggestion_engine import build_suggestion_payload
from backend.services.profiling_engine import DatasetProfileReport, build_profile_report
from backend.services.target_detector import detect_target_column


_DOMAIN_KEYWORDS = {
    "sales": {"sales", "revenue", "profit", "customer", "order", "product", "quantity", "discount"},
    "medical": {"patient", "diagnosis", "doctor", "hospital", "symptom", "disease", "lab", "vital"},
    "finance": {"ticker", "portfolio", "return", "asset", "price", "balance", "loan", "cashflow"},
    "student": {"student", "attendance", "grade", "exam", "score", "gpa", "semester", "course"},
}


def _dataset_tokens(df: pd.DataFrame) -> set[str]:
    tokens: set[str] = set()
    for column in df.columns:
        tokens.update(str(column).strip().lower().replace("_", " ").replace("-", " ").split())
    sample = df.head(20)
    for column in sample.columns:
        if pd.api.types.is_numeric_dtype(sample[column]):
            continue
        for value in sample[column].dropna().astype("string").head(20):
            tokens.update(str(value).strip().lower().replace("_", " ").replace("-", " ").split())
    return tokens


def detect_dataset_domain(df: pd.DataFrame) -> str:
    tokens = _dataset_tokens(df)
    ranked = sorted(
        (
            (domain, len(tokens.intersection(keywords)))
            for domain, keywords in _DOMAIN_KEYWORDS.items()
        ),
        key=lambda item: (-item[1], item[0]),
    )
    if not ranked or ranked[0][1] <= 0:
        context = analyze_dataset(df)
        fallback_domain = str(context.get("domain") or "generic")
        return "sales" if fallback_domain == "business" else fallback_domain
    return ranked[0][0]


def _analysis_ideas(domain: str, profile: DatasetProfileReport) -> list[str]:
    ideas: list[str] = []
    if profile.is_time_series and profile.numeric_columns:
        ideas.append(f"Track {profile.numeric_columns[0]} trend over time.")
        ideas.append("Build a short-horizon forecast for the leading metric.")
    if domain == "sales":
        ideas.extend(
            [
                "Compare revenue or sales by region, product, or segment.",
                "Measure top and bottom performing categories after cleaning missing records.",
            ]
        )
    elif domain == "medical":
        ideas.extend(
            [
                "Review patient outcomes by diagnosis, severity, or treatment group.",
                "Check for abnormal measurements and elevated risk cohorts.",
            ]
        )
    elif domain == "finance":
        ideas.extend(
            [
                "Evaluate portfolio exposure and returns by asset or account segment.",
                "Review cashflow, balances, or volatility drivers over time.",
            ]
        )
    elif domain == "student":
        ideas.extend(
            [
                "Find the strongest drivers of grades, attendance, or exam performance.",
                "Group students into at-risk and high-performing cohorts.",
            ]
        )
    else:
        ideas.extend(
            [
                "Compare major categories and identify the largest drivers of change.",
                "Check for anomalies, outliers, and distribution shifts before modeling.",
            ]
        )

    if profile.target_column:
        ideas.append(f"Use {profile.target_column} as a likely target for predictive analysis.")

    deduped: list[str] = []
    for idea in ideas:
        if idea and idea not in deduped:
            deduped.append(idea)
    return deduped[:8]


def _cleaning_actions(profile: DatasetProfileReport) -> list[str]:
    actions = list(profile.suggested_fixes)
    if profile.duplicate_row_count > 0 and "Remove duplicate rows" not in actions:
        actions.append("Remove duplicate rows")
    if profile.quality_score < 7.0:
        actions.append("Prioritize cleaning before running downstream analysis.")
    return actions[:8]


def build_assistant_payload(
    df: pd.DataFrame,
    *,
    dataset_name: str = "dataset",
    profile: DatasetProfileReport | None = None,
    recent_queries: list[str] | None = None,
) -> dict[str, Any]:
    resolved_profile = profile if profile is not None else build_profile_report(df, dataset_name=dataset_name)
    domain = detect_dataset_domain(df)
    target_details = detect_target_column(df)
    suggestion_payload = build_suggestion_payload(df, recent_queries=recent_queries, limit=6)
    context = analyze_dataset(df)

    if target_details.get("target") and target_details.get("target") not in resolved_profile.insights:
        resolved_profile.insights.append(f"Detected target candidate: {target_details['target']}.")

    relevant_questions = [str(item) for item in suggestion_payload.get("suggested_questions", []) if str(item).strip()]
    recommended_next_step = str(suggestion_payload.get("recommended_next_step") or "").strip() or None

    return {
        "dataset_name": dataset_name,
        "dataset_type": domain,
        "detected_domain": domain,
        "is_time_series": bool(resolved_profile.is_time_series),
        "target_column": str(target_details.get("target") or resolved_profile.target_column or "") or None,
        "target_type": str(target_details.get("type") or resolved_profile.target_type or "") or None,
        "quality_score": float(resolved_profile.quality_score),
        "relevant_questions": relevant_questions,
        "analysis_ideas": _analysis_ideas(domain, resolved_profile),
        "cleaning_actions": _cleaning_actions(resolved_profile),
        "recommended_next_step": recommended_next_step,
        "detected_columns": {
            "datetime": list(resolved_profile.datetime_columns),
            "numeric": list(resolved_profile.numeric_columns),
            "categorical": list(resolved_profile.categorical_columns),
        },
        "ml_readiness": {
            "target_column": str(target_details.get("target") or "") or None,
            "target_type": str(target_details.get("type") or "") or None,
            "can_forecast": bool(resolved_profile.is_time_series and resolved_profile.numeric_columns),
            "can_model": bool(context.get("primary_metrics") or target_details.get("target")),
        },
        "profile_highlights": list(resolved_profile.insights[:6]),
        "suggestions": list(suggestion_payload.get("suggestions", [])),
    }


__all__ = [
    "build_assistant_payload",
    "detect_dataset_domain",
]
