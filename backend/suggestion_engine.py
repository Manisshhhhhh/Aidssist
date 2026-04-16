from __future__ import annotations

from typing import Any

import pandas as pd

from backend.aidssist_runtime.cache import get_cache_store
from backend.dataset_understanding import analyze_dataset
from backend.services.target_detector import build_target_suggested_questions, detect_target_column


_MEMORY_TTL_SECONDS = 60 * 60 * 24 * 30
_MEMORY_LIMIT = 12


def _memory_key(source_fingerprint: str | None) -> str | None:
    normalized = str(source_fingerprint or "").strip()
    if not normalized:
        return None
    return f"interaction-memory:{normalized}"


def _normalize_memory(payload: dict[str, Any] | None) -> dict[str, Any]:
    payload = dict(payload or {})
    return {
        "dataset_type": str(payload.get("dataset_type") or "generic"),
        "queries": [str(item) for item in payload.get("queries", []) if str(item).strip()][-_MEMORY_LIMIT:],
        "successful_actions": [str(item) for item in payload.get("successful_actions", []) if str(item).strip()][-_MEMORY_LIMIT:],
    }


def get_user_interaction_memory(source_fingerprint: str | None) -> dict[str, Any]:
    key = _memory_key(source_fingerprint)
    if key is None:
        return _normalize_memory(None)
    return _normalize_memory(get_cache_store().get_json(key))


def record_user_interaction_memory(
    *,
    source_fingerprint: str | None,
    dataset_type: str | None,
    query: str | None = None,
    successful_action: str | None = None,
) -> dict[str, Any]:
    key = _memory_key(source_fingerprint)
    memory = get_user_interaction_memory(source_fingerprint)
    if dataset_type:
        memory["dataset_type"] = str(dataset_type)

    for field_name, value in (("queries", query), ("successful_actions", successful_action)):
        normalized = str(value or "").strip()
        if not normalized:
            continue
        items = [item for item in memory[field_name] if item != normalized]
        items.append(normalized)
        memory[field_name] = items[-_MEMORY_LIMIT:]

    if key is not None:
        get_cache_store().set_json(key, memory, ttl_seconds=_MEMORY_TTL_SECONDS)
    return memory


def _first_matching(values: list[str], preferred_tokens: tuple[str, ...]) -> str | None:
    for token in preferred_tokens:
        for value in values:
            if token in str(value).lower():
                return str(value)
    return values[0] if values else None


def _prompt_for_suggestion(suggestion: dict[str, Any], context: dict[str, Any]) -> str:
    primary_metrics = [str(item) for item in context.get("primary_metrics", []) if str(item).strip()]
    categorical_features = [str(item) for item in context.get("categorical_features", []) if str(item).strip()]
    time_columns = [str(item) for item in context.get("time_columns", []) if str(item).strip()]
    metric = _first_matching(primary_metrics, ("revenue", "sales", "profit", "cases", "price", "amount")) or "value"
    time_column = time_columns[0] if time_columns else "date"
    product_column = _first_matching(categorical_features, ("product", "sku", "item", "brand", "service"))
    region_column = _first_matching(categorical_features, ("region", "territory", "market", "country"))
    customer_column = _first_matching(categorical_features, ("customer", "segment", "cohort"))
    diagnosis_column = _first_matching(categorical_features, ("diagnosis", "condition", "disease"))
    asset_column = _first_matching(categorical_features, ("asset", "ticker", "portfolio", "instrument"))
    goal = str(suggestion.get("goal") or "").strip().lower()
    target_column = str(context.get("target_column") or metric).strip() or metric

    if goal == "trend":
        return f"Show {metric} trend over time using {time_column}"
    if goal == "forecast":
        return f"Forecast future values for {metric} using {time_column}"
    if goal == "seasonality":
        return f"Detect seasonality and repeating patterns in {metric} over time"
    if goal == "product_revenue":
        group_column = product_column or "product"
        metric_name = _first_matching(primary_metrics, ("revenue", "sales", "profit")) or metric
        return f"Show top {group_column} by {metric_name}"
    if goal == "regional_performance":
        group_column = region_column or "region"
        metric_name = _first_matching(primary_metrics, ("revenue", "sales", "profit", "cases")) or metric
        return f"Compare {metric_name} by {group_column}"
    if goal == "customer_segmentation":
        group_column = customer_column or "segment"
        metric_name = _first_matching(primary_metrics, ("revenue", "sales", "profit")) or metric
        return f"Segment customers by {metric_name} and identify the highest-value groups"
    if goal == "patient_risk":
        group_column = diagnosis_column or "patient cohort"
        return f"Identify high-risk patients and compare outcomes by {group_column}"
    if goal == "abnormal_vitals":
        return f"Find abnormal values and outliers in the primary medical metrics"
    if goal == "portfolio_exposure":
        group_column = asset_column or "asset"
        metric_name = _first_matching(primary_metrics, ("return", "price", "balance", "amount")) or metric
        return f"Analyze exposure and performance by {group_column} using {metric_name}"
    if goal == "cashflow":
        metric_name = _first_matching(primary_metrics, ("cashflow", "cash", "balance", "amount")) or metric
        return f"Show {metric_name} trend over time and highlight major shifts"
    if goal == "outliers":
        return f"Detect outliers and unusual patterns in {metric}"
    if goal == "target_drivers":
        return f"What affects {target_column}?"
    if goal == "predict_target":
        return f"Predict {target_column}"
    if goal == "improve_target":
        return f"Improve {target_column}"
    if goal == "compare_categories":
        group_column = categorical_features[0] if categorical_features else "category"
        return f"Compare {metric} by {group_column}"
    return str(suggestion.get("title") or f"Analyze {metric}")


def generate_suggestions(context: dict[str, Any]) -> list[dict[str, Any]]:
    suggestions: list[dict[str, Any]] = []
    domain = str(context.get("domain") or "generic")
    is_time_series = bool(context.get("is_time_series"))
    target_column = str(context.get("target_column") or "").strip()
    categorical_features = [str(item) for item in context.get("categorical_features", []) if str(item).strip()]
    has_product = any(token in column.lower() for column in categorical_features for token in ("product", "sku", "item", "brand", "service"))
    has_region = any(token in column.lower() for column in categorical_features for token in ("region", "territory", "market", "country"))
    has_customer = any(token in column.lower() for column in categorical_features for token in ("customer", "segment", "cohort"))

    if is_time_series:
        suggestions.extend(
            [
                {"title": "Analyze trend over time", "goal": "trend", "action_type": "analysis", "category": "time_series"},
                {"title": "Forecast future values", "goal": "forecast", "action_type": "forecast", "category": "time_series"},
                {"title": "Detect seasonality", "goal": "seasonality", "action_type": "analysis", "category": "time_series"},
            ]
        )

    for question in build_target_suggested_questions(target_column):
        goal = "predict_target" if question.lower().startswith("predict ") else "improve_target"
        if question.lower().startswith("what affects "):
            goal = "target_drivers"
        suggestions.append(
            {
                "title": question,
                "goal": goal,
                "action_type": "analysis",
                "category": "ml_intelligence",
                "prompt": question,
                "rationale": "Recommended because a likely target column was detected automatically.",
            }
        )

    if domain == "business":
        if has_product:
            suggestions.append({"title": "Top products by revenue", "goal": "product_revenue", "action_type": "analysis", "category": "business"})
        if has_region:
            suggestions.append({"title": "Region-wise performance", "goal": "regional_performance", "action_type": "analysis", "category": "business"})
        if has_customer:
            suggestions.append({"title": "Customer segmentation", "goal": "customer_segmentation", "action_type": "analysis", "category": "business"})
        if not any((has_product, has_region, has_customer)):
            suggestions.append({"title": "Compare business segments", "goal": "compare_categories", "action_type": "analysis", "category": "business"})
    elif domain == "medical":
        suggestions.extend(
            [
                {"title": "Find abnormal vitals", "goal": "abnormal_vitals", "action_type": "analysis", "category": "medical"},
                {"title": "Identify high-risk patients", "goal": "patient_risk", "action_type": "analysis", "category": "medical"},
            ]
        )
    elif domain == "finance":
        suggestions.extend(
            [
                {"title": "Portfolio exposure analysis", "goal": "portfolio_exposure", "action_type": "analysis", "category": "finance"},
                {"title": "Cash flow trend review", "goal": "cashflow", "action_type": "analysis", "category": "finance"},
            ]
        )
    else:
        suggestions.extend(
            [
                {"title": "Compare categories", "goal": "compare_categories", "action_type": "analysis", "category": "generic"},
                {"title": "Detect outliers", "goal": "outliers", "action_type": "analysis", "category": "generic"},
            ]
        )

    if not categorical_features:
        suggestions = [item for item in suggestions if item.get("goal") not in {"product_revenue", "regional_performance", "customer_segmentation", "portfolio_exposure", "compare_categories"}]

    for suggestion in suggestions:
        suggestion["prompt"] = _prompt_for_suggestion(suggestion, context)
        suggestion["rationale"] = (
            "Recommended because the dataset structure and prior usage patterns indicate this is a strong next step."
        )
    return suggestions


def rank_suggestions(suggestions: list[dict[str, Any]], context: dict[str, Any]) -> list[dict[str, Any]]:
    memory = dict(context.get("interaction_memory") or {})
    memory_queries = [str(item).lower() for item in memory.get("queries", [])]
    memory_successes = [str(item).lower() for item in memory.get("successful_actions", [])]
    domain = str(context.get("domain") or "generic").lower()
    is_time_series = bool(context.get("is_time_series"))

    ranked: list[dict[str, Any]] = []
    for index, suggestion in enumerate(suggestions):
        payload = dict(suggestion)
        title = str(payload.get("title") or "").lower()
        prompt = str(payload.get("prompt") or "").lower()
        goal = str(payload.get("goal") or "").lower()
        score = 1.0

        if is_time_series and goal in {"trend", "forecast", "seasonality"}:
            score += 5.0
        if goal == "forecast" and is_time_series:
            score += 1.5
        if goal in {"target_drivers", "predict_target", "improve_target"}:
            score += 4.5
        if domain in str(payload.get("category") or "").lower():
            score += 2.0
        if any(token in title or token in prompt for token in ("revenue", "sales", "profit")) and domain == "business":
            score += 1.0
        if any(token in title or token in prompt for token in ("risk", "patient", "vital")) and domain == "medical":
            score += 1.0
        if any(token in title or token in prompt for token in ("portfolio", "cash", "balance")) and domain == "finance":
            score += 1.0
        if any(prompt in memory_item or title in memory_item for memory_item in memory_queries):
            score += 0.75
        if any(prompt in memory_item or title in memory_item for memory_item in memory_successes):
            score += 1.25

        payload["score"] = round(score, 2)
        payload["rank"] = index + 1
        ranked.append(payload)

    ranked.sort(key=lambda item: (-float(item.get("score") or 0.0), str(item.get("title") or "")))
    for rank, item in enumerate(ranked, start=1):
        item["rank"] = rank
    return ranked


def build_suggestion_payload(
    df: pd.DataFrame,
    *,
    source_fingerprint: str | None = None,
    recent_queries: list[str] | None = None,
    limit: int = 6,
) -> dict[str, Any]:
    context = analyze_dataset(df)
    detected_target = detect_target_column(df)
    if detected_target.get("target"):
        context["target_column"] = detected_target.get("target")
        context["target_type"] = detected_target.get("type")
        context["target_confidence"] = detected_target.get("confidence")
    memory = get_user_interaction_memory(source_fingerprint)
    if recent_queries:
        merged_queries = [str(item) for item in memory.get("queries", []) if str(item).strip()]
        for query in recent_queries:
            normalized = str(query or "").strip()
            if not normalized:
                continue
            merged_queries = [item for item in merged_queries if item != normalized]
            merged_queries.append(normalized)
        memory["queries"] = merged_queries[-_MEMORY_LIMIT:]

    context["interaction_memory"] = memory
    suggestions = rank_suggestions(generate_suggestions(context), context)
    limited_suggestions = suggestions[: max(1, int(limit))]
    recommended_next_step = str((limited_suggestions[0] or {}).get("prompt") or "") if limited_suggestions else ""
    return {
        "domain": str(context.get("domain") or "generic"),
        "context": context,
        "suggestions": limited_suggestions,
        "recommended_next_step": recommended_next_step or None,
        "suggested_questions": [str(item.get("prompt") or item.get("title") or "") for item in limited_suggestions if str(item.get("prompt") or item.get("title") or "").strip()],
    }
