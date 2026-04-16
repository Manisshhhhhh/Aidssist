from __future__ import annotations

import math
import re
from typing import Any

from backend.aidssist_runtime.cache import get_cache_store


_CONFIDENCE_ORDER = {"low": 0, "medium": 1, "high": 2}
_RISK_ORDER = {"low": 0, "medium": 1, "high": 2}
_SUCCESS_TOKENS = (
    "success",
    "succeeded",
    "improved",
    "increase",
    "increased",
    "grew",
    "growth",
    "outperformed",
    "achieved",
    "beat",
    "positive",
    "completed",
    "better",
)
_FAILURE_TOKENS = (
    "fail",
    "failed",
    "failure",
    "decline",
    "declined",
    "drop",
    "dropped",
    "worse",
    "underperformed",
    "missed",
    "negative",
    "loss",
    "lost",
)
_UNCERTAIN_TOKENS = (
    "mixed",
    "unclear",
    "unknown",
    "uncertain",
    "partial",
    "conflicting",
    "neutral",
)
_POSITIVE_REDUCTION_CONTEXTS = (
    "churn",
    "risk",
    "cost",
    "expense",
    "waste",
    "delay",
    "error",
    "stockout",
    "defect",
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return number


def _normalize_confidence(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in _CONFIDENCE_ORDER:
        return normalized
    return "low"


def _normalize_risk(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if normalized.startswith("high"):
        return "high"
    if normalized.startswith("low"):
        return "low"
    return "medium"


def _decision_learning_cache_key(source_fingerprint: str | None) -> str:
    return f"decision-learning:{str(source_fingerprint or '').strip() or 'global'}"


def infer_decision_type(action: str | None) -> str:
    normalized = str(action or "").strip().lower()
    if not normalized:
        return "general_action"

    if any(token in normalized for token in ("pricing", "price", "discount")):
        return "pricing_adjustment"
    if any(token in normalized for token in ("inventory", "stock", "supply")):
        return "inventory_adjustment"
    if any(token in normalized for token in ("retention", "churn", "customer", "subscriber")):
        return "retention_action"
    if any(token in normalized for token in ("data quality", "cleaned derived dataset", "clean data", "improve data")):
        return "data_reliability_improvement"
    if any(token in normalized for token in ("validate", "review", "investigate", "monitor", "re-run")):
        return "validation_gate"
    if any(token in normalized for token in ("region", "market", "territory", "channel", "segment")):
        return "expand_region"
    if any(token in normalized for token in ("north", "south", "east", "west")) and any(
        token in normalized for token in ("increase", "expand", "focus", "shift", "rebalance", "invest")
    ):
        return "expand_region"

    cleaned = re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")
    if not cleaned:
        return "general_action"
    return "_".join(cleaned.split("_")[:3])


def _extract_outcome_text(entry: Any) -> str:
    if isinstance(entry, dict):
        return str(entry.get("outcome") or "")
    return str(getattr(entry, "outcome", "") or "")


def _extract_action(entry: Any) -> str:
    decision_json = {}
    if isinstance(entry, dict):
        decision_json = entry.get("decision_json") or {}
    else:
        decision_json = getattr(entry, "decision_json", {}) or {}
    if isinstance(decision_json, dict) and str(decision_json.get("action") or "").strip():
        return str(decision_json.get("action") or "")
    if isinstance(entry, dict):
        return str(entry.get("decision") or entry.get("action") or "")
    return str(getattr(entry, "decision", "") or getattr(entry, "action", "") or "")


def _parse_outcome_signal(outcome: str | None) -> dict[str, Any] | None:
    normalized = str(outcome or "").strip().lower()
    if not normalized:
        return None

    positive_reduction = (
        ("reduce" in normalized or "reduced" in normalized or "decrease" in normalized or "decreased" in normalized)
        and any(token in normalized for token in _POSITIVE_REDUCTION_CONTEXTS)
    )
    has_success = positive_reduction or any(token in normalized for token in _SUCCESS_TOKENS)
    has_failure = any(token in normalized for token in _FAILURE_TOKENS)
    has_uncertain = any(token in normalized for token in _UNCERTAIN_TOKENS)

    if has_uncertain or (has_success and has_failure):
        status = "uncertain"
    elif has_success:
        status = "success"
    elif has_failure:
        status = "failure"
    else:
        status = "unknown"

    impact_match = re.search(r"([+-]?\d+(?:\.\d+)?)\s*%", normalized)
    impact = None
    if impact_match:
        impact = _safe_float(impact_match.group(1), default=0.0)
        if not impact_match.group(1).startswith(("+", "-")):
            if status == "failure":
                impact = -abs(impact)
            elif status == "success":
                impact = abs(impact)

    return {
        "status": status,
        "impact": impact,
    }


def _pattern_entry(decision_type: str) -> dict[str, Any]:
    return {
        "decision_type": decision_type,
        "success_count": 0,
        "failure_count": 0,
        "uncertain_count": 0,
        "sample_size": 0,
        "impact_sum": 0.0,
        "impact_count": 0,
    }


def _uncertainty_label(*, sample_size: int, success_rate: float, uncertain_count: int) -> str:
    if sample_size < 3:
        return "high"
    if uncertain_count > 0:
        return "high"
    if 0.4 <= success_rate <= 0.6:
        return "medium"
    return "low"


def learn_from_outcomes(decision_history):
    aggregated: dict[str, dict[str, Any]] = {}

    for entry in list(decision_history or []):
        outcome_text = _extract_outcome_text(entry)
        signal = _parse_outcome_signal(outcome_text)
        if signal is None:
            continue

        decision_type = infer_decision_type(_extract_action(entry))
        pattern = aggregated.setdefault(decision_type, _pattern_entry(decision_type))
        pattern["sample_size"] += 1

        if signal["status"] == "success":
            pattern["success_count"] += 1
        elif signal["status"] == "failure":
            pattern["failure_count"] += 1
        else:
            pattern["uncertain_count"] += 1

        if signal["impact"] is not None:
            pattern["impact_sum"] += float(signal["impact"])
            pattern["impact_count"] += 1

    learning_patterns: dict[str, dict[str, Any]] = {}
    for decision_type, pattern in aggregated.items():
        decisive_count = int(pattern["success_count"] + pattern["failure_count"])
        success_rate = (
            float(pattern["success_count"]) / decisive_count
            if decisive_count > 0
            else 0.5
        )
        avg_impact = (
            float(pattern["impact_sum"]) / int(pattern["impact_count"])
            if int(pattern["impact_count"]) > 0
            else 0.0
        )
        learning_patterns[decision_type] = {
            "decision_type": decision_type,
            "success_rate": round(success_rate, 4),
            "avg_impact": round(avg_impact, 4),
            "sample_size": int(pattern["sample_size"]),
            "success_count": int(pattern["success_count"]),
            "failure_count": int(pattern["failure_count"]),
            "uncertain_count": int(pattern["uncertain_count"]),
            "uncertainty": _uncertainty_label(
                sample_size=int(pattern["sample_size"]),
                success_rate=success_rate,
                uncertain_count=int(pattern["uncertain_count"]),
            ),
        }
    return learning_patterns


def _step_confidence(value: str, delta: int) -> str:
    order = ["low", "medium", "high"]
    index = _CONFIDENCE_ORDER[_normalize_confidence(value)]
    return order[max(0, min(len(order) - 1, index + int(delta)))]


def _step_risk(value: str, delta: int) -> str:
    order = ["low", "medium", "high"]
    index = _RISK_ORDER[_normalize_risk(value)]
    return order[max(0, min(len(order) - 1, index + int(delta)))]


def decision_performance_snapshot(learning_patterns, decision_type: str | None) -> dict[str, Any]:
    pattern = dict((learning_patterns or {}).get(str(decision_type or ""), {}) or {})
    return {
        "historical_success_rate": round(_safe_float(pattern.get("success_rate"), default=0.0), 4),
        "avg_impact": round(_safe_float(pattern.get("avg_impact"), default=0.0), 4),
        "sample_size": int(pattern.get("sample_size") or 0),
    }


def adjust_confidence(base_confidence, learning_patterns, decision_type):
    base = _normalize_confidence(base_confidence)
    pattern = dict((learning_patterns or {}).get(str(decision_type or ""), {}) or {})
    sample_size = int(pattern.get("sample_size") or 0)
    if sample_size <= 0:
        return base
    if sample_size < 3:
        return "low"

    success_rate = _safe_float(pattern.get("success_rate"), default=0.5)
    avg_impact = _safe_float(pattern.get("avg_impact"), default=0.0)
    uncertainty = str(pattern.get("uncertainty") or "low").lower()

    if uncertainty == "high":
        return _step_confidence(base, -1)
    if success_rate >= 0.75 and avg_impact >= 0:
        return _step_confidence(base, 1)
    if success_rate <= 0.4 or avg_impact < 0:
        return _step_confidence(base, -1)
    return base


def adjust_risk(base_risk, learning_patterns, decision_type):
    base = _normalize_risk(base_risk)
    pattern = dict((learning_patterns or {}).get(str(decision_type or ""), {}) or {})
    sample_size = int(pattern.get("sample_size") or 0)
    if sample_size <= 0:
        return base

    success_rate = _safe_float(pattern.get("success_rate"), default=0.5)
    avg_impact = _safe_float(pattern.get("avg_impact"), default=0.0)
    uncertainty = str(pattern.get("uncertainty") or "low").lower()

    if sample_size < 3 or uncertainty == "high":
        return _step_risk(base, 1)
    if success_rate >= 0.75 and avg_impact >= 0:
        return _step_risk(base, -1)
    if success_rate <= 0.4 or avg_impact < 0:
        return _step_risk(base, 1)
    return base


def build_learning_pattern_summaries(learning_patterns, *, decision_types: list[str] | None = None) -> list[dict[str, Any]]:
    decision_types = [str(item) for item in list(decision_types or []) if str(item).strip()]
    selected_patterns: list[dict[str, Any]] = []

    if decision_types:
        for decision_type in decision_types:
            pattern = (learning_patterns or {}).get(decision_type)
            if not pattern:
                continue
            selected_patterns.append(
                {
                    "decision_type": decision_type,
                    "success_rate": round(_safe_float(pattern.get("success_rate"), default=0.0), 4),
                    "avg_impact": round(_safe_float(pattern.get("avg_impact"), default=0.0), 4),
                    "sample_size": int(pattern.get("sample_size") or 0),
                    "uncertainty": str(pattern.get("uncertainty") or "low"),
                }
            )
    else:
        for decision_type, pattern in list((learning_patterns or {}).items())[:5]:
            selected_patterns.append(
                {
                    "decision_type": decision_type,
                    "success_rate": round(_safe_float(pattern.get("success_rate"), default=0.0), 4),
                    "avg_impact": round(_safe_float(pattern.get("avg_impact"), default=0.0), 4),
                    "sample_size": int(pattern.get("sample_size") or 0),
                    "uncertainty": str(pattern.get("uncertainty") or "low"),
                }
            )
    return selected_patterns


def get_learning_patterns(
    store,
    source_fingerprint: str | None,
    *,
    refresh: bool = False,
    max_records: int = 500,
) -> dict[str, Any]:
    if store is None or not str(source_fingerprint or "").strip():
        return {}

    cache_key = _decision_learning_cache_key(source_fingerprint)
    if not refresh:
        cached = get_cache_store().get_json(cache_key)
        if isinstance(cached, dict):
            return cached

    decision_history = store.list_decision_history(
        source_fingerprint=str(source_fingerprint),
        limit=max_records,
    )
    learning_patterns = learn_from_outcomes(decision_history)
    get_cache_store().set_json(cache_key, learning_patterns)
    return learning_patterns


def refresh_learning_patterns(store, source_fingerprint: str | None) -> dict[str, Any]:
    return get_learning_patterns(store, source_fingerprint, refresh=True)


def invalidate_learning_patterns(source_fingerprint: str | None) -> None:
    if not str(source_fingerprint or "").strip():
        return
    get_cache_store().delete(_decision_learning_cache_key(source_fingerprint))
