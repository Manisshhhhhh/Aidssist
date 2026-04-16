from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any

import pandas as pd

from backend.services.learning_engine import (
    adjust_confidence,
    adjust_risk,
    build_learning_pattern_summaries,
    decision_performance_snapshot,
    infer_decision_type,
)
from backend.services.recommendation_engine import generate_recommendations


_CONFIDENCE_LEVELS = {"high": 1.0, "medium": 0.6, "low": 0.3}
_RISK_LEVELS = {"low": 0.2, "medium": 0.55, "high": 0.9}
_PRIORITY_ORDER = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
_DEFAULT_IMPACT_TEXT = "Impact should be validated with a monitored rollout."


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
    if normalized in _CONFIDENCE_LEVELS:
        return normalized
    return "low"


def _normalize_risk(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if normalized.startswith("high"):
        return "high"
    if normalized.startswith("low"):
        return "low"
    return "medium"


def _normalize_priority(value: Any) -> str:
    normalized = str(value or "").strip().upper()
    if normalized in _PRIORITY_ORDER:
        return normalized
    return "LOW"


def _parse_percent(value: str | None) -> float | None:
    match = re.search(r"(-?\d+(?:\.\d+)?)\s*%", str(value or ""))
    if not match:
        return None
    return abs(_safe_float(match.group(1), default=0.0))


def _impact_score(expected_impact: str | None) -> float:
    percent = _parse_percent(expected_impact)
    if percent is not None:
        return max(0.05, min(percent / 20.0, 1.0))

    lowered = str(expected_impact or "").lower()
    if any(token in lowered for token in ("increase revenue", "protect against", "lift", "reduce churn", "reduce decline")):
        return 0.7
    if any(token in lowered for token in ("improve", "protect", "stabilize", "monitor")):
        return 0.5
    return 0.3


def _data_quality_score(data_quality: dict[str, Any] | None) -> float:
    if not isinstance(data_quality, dict):
        return 0.0
    return _safe_float(data_quality.get("score"), default=0.0)


def _max_missing_percent(data_quality: dict[str, Any] | None) -> float:
    profile = (data_quality or {}).get("profile") if isinstance(data_quality, dict) else {}
    if not isinstance(profile, dict):
        return 0.0
    missing_by_column = profile.get("missing_percent")
    if not isinstance(missing_by_column, dict):
        return 0.0
    if not missing_by_column:
        return 0.0
    return max(_safe_float(value, default=0.0) for value in missing_by_column.values())


def _result_frame(result: Any, frame_key: str | None = None) -> pd.DataFrame | None:
    if isinstance(result, pd.DataFrame):
        return result.copy()
    if isinstance(result, dict):
        candidate = result.get(frame_key) if frame_key else None
        if isinstance(candidate, pd.DataFrame):
            return candidate.copy()
        for key in ("forecast_table", "result", "table", "comparison_table"):
            candidate = result.get(key)
            if isinstance(candidate, pd.DataFrame):
                return candidate.copy()
            if isinstance(candidate, list) and candidate and isinstance(candidate[0], dict):
                return pd.DataFrame(candidate)
    if isinstance(result, list) and result and isinstance(result[0], dict):
        return pd.DataFrame(result)
    return None


def _preferred_numeric_column(frame: pd.DataFrame) -> str | None:
    numeric_columns = frame.select_dtypes(include="number").columns.tolist()
    if not numeric_columns:
        return None

    priorities = ("forecast", "predicted", "prediction", "value", "sales", "revenue", "amount", "score")
    for token in priorities:
        for column in numeric_columns:
            if token in str(column).lower():
                return str(column)
    return str(numeric_columns[0])


def _extract_prediction_context(result: Any, plan: dict[str, Any]) -> dict[str, Any]:
    metric_name = str(plan.get("target_column") or plan.get("metric_column") or "metric")
    baseline_value = None
    projected_value = None
    delta_ratio = None

    if isinstance(result, dict):
        metric_name = str(
            result.get("target_column")
            or (result.get("config") or {}).get("target_column")
            or metric_name
        )
        baseline_value = result.get("history_baseline") or result.get("current_baseline") or result.get("baseline")
        delta_ratio = result.get("delta_ratio")
        if delta_ratio is None:
            delta_ratio = result.get("trend_delta_ratio")

        forecast_frame = _result_frame(result, "forecast_table")
        if forecast_frame is not None and not forecast_frame.empty:
            forecast_column = _preferred_numeric_column(forecast_frame)
            if forecast_column:
                projected_value = _safe_float(forecast_frame[forecast_column].mean(), default=0.0)

    if projected_value is None:
        frame = _result_frame(result)
        if frame is not None and not frame.empty:
            forecast_column = _preferred_numeric_column(frame)
            if forecast_column:
                projected_value = _safe_float(frame[forecast_column].mean(), default=0.0)

    baseline_value = _safe_float(baseline_value, default=0.0) if baseline_value is not None else None
    projected_value = _safe_float(projected_value, default=0.0) if projected_value is not None else None

    if delta_ratio is None and baseline_value is not None and projected_value is not None and abs(baseline_value) > 1e-9:
        delta_ratio = (projected_value - baseline_value) / abs(baseline_value)

    if delta_ratio is not None:
        delta_ratio = _safe_float(delta_ratio, default=0.0)

    return {
        "metric_name": metric_name,
        "baseline_value": baseline_value,
        "projected_value": projected_value,
        "delta_ratio": delta_ratio,
    }


def _extract_aggregation_context(result: Any, plan: dict[str, Any]) -> dict[str, Any]:
    frame = _result_frame(result)
    if frame is None or frame.empty:
        return {}

    group_column = str(plan.get("group_column") or "")
    metric_column = str(plan.get("metric_column") or "") or _preferred_numeric_column(frame) or ""
    if not group_column or group_column not in frame.columns:
        non_numeric = [str(column) for column in frame.columns if not pd.api.types.is_numeric_dtype(frame[column])]
        if non_numeric:
            group_column = non_numeric[0]
    if not metric_column or metric_column not in frame.columns:
        metric_column = _preferred_numeric_column(frame) or ""

    if not group_column or not metric_column or group_column not in frame.columns or metric_column not in frame.columns:
        return {}

    working = frame[[group_column, metric_column]].copy()
    working[metric_column] = pd.to_numeric(working[metric_column], errors="coerce")
    working = working.dropna(subset=[metric_column])
    if working.empty:
        return {}

    sorted_frame = working.sort_values(metric_column, ascending=False).reset_index(drop=True)
    top_row = sorted_frame.iloc[0]
    bottom_row = sorted_frame.iloc[-1]
    top_value = _safe_float(top_row[metric_column], default=0.0)
    bottom_value = _safe_float(bottom_row[metric_column], default=0.0)
    spread_ratio = None
    if abs(bottom_value) > 1e-9:
        spread_ratio = max((top_value - bottom_value) / abs(bottom_value), 0.0)
    elif abs(top_value) > 1e-9:
        spread_ratio = 1.0

    return {
        "group_column": group_column,
        "metric_column": metric_column,
        "top_group": str(top_row[group_column]),
        "top_value": top_value,
        "bottom_group": str(bottom_row[group_column]),
        "bottom_value": bottom_value,
        "spread_ratio": spread_ratio,
    }


def _impact_text_from_prediction(context: dict[str, Any], action: str) -> str:
    metric_name = str(context.get("metric_name") or "metric")
    delta_ratio = context.get("delta_ratio")
    if delta_ratio is None:
        if "protect" in action.lower() or "risk" in action.lower():
            return "Protect against downside risk with a staged rollout."
        return f"Improve {metric_name} with a measured pilot before scaling."

    percent = abs(_safe_float(delta_ratio, default=0.0) * 100.0)
    if delta_ratio < 0:
        return f"Protect against ~{percent:.0f}% decline in {metric_name}"
    return f"Increase {metric_name} by ~{percent:.0f}%"


def _impact_text_from_aggregation(context: dict[str, Any], action: str) -> str:
    spread_ratio = context.get("spread_ratio")
    metric_name = str(context.get("metric_column") or "performance")
    group_column = str(context.get("group_column") or "segment")
    if spread_ratio is not None:
        return (
            f"Lift {group_column} {metric_name} by ~{abs(_safe_float(spread_ratio) * 100.0):.0f}% "
            "if lagging groups close the visible gap"
        )
    if "investigate" in action.lower() or "review" in action.lower():
        return f"Reduce underperformance risk in the weakest {group_column}."
    return f"Improve {metric_name} through a targeted {group_column} action."


def _fallback_impact_text(action: str, data_quality_score: float) -> str:
    lowered = action.lower()
    if "churn" in lowered:
        return "Reduce churn risk by ~5%"
    if "inventory" in lowered:
        return "Reduce stock imbalance risk over the next planning cycle."
    if data_quality_score < 6.0 or "data quality" in lowered:
        return "Improve data reliability for the next decision cycle"
    if "investigate" in lowered or "validate" in lowered:
        return "Reduce decision risk before scaling the next action."
    return "Improve performance by ~3-5% with a targeted trial."


def _reasoning_text(
    *,
    action: str,
    insights: list[str],
    plan: dict[str, Any],
    prediction_context: dict[str, Any],
    aggregation_context: dict[str, Any],
    confidence: str,
    risk_level: str,
    data_quality_score: float,
    model_quality: str | None,
) -> str:
    reasoning_parts: list[str] = []
    action_lower = action.lower()

    if prediction_context.get("delta_ratio") is not None:
        reasoning_parts.append(
            f"The predictive path shows approximately {prediction_context['delta_ratio']:.1%} movement in "
            f"{prediction_context.get('metric_name') or 'the target metric'} versus the available baseline."
        )
    elif aggregation_context.get("spread_ratio") is not None:
        reasoning_parts.append(
            f"{aggregation_context.get('top_group')} leads {aggregation_context.get('metric_column')} at "
            f"{aggregation_context.get('top_value')}, while {aggregation_context.get('bottom_group')} trails at "
            f"{aggregation_context.get('bottom_value')}."
        )
    elif insights:
        reasoning_parts.append(f"The strongest visible signal is: {insights[0]}")
    else:
        reasoning_parts.append("The action is based on the visible output pattern and trust-layer checks.")

    if plan.get("analysis_type") == "time_series":
        reasoning_parts.append("Time-series signals were used, so the recommendation follows the projected trend rather than a static snapshot.")
    elif plan.get("analysis_type") == "aggregation":
        reasoning_parts.append("Segment-level performance differences support a targeted business response instead of a broad rollout.")
    elif "validate" in action_lower or "investigate" in action_lower:
        reasoning_parts.append("This is framed as a validation step because the current evidence is directional rather than definitive.")

    if risk_level == "high" or confidence == "low":
        resolved_model_quality = str(model_quality or "weak").lower()
        if resolved_model_quality == "not_applicable":
            reasoning_parts.append(
                f"Treat this cautiously because data quality is {data_quality_score:.1f}/10 and this route does not include model-based confidence signals."
            )
        else:
            reasoning_parts.append(
                f"Treat this cautiously because data quality is {data_quality_score:.1f}/10 and model quality is {resolved_model_quality}."
            )
    elif confidence == "high":
        reasoning_parts.append("The trust signals are strong enough for a prioritized decision, assuming standard monitoring remains in place.")

    return " ".join(part.strip() for part in reasoning_parts if part).strip()


def _build_decision_id(action: str, result_hash: str | None) -> str:
    payload = json.dumps(
        {
            "action": str(action or "").strip(),
            "result_hash": str(result_hash or "").strip(),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def estimate_expected_impact(decision, *, result, plan, insights) -> str:
    impact_context = decision.get("_impact_context") if isinstance(decision, dict) else {}
    prediction_context = impact_context.get("prediction") if isinstance(impact_context, dict) else {}
    aggregation_context = impact_context.get("aggregation") if isinstance(impact_context, dict) else {}
    data_quality_score = _safe_float(decision.get("_data_quality_score"), default=0.0) if isinstance(decision, dict) else 0.0
    action = str((decision or {}).get("action") or "").strip() if isinstance(decision, dict) else ""

    analysis_type = str((plan or {}).get("analysis_type") or "").strip().lower()
    intent = str((plan or {}).get("intent") or "").strip().lower()
    if analysis_type in {"time_series", "ml"} or intent == "prediction":
        return _impact_text_from_prediction(prediction_context, action)
    if analysis_type == "aggregation" or intent == "comparison":
        return _impact_text_from_aggregation(aggregation_context, action)
    return _fallback_impact_text(action, data_quality_score)


def compute_decision_confidence(
    *,
    model_quality,
    data_quality_score,
    consistency_validated,
    inconsistency_detected,
) -> str:
    resolved_model_quality = str(model_quality or "weak").strip().lower()
    score = _safe_float(data_quality_score, default=0.0)

    if inconsistency_detected:
        return "low"
    if resolved_model_quality == "weak" or score < 5.0:
        return "low"
    if resolved_model_quality == "not_applicable":
        if score >= 8.0 and consistency_validated:
            return "high"
        if score >= 6.0:
            return "medium"
        return "low"
    if resolved_model_quality == "strong" and score >= 8.0 and consistency_validated:
        return "high"
    if resolved_model_quality in {"strong", "moderate"} and score >= 6.0:
        return "medium"
    return "low"


def compute_decision_risk(
    *,
    data_quality_score,
    model_quality,
    missing_percent,
    global_risk,
) -> str:
    resolved_global_risk = _normalize_risk(global_risk)
    resolved_model_quality = str(model_quality or "weak").strip().lower()
    score = _safe_float(data_quality_score, default=0.0)
    missing = _safe_float(missing_percent, default=0.0)

    if resolved_global_risk == "high" or score < 5.0 or resolved_model_quality == "weak" or missing >= 30.0:
        return "high"
    if resolved_model_quality == "not_applicable":
        if resolved_global_risk == "low" and score >= 8.0 and missing < 10.0:
            return "low"
        return "medium"
    if resolved_global_risk == "low" and score >= 8.0 and resolved_model_quality == "strong" and missing < 10.0:
        return "low"
    return "medium"


def build_decision_candidates(
    result,
    insights,
    *,
    plan,
    model_quality,
    data_quality,
    reproducibility,
    risk,
    warnings,
    seed_recommendations: list[str] | None = None,
) -> list[dict]:
    normalized_insights = [str(item).strip() for item in list(insights or []) if str(item).strip()]
    normalized_warnings = [str(item).strip() for item in list(warnings or []) if str(item).strip()]
    result_frame = _result_frame(result)
    has_result_signal = result is not None
    if isinstance(result_frame, pd.DataFrame):
        has_result_signal = not result_frame.empty
    elif isinstance(result, dict):
        has_result_signal = bool(result)
    elif isinstance(result, (list, tuple, set)):
        has_result_signal = bool(result)

    if not has_result_signal and not normalized_insights:
        return []

    data_quality_score = _data_quality_score(data_quality)
    missing_percent = _max_missing_percent(data_quality)
    consistency_validated = bool((reproducibility or {}).get("consistency_validated"))
    inconsistency_detected = not bool((reproducibility or {}).get("consistent_with_prior_runs", True))
    decision_confidence = compute_decision_confidence(
        model_quality=model_quality,
        data_quality_score=data_quality_score,
        consistency_validated=consistency_validated,
        inconsistency_detected=inconsistency_detected,
    )
    prediction_context = _extract_prediction_context(result, plan or {})
    aggregation_context = _extract_aggregation_context(result, plan or {})
    global_risk = compute_decision_risk(
        data_quality_score=data_quality_score,
        model_quality=model_quality,
        missing_percent=missing_percent,
        global_risk=risk,
    )

    base_actions = generate_recommendations(
        result,
        normalized_insights,
        plan=plan,
        warnings=normalized_warnings,
        model_quality=model_quality,
        risk=risk,
    )
    base_actions.extend(
        [
            str(item).strip()
            for item in list(seed_recommendations or [])
            if str(item).strip()
        ]
    )

    intent = str((plan or {}).get("intent") or "").strip().lower()
    if intent == "data_cleaning":
        base_actions.append("Persist a cleaned derived dataset so downstream decisions stay reproducible.")
    elif intent == "root_cause":
        base_actions.append("Validate the suspected driver with a targeted slice of raw source data before rollout.")

    if inconsistency_detected:
        base_actions.append("Re-run this decision on a validated recent slice before operational rollout.")
    elif data_quality_score < 5.0:
        base_actions.append("Improve data quality before taking a high-impact action.")

    unique_actions: list[str] = []
    seen_actions: set[str] = set()
    for action in base_actions:
        normalized_action = str(action or "").strip()
        if not normalized_action:
            continue
        key = normalized_action.lower()
        if key in seen_actions:
            continue
        seen_actions.add(key)
        unique_actions.append(normalized_action)

    if not unique_actions and not normalized_insights:
        return []

    result_hash = str((reproducibility or {}).get("result_hash") or "")
    decisions: list[dict[str, Any]] = []
    for action in unique_actions[:5]:
        decision = {
            "decision_id": _build_decision_id(action, result_hash),
            "action": action,
            "_impact_context": {
                "prediction": prediction_context,
                "aggregation": aggregation_context,
            },
            "_data_quality_score": data_quality_score,
            "confidence": decision_confidence,
            "risk_level": global_risk,
        }
        decision["expected_impact"] = estimate_expected_impact(decision, result=result, plan=plan, insights=normalized_insights)
        decision["reasoning"] = _reasoning_text(
            action=action,
            insights=normalized_insights,
            plan=plan or {},
            prediction_context=prediction_context,
            aggregation_context=aggregation_context,
            confidence=decision_confidence,
            risk_level=global_risk,
            data_quality_score=data_quality_score,
            model_quality=model_quality,
        )
        decisions.append(decision)
    return decisions


def rank_decisions(decisions):
    ranked_payload: list[dict[str, Any]] = []
    for decision in list(decisions or []):
        working = dict(decision or {})
        impact_score = _impact_score(str(working.get("expected_impact") or ""))
        confidence_score = _CONFIDENCE_LEVELS[_normalize_confidence(working.get("confidence"))]
        risk_penalty = _RISK_LEVELS[_normalize_risk(working.get("risk_level"))]
        ranking_score = round((impact_score * 0.5) + (confidence_score * 0.3) - (risk_penalty * 0.2), 6)

        priority = "LOW"
        if ranking_score >= 0.55 and _normalize_confidence(working.get("confidence")) != "low" and _normalize_risk(working.get("risk_level")) != "high":
            priority = "HIGH"
        elif ranking_score >= 0.25:
            priority = "MEDIUM"

        working["priority"] = priority
        working["_ranking_score"] = ranking_score
        ranked_payload.append(working)

    ranked_payload.sort(
        key=lambda item: (
            item.get("_ranking_score", 0.0),
            _PRIORITY_ORDER.get(str(item.get("priority") or "LOW"), 0),
            _CONFIDENCE_LEVELS.get(_normalize_confidence(item.get("confidence")), 0.0),
            -_RISK_LEVELS.get(_normalize_risk(item.get("risk_level")), 1.0),
        ),
        reverse=True,
    )

    cleaned_ranked: list[dict[str, Any]] = []
    for item in ranked_payload:
        cleaned_decision = {
            "decision_id": str(item.get("decision_id") or ""),
            "action": str(item.get("action") or ""),
            "expected_impact": str(item.get("expected_impact") or _DEFAULT_IMPACT_TEXT),
            "confidence": _normalize_confidence(item.get("confidence")),
            "risk_level": _normalize_risk(item.get("risk_level")),
            "priority": _normalize_priority(item.get("priority")),
            "reasoning": str(item.get("reasoning") or "").strip(),
        }
        if isinstance(item.get("decision_performance"), dict):
            cleaned_decision["decision_performance"] = {
                "historical_success_rate": round(
                    _safe_float(item["decision_performance"].get("historical_success_rate"), default=0.0),
                    4,
                ),
                "avg_impact": round(_safe_float(item["decision_performance"].get("avg_impact"), default=0.0), 4),
                "sample_size": int(item["decision_performance"].get("sample_size") or 0),
            }
        cleaned_ranked.append(cleaned_decision)
    return cleaned_ranked


def _build_risk_summary(decisions: list[dict[str, Any]], risk: str | None) -> str:
    if not decisions:
        return str(risk or "medium")

    counts = {"high": 0, "medium": 0, "low": 0}
    for decision in decisions:
        counts[_normalize_risk(decision.get("risk_level"))] += 1

    fragments = []
    if counts["high"]:
        fragments.append(f"{counts['high']} high-risk")
    if counts["medium"]:
        fragments.append(f"{counts['medium']} medium-risk")
    if counts["low"]:
        fragments.append(f"{counts['low']} low-risk")

    base = str(risk or "").strip()
    if not fragments:
        return base or "medium"
    if not base:
        return f"Decision mix includes {', '.join(fragments)} actions."
    return f"{base}. Decision mix includes {', '.join(fragments)} actions."


def _default_learning_insights() -> dict[str, Any]:
    return {
        "patterns": [],
        "confidence_adjustment": "No historical outcomes available; using base confidence.",
        "risk_adjustment": "No historical outcomes available; using base risk.",
    }


def _build_learning_adjustment_summary(
    *,
    decisions: list[dict[str, Any]],
    learning_patterns: dict[str, Any] | None,
) -> dict[str, Any]:
    if not decisions or not learning_patterns:
        return _default_learning_insights()

    decision_types = []
    confidence_messages: list[str] = []
    risk_messages: list[str] = []
    for decision in decisions:
        decision_type = str(decision.get("_decision_type") or "")
        if not decision_type:
            continue
        decision_types.append(decision_type)
        base_confidence = _normalize_confidence(decision.get("_base_confidence"))
        adjusted_confidence = _normalize_confidence(decision.get("confidence"))
        base_risk = _normalize_risk(decision.get("_base_risk"))
        adjusted_risk = _normalize_risk(decision.get("risk_level"))
        performance = dict(decision.get("decision_performance") or {})
        sample_size = int(performance.get("sample_size") or 0)

        if base_confidence != adjusted_confidence:
            confidence_messages.append(
                f"{decision_type} confidence moved from {base_confidence} to {adjusted_confidence} using {sample_size} historical outcomes."
            )
        elif sample_size:
            confidence_messages.append(
                f"{decision_type} confidence stayed {adjusted_confidence} after reviewing {sample_size} historical outcomes."
            )

        if base_risk != adjusted_risk:
            risk_messages.append(
                f"{decision_type} risk moved from {base_risk} to {adjusted_risk} using outcome history."
            )
        elif sample_size:
            risk_messages.append(
                f"{decision_type} risk stayed {adjusted_risk} after reviewing {sample_size} historical outcomes."
            )

    return {
        "patterns": build_learning_pattern_summaries(learning_patterns, decision_types=decision_types),
        "confidence_adjustment": confidence_messages[0] if confidence_messages else _default_learning_insights()["confidence_adjustment"],
        "risk_adjustment": risk_messages[0] if risk_messages else _default_learning_insights()["risk_adjustment"],
    }


def build_decision_layer(
    result,
    insights,
    *,
    plan,
    model_quality,
    data_quality,
    reproducibility,
    risk,
    warnings,
    learning_patterns: dict[str, Any] | None = None,
    seed_recommendations: list[str] | None = None,
) -> dict[str, Any]:
    data_quality_score = _data_quality_score(data_quality)
    consistency_validated = bool((reproducibility or {}).get("consistency_validated"))
    inconsistency_detected = not bool((reproducibility or {}).get("consistent_with_prior_runs", True))

    base_decision_confidence = compute_decision_confidence(
        model_quality=model_quality,
        data_quality_score=data_quality_score,
        consistency_validated=consistency_validated,
        inconsistency_detected=inconsistency_detected,
    )
    base_decisions = build_decision_candidates(
            result,
            insights,
            plan=plan,
            model_quality=model_quality,
            data_quality=data_quality,
            reproducibility=reproducibility,
            risk=risk,
            warnings=warnings,
            seed_recommendations=seed_recommendations,
    )
    learning_patterns = dict(learning_patterns or {})
    learned_decisions: list[dict[str, Any]] = []
    for decision in base_decisions:
        working = dict(decision)
        decision_type = infer_decision_type(working.get("action"))
        base_confidence = _normalize_confidence(working.get("confidence") or base_decision_confidence)
        base_risk = _normalize_risk(working.get("risk_level") or risk)
        working["_decision_type"] = decision_type
        working["_base_confidence"] = base_confidence
        working["_base_risk"] = base_risk
        working["decision_performance"] = decision_performance_snapshot(learning_patterns, decision_type)
        working["confidence"] = adjust_confidence(base_confidence, learning_patterns, decision_type)
        working["risk_level"] = adjust_risk(base_risk, learning_patterns, decision_type)
        learned_decisions.append(working)

    decisions = rank_decisions(learned_decisions)
    learning_insights = _build_learning_adjustment_summary(
        decisions=learned_decisions,
        learning_patterns=learning_patterns,
    )
    if not decisions:
        return {
            "decisions": [],
            "top_decision": None,
            "decision_confidence": "low",
            "risk_summary": str(risk or "medium"),
            "learning_insights": learning_insights,
        }

    top_decision = dict(decisions[0])
    decision_confidence = _normalize_confidence(top_decision.get("confidence") or base_decision_confidence)
    return {
        "decisions": decisions,
        "top_decision": top_decision,
        "decision_confidence": decision_confidence,
        "risk_summary": _build_risk_summary(decisions, risk),
        "learning_insights": learning_insights,
    }


def ensure_decision_layer_defaults(
    value: Any,
    *,
    risk: str | None = None,
    recommendations: list[str] | None = None,
) -> dict[str, Any]:
    payload = value if isinstance(value, dict) else {}
    decisions = payload.get("decisions")
    normalized_decisions: list[dict[str, Any]] = []
    if isinstance(decisions, list):
        normalized_decisions = rank_decisions(decisions)

    if not normalized_decisions and recommendations:
        fallback_risk = _normalize_risk(risk)
        fallback_priority = "LOW" if fallback_risk == "high" else "MEDIUM"
        normalized_decisions = rank_decisions(
            [
                {
                    "decision_id": _build_decision_id(action, None),
                    "action": str(action),
                    "expected_impact": _DEFAULT_IMPACT_TEXT,
                    "confidence": "low",
                    "risk_level": fallback_risk,
                    "priority": fallback_priority,
                    "reasoning": "This structured decision was reconstructed from a legacy recommendation payload.",
                }
                for action in recommendations
                if str(action).strip()
            ]
        )

    top_decision = payload.get("top_decision")
    normalized_top = dict(top_decision) if isinstance(top_decision, dict) else (dict(normalized_decisions[0]) if normalized_decisions else None)
    if isinstance(normalized_top, dict):
        normalized_top = rank_decisions([normalized_top])[0]

    decision_confidence = _normalize_confidence(
        payload.get("decision_confidence")
        or (normalized_top or {}).get("confidence")
        or "low"
    )

    learning_insights = _default_learning_insights()
    learning_insights.update(dict(payload.get("learning_insights") or {}))

    return {
        "decisions": normalized_decisions,
        "top_decision": normalized_top,
        "decision_confidence": decision_confidence if normalized_decisions else "low",
        "risk_summary": str(payload.get("risk_summary") or _build_risk_summary(normalized_decisions, risk)),
        "learning_insights": learning_insights,
    }


def derive_recommendations_from_decision_layer(decision_layer: dict[str, Any] | None) -> list[str]:
    decisions = list((decision_layer or {}).get("decisions") or [])
    return [str(item.get("action") or "").strip() for item in decisions if str(item.get("action") or "").strip()]


def build_business_decisions_text(decision_layer: dict[str, Any] | None) -> str | None:
    recommendations = derive_recommendations_from_decision_layer(decision_layer)
    if not recommendations:
        return None
    return "\n".join(recommendations)


def build_forecast_recommendations(decision_layer: dict[str, Any] | None) -> list[dict[str, Any]]:
    decisions = list((decision_layer or {}).get("decisions") or [])
    forecast_recommendations: list[dict[str, Any]] = []
    for index, decision in enumerate(decisions, start=1):
        expected_impact = str(decision.get("expected_impact") or "")
        impact_direction = "increase_profit"
        if any(token in expected_impact.lower() for token in ("decline", "protect", "reduce")):
            impact_direction = "protect_revenue"
        forecast_recommendations.append(
            {
                "category": re.sub(r"[^a-z0-9]+", "_", str(decision.get("action") or "").strip().lower()).strip("_") or f"decision_{index}",
                "priority": index,
                "priority_label": str(decision.get("priority") or "LOW"),
                "title": str(decision.get("action") or ""),
                "recommended_action": str(decision.get("action") or ""),
                "rationale": str(decision.get("reasoning") or ""),
                "impact_direction": impact_direction,
                "expected_impact": expected_impact,
                "confidence": _normalize_confidence(decision.get("confidence")),
                "risk_level": _normalize_risk(decision.get("risk_level")),
                "decision_id": str(decision.get("decision_id") or ""),
            }
        )
    return forecast_recommendations
