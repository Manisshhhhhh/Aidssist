from __future__ import annotations

from threading import Lock
from typing import Any

import pandas as pd


_COST_LEVELS = ("low", "medium", "high")
_COST_SCORE = {"low": 1, "medium": 2, "high": 3}
_DEFAULT_TOOL_COSTS = {
    "SQL": "low",
    "EXCEL": "low",
    "PYTHON": "low",
    "BI": "medium",
    "LLM": "high",
}
_DEFAULT_TOOL_PERFORMANCE = {
    "SQL": {"avg_time": 30.0, "avg_time_ms": 30.0, "run_count": 0},
    "EXCEL": {"avg_time": 20.0, "avg_time_ms": 20.0, "run_count": 0},
    "PYTHON": {"avg_time": 120.0, "avg_time_ms": 120.0, "run_count": 0},
    "BI": {"avg_time": 80.0, "avg_time_ms": 80.0, "run_count": 0},
    "LLM": {"avg_time": 250.0, "avg_time_ms": 250.0, "run_count": 0},
}
_DEFAULT_TOOL_RELIABILITY = {
    "SQL": 0.96,
    "EXCEL": 0.94,
    "PYTHON": 0.91,
    "BI": 0.9,
    "LLM": 0.82,
}
_TOOL_BASE_NUMERIC_COST = {
    "SQL": 1.0,
    "EXCEL": 0.9,
    "PYTHON": 1.8,
    "BI": 1.4,
    "LLM": 3.0,
}
_COMPLEXITY_TOKENS = {
    "join": 0.8,
    "merge": 0.8,
    "rank": 0.4,
    "top": 0.3,
    "bottom": 0.3,
    "forecast": 1.0,
    "predict": 1.0,
    "prediction": 1.0,
    "trend": 0.6,
    "correlation": 0.7,
    "regression": 1.0,
    "dashboard": 0.6,
    "chart": 0.4,
    "visual": 0.4,
    "clean": 0.4,
    "pivot": 0.3,
}
_TOOL_PERFORMANCE_MEMORY = {
    tool: {
        **dict(metrics),
        "success_count": 0,
        "failure_count": 0,
        "success_rate": _DEFAULT_TOOL_RELIABILITY.get(tool, 0.9),
    }
    for tool, metrics in _DEFAULT_TOOL_PERFORMANCE.items()
}
_MEMORY_LOCK = Lock()


def _normalize_tool(value: str | None) -> str:
    return str(value or "").strip().upper() or "PYTHON"


def _normalize_cost(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in _COST_LEVELS:
        return normalized
    return "low"


def _cost_from_numeric(value: int | float | None) -> str:
    numeric_value = max(0.0, float(value or 0.0))
    if numeric_value >= 8.0:
        return "high"
    if numeric_value >= 4.0:
        return "medium"
    return "low"


def _normalize_tool_payload(step: dict[str, Any] | None) -> dict[str, Any]:
    return dict(step or {})


def _dataset_scale_factor(df: pd.DataFrame | None) -> float:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return 1.0
    row_factor = min(3.0, max(0.0, len(df) / 5000.0))
    column_factor = min(1.5, max(0.0, len(df.columns) / 40.0))
    return 1.0 + row_factor + column_factor


def _task_complexity_factor(step: dict[str, Any] | None) -> float:
    payload = _normalize_tool_payload(step)
    task_text = " ".join(
        str(payload.get(key) or "").strip().lower()
        for key in ("task", "query", "sql_plan")
    )
    complexity = 1.0
    for token, weight in _COMPLEXITY_TOKENS.items():
        if token in task_text:
            complexity += weight
    if list(payload.get("depends_on") or []):
        complexity += 0.2 * len(list(payload.get("depends_on") or []))
    if payload.get("uses_context"):
        complexity += 0.25
    if list(payload.get("python_steps") or []):
        complexity += min(0.8, 0.15 * len(list(payload.get("python_steps") or [])))
    if isinstance(payload.get("excel_logic"), dict) and payload.get("excel_logic"):
        complexity += 0.2
    return complexity


def estimate_numeric_cost(step: dict[str, Any] | None, df: pd.DataFrame | None = None) -> float:
    payload = _normalize_tool_payload(step)
    explicit_cost = payload.get("cost_estimate")
    if isinstance(explicit_cost, (int, float)) and not isinstance(explicit_cost, bool):
        return round(max(0.0, float(explicit_cost)), 2)

    tool = _normalize_tool(payload.get("tool"))
    base_cost = float(_TOOL_BASE_NUMERIC_COST.get(tool, 1.5))
    complexity_factor = _task_complexity_factor(payload)
    dataset_factor = _dataset_scale_factor(df)
    numeric_cost = base_cost * complexity_factor * dataset_factor
    return round(numeric_cost, 2)


def estimate_cost(step: dict[str, Any] | None, df: pd.DataFrame | None = None) -> str:
    payload = dict(step or {})
    explicit_cost = payload.get("cost_estimate")
    if isinstance(explicit_cost, str):
        return _normalize_cost(explicit_cost)
    if isinstance(explicit_cost, (int, float)) and not isinstance(explicit_cost, bool):
        return _cost_from_numeric(explicit_cost)
    if explicit_cost is not None:
        return _normalize_cost(explicit_cost)
    numeric_cost = estimate_numeric_cost(payload, df=df)
    return _cost_from_numeric(numeric_cost)


def get_tool_performance_memory() -> dict[str, dict[str, float | int]]:
    with _MEMORY_LOCK:
        return {
            tool: dict(metrics)
            for tool, metrics in _TOOL_PERFORMANCE_MEMORY.items()
        }


def get_average_tool_time(tool: str | None) -> float:
    normalized_tool = _normalize_tool(tool)
    with _MEMORY_LOCK:
        metrics = dict(_TOOL_PERFORMANCE_MEMORY.get(normalized_tool) or _DEFAULT_TOOL_PERFORMANCE.get(normalized_tool) or {})
    avg_time = float(metrics.get("avg_time_ms") or metrics.get("avg_time") or 0.0)
    if avg_time > 0:
        return avg_time
    return float((_DEFAULT_TOOL_PERFORMANCE.get(normalized_tool) or {}).get("avg_time_ms") or 100.0)


def get_tool_reliability(tool: str | None) -> float:
    normalized_tool = _normalize_tool(tool)
    with _MEMORY_LOCK:
        metrics = dict(_TOOL_PERFORMANCE_MEMORY.get(normalized_tool) or {})
    success_rate = float(metrics.get("success_rate") or 0.0)
    if success_rate > 0:
        return max(0.0, min(1.0, success_rate))
    return float(_DEFAULT_TOOL_RELIABILITY.get(normalized_tool, 0.9))


def record_tool_performance(
    tool: str | None,
    execution_time_ms: int | float | None,
    *,
    status: str | None = None,
) -> dict[str, float | int]:
    normalized_tool = _normalize_tool(tool)
    measured_time = max(0.0, float(execution_time_ms or 0.0))
    normalized_status = str(status or "completed").strip().lower()
    succeeded = normalized_status in {"completed", "fallback_completed"}
    with _MEMORY_LOCK:
        metrics = dict(
            _TOOL_PERFORMANCE_MEMORY.get(normalized_tool)
            or _DEFAULT_TOOL_PERFORMANCE.get(normalized_tool)
            or {"avg_time": 0.0, "avg_time_ms": 0.0, "run_count": 0}
        )
        prior_runs = int(metrics.get("run_count") or 0)
        prior_avg = float(metrics.get("avg_time_ms") or metrics.get("avg_time") or 0.0)
        prior_success = int(metrics.get("success_count") or 0)
        prior_failure = int(metrics.get("failure_count") or 0)
        updated_avg = (
            measured_time
            if prior_runs <= 0
            else ((prior_avg * prior_runs) + measured_time) / float(prior_runs + 1)
        )
        success_count = prior_success + (1 if succeeded else 0)
        failure_count = prior_failure + (0 if succeeded else 1)
        observed_runs = success_count + failure_count
        default_success_rate = float(_DEFAULT_TOOL_RELIABILITY.get(normalized_tool, 0.9))
        success_rate = (
            (success_count / float(observed_runs))
            if observed_runs > 0
            else default_success_rate
        )
        updated_metrics = {
            "avg_time": round(updated_avg, 2),
            "avg_time_ms": round(updated_avg, 2),
            "run_count": prior_runs + 1,
            "success_count": success_count,
            "failure_count": failure_count,
            "success_rate": round(success_rate, 4),
        }
        _TOOL_PERFORMANCE_MEMORY[normalized_tool] = updated_metrics
        return dict(updated_metrics)


def combine_cost_estimates(costs: list[str] | tuple[str, ...] | None) -> str:
    normalized_costs = [_normalize_cost(item) for item in list(costs or []) if str(item or "").strip()]
    if not normalized_costs:
        return "low"

    score_total = sum(_COST_SCORE[cost] for cost in normalized_costs)
    highest_cost = max(normalized_costs, key=lambda item: _COST_SCORE[item])
    if highest_cost == "high" or score_total >= 6:
        return "high"
    if highest_cost == "medium" or score_total >= 3:
        return "medium"
    return "low"


def track_performance(
    step: dict[str, Any],
    start_time: float,
    end_time: float,
    *,
    df: pd.DataFrame | None = None,
    status: str,
    warnings: list[str] | None = None,
    error: str | None = None,
    fallback_tool: str | None = None,
) -> dict[str, Any]:
    execution_time_ms = max(0, int(round((float(end_time) - float(start_time)) * 1000.0)))
    return {
        "step": int(step.get("step") or 0),
        "tool": _normalize_tool(step.get("tool")),
        "task": str(step.get("task") or "").strip() or None,
        "status": str(status or "completed").strip().lower() or "completed",
        "execution_time_ms": execution_time_ms,
        "cost_estimate": estimate_cost(step, df=df),
        "warnings": [str(item).strip() for item in list(warnings or []) if str(item).strip()],
        "error": str(error).strip() if str(error or "").strip() else None,
        "fallback_tool": _normalize_tool(fallback_tool) if str(fallback_tool or "").strip() else None,
    }


def build_optimization_summary(
    execution_trace: list[dict[str, Any]] | None,
    *,
    optimized: bool,
    parallel_execution: bool,
    plans_considered: int | None = None,
    selected_plan_score: float | None = None,
    constraints_applied: dict[str, Any] | None = None,
) -> dict[str, Any]:
    trace = [dict(item) for item in list(execution_trace or []) if isinstance(item, dict)]
    execution_time_total = int(sum(int(item.get("execution_time_ms") or 0) for item in trace))
    cost_estimate = combine_cost_estimates([str(item.get("cost_estimate") or "") for item in trace])
    return {
        "execution_time_total": execution_time_total,
        "cost_estimate": cost_estimate,
        "optimized": bool(optimized),
        "parallel_execution": bool(parallel_execution),
        "plans_considered": max(1, int(plans_considered or 1)),
        "selected_plan_score": round(max(0.0, min(1.0, float(selected_plan_score or 0.0))), 4),
        "constraints_applied": dict(constraints_applied or {}),
    }
