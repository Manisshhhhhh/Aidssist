from __future__ import annotations

import math
import re
from typing import Any

import pandas as pd

from backend.services.dag_execution import build_execution_graph
from backend.services.optimizer import optimize_plan as optimize_single_plan
from backend.services.performance_tracker import (
    estimate_cost,
    estimate_numeric_cost,
    get_average_tool_time,
    get_tool_performance_memory,
    get_tool_reliability,
)


TOOL_SQL = "SQL"
TOOL_EXCEL = "EXCEL"
TOOL_PYTHON = "PYTHON"
TOOL_BI = "BI"
_SUPPORTED_TOOLS = {TOOL_SQL, TOOL_EXCEL, TOOL_PYTHON, TOOL_BI}
_PREDICTION_TOKENS = ("predict", "prediction", "forecast", "estimate", "project")
_DASHBOARD_TOKENS = ("dashboard", "chart", "plot", "graph", "visual", "kpi")
_SQL_TOKENS = ("top", "bottom", "rank", "filter", "join", "merge", "customer")
_CLEANING_TOKENS = ("clean", "cleanup", "dedupe", "duplicate", "missing", "null", "normalize")
_PYTHON_TOKENS = ("trend", "correlation", "regression", "analyze", "analyse", "explain")
_COST_LABEL_TO_LIMIT = {"low": 6.0, "medium": 12.0, "high": 24.0}
_DEFAULT_CONSTRAINTS = {
    "max_execution_time": None,
    "budget": "medium",
    "priority": "reliability",
}


def _normalize_text(value: str | None) -> str:
    return str(value or "").strip().lower()


def _contains_any(text: str, tokens: tuple[str, ...]) -> bool:
    return any(token in text for token in tokens)


def _normalize_dependencies(value: Any) -> list[int]:
    dependencies: list[int] = []
    for item in list(value or []):
        try:
            dependency = int(item)
        except (TypeError, ValueError):
            continue
        if dependency > 0 and dependency not in dependencies:
            dependencies.append(dependency)
    return dependencies


def _step(step_number: int, tool: str, task: str, *, query: str, depends_on: list[int] | None = None, uses_context: bool = False) -> dict[str, Any]:
    return {
        "step": int(step_number),
        "tool": str(tool).strip().upper() or TOOL_PYTHON,
        "task": str(task).strip() or "Run the requested analysis step.",
        "query": str(query).strip() or str(task).strip(),
        "depends_on": _normalize_dependencies(depends_on),
        "uses_context": bool(uses_context),
    }


def _standard_task(tool: str, *, query: str, prediction: bool = False, dashboard: bool = False, cleaning: bool = False) -> str:
    normalized_tool = str(tool).strip().upper()
    if normalized_tool == TOOL_SQL:
        if "customer" in _normalize_text(query):
            return "Get the top customers using SQL-style grouping, ranking, and filtering."
        return "Use SQL-style reasoning to filter, group, or rank the requested data."
    if normalized_tool == TOOL_EXCEL:
        if cleaning:
            return "Profile data quality issues, clean obvious anomalies, and prepare an analyst-friendly summary table."
        return "Build an Excel-style summary or pivot for the request."
    if normalized_tool == TOOL_BI:
        return "Build dashboard-ready charts and KPI structure from the current context." if dashboard else "Build a BI-ready chart and KPI payload."
    if prediction:
        return "Run Python prediction or forecasting on the current context."
    if "trend" in _normalize_text(query):
        return "Analyze trends and derived metrics in Python before presentation."
    return "Run Python analysis on the current context."


def _normalize_plan(plan: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, step in enumerate(list(plan or []), start=1):
        if not isinstance(step, dict):
            continue
        tool = str(step.get("tool") or TOOL_PYTHON).strip().upper()
        if tool not in _SUPPORTED_TOOLS:
            continue
        task = str(step.get("task") or "").strip()
        if not task:
            continue
        normalized.append(
            {
                "step": int(step.get("step") or index),
                "tool": tool,
                "task": task,
                "query": str(step.get("query") or task).strip() or task,
                "depends_on": _normalize_dependencies(step.get("depends_on")),
                "uses_context": bool(step.get("uses_context")),
            }
        )
    return normalized


def _renumber(plan: list[dict[str, Any]]) -> list[dict[str, Any]]:
    mapping = {
        int(step.get("step") or index): index
        for index, step in enumerate(plan, start=1)
    }
    renumbered: list[dict[str, Any]] = []
    for index, step in enumerate(plan, start=1):
        updated = dict(step)
        updated["step"] = index
        updated["depends_on"] = [
            mapping[dependency]
            for dependency in list(updated.get("depends_on") or [])
            if dependency in mapping and mapping[dependency] < index
        ]
        renumbered.append(updated)
    return renumbered


def _normalize_constraints(constraints: dict[str, Any] | None) -> dict[str, Any]:
    payload = dict(_DEFAULT_CONSTRAINTS)
    payload.update(dict(constraints or {}))
    max_execution_time = payload.get("max_execution_time")
    try:
        payload["max_execution_time"] = int(max_execution_time) if max_execution_time is not None else None
    except (TypeError, ValueError):
        payload["max_execution_time"] = None
    budget = str(payload.get("budget") or "medium").strip().lower()
    payload["budget"] = budget if budget in _COST_LABEL_TO_LIMIT else "medium"
    priority = str(payload.get("priority") or "reliability").strip().lower()
    if priority not in {"speed", "cost", "reliability"}:
        priority = "reliability"
    payload["priority"] = priority
    return payload


def _estimate_step_time_ms(step: dict[str, Any], df: pd.DataFrame | None) -> float:
    average_time = get_average_tool_time(step.get("tool"))
    numeric_cost = estimate_numeric_cost(step, df=df)
    scaling_factor = 0.75 + min(2.0, numeric_cost / 5.0)
    return round(average_time * scaling_factor, 2)


def _semantic_reliability(plan: list[dict[str, Any]], *, query: str, preflight: dict[str, Any] | None = None) -> float:
    normalized_query = _normalize_text(query)
    tools = {str(step.get("tool") or "").strip().upper() for step in plan}
    reliability = sum(get_tool_reliability(step.get("tool")) for step in plan) / float(len(plan) or 1)

    if _contains_any(normalized_query, _PREDICTION_TOKENS) and TOOL_PYTHON not in tools:
        reliability *= 0.72
    if _contains_any(normalized_query, _SQL_TOKENS) and TOOL_SQL not in tools:
        reliability *= 0.82
    if _contains_any(normalized_query, _DASHBOARD_TOKENS) and TOOL_BI not in tools:
        reliability *= 0.8
    if _contains_any(normalized_query, _CLEANING_TOKENS) and TOOL_EXCEL not in tools:
        reliability *= 0.88
    if list((preflight or {}).get("blocking_errors") or []) and TOOL_EXCEL not in tools:
        reliability *= 0.78
    return round(max(0.0, min(0.99, reliability)), 4)


def _build_plan_from_tools(query: str, tools: list[str]) -> list[dict[str, Any]]:
    normalized_query = _normalize_text(query)
    needs_prediction = _contains_any(normalized_query, _PREDICTION_TOKENS)
    needs_dashboard = _contains_any(normalized_query, _DASHBOARD_TOKENS)
    needs_cleaning = _contains_any(normalized_query, _CLEANING_TOKENS)

    plan: list[dict[str, Any]] = []
    dependency: list[int] = []
    for index, tool in enumerate(tools, start=1):
        uses_context = bool(dependency)
        plan.append(
            _step(
                index,
                tool,
                _standard_task(
                    tool,
                    query=query,
                    prediction=needs_prediction and tool == TOOL_PYTHON,
                    dashboard=needs_dashboard and tool == TOOL_BI,
                    cleaning=needs_cleaning and tool == TOOL_EXCEL,
                ),
                query=query,
                depends_on=dependency,
                uses_context=uses_context,
            )
        )
        dependency = [index]
    return plan


def generate_candidate_plans(
    query: str,
    df: pd.DataFrame,
    *,
    base_plan: list[dict[str, Any]] | None = None,
    plan: dict[str, Any] | None = None,
    preflight: dict[str, Any] | None = None,
) -> list[list[dict[str, Any]]]:
    del plan
    normalized_query = _normalize_text(query)
    normalized_base_plan = optimize_single_plan(_normalize_plan(base_plan), df=df, preflight=preflight)
    candidates: list[list[dict[str, Any]]] = [normalized_base_plan] if normalized_base_plan else []

    needs_prediction = _contains_any(normalized_query, _PREDICTION_TOKENS)
    needs_dashboard = _contains_any(normalized_query, _DASHBOARD_TOKENS)
    needs_sql = _contains_any(normalized_query, _SQL_TOKENS) or any(step.get("tool") == TOOL_SQL for step in normalized_base_plan)
    needs_cleaning = _contains_any(normalized_query, _CLEANING_TOKENS)
    needs_python = needs_prediction or _contains_any(normalized_query, _PYTHON_TOKENS) or any(step.get("tool") == TOOL_PYTHON for step in normalized_base_plan)

    tool_variants: list[list[str]] = []
    if needs_prediction and needs_dashboard and needs_sql:
        tool_variants.extend(
            [
                [TOOL_SQL, TOOL_PYTHON, TOOL_BI],
                [TOOL_PYTHON, TOOL_BI],
                [TOOL_EXCEL, TOOL_PYTHON, TOOL_BI],
            ]
        )
    elif needs_prediction and needs_sql:
        tool_variants.extend(
            [
                [TOOL_SQL, TOOL_PYTHON],
                [TOOL_PYTHON],
                [TOOL_EXCEL, TOOL_PYTHON],
            ]
        )
    elif needs_cleaning and needs_dashboard:
        tool_variants.extend(
            [
                [TOOL_EXCEL, TOOL_PYTHON, TOOL_BI],
                [TOOL_PYTHON, TOOL_BI],
                [TOOL_EXCEL, TOOL_BI],
            ]
        )
    elif needs_dashboard and needs_python:
        tool_variants.extend(
            [
                [TOOL_PYTHON, TOOL_BI],
                [TOOL_EXCEL, TOOL_BI],
                [TOOL_SQL, TOOL_BI] if needs_sql else [TOOL_EXCEL, TOOL_PYTHON, TOOL_BI],
            ]
        )
    elif needs_dashboard:
        tool_variants.extend(
            [
                [TOOL_EXCEL, TOOL_BI],
                [TOOL_PYTHON, TOOL_BI],
                [TOOL_SQL, TOOL_BI] if needs_sql else [TOOL_EXCEL],
            ]
        )
    elif needs_prediction:
        tool_variants.extend(
            [
                [TOOL_PYTHON],
                [TOOL_SQL, TOOL_PYTHON] if needs_sql else [TOOL_EXCEL, TOOL_PYTHON],
                [TOOL_EXCEL, TOOL_PYTHON],
            ]
        )
    else:
        tool_variants.extend(
            [
                [TOOL_EXCEL],
                [TOOL_SQL] if needs_sql else [TOOL_PYTHON],
                [TOOL_PYTHON],
            ]
        )

    signatures: set[tuple[str, ...]] = set()
    deduped_candidates: list[list[dict[str, Any]]] = []
    for candidate in candidates + [
        optimize_single_plan(_build_plan_from_tools(query, tools), df=df, preflight=preflight)
        for tools in tool_variants
    ]:
        normalized_candidate = _renumber(_normalize_plan(candidate))
        if not normalized_candidate:
            continue
        signature = tuple(str(step.get("tool") or "").strip().upper() for step in normalized_candidate)
        if signature in signatures:
            continue
        signatures.add(signature)
        deduped_candidates.append(
            [
                {
                    **dict(step),
                    "cost_estimate": estimate_cost(step, df=df),
                }
                for step in normalized_candidate
            ]
        )
    return deduped_candidates


def score_plan(
    plan: list[dict[str, Any]] | None,
    constraints: dict[str, Any] | None,
    *,
    df: pd.DataFrame | None = None,
    query: str | None = None,
    preflight: dict[str, Any] | None = None,
    tool_performance: dict[str, dict[str, float | int]] | None = None,
) -> dict[str, Any]:
    del tool_performance
    normalized_plan = _renumber(_normalize_plan(plan))
    normalized_constraints = _normalize_constraints(constraints)
    if not normalized_plan:
        return {
            "score": 0.0,
            "estimated_execution_time_ms": 0,
            "estimated_cost_numeric": 0.0,
            "estimated_cost_estimate": "low",
            "reliability": 0.0,
            "constraints_applied": normalized_constraints,
            "parallel_batches": [],
            "parallel_execution": False,
            "within_constraints": False,
        }

    graph = build_execution_graph(normalized_plan)
    batch_time_estimates: list[float] = []
    for batch in list(graph.get("batches") or []):
        batch_times = [
            _estimate_step_time_ms(graph["nodes"][step_id], df)
            for step_id in batch
            if step_id in graph.get("nodes", {})
        ]
        if batch_times:
            batch_time_estimates.append(max(batch_times))
    estimated_execution_time_ms = int(round(sum(batch_time_estimates)))

    estimated_cost_numeric = round(
        sum(estimate_numeric_cost(step, df=df) for step in normalized_plan),
        2,
    )
    estimated_cost_estimate = estimate_cost({"tool": TOOL_PYTHON, "cost_estimate": estimated_cost_numeric}, df=df)
    reliability = _semantic_reliability(
        normalized_plan,
        query=str(query or ""),
        preflight=preflight,
    )

    budget_limit = _COST_LABEL_TO_LIMIT.get(normalized_constraints["budget"], _COST_LABEL_TO_LIMIT["medium"])
    time_limit = normalized_constraints.get("max_execution_time")
    within_budget = estimated_cost_numeric <= budget_limit
    within_time = time_limit is None or estimated_execution_time_ms <= time_limit
    within_constraints = within_budget and within_time

    if normalized_constraints["priority"] == "speed":
        weights = {"reliability": 0.5, "time": 0.35, "cost": 0.15}
    elif normalized_constraints["priority"] == "cost":
        weights = {"reliability": 0.5, "time": 0.15, "cost": 0.35}
    else:
        weights = {"reliability": 0.6, "time": 0.2, "cost": 0.2}

    time_scale = float(time_limit or max(200, estimated_execution_time_ms or 1))
    time_score = 1.0 / (1.0 + (estimated_execution_time_ms / time_scale))
    cost_scale = float(budget_limit or 1.0)
    cost_score = 1.0 / (1.0 + (estimated_cost_numeric / cost_scale))

    score = (
        (reliability * weights["reliability"])
        + (time_score * weights["time"])
        + (cost_score * weights["cost"])
    )
    if not within_budget:
        score -= min(0.35, 0.15 + ((estimated_cost_numeric - budget_limit) / max(1.0, budget_limit)))
    if not within_time and time_limit is not None:
        score -= min(0.35, 0.15 + ((estimated_execution_time_ms - time_limit) / max(1.0, float(time_limit))))
    score = round(max(0.0, min(1.0, score)), 4)

    return {
        "score": score,
        "estimated_execution_time_ms": estimated_execution_time_ms,
        "estimated_cost_numeric": estimated_cost_numeric,
        "estimated_cost_estimate": estimated_cost_estimate,
        "reliability": reliability,
        "constraints_applied": normalized_constraints,
        "parallel_batches": [list(batch) for batch in list(graph.get("batches") or []) if len(batch) > 1],
        "parallel_execution": any(len(batch) > 1 for batch in list(graph.get("batches") or [])),
        "within_constraints": within_constraints,
        "within_budget": within_budget,
        "within_time": within_time,
    }


def select_best_plan(
    query: str,
    df: pd.DataFrame,
    *,
    base_plan: list[dict[str, Any]] | None,
    constraints: dict[str, Any] | None = None,
    preflight: dict[str, Any] | None = None,
    tool_performance: dict[str, dict[str, float | int]] | None = None,
) -> dict[str, Any]:
    performance_memory = tool_performance or get_tool_performance_memory()
    candidates = generate_candidate_plans(
        query,
        df,
        base_plan=base_plan,
        preflight=preflight,
    )
    if not candidates:
        fallback_plan = optimize_single_plan(_normalize_plan(base_plan), df=df, preflight=preflight)
        candidates = [fallback_plan] if fallback_plan else []

    scored_candidates: list[dict[str, Any]] = []
    best_candidate: dict[str, Any] | None = None
    best_score = -math.inf

    for candidate_plan in candidates:
        scoring = score_plan(
            candidate_plan,
            constraints,
            df=df,
            query=query,
            preflight=preflight,
            tool_performance=performance_memory,
        )
        candidate_payload = {
            "plan": candidate_plan,
            "score": scoring["score"],
            "scoring": scoring,
        }
        scored_candidates.append(candidate_payload)
        if scoring["score"] > best_score:
            best_score = scoring["score"]
            best_candidate = candidate_payload

    if best_candidate is None:
        normalized_constraints = _normalize_constraints(constraints)
        return {
            "selected_plan": [],
            "optimization": {
                "plans_considered": 0,
                "selected_plan_score": 0.0,
                "constraints_applied": normalized_constraints,
                "estimated_execution_time_ms": 0,
                "estimated_cost_numeric": 0.0,
                "cost_estimate": "low",
                "parallel_execution": False,
                "optimized": False,
            },
            "candidate_plans": [],
        }

    selected_scoring = dict(best_candidate["scoring"])
    return {
        "selected_plan": list(best_candidate["plan"]),
        "optimization": {
            "plans_considered": len(scored_candidates),
            "selected_plan_score": selected_scoring["score"],
            "constraints_applied": dict(selected_scoring["constraints_applied"]),
            "estimated_execution_time_ms": selected_scoring["estimated_execution_time_ms"],
            "estimated_cost_numeric": selected_scoring["estimated_cost_numeric"],
            "cost_estimate": selected_scoring["estimated_cost_estimate"],
            "parallel_execution": bool(selected_scoring["parallel_execution"]),
            "optimized": len(scored_candidates) > 1,
        },
        "candidate_plans": scored_candidates,
    }
