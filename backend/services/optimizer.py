from __future__ import annotations

import re
from typing import Any

import pandas as pd

from backend.dashboard_helpers import classify_columns, infer_datetime_columns
from backend.services.performance_tracker import estimate_cost, get_average_tool_time


TOOL_SQL = "SQL"
TOOL_EXCEL = "EXCEL"
TOOL_PYTHON = "PYTHON"
TOOL_BI = "BI"
_SUPPORTED_TOOLS = {TOOL_SQL, TOOL_EXCEL, TOOL_PYTHON, TOOL_BI}
_SQL_HINTS = ("sql", "select", "where", "join", "merge", "rank", "top", "bottom", "filter")
_EXCEL_HINTS = ("pivot", "summary", "summarize", "show", "breakdown", "by ", "total", "average")
_BI_HINTS = ("dashboard", "chart", "plot", "graph", "visual", "kpi")


def _normalize_text(value: str | None) -> str:
    return str(value or "").strip().lower()


def _has_chartable_fields(df: pd.DataFrame | None) -> bool:
    if not isinstance(df, pd.DataFrame) or df.empty or len(df.columns) == 0:
        return False
    datetime_columns = infer_datetime_columns(df)
    numeric_columns, categorical_columns, datetime_column_names = classify_columns(df, datetime_columns)
    return bool(numeric_columns) and bool(categorical_columns or datetime_column_names)


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


def _normalize_step(value: Any, *, step_number: int, df: pd.DataFrame | None = None) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None

    tool = str(value.get("tool") or TOOL_PYTHON).strip().upper()
    if tool not in _SUPPORTED_TOOLS:
        return None

    task = str(value.get("task") or "").strip()
    if not task:
        return None

    normalized_step = {
        "step": max(1, int(value.get("step") or step_number)),
        "tool": tool,
        "task": task,
        "query": str(value.get("query") or task).strip() or task,
        "depends_on": _normalize_dependencies(value.get("depends_on")),
        "uses_context": bool(value.get("uses_context")),
        "cost_estimate": estimate_cost(value, df=df),
    }

    if value.get("sql_plan") is not None:
        normalized_step["sql_plan"] = str(value.get("sql_plan") or "").strip() or None
    python_steps = [str(item).strip() for item in list(value.get("python_steps") or []) if str(item).strip()]
    if python_steps:
        normalized_step["python_steps"] = python_steps
    if isinstance(value.get("excel_logic"), dict) and value.get("excel_logic"):
        normalized_step["excel_logic"] = dict(value.get("excel_logic") or {})
    if value.get("fallback_reason") is not None:
        normalized_step["fallback_reason"] = str(value.get("fallback_reason") or "").strip() or None
    return normalized_step


def _signature(step: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(step.get("tool") or ""),
        str(step.get("task") or ""),
        str(step.get("query") or ""),
        tuple(step.get("depends_on") or []),
    )


def _can_use_excel(step: dict[str, Any]) -> bool:
    task_text = _normalize_text(step.get("task") or step.get("query"))
    if any(token in task_text for token in _SQL_HINTS):
        return False
    if "current context" in task_text:
        return False
    return True


def _prefers_sql(step: dict[str, Any]) -> bool:
    task_text = _normalize_text(step.get("task") or step.get("query"))
    return any(token in task_text for token in _SQL_HINTS)


def _prefer_faster_tool(step: dict[str, Any]) -> dict[str, Any]:
    updated_step = dict(step)
    tool = str(updated_step.get("tool") or TOOL_PYTHON).strip().upper()

    if tool == TOOL_SQL and _can_use_excel(updated_step):
        sql_time = get_average_tool_time(TOOL_SQL)
        excel_time = get_average_tool_time(TOOL_EXCEL)
        if excel_time < sql_time:
            updated_step["tool"] = TOOL_EXCEL
            updated_step["task"] = "Build an Excel-style summary or pivot for the request."
    elif tool == TOOL_EXCEL and _prefers_sql(updated_step):
        sql_time = get_average_tool_time(TOOL_SQL)
        excel_time = get_average_tool_time(TOOL_EXCEL)
        if sql_time <= (excel_time * 1.15):
            updated_step["tool"] = TOOL_SQL
            updated_step["task"] = "Use SQL-style reasoning to answer the request."

    updated_step["cost_estimate"] = estimate_cost(updated_step)
    return updated_step


def _remove_duplicates(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen_signatures: set[tuple[Any, ...]] = set()
    deduped_steps: list[dict[str, Any]] = []
    for step in steps:
        signature = _signature(step)
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        deduped_steps.append(dict(step))
    return deduped_steps


def _merge_adjacent_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged_steps: list[dict[str, Any]] = []
    for step in steps:
        if (
            merged_steps
            and merged_steps[-1]["tool"] == step["tool"]
            and merged_steps[-1].get("depends_on") == step.get("depends_on")
        ):
            if step["task"] not in merged_steps[-1]["task"]:
                merged_steps[-1]["task"] = f"{merged_steps[-1]['task']} Then {step['task']}".strip()
            if step.get("query") and step["query"] not in merged_steps[-1].get("query", ""):
                merged_steps[-1]["query"] = f"{merged_steps[-1]['query']} | {step['query']}"
            merged_steps[-1]["cost_estimate"] = estimate_cost(merged_steps[-1])
            continue
        merged_steps.append(dict(step))
    return merged_steps


def _renumber_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    step_mapping = {
        int(step.get("step") or index): index
        for index, step in enumerate(steps, start=1)
    }
    renumbered_steps: list[dict[str, Any]] = []
    for index, step in enumerate(steps, start=1):
        updated_step = dict(step)
        updated_step["step"] = index
        updated_step["depends_on"] = [
            step_mapping[dependency]
            for dependency in list(updated_step.get("depends_on") or [])
            if dependency in step_mapping and step_mapping[dependency] < index
        ]
        renumbered_steps.append(updated_step)
    return renumbered_steps


def optimize_plan(
    plan: list[dict[str, Any]] | None,
    *,
    df: pd.DataFrame | None = None,
    preflight: dict[str, Any] | None = None,
    tool_performance: dict[str, dict[str, float | int]] | None = None,
) -> list[dict[str, Any]]:
    del tool_performance
    normalized_steps = [
        _normalize_step(step, step_number=index, df=df)
        for index, step in enumerate(list(plan or []), start=1)
    ]
    normalized_steps = [step for step in normalized_steps if step]

    blockers = list((preflight or {}).get("blocking_errors") or [])
    if blockers and isinstance(df, pd.DataFrame) and not df.empty and len(df.columns) > 0:
        fallback_step = {
            "step": 1,
            "tool": TOOL_EXCEL,
            "task": "Fallback to an Excel-style analyst summary because the dataset is too weak for higher-cost automation.",
            "query": "Fallback to Excel-style summary",
            "depends_on": [],
            "uses_context": False,
            "fallback_reason": str(blockers[0]),
            "cost_estimate": estimate_cost({"tool": TOOL_EXCEL}, df=df),
        }
        return [fallback_step]

    if isinstance(df, pd.DataFrame) and not _has_chartable_fields(df):
        normalized_steps = [step for step in normalized_steps if step.get("tool") != TOOL_BI]

    normalized_steps = _remove_duplicates(normalized_steps)
    normalized_steps = [_prefer_faster_tool(step) for step in normalized_steps]
    normalized_steps = _merge_adjacent_steps(normalized_steps)
    normalized_steps = _renumber_steps(normalized_steps)

    if not normalized_steps:
        normalized_steps = [
            {
                "step": 1,
                "tool": TOOL_PYTHON,
                "task": "Run Python analysis for the request.",
                "query": "Run Python analysis for the request.",
                "depends_on": [],
                "uses_context": False,
                "cost_estimate": estimate_cost({"tool": TOOL_PYTHON}),
            }
        ]

    for step in normalized_steps:
        step["cost_estimate"] = estimate_cost(step, df=df)
    return normalized_steps


def find_parallel_steps(plan: list[dict[str, Any]] | None) -> list[list[int]]:
    normalized_steps = [
        _normalize_step(step, step_number=index)
        for index, step in enumerate(list(plan or []), start=1)
    ]
    normalized_steps = [step for step in normalized_steps if step]
    if not normalized_steps:
        return []

    dependents: dict[int, set[int]] = {int(step["step"]): set() for step in normalized_steps}
    for step in normalized_steps:
        for dependency in list(step.get("depends_on") or []):
            dependents.setdefault(int(dependency), set()).add(int(step["step"]))

    groups: dict[tuple[int, ...], list[int]] = {}
    for step in normalized_steps:
        step_id = int(step["step"])
        dependency_key = tuple(step.get("depends_on") or [])
        task_text = _normalize_text(step.get("task") or step.get("query"))
        if step.get("uses_context"):
            continue
        if dependents.get(step_id):
            continue
        if str(step.get("tool") or "").strip().upper() not in {TOOL_SQL, TOOL_EXCEL, TOOL_BI}:
            continue
        if "current context" in task_text:
            continue
        groups.setdefault(dependency_key, []).append(step_id)

    return [
        sorted(step_ids)
        for step_ids in groups.values()
        if len(step_ids) > 1
    ]
