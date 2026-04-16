from __future__ import annotations

import re
from typing import Any

import pandas as pd

from backend.dashboard_helpers import classify_columns, infer_datetime_columns
from backend.services.optimizer import optimize_plan as optimize_execution_plan
from backend.services.tool_selector import (
    ANALYSIS_MODE_AD_HOC,
    ANALYSIS_MODE_DASHBOARD,
    ANALYSIS_MODE_PREDICTION,
    TOOL_BI,
    TOOL_EXCEL,
    TOOL_PYTHON,
    TOOL_SQL,
    select_tool,
)


_CLEANING_TOKENS = (
    "clean",
    "cleanup",
    "dedupe",
    "duplicate",
    "missing",
    "null",
    "fill",
    "normalize",
    "standardize",
)
_PREDICTION_TOKENS = (
    "predict",
    "prediction",
    "forecast",
    "estimate",
    "project",
)
_PYTHON_TOKENS = (
    "trend",
    "trends",
    "correlation",
    "regression",
    "analyze",
    "analyse",
    "explain",
    "why",
    "driver",
    "scenario",
)
_DASHBOARD_TOKENS = (
    "dashboard",
    "visualize",
    "visualise",
    "visualization",
    "chart",
    "plot",
    "graph",
    "kpi",
)
_SQL_TOKENS = (
    "top",
    "bottom",
    "rank",
    "ranking",
    "filter",
    "where",
    "join",
    "merge",
    "customer",
    "customers",
)
_MULTI_STEP_MARKERS = (
    "+",
    " then ",
    " after ",
    " followed by ",
)


def _normalize_text(value: str | None) -> str:
    return str(value or "").strip().lower()


def _contains_any(text: str, tokens: tuple[str, ...]) -> bool:
    return any(token in text for token in tokens)


def _has_chartable_fields(df: pd.DataFrame) -> bool:
    datetime_columns = infer_datetime_columns(df)
    numeric_columns, categorical_columns, datetime_column_names = classify_columns(df, datetime_columns)
    return bool(numeric_columns) and bool(categorical_columns or datetime_column_names)


def _has_non_critical_blocker(preflight: dict[str, Any] | None, df: pd.DataFrame) -> bool:
    blockers = list((preflight or {}).get("blocking_errors") or [])
    return bool(blockers) and isinstance(df, pd.DataFrame) and not df.empty and len(df.columns) > 0


def _step(
    step_number: int,
    tool: str,
    task: str,
    *,
    query: str | None = None,
    depends_on: list[int] | None = None,
    uses_context: bool = False,
) -> dict[str, Any]:
    payload = {
        "step": int(step_number),
        "tool": str(tool).strip().upper() or TOOL_PYTHON,
        "task": str(task).strip() or "Run the requested analysis step.",
        "query": str(query).strip() if str(query or "").strip() else str(task).strip(),
        "depends_on": [int(item) for item in list(depends_on or []) if int(item) > 0],
        "uses_context": bool(uses_context),
    }
    return payload


def _is_multi_step_request(query: str, *, needs_cleaning: bool, needs_sql: bool, needs_python: bool, needs_dashboard: bool) -> bool:
    normalized = _normalize_text(query)
    action_groups = sum(
        1
        for flag in (needs_cleaning, needs_sql, needs_python, needs_dashboard)
        if flag
    )
    if action_groups >= 3:
        return True
    if action_groups >= 2 and (
        _contains_any(normalized, _MULTI_STEP_MARKERS)
        or re.search(r"\b(and|with|plus)\b", normalized)
    ):
        return True
    return False


def determine_primary_tool(execution_plan: list[dict[str, Any]] | None, *, default: str | None = None) -> str:
    plan = list(execution_plan or [])
    tools = [str(step.get("tool") or "").strip().upper() for step in plan]
    if TOOL_PYTHON in tools:
        return TOOL_PYTHON
    if TOOL_SQL in tools:
        return TOOL_SQL
    if TOOL_EXCEL in tools:
        return TOOL_EXCEL
    if TOOL_BI in tools:
        return TOOL_BI
    return str(default or TOOL_PYTHON).strip().upper() or TOOL_PYTHON


def determine_analysis_mode(
    execution_plan: list[dict[str, Any]] | None,
    *,
    intent: str | None = None,
    default: str | None = None,
) -> str:
    normalized_intent = _normalize_text(intent)
    plan = list(execution_plan or [])
    tasks = " ".join(str(step.get("task") or "") for step in plan).lower()
    tools = {str(step.get("tool") or "").strip().upper() for step in plan}
    if normalized_intent == "prediction" or _contains_any(tasks, _PREDICTION_TOKENS):
        return ANALYSIS_MODE_PREDICTION
    if TOOL_BI in tools or str(default or "").strip().lower() == ANALYSIS_MODE_DASHBOARD:
        return ANALYSIS_MODE_DASHBOARD
    return str(default or ANALYSIS_MODE_AD_HOC).strip().lower() or ANALYSIS_MODE_AD_HOC


def optimize_plan(
    plan: list[dict[str, Any]] | None,
    *,
    df: pd.DataFrame | None = None,
    preflight: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    return optimize_execution_plan(
        plan,
        df=df,
        preflight=preflight,
    )


def build_execution_plan(
    query: str,
    df: pd.DataFrame,
    *,
    plan: dict[str, Any] | None = None,
    preflight: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    plan = dict(plan or {})
    preflight = dict(preflight or {})
    selection = select_tool(query, df, plan=plan, preflight=preflight)
    normalized_query = _normalize_text(query)

    needs_cleaning = (
        str(plan.get("intent") or "").strip().lower() == "data_cleaning"
        or _contains_any(normalized_query, _CLEANING_TOKENS)
    )
    needs_prediction = (
        str(plan.get("intent") or "").strip().lower() == "prediction"
        or _contains_any(normalized_query, _PREDICTION_TOKENS)
    )
    needs_dashboard = (
        str(selection.get("analysis_mode") or "").strip().lower() == ANALYSIS_MODE_DASHBOARD
        or _contains_any(normalized_query, _DASHBOARD_TOKENS)
    )
    needs_sql = (
        _contains_any(normalized_query, _SQL_TOKENS)
        and not _has_non_critical_blocker(preflight, df)
    )
    needs_python = needs_prediction or _contains_any(normalized_query, _PYTHON_TOKENS)

    multi_step_requested = _is_multi_step_request(
        query,
        needs_cleaning=needs_cleaning,
        needs_sql=needs_sql,
        needs_python=needs_python,
        needs_dashboard=needs_dashboard,
    )

    if not multi_step_requested:
        selected_tool = str(selection.get("tool_used") or TOOL_PYTHON).strip().upper() or TOOL_PYTHON
        single_task = {
            TOOL_SQL: "Use SQL-style reasoning to answer the request.",
            TOOL_EXCEL: "Build an Excel-style summary or pivot for the request.",
            TOOL_PYTHON: "Run Python analysis for the request.",
            TOOL_BI: "Build a BI dashboard specification for the request.",
        }.get(selected_tool, "Run Python analysis for the request.")
        return optimize_plan(
            [_step(1, selected_tool, single_task, query=query, depends_on=[], uses_context=False)],
            df=df,
            preflight=preflight,
        )

    steps: list[dict[str, Any]] = []
    if needs_cleaning:
        cleaning_step = len(steps) + 1
        steps.append(
            _step(
                cleaning_step,
                TOOL_EXCEL,
                "Profile data quality issues, clean obvious anomalies, and prepare an analyst-friendly summary table.",
                query="Clean and summarize the dataset",
                depends_on=[],
                uses_context=False,
            )
        )

    if needs_sql:
        sql_task = "Filter, group, and rank the most relevant entities using SQL-style logic."
        if "customer" in normalized_query:
            sql_task = "Get the top customers using SQL-style grouping, ranking, and filtering."
        dependency = [len(steps)] if steps else []
        steps.append(
            _step(
                len(steps) + 1,
                TOOL_SQL,
                sql_task,
                query=query,
                depends_on=dependency,
                uses_context=bool(dependency),
            )
        )

    if needs_dashboard and not any(step.get("tool") == TOOL_PYTHON for step in steps) and _contains_any(normalized_query, ("trend", "trends", "time", "over time")):
        needs_python = True

    if needs_python:
        python_task = "Run Python analysis on the current context."
        if needs_prediction:
            python_task = "Run Python prediction or forecasting on the current context."
        elif "trend" in normalized_query or "trends" in normalized_query:
            python_task = "Analyze trends and derived metrics in Python before presentation."
        dependency = [len(steps)] if steps else []
        steps.append(
            _step(
                len(steps) + 1,
                TOOL_PYTHON,
                python_task,
                query=query,
                depends_on=dependency,
                uses_context=bool(dependency),
            )
        )

    if needs_dashboard and _has_chartable_fields(df):
        dependency = [len(steps)] if steps else []
        steps.append(
            _step(
                len(steps) + 1,
                TOOL_BI,
                "Build dashboard-ready charts and KPI structure from the current context.",
                query=query,
                depends_on=dependency,
                uses_context=bool(dependency),
            )
        )

    if not steps:
        selected_tool = str(selection.get("tool_used") or TOOL_PYTHON).strip().upper() or TOOL_PYTHON
        steps.append(_step(1, selected_tool, "Run the selected analysis tool for the request.", query=query, depends_on=[], uses_context=False))

    return optimize_plan(steps, df=df, preflight=preflight)
