from __future__ import annotations

import re
from typing import Any

import pandas as pd

from backend.dashboard_helpers import classify_columns, infer_datetime_columns


TOOL_SQL = "SQL"
TOOL_PYTHON = "PYTHON"
TOOL_EXCEL = "EXCEL"
TOOL_BI = "BI"

ANALYSIS_MODE_AD_HOC = "ad-hoc"
ANALYSIS_MODE_DASHBOARD = "dashboard"
ANALYSIS_MODE_PREDICTION = "prediction"

_DASHBOARD_TOKENS = (
    "dashboard",
    "visualization",
    "visualise",
    "visualize",
    "chart",
    "plot",
    "graph",
    "kpi",
    "scorecard",
)
_SQL_TOKENS = (
    "sql",
    "select",
    "where",
    "group by",
    "order by",
    "limit",
    "join",
    "top ",
    "bottom ",
    "highest",
    "lowest",
    "rank",
    "filter",
)
_EXCEL_TOKENS = (
    "pivot",
    "summary",
    "summarize",
    "breakdown",
    "show",
    "by ",
    "total",
    "sum",
    "average",
    "avg",
    "mean",
)


def _normalize_text(value: str | None) -> str:
    return str(value or "").strip().lower()


def _has_chartable_fields(df: pd.DataFrame) -> bool:
    datetime_columns = infer_datetime_columns(df)
    numeric_columns, categorical_columns, datetime_column_names = classify_columns(df, datetime_columns)
    has_dimension = bool(categorical_columns or datetime_column_names)
    return bool(numeric_columns) and has_dimension


def _is_empty_or_unusable(df: pd.DataFrame) -> bool:
    return not isinstance(df, pd.DataFrame) or df.empty or len(df.columns) == 0


def _has_non_critical_blocker(preflight: dict[str, Any] | None, df: pd.DataFrame) -> bool:
    blockers = list((preflight or {}).get("blocking_errors") or [])
    return bool(blockers) and not _is_empty_or_unusable(df)


def _looks_like_dashboard_request(query: str, plan: dict[str, Any]) -> bool:
    if plan.get("analysis_mode") == ANALYSIS_MODE_DASHBOARD:
        return True
    if str(plan.get("intent") or "").strip().lower() == "visualization":
        return True
    normalized = _normalize_text(query)
    return any(token in normalized for token in _DASHBOARD_TOKENS)


def _looks_like_sql_request(query: str, plan: dict[str, Any]) -> bool:
    normalized = _normalize_text(query)
    if any(token in normalized for token in _SQL_TOKENS):
        return True

    metric_column = str(plan.get("metric_column") or "")
    group_column = str(plan.get("group_column") or "")
    required_columns = {str(column) for column in list(plan.get("required_columns") or [])}
    if group_column and metric_column and any(token in normalized for token in ("top ", "bottom ", "highest", "lowest", "rank")):
        return True
    if any(token in normalized for token in ("customer", "customers")) and (
        "top" in normalized or "bottom" in normalized or required_columns
    ):
        return True
    return False


def _looks_like_excel_request(query: str, plan: dict[str, Any]) -> bool:
    normalized = _normalize_text(query)
    metric_column = str(plan.get("metric_column") or "")
    group_column = str(plan.get("group_column") or "")

    if plan.get("analysis_type") == "aggregation" and metric_column and group_column:
        if not _looks_like_sql_request(query, plan):
            return True

    if any(token in normalized for token in _EXCEL_TOKENS) and metric_column and group_column:
        return True

    if re.search(r"\bshow\b.+\bby\b", normalized) and metric_column and group_column:
        return True

    return False


def select_tool(query: str, df: pd.DataFrame, *, plan: dict[str, Any], preflight: dict[str, Any]) -> dict[str, str | None]:
    intent = str(plan.get("intent") or "").strip().lower()
    normalized_query = _normalize_text(query)
    requested_dashboard = _looks_like_dashboard_request(normalized_query, plan)
    weak_data_fallback = _has_non_critical_blocker(preflight, df)

    if intent == "prediction":
        if weak_data_fallback:
            return {
                "tool_used": TOOL_EXCEL,
                "analysis_mode": ANALYSIS_MODE_PREDICTION,
                "reason": "Prediction-style request fell back to analyst summary because the dataset does not support safe model execution.",
                "fallback_reason": "Prediction inputs are too weak for safe model execution, so the engine returned an Excel-style summary instead.",
            }
        return {
            "tool_used": TOOL_PYTHON,
            "analysis_mode": ANALYSIS_MODE_PREDICTION,
            "reason": "Prediction and forecasting requests should run through the Python analytics engine.",
            "fallback_reason": None,
        }

    if requested_dashboard:
        if _has_chartable_fields(df):
            return {
                "tool_used": TOOL_BI,
                "analysis_mode": ANALYSIS_MODE_DASHBOARD,
                "reason": "The request asks for dashboard or chart structure and the dataset has chartable dimensions and metrics.",
                "fallback_reason": None,
            }
        return {
            "tool_used": TOOL_EXCEL,
            "analysis_mode": ANALYSIS_MODE_DASHBOARD,
            "reason": "The request asked for dashboard output, but the dataset is not chartable enough for BI layout generation.",
            "fallback_reason": "No reliable chartable field combination was detected, so the engine returned an Excel-style summary.",
        }

    if weak_data_fallback:
        return {
            "tool_used": TOOL_EXCEL,
            "analysis_mode": ANALYSIS_MODE_AD_HOC,
            "reason": "The dataset has blocking quality issues for deeper automation, so the engine is falling back to a conservative analyst summary.",
            "fallback_reason": "Data quality is too weak for higher-risk automation, so the engine returned an Excel-style summary.",
        }

    if _looks_like_sql_request(normalized_query, plan):
        return {
            "tool_used": TOOL_SQL,
            "analysis_mode": ANALYSIS_MODE_AD_HOC,
            "reason": "The query reads like a ranking, filtering, or relational lookup that fits SQL behavior.",
            "fallback_reason": None,
        }

    if _looks_like_excel_request(normalized_query, plan):
        return {
            "tool_used": TOOL_EXCEL,
            "analysis_mode": ANALYSIS_MODE_AD_HOC,
            "reason": "The query is a quick summary or pivot-style request that fits Excel behavior.",
            "fallback_reason": None,
        }

    return {
        "tool_used": TOOL_PYTHON,
        "analysis_mode": ANALYSIS_MODE_AD_HOC,
        "reason": "The request needs the flexible Python execution path.",
        "fallback_reason": None,
    }
