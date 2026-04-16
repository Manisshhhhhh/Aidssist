from __future__ import annotations

from typing import Any

import pandas as pd


def _normalize_text(value: str | None) -> str:
    return str(value or "").strip().lower()


def _resolve_aggregation(query: str) -> str:
    normalized = _normalize_text(query)
    if any(token in normalized for token in ("average", "avg", "mean")):
        return "mean"
    if "median" in normalized:
        return "median"
    if "min" in normalized or "lowest" in normalized:
        return "min"
    if "max" in normalized or "highest" in normalized:
        return "max"
    return "sum"


def _first_numeric_column(frame: pd.DataFrame) -> str | None:
    numeric_columns = [str(column) for column in frame.select_dtypes(include="number").columns.tolist()]
    return numeric_columns[0] if numeric_columns else None


def _first_dimension_column(frame: pd.DataFrame, *, excluded: set[str] | None = None) -> str | None:
    excluded = excluded or set()
    for column in frame.columns:
        column_name = str(column)
        if column_name in excluded:
            continue
        if not pd.api.types.is_numeric_dtype(frame[column]):
            return column_name
    for column in frame.columns:
        column_name = str(column)
        if column_name not in excluded:
            return column_name
    return None


def _overall_aggregations(frame: pd.DataFrame, metric_column: str | None) -> dict[str, Any]:
    if not metric_column or metric_column not in frame.columns:
        return {
            "row_count": int(len(frame)),
            "column_count": int(len(frame.columns)),
            "missing_cell_count": int(frame.isna().sum().sum()),
        }

    numeric_series = pd.to_numeric(frame[metric_column], errors="coerce").dropna()
    if numeric_series.empty:
        return {
            "row_count": int(len(frame)),
            "column_count": int(len(frame.columns)),
            "usable_metric_rows": 0,
        }

    return {
        f"{metric_column}_sum": float(numeric_series.sum()),
        f"{metric_column}_mean": float(numeric_series.mean()),
        f"{metric_column}_min": float(numeric_series.min()),
        f"{metric_column}_max": float(numeric_series.max()),
        "usable_metric_rows": int(len(numeric_series)),
    }


def run_excel_analysis(
    query: str,
    df: pd.DataFrame,
    *,
    plan: dict[str, Any],
    preflight: dict[str, Any],
) -> dict[str, Any]:
    warnings = list((preflight or {}).get("blocking_errors") or [])
    aggregation_name = _resolve_aggregation(query)
    metric_column = str(plan.get("metric_column") or "") or _first_numeric_column(df) or ""
    group_column = str(plan.get("group_column") or "")
    if not group_column:
        group_column = _first_dimension_column(df, excluded={metric_column} if metric_column else set()) or ""

    pivot_config: dict[str, Any] = {}
    result_frame: pd.DataFrame
    if group_column and metric_column and group_column in df.columns and metric_column in df.columns:
        pivot = pd.pivot_table(
            df,
            index=[group_column],
            values=[metric_column],
            aggfunc=aggregation_name,
            dropna=False,
        ).reset_index()
        value_column = str(pivot.columns[-1])
        result_frame = pivot.sort_values(value_column, ascending=False).reset_index(drop=True)
        pivot_config = {
            "index": [group_column],
            "values": [metric_column],
            "aggfunc": aggregation_name,
            "rows": result_frame.to_dict(orient="records"),
        }
    else:
        if not metric_column:
            warnings.append("Excel summary fell back to dataset-level totals because no numeric metric column was detected.")
        elif metric_column not in df.columns:
            warnings.append(f"Excel summary could not find `{metric_column}`, so it used dataset-level totals instead.")
        if not group_column and metric_column:
            warnings.append("Excel summary could not find a grouping column, so it returned overall metric aggregates.")
        result_frame = pd.DataFrame([_overall_aggregations(df, metric_column if metric_column in df.columns else None)])

    aggregations = _overall_aggregations(df, metric_column if metric_column in df.columns else None)
    excel_analysis = {
        "pivot_table": pivot_config,
        "aggregations": aggregations,
        "summary": {
            "aggregation": aggregation_name,
            "group_column": group_column or None,
            "metric_column": metric_column or None,
            "warning_count": len(warnings),
        },
    }
    return {
        "result": result_frame,
        "excel_analysis": excel_analysis,
        "warnings": list(dict.fromkeys(str(item) for item in warnings if str(item).strip())),
    }
