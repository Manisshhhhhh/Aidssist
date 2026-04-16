from __future__ import annotations

from typing import Any

import pandas as pd

from backend.dashboard_helpers import classify_columns, infer_datetime_columns
from backend.question_engine import build_question_payload
from backend.time_filter_service import apply_time_filter, build_time_filter_options


def _coerce_result_frame(result: Any) -> pd.DataFrame | None:
    if isinstance(result, pd.DataFrame):
        return result.copy()
    if isinstance(result, dict):
        for key in ("result", "table"):
            candidate = result.get(key)
            if isinstance(candidate, pd.DataFrame):
                return candidate.copy()
    return None


def _first_numeric_column(frame: pd.DataFrame, *, excluded: set[str] | None = None) -> str | None:
    excluded = excluded or set()
    for column in frame.select_dtypes(include="number").columns.tolist():
        column_name = str(column)
        if column_name not in excluded:
            return column_name
    return None


def _safe_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return frame.where(pd.notna(frame), None).to_dict(orient="records")


def _build_bar_chart(frame: pd.DataFrame, group_column: str, metric_column: str) -> dict[str, Any] | None:
    if group_column not in frame.columns or metric_column not in frame.columns:
        return None

    chart_frame = (
        frame[[group_column, metric_column]]
        .assign(**{metric_column: pd.to_numeric(frame[metric_column], errors="coerce")})
        .dropna(subset=[group_column, metric_column])
        .groupby(group_column, dropna=False)[metric_column]
        .sum()
        .reset_index()
        .sort_values(metric_column, ascending=False)
        .head(12)
    )
    if chart_frame.empty:
        return None

    return {
        "type": "bar",
        "purpose": "comparison",
        "x": group_column,
        "y": metric_column,
        "title": f"{metric_column} by {group_column}",
        "rows": _safe_rows(chart_frame),
        "layout": {"x": 8, "y": 0, "w": 4, "h": 2},
        "drilldown": {"enabled": True, "field": group_column},
    }


def _build_line_chart(frame: pd.DataFrame, datetime_column: str, metric_column: str) -> dict[str, Any] | None:
    if datetime_column not in frame.columns or metric_column not in frame.columns:
        return None

    datetime_series = pd.to_datetime(frame[datetime_column], errors="coerce", format="mixed")
    metric_series = pd.to_numeric(frame[metric_column], errors="coerce")
    chart_frame = pd.DataFrame(
        {
            datetime_column: datetime_series,
            metric_column: metric_series,
        }
    ).dropna()
    if chart_frame.empty:
        return None

    chart_frame = (
        chart_frame.assign(**{datetime_column: chart_frame[datetime_column].dt.floor("D")})
        .groupby(datetime_column, dropna=False)[metric_column]
        .sum()
        .reset_index()
        .sort_values(datetime_column)
    )
    if len(chart_frame) < 2:
        return None

    chart_frame[datetime_column] = chart_frame[datetime_column].astype(str)
    return {
        "type": "line",
        "purpose": "trend",
        "x": datetime_column,
        "y": metric_column,
        "title": f"{metric_column} over time",
        "time_column": datetime_column,
        "rows": _safe_rows(chart_frame),
        "layout": {"x": 0, "y": 0, "w": 8, "h": 4},
        "drilldown": {"enabled": True, "field": datetime_column},
    }


def _build_pie_chart(frame: pd.DataFrame, group_column: str, metric_column: str) -> dict[str, Any] | None:
    if group_column not in frame.columns or metric_column not in frame.columns:
        return None

    chart_frame = (
        frame[[group_column, metric_column]]
        .assign(**{metric_column: pd.to_numeric(frame[metric_column], errors="coerce")})
        .dropna(subset=[group_column, metric_column])
        .groupby(group_column, dropna=False)[metric_column]
        .sum()
        .reset_index()
        .sort_values(metric_column, ascending=False)
        .head(6)
    )
    if chart_frame.empty:
        return None

    total = float(chart_frame[metric_column].sum()) or 1.0
    chart_frame["share"] = chart_frame[metric_column].astype(float) / total
    return {
        "type": "pie",
        "purpose": "distribution",
        "x": group_column,
        "y": metric_column,
        "title": f"{group_column} distribution",
        "rows": _safe_rows(chart_frame),
        "layout": {"x": 8, "y": 2, "w": 4, "h": 2},
        "drilldown": {"enabled": True, "field": group_column},
    }


def _build_kpis(frame: pd.DataFrame, metric_column: str, datetime_column: str | None = None) -> list[dict[str, Any]]:
    numeric_series = pd.to_numeric(frame[metric_column], errors="coerce").dropna()
    if numeric_series.empty:
        return []

    kpis: list[dict[str, Any]] = [
        {"metric": f"total_{metric_column}", "value": float(numeric_series.sum())},
        {"metric": f"average_{metric_column}", "value": float(numeric_series.mean())},
        {"metric": "record_count", "value": int(len(frame))},
    ]
    if datetime_column and datetime_column in frame.columns:
        time_series = pd.DataFrame(
            {
                datetime_column: pd.to_datetime(frame[datetime_column], errors="coerce", format="mixed"),
                metric_column: pd.to_numeric(frame[metric_column], errors="coerce"),
            }
        ).dropna()
        time_series = time_series.sort_values(datetime_column)
        if len(time_series) >= 2:
            first_value = float(time_series.iloc[0][metric_column])
            last_value = float(time_series.iloc[-1][metric_column])
            if abs(first_value) > 1e-9:
                kpis.append(
                    {
                        "metric": "growth_rate",
                        "value": float((last_value - first_value) / abs(first_value)),
                    }
                )
    return kpis


def build_dashboard_output(
    query: str,
    df: pd.DataFrame,
    *,
    result: Any = None,
    plan: dict[str, Any],
    preflight: dict[str, Any],
) -> dict[str, Any]:
    result_frame = _coerce_result_frame(result)
    base_frame = result_frame if result_frame is not None else df.copy()
    datetime_columns = infer_datetime_columns(base_frame)
    numeric_columns, categorical_columns, datetime_column_names = classify_columns(base_frame, datetime_columns)
    warnings: list[str] = []

    metric_column = str(plan.get("metric_column") or "") or (numeric_columns[0] if numeric_columns else "")
    if metric_column and metric_column not in base_frame.columns:
        metric_column = numeric_columns[0] if numeric_columns else ""

    secondary_metric = _first_numeric_column(base_frame, excluded={metric_column} if metric_column else set())
    group_column = str(plan.get("group_column") or "")
    if group_column and group_column not in base_frame.columns:
        group_column = ""
    if not group_column and categorical_columns:
        group_column = categorical_columns[0]

    datetime_column = str(plan.get("datetime_column") or "")
    if datetime_column and datetime_column not in base_frame.columns:
        datetime_column = ""
    if not datetime_column and datetime_column_names:
        datetime_column = datetime_column_names[0]

    filtered_frame = base_frame.copy()
    applied_time_filter = str(plan.get("time_filter") or preflight.get("time_filter") or "").strip().lower() or None
    custom_time_range = dict(plan.get("custom_time_range") or preflight.get("custom_time_range") or {})
    if applied_time_filter and datetime_column:
        try:
            filtered_frame = apply_time_filter(
                filtered_frame,
                applied_time_filter,
                time_column=datetime_column,
                custom_range=custom_time_range,
            )
        except ValueError as error:
            warnings.append(str(error))

    charts: list[dict[str, Any]] = []
    line_metric = secondary_metric or metric_column
    if datetime_column and line_metric:
        line_chart = _build_line_chart(filtered_frame, datetime_column, line_metric)
        if line_chart is not None:
            charts.append(line_chart)

    if group_column and metric_column:
        bar_chart = _build_bar_chart(filtered_frame, group_column, metric_column)
        if bar_chart is not None:
            charts.append(bar_chart)
        pie_chart = _build_pie_chart(filtered_frame, group_column, metric_column)
        if pie_chart is not None:
            charts.append(pie_chart)

    if not charts:
        warnings.append("Dashboard generation could not find a reliable chartable combination in the available dataset.")

    kpis = _build_kpis(filtered_frame, metric_column, datetime_column=datetime_column or None) if metric_column else []
    if not kpis and metric_column:
        warnings.append("Dashboard generation could not compute KPI values from the selected metric.")

    question_payload = build_question_payload(
        base_frame,
        source_fingerprint=str((plan or {}).get("source_fingerprint") or ""),
        recent_queries=[query] if str(query or "").strip() else None,
    )
    visualization_type = charts[0]["type"] if charts else None
    return {
        "dashboard": {
            "charts": charts,
            "filters": build_time_filter_options(),
            "kpis": kpis,
            "layout": {"type": "grid", "columns": 12, "row_height": 120, "mode": "tableau_like"},
            "drilldown_ready": True,
            "time_column": datetime_column or None,
            "applied_time_filter": applied_time_filter,
            "active_filter": applied_time_filter,
            "visualization_type": visualization_type,
        },
        "active_filter": applied_time_filter,
        "visualization_type": visualization_type,
        "warnings": warnings,
        "context": question_payload["context"],
        "suggestions": question_payload["suggestions"],
        "recommended_next_step": question_payload["recommended_next_step"],
        "suggested_questions": question_payload["suggested_questions"],
        "domain": question_payload["domain"],
    }
