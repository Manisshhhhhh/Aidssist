from __future__ import annotations

from typing import Any

import pandas as pd

from backend.services.agent_engine import run_analysis_agent


MAX_AUTO_TASKS = 5


def _normalize_name(value: str) -> str:
    return str(value or "").strip().lower().replace("_", " ")


def _is_date_like_name(column_name: str) -> bool:
    normalized = _normalize_name(column_name)
    return any(keyword in normalized for keyword in ("date", "time", "day", "month", "year", "week"))


def has_date_column(df: pd.DataFrame) -> bool:
    for column in df.columns:
        series = df[column]
        if pd.api.types.is_datetime64_any_dtype(series):
            return True
        if _is_date_like_name(str(column)):
            return True
        if pd.api.types.is_object_dtype(series):
            sample = series.dropna().head(10)
            if sample.empty:
                continue
            parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
            if parsed.notna().sum() >= max(1, len(sample) // 2):
                return True
    return False


def _has_numeric_columns(df: pd.DataFrame) -> bool:
    return not df.select_dtypes(include="number").empty


def _pick_primary_category_column(df: pd.DataFrame) -> str | None:
    best_column = None
    best_score = -1.0
    row_count = max(len(df.index), 1)

    for column in df.columns:
        series = df[column]
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_datetime64_any_dtype(series):
            continue
        if _is_date_like_name(str(column)):
            continue

        non_null = int(series.notna().sum())
        unique_count = int(series.nunique(dropna=True))
        if non_null == 0 or unique_count <= 1:
            continue

        uniqueness_ratio = unique_count / max(non_null, 1)
        if uniqueness_ratio >= 0.9 and unique_count > 5:
            continue

        score = float(non_null - unique_count)
        preferred_words = ("country", "region", "category", "segment", "product", "diagnosis", "status", "type")
        if any(word in _normalize_name(str(column)) for word in preferred_words):
            score += 10
        if unique_count <= min(20, row_count):
            score += 3

        if score > best_score:
            best_score = score
            best_column = str(column)

    return best_column


def _pick_primary_metric_column(df: pd.DataFrame) -> str | None:
    preferred_words = ("sales", "revenue", "profit", "amount", "value", "confirmed", "cases", "deaths", "cost")
    numeric_columns = [str(column) for column in df.select_dtypes(include="number").columns]
    if not numeric_columns:
        return None

    for column in numeric_columns:
        normalized = _normalize_name(column)
        if any(word in normalized for word in preferred_words):
            return column

    for column in numeric_columns:
        normalized = _normalize_name(column)
        if "id" not in normalized:
            return column

    return numeric_columns[0]


def _dedupe_tasks(tasks: list[str]) -> list[str]:
    unique_tasks: list[str] = []
    seen: set[str] = set()
    for task in tasks:
        normalized_task = str(task).strip()
        if not normalized_task or normalized_task in seen:
            continue
        seen.add(normalized_task)
        unique_tasks.append(normalized_task)
        if len(unique_tasks) >= MAX_AUTO_TASKS:
            break
    return unique_tasks


def plan_analysis(df: pd.DataFrame) -> list[str]:
    plan: list[str] = []
    plan.append("Show dataset summary")
    plan.append("Check missing values")

    category_column = _pick_primary_category_column(df)
    if category_column:
        plan.append(f"Find top categories in {category_column}")

    if _has_numeric_columns(df):
        plan.append("Compute averages")

    if has_date_column(df):
        metric_column = _pick_primary_metric_column(df)
        if metric_column:
            plan.append(f"Analyze {metric_column} trend over time")
        else:
            plan.append("Analyze trend over time")

    return _dedupe_tasks(plan)


def _clean_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, pd.Timedelta):
        return str(value)
    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def _serialize_result(result: Any) -> Any:
    if isinstance(result, pd.DataFrame):
        preview = result.head(20).copy()
        preview = preview.where(pd.notna(preview), None)
        rows = [
            {str(key): _clean_value(value) for key, value in row.items()}
            for row in preview.to_dict(orient="records")
        ]
        return {
            "type": "table",
            "columns": [str(column) for column in preview.columns.tolist()],
            "rows": rows,
        }

    if isinstance(result, pd.Series):
        preview = result.head(10)
        return {
            "type": "series",
            "values": [
                {"label": str(index), "value": _clean_value(value)}
                for index, value in preview.items()
            ],
        }

    return _clean_value(result)


def run_auto_analysis(df: pd.DataFrame) -> list[dict[str, Any]]:
    plan = plan_analysis(df)
    results: list[dict[str, Any]] = []

    for task in plan[:MAX_AUTO_TASKS]:
        response = run_analysis_agent(df, task, prefer_rule_based=True)
        results.append(
            {
                "task": task,
                "result": _serialize_result(response.get("result")),
                "insight": str(response.get("insight") or response.get("error") or ""),
            }
        )

    return results


def summarize_insights(results: list[dict[str, Any]]) -> list[str]:
    summary: list[str] = []
    seen: set[str] = set()

    for result in results:
        insight = str(result.get("insight") or "").strip()
        if not insight or insight in seen:
            continue
        seen.add(insight)
        summary.append(insight)
        if len(summary) >= MAX_AUTO_TASKS:
            break

    return summary[:MAX_AUTO_TASKS]


def build_auto_analysis_payload(df: pd.DataFrame) -> dict[str, Any]:
    results = run_auto_analysis(df)
    return {
        "auto_analysis": {
            "tasks": [item["task"] for item in results],
            "results": results,
            "summary": summarize_insights(results),
        }
    }


__all__ = [
    "build_auto_analysis_payload",
    "has_date_column",
    "plan_analysis",
    "run_auto_analysis",
    "summarize_insights",
]
