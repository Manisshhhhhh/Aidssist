from __future__ import annotations

import re
from typing import Any

import pandas as pd


def _normalize_text(value: str | None) -> str:
    return str(value or "").strip().lower()


def _normalize_column_token(value: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", "", _normalize_text(value))


def _find_matching_column(term: str | None, frame: pd.DataFrame) -> str | None:
    normalized_term = _normalize_column_token(term)
    if not normalized_term:
        return None

    exact_match = None
    partial_match = None
    for column in frame.columns:
        column_name = str(column)
        normalized_column = _normalize_column_token(column_name)
        if normalized_column == normalized_term:
            exact_match = column_name
            break
        if normalized_term in normalized_column and partial_match is None:
            partial_match = column_name
    return exact_match or partial_match


def query_requires_join(query: str) -> bool:
    normalized = _normalize_text(query)
    return any(
        token in normalized
        for token in ("join ", " merge ", "combine ", "match ", "merge ", "joined ")
    )


def _resolve_limit(query: str) -> int | None:
    match = re.search(r"\btop\s+(\d+)\b", _normalize_text(query))
    if match:
        return max(1, int(match.group(1)))
    match = re.search(r"\blimit\s+(\d+)\b", _normalize_text(query))
    if match:
        return max(1, int(match.group(1)))
    return None


def _resolve_aggregation(query: str) -> tuple[str, str]:
    normalized = _normalize_text(query)
    if any(token in normalized for token in ("sum", "total")):
        return "sum", "SUM"
    if any(token in normalized for token in ("average", "avg", "mean")):
        return "mean", "AVG"
    if "median" in normalized:
        return "median", "MEDIAN"
    if "min" in normalized or "lowest" in normalized:
        return "min", "MIN"
    if "max" in normalized or "highest" in normalized:
        return "max", "MAX"
    return "sum", "SUM"


def _resolve_sort_direction(query: str) -> bool:
    normalized = _normalize_text(query)
    return not any(token in normalized for token in ("bottom", "lowest", "asc", "ascending"))


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


def _extract_simple_filter(query: str, frame: pd.DataFrame) -> tuple[str | None, str | None]:
    normalized = _normalize_text(query)
    match = re.search(r"\bwhere\s+([a-z0-9_ \-]+?)\s*(?:=|is)\s*['\"]?([a-z0-9_ \-]+)['\"]?\b", normalized)
    if not match:
        return None, None

    column_name = _find_matching_column(match.group(1), frame)
    if column_name is None:
        return None, None

    return column_name, match.group(2).strip()


def _plan_single_table_query(query: str, frame: pd.DataFrame, plan: dict[str, Any]) -> tuple[pd.DataFrame, str]:
    metric_column = str(plan.get("metric_column") or "") or _first_numeric_column(frame) or ""
    group_column = str(plan.get("group_column") or "")
    if not group_column:
        group_column = _first_dimension_column(frame, excluded={metric_column} if metric_column else set()) or ""

    filter_column, filter_value = _extract_simple_filter(query, frame)
    working = frame.copy()
    where_clause = ""
    if filter_column and filter_value:
        filter_mask = working[filter_column].astype(str).str.strip().str.lower() == filter_value.lower()
        working = working.loc[filter_mask].copy()
        where_clause = f" WHERE {filter_column} = '{filter_value}'"

    aggregation_name, sql_aggregation = _resolve_aggregation(query)
    sort_descending = _resolve_sort_direction(query)
    limit_value = _resolve_limit(query)

    if group_column and group_column in working.columns:
        if metric_column and metric_column in working.columns and pd.api.types.is_numeric_dtype(working[metric_column]):
            aggregate_series = working.groupby(group_column, dropna=False)[metric_column]
            if aggregation_name == "mean":
                result = aggregate_series.mean()
            elif aggregation_name == "median":
                result = aggregate_series.median()
            elif aggregation_name == "min":
                result = aggregate_series.min()
            elif aggregation_name == "max":
                result = aggregate_series.max()
            else:
                result = aggregate_series.sum()

            metric_alias = f"{aggregation_name}_{metric_column}"
            result_frame = result.sort_values(ascending=not sort_descending).reset_index(name=metric_alias)
            if limit_value:
                result_frame = result_frame.head(limit_value).reset_index(drop=True)
            sql_plan = (
                f"SELECT {group_column}, {sql_aggregation}({metric_column}) AS {metric_alias} "
                f"FROM df{where_clause} GROUP BY {group_column} "
                f"ORDER BY {metric_alias} {'DESC' if sort_descending else 'ASC'}"
            )
            if limit_value:
                sql_plan += f" LIMIT {limit_value}"
            return result_frame, sql_plan

        count_alias = "row_count"
        result_frame = (
            working.groupby(group_column, dropna=False)
            .size()
            .sort_values(ascending=not sort_descending)
            .reset_index(name=count_alias)
        )
        if limit_value:
            result_frame = result_frame.head(limit_value).reset_index(drop=True)
        sql_plan = (
            f"SELECT {group_column}, COUNT(*) AS {count_alias} FROM df{where_clause} "
            f"GROUP BY {group_column} ORDER BY {count_alias} {'DESC' if sort_descending else 'ASC'}"
        )
        if limit_value:
            sql_plan += f" LIMIT {limit_value}"
        return result_frame, sql_plan

    preview_limit = limit_value or 10
    result_frame = working.head(preview_limit).reset_index(drop=True)
    sql_plan = f"SELECT * FROM df{where_clause} LIMIT {preview_limit}"
    return result_frame, sql_plan


def _run_join_query(query: str, tables: dict[str, pd.DataFrame], plan: dict[str, Any]) -> dict[str, Any]:
    warnings: list[str] = []
    table_items = [(str(name), frame.copy()) for name, frame in tables.items() if isinstance(frame, pd.DataFrame)]
    if len(table_items) < 2:
        return {
            "result": None,
            "sql_plan": "SELECT * FROM df",
            "warnings": ["Join-style query requested multiple tables, but only one table is available."],
            "unsupported": True,
        }

    (left_name, left_frame), (right_name, right_frame) = table_items[:2]
    join_match = re.search(r"\bon\s+([a-z0-9_]+)\b", _normalize_text(query))
    join_key = _find_matching_column(join_match.group(1), left_frame) if join_match else None
    if join_key is None:
        common_columns = [str(column) for column in left_frame.columns if str(column) in {str(item) for item in right_frame.columns}]
        join_key = common_columns[0] if common_columns else None

    if join_key is None:
        return {
            "result": None,
            "sql_plan": f"SELECT * FROM {left_name} JOIN {right_name}",
            "warnings": ["Join-style query could not find a common join key across the available tables."],
            "unsupported": True,
        }

    merged = left_frame.merge(right_frame, on=join_key, how="inner", suffixes=("_left", "_right"))
    result_frame, base_sql_plan = _plan_single_table_query(query, merged, plan)
    sql_plan = base_sql_plan.replace("FROM df", f"FROM {left_name} JOIN {right_name} ON {left_name}.{join_key} = {right_name}.{join_key}")
    warnings.append(f"Join simulation used `{join_key}` as the shared key between `{left_name}` and `{right_name}`.")
    return {
        "result": result_frame,
        "sql_plan": sql_plan,
        "warnings": warnings,
        "unsupported": False,
    }


def run_sql_analysis(
    query: str,
    df: pd.DataFrame,
    *,
    tables: dict[str, pd.DataFrame] | None = None,
    plan: dict[str, Any],
    preflight: dict[str, Any],
) -> dict[str, Any]:
    del preflight
    resolved_tables = dict(tables or {"df": df})
    if query_requires_join(query):
        return _run_join_query(query, resolved_tables, plan)

    result_frame, sql_plan = _plan_single_table_query(query, df.copy(), plan)
    warnings: list[str] = []
    if result_frame.empty:
        warnings.append("SQL simulation returned no rows after applying the available filters and aggregations.")
    return {
        "result": result_frame,
        "sql_plan": sql_plan,
        "warnings": warnings,
        "unsupported": False,
    }
