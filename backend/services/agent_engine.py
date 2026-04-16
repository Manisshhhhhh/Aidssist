from __future__ import annotations

import math
import re
from typing import Any

import pandas as pd

from backend.services.llm_code_generator import (
    generate_code_with_llm,
    validate_generated_code,
)
from backend.services.memory_engine import MemoryEngine
from backend.services.suggestion_engine import generate_suggestions


BLOCKED_CODE_PATTERNS = {
    "import os": "Operating system access is blocked.",
    "import sys": "System access is blocked.",
    "open(": "File access is blocked.",
    "open (": "File access is blocked.",
    "exec(": "Nested execution is blocked.",
    "exec (": "Nested execution is blocked.",
    "eval(": "Dynamic evaluation is blocked.",
    "eval (": "Dynamic evaluation is blocked.",
    "subprocess": "Process spawning is blocked.",
    "__": "Dunder access is blocked.",
}

AGGREGATION_WORDS = {
    "sum",
    "total",
    "average",
    "averages",
    "avg",
    "mean",
    "count",
    "top",
    "group",
    "by",
    "show",
    "dataset",
    "summary",
    "compute",
    "analyze",
    "trend",
    "over",
    "time",
    "check",
    "missing",
    "values",
    "find",
    "categories",
}
analysis_memory = MemoryEngine()


def _singularize_token(token: str) -> str:
    if token.endswith("ies") and len(token) > 3:
        return token[:-3] + "y"
    if token.endswith("ses") and len(token) > 3:
        return token[:-2]
    if token.endswith("s") and not token.endswith("ss") and len(token) > 3:
        return token[:-1]
    return token


def _normalize_text(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()
    tokens = [_singularize_token(token) for token in cleaned.split() if token]
    return " ".join(tokens)


def _extract_top_n(query: str, default: int = 5) -> int:
    match = re.search(r"\btop\s+(\d+)\b", query.lower())
    if match:
        return max(int(match.group(1)), 1)
    return default


def _quoted_column(column_name: str) -> str:
    return repr(str(column_name))


def _is_time_like_column_name(column_name: str) -> bool:
    normalized_column = _normalize_text(str(column_name))
    return any(keyword in normalized_column for keyword in ("date", "time", "day", "month", "year", "week"))


def _find_matching_column(
    df_head: pd.DataFrame,
    text: str,
    *,
    numeric_only: bool | None = None,
) -> str | None:
    normalized_text = _normalize_text(text)
    query_tokens = set(normalized_text.split())
    best_column = None
    best_score = 0

    for column in df_head.columns:
        is_numeric = pd.api.types.is_numeric_dtype(df_head[column])
        if numeric_only is True and not is_numeric:
            continue
        if numeric_only is False and is_numeric:
            continue

        normalized_column = _normalize_text(str(column))
        column_tokens = set(normalized_column.split())
        score = 0

        if normalized_column and normalized_column in normalized_text:
            score += 100
        score += len(query_tokens & column_tokens) * 10

        if score > best_score:
            best_score = score
            best_column = str(column)

    return best_column


def _pick_group_column(df_head: pd.DataFrame, user_query: str) -> str | None:
    group_column = _find_matching_column(df_head, user_query, numeric_only=False)
    if group_column and not _is_time_like_column_name(group_column):
        return group_column

    for column in df_head.columns:
        if not pd.api.types.is_numeric_dtype(df_head[column]) and not _is_time_like_column_name(column):
            return str(column)

    if group_column:
        return group_column

    for column in df_head.columns:
        if not pd.api.types.is_numeric_dtype(df_head[column]):
            return str(column)
    return None


def _pick_metric_column(df_head: pd.DataFrame, user_query: str) -> str | None:
    metric_column = _find_matching_column(df_head, user_query, numeric_only=True)
    if metric_column:
        return metric_column

    preferred_keywords = ["confirmed", "case", "death", "sale", "revenue", "amount", "value"]
    for keyword in preferred_keywords:
        metric_column = _find_matching_column(df_head, keyword, numeric_only=True)
        if metric_column:
            return metric_column

    for column in df_head.columns:
        if pd.api.types.is_numeric_dtype(df_head[column]):
            return str(column)
    return None


def _pick_time_column(df_head: pd.DataFrame, user_query: str) -> str | None:
    time_column = _find_matching_column(df_head, user_query, numeric_only=False)
    if time_column and any(
        keyword in _normalize_text(time_column)
        for keyword in ("date", "time", "day", "month", "year", "week")
    ):
        return time_column

    for column in df_head.columns:
        normalized_column = _normalize_text(str(column))
        if any(keyword in normalized_column for keyword in ("date", "time", "day", "month", "year", "week")):
            return str(column)

    for column in df_head.columns:
        if pd.api.types.is_datetime64_any_dtype(df_head[column]):
            return str(column)

    return None


def _pick_categorical_column(df_head: pd.DataFrame) -> str | None:
    best_column = None
    best_score = -1.0
    row_count = max(len(df_head.index), 1)

    for column in df_head.columns:
        series = df_head[column]
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_datetime64_any_dtype(series):
            continue
        if _is_time_like_column_name(column):
            continue

        non_null = int(series.notna().sum())
        unique_count = int(series.nunique(dropna=True))
        if non_null == 0 or unique_count <= 1:
            continue

        uniqueness_ratio = unique_count / max(non_null, 1)
        if uniqueness_ratio >= 0.9 and unique_count > 5:
            continue

        score = float((non_null * 2) - unique_count)
        if unique_count <= min(12, row_count):
            score += 5

        if score > best_score:
            best_score = score
            best_column = str(column)

    return best_column


def _pick_meaningful_numeric_columns(df_head: pd.DataFrame) -> list[str]:
    numeric_columns = [str(column) for column in df_head.select_dtypes(include="number").columns]
    filtered_columns = [
        column
        for column in numeric_columns
        if "id" not in _normalize_text(column)
    ]
    return filtered_columns or numeric_columns


def _metric_from_query(df_head: pd.DataFrame, user_query: str) -> str | None:
    normalized_query = _normalize_text(user_query)
    filtered_tokens = [
        token
        for token in normalized_query.split()
        if token not in AGGREGATION_WORDS and not token.isdigit()
    ]
    if not filtered_tokens:
        return None
    return _find_matching_column(df_head, " ".join(filtered_tokens), numeric_only=True)


def generate_code(df_head: pd.DataFrame, user_query: str) -> str:
    normalized_query = _normalize_text(user_query)
    if not normalized_query:
        raise ValueError("Query is empty.")

    if "dataset summary" in normalized_query or "dataset overview" in normalized_query:
        return (
            "result = (\n"
            "    df.dtypes.astype('string')\n"
            "    .rename('dtype')\n"
            "    .reset_index()\n"
            "    .rename(columns={'index': 'column'})\n"
            ")"
        )

    if "missing value" in normalized_query or "missing values" in normalized_query:
        return (
            "result = (\n"
            "    df.isna()\n"
            "    .sum()\n"
            "    .sort_values(ascending=False)\n"
            "    .rename('missing_values')\n"
            "    .reset_index()\n"
            "    .rename(columns={'index': 'column'})\n"
            ")"
        )

    if "top categories" in normalized_query:
        category_column = _pick_categorical_column(df_head) or _pick_group_column(df_head, user_query)
        if not category_column:
            raise ValueError("Could not find a categorical column for the top categories request.")
        top_n = _extract_top_n(user_query)
        return (
            "result = (\n"
            f"    df[{_quoted_column(category_column)}]\n"
            "    .value_counts(dropna=False)\n"
            f"    .head({top_n})\n"
            f"    .rename_axis({_quoted_column(category_column)})\n"
            "    .reset_index(name='count')\n"
            ")"
        )

    if "top" in normalized_query:
        top_n = _extract_top_n(user_query)
        group_column = _pick_group_column(df_head, user_query)
        metric_column = _metric_from_query(df_head, user_query)

        if not group_column:
            raise ValueError("Could not find a grouping column for the top request.")

        if metric_column:
            return (
                "result = (\n"
                f"    df.groupby({_quoted_column(group_column)}, dropna=False)[{_quoted_column(metric_column)}]\n"
                "    .sum()\n"
                "    .sort_values(ascending=False)\n"
                f"    .head({top_n})\n"
                "    .reset_index()\n"
                ")"
            )

        return (
            "result = (\n"
            f"    df[{_quoted_column(group_column)}]\n"
            "    .value_counts(dropna=False)\n"
            f"    .head({top_n})\n"
            f"    .rename_axis({_quoted_column(group_column)})\n"
            "    .reset_index(name='count')\n"
            ")"
        )

    if (
        "average" in normalized_query
        or "averages" in normalized_query
        or "avg" in normalized_query
        or "mean" in normalized_query
    ):
        metric_column = _metric_from_query(df_head, user_query)
        if metric_column:
            return f"result = df[{_quoted_column(metric_column)}].mean()"
        numeric_columns = _pick_meaningful_numeric_columns(df_head)
        if not numeric_columns:
            raise ValueError("Could not find a numeric column for the average request.")
        if len(numeric_columns) == 1:
            return f"result = df[{_quoted_column(numeric_columns[0])}].mean()"
        quoted_columns = ", ".join(_quoted_column(column) for column in numeric_columns)
        return f"result = df[[{quoted_columns}]].mean().sort_values(ascending=False)"

    if "group by" in normalized_query:
        group_column = _pick_group_column(df_head, user_query)
        if not group_column:
            raise ValueError("Could not find a grouping column for the group-by request.")
        metric_column = _metric_from_query(df_head, user_query)
        if metric_column:
            return (
                "result = (\n"
                f"    df.groupby({_quoted_column(group_column)}, dropna=False)[{_quoted_column(metric_column)}]\n"
                "    .sum()\n"
                "    .reset_index()\n"
                ")"
            )
        return (
            "result = (\n"
            f"    df.groupby({_quoted_column(group_column)}, dropna=False)\n"
            "    .sum(numeric_only=True)\n"
            "    .reset_index()\n"
            ")"
        )

    if "sum by" in normalized_query or ("sum" in normalized_query and "by" in normalized_query):
        group_column = _pick_group_column(df_head, user_query)
        metric_column = _metric_from_query(df_head, user_query)

        if not group_column:
            raise ValueError("Could not find a grouping column for the sum request.")
        if metric_column:
            return (
                "result = (\n"
                f"    df.groupby({_quoted_column(group_column)}, dropna=False)[{_quoted_column(metric_column)}]\n"
                "    .sum()\n"
                "    .reset_index()\n"
                ")"
            )
        return (
            "result = (\n"
            f"    df.groupby({_quoted_column(group_column)}, dropna=False)\n"
            "    .sum(numeric_only=True)\n"
            "    .reset_index()\n"
            ")"
        )

    if "trend" in normalized_query or "over time" in normalized_query:
        time_column = _pick_time_column(df_head, user_query)
        if not time_column:
            raise ValueError("Could not find a time column for the trend request.")

        metric_column = _metric_from_query(df_head, user_query) or _pick_metric_column(df_head, user_query)
        if metric_column:
            return (
                "result = (\n"
                f"    df.groupby({_quoted_column(time_column)}, dropna=False)[{_quoted_column(metric_column)}]\n"
                "    .sum()\n"
                "    .reset_index()\n"
                f"    .sort_values(by={_quoted_column(time_column)})\n"
                ")"
            )

        return (
            "result = (\n"
            f"    df.groupby({_quoted_column(time_column)}, dropna=False)\n"
            "    .size()\n"
            "    .reset_index(name='count')\n"
            f"    .sort_values(by={_quoted_column(time_column)})\n"
            ")"
        )

    raise ValueError(
        "Unsupported query. Try a dataset summary, missing values, top categories, average, group by, sum by, or trend request."
    )


def validate_code(code: str) -> str | None:
    lowered_code = str(code).lower()
    for pattern, message in BLOCKED_CODE_PATTERNS.items():
        if pattern in lowered_code:
            return f"Unsafe code blocked: {message}"

    try:
        compile(code, "<analysis-agent>", "exec")
    except SyntaxError as exc:
        return f"Generated code is invalid: {exc}"

    if "result" not in code:
        return "Generated code must assign to result."

    return None


def execute_code(code: str, df: pd.DataFrame) -> tuple[Any, str | None]:
    local_vars = {"df": df.copy(), "result": None}
    try:
        exec(code, {"__builtins__": {}}, local_vars)
        return local_vars.get("result"), None
    except Exception as exc:  # pragma: no cover - exercised through run_analysis_agent
        return None, str(exc)


def _series_preview(series: pd.Series, limit: int = 3) -> list[str]:
    preview: list[str] = []
    for index, value in series.head(limit).items():
        preview.append(f"{index}: {value}")
    return preview


def generate_insight(result: Any) -> str:
    if isinstance(result, pd.DataFrame):
        if result.empty:
            return "- No rows matched the request.\n- Try a broader question if you expected data."

        result_columns = set(str(column) for column in result.columns)
        if {"column", "dtype"}.issubset(result_columns):
            unique_dtypes = sorted(str(dtype) for dtype in result["dtype"].dropna().unique().tolist())
            bullets = [f"- The dataset summary covers {len(result)} columns."]
            if unique_dtypes:
                bullets.append(f"- Data types present: {', '.join(unique_dtypes[:4])}.")
            bullets.append(f"- Sample columns: {', '.join(result['column'].astype(str).head(3).tolist())}.")
            return "\n".join(bullets[:3])

        if {"column", "missing_values"}.issubset(result_columns):
            top_row = result.sort_values(by="missing_values", ascending=False).iloc[0]
            if float(top_row["missing_values"]) == 0:
                return (
                    "- No missing values were found in the checked columns.\n"
                    "- The dataset looks complete for the fields reviewed."
                )
            return (
                f"- {top_row['column']} has the most missing values at {int(top_row['missing_values'])}.\n"
                "- Missing data is concentrated in a small set of fields."
            )

        bullets = [f"- The result has {len(result)} rows and {len(result.columns)} columns."]
        numeric_columns = list(result.select_dtypes(include="number").columns)
        if numeric_columns:
            lead_metric = numeric_columns[0]
            top_row = result.sort_values(by=lead_metric, ascending=False).iloc[0]
            label_columns = [
                column
                for column in result.columns
                if column != lead_metric and not pd.api.types.is_numeric_dtype(result[column])
            ]
            if label_columns:
                label = ", ".join(str(top_row[column]) for column in label_columns[:2])
                bullets.append(f"- The highest {lead_metric} is {top_row[lead_metric]} for {label}.")
            else:
                bullets.append(f"- The highest {lead_metric} in the result is {top_row[lead_metric]}.")
        bullets.append(f"- Columns returned: {', '.join(str(column) for column in result.columns[:4])}.")
        return "\n".join(bullets[:3])

    if isinstance(result, pd.Series):
        if result.empty:
            return "- The result is empty.\n- There is nothing to summarize yet."

        preview = _series_preview(result)
        bullets = [f"- The result has {len(result)} values."]
        if preview:
            bullets.append(f"- Top values: {', '.join(preview)}.")
        bullets.append("- This gives a quick view of the strongest values in the answer.")
        return "\n".join(bullets[:3])

    if isinstance(result, (int, float)) and not isinstance(result, bool):
        if isinstance(result, float) and math.isnan(result):
            return "- The answer is not available from the current data."
        rounded_result = round(float(result), 2)
        return (
            f"- The answer is {rounded_result}.\n"
            "- This is the single value that matches your question."
        )

    if result is None:
        return "- No result was produced."

    return f"- Result summary: {result}"


def run_analysis_agent(df: pd.DataFrame, user_query: str, *, prefer_rule_based: bool = False) -> dict[str, Any]:
    payload = {
        "code": "",
        "result": None,
        "insight": "",
        "suggestions": [],
        "error": None,
    }

    try:
        llm_error: str | None = None
        used_llm_code = False
        memory_context = analysis_memory.get_context()

        if prefer_rule_based:
            code = generate_code(df.head(), user_query)
            payload["code"] = code
        else:
            try:
                code = generate_code_with_llm(df.head(), user_query, memory_context=memory_context)
                is_valid, validation_error = validate_generated_code(code)
                if not is_valid:
                    llm_error = validation_error
                    code = generate_code(df.head(), user_query)
                else:
                    used_llm_code = True
                payload["code"] = code
            except Exception as exc:
                llm_error = str(exc)
                code = generate_code(df.head(), user_query)
                payload["code"] = code

        validation_error = validate_code(code)
        if validation_error:
            payload["error"] = validation_error
            return payload

        result, execution_error = execute_code(code, df)
        if execution_error and used_llm_code:
            llm_error = execution_error
            code = generate_code(df.head(), user_query)
            payload["code"] = code

            validation_error = validate_code(code)
            if validation_error:
                payload["error"] = validation_error
                return payload

            result, execution_error = execute_code(code, df)

        payload["result"] = result
        if execution_error:
            payload["error"] = execution_error
            return payload

        payload["insight"] = generate_insight(result)
        analysis_memory.add_entry(user_query, code, result)
        payload["suggestions"] = generate_suggestions(analysis_memory.get_context(), df)
        if llm_error and not payload["error"]:
            payload["llm_fallback_reason"] = llm_error
        return payload
    except Exception as exc:
        payload["error"] = str(exc)
        return payload


__all__ = [
    "execute_code",
    "generate_code",
    "generate_insight",
    "analysis_memory",
    "run_analysis_agent",
    "validate_code",
]
