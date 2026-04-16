from __future__ import annotations

from typing import Any

import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
)


DATETIME_PARSE_THRESHOLD = 0.70
MIN_TIME_SERIES_UNIQUE_TIMESTAMPS = 10
IDENTIFIER_HINTS = ("id", "key", "code", "index", "row", "sku", "zip", "postal")


def _normalize_column_name(column_name: Any) -> str:
    return str(column_name or "").strip().lower()


def _looks_like_identifier(column_name: Any, series: pd.Series) -> bool:
    normalized_name = _normalize_column_name(column_name)
    if any(token in normalized_name for token in IDENTIFIER_HINTS):
        return True

    numeric_series = pd.to_numeric(series, errors="coerce").dropna()
    if numeric_series.empty or numeric_series.nunique() < 4:
        return False

    unique_ratio = float(numeric_series.nunique() / max(len(numeric_series), 1))
    monotonic = bool(
        numeric_series.is_monotonic_increasing or numeric_series.is_monotonic_decreasing
    )
    integer_like = bool((numeric_series % 1).abs().max() < 1e-9)
    return unique_ratio >= 0.98 and monotonic and integer_like


def _parse_datetime_series(
    series: pd.Series,
    *,
    column_name: Any | None = None,
) -> pd.Series | None:
    if is_datetime64_any_dtype(series):
        parsed = pd.to_datetime(series, errors="coerce")
        return parsed if parsed.notna().any() else None

    if not (
        is_object_dtype(series)
        or is_string_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
    ):
        return None

    non_null = series.dropna()
    if non_null.empty:
        return None

    normalized_name = _normalize_column_name(column_name)
    sampled_text = non_null.astype("string").str.strip()
    looks_datetime_like = bool(
        sampled_text.str.contains(r"[-/:]|\b\d{4}\b", regex=True, na=False).mean() >= 0.5
    )
    if not looks_datetime_like and not any(
        token in normalized_name for token in ("date", "time", "day", "month", "year", "week")
    ):
        return None

    parsed = pd.to_datetime(series.astype("string"), errors="coerce")
    valid_ratio = float(parsed.notna().sum() / max(non_null.shape[0], 1))
    if valid_ratio < DATETIME_PARSE_THRESHOLD:
        return None
    return parsed


def _detect_datetime_columns(df: pd.DataFrame) -> tuple[list[str], dict[str, pd.Series]]:
    datetime_columns: list[str] = []
    parsed_columns: dict[str, pd.Series] = {}

    for column in df.columns:
        column_name = str(column)
        parsed = _parse_datetime_series(df[column], column_name=column_name)
        if parsed is None:
            continue
        datetime_columns.append(column_name)
        parsed_columns[column_name] = parsed

    return datetime_columns, parsed_columns


def _detect_numeric_columns(df: pd.DataFrame, *, excluded: set[str]) -> list[str]:
    numeric_columns: list[str] = []
    for column in df.columns:
        column_name = str(column)
        if column_name in excluded:
            continue
        if is_bool_dtype(df[column]):
            continue
        if is_numeric_dtype(df[column]):
            numeric_columns.append(column_name)
    return numeric_columns


def _detect_categorical_columns(df: pd.DataFrame, *, excluded: set[str]) -> list[str]:
    categorical_columns: list[str] = []
    for column in df.columns:
        column_name = str(column)
        if column_name in excluded:
            continue
        if column_name in categorical_columns:
            continue
        if (
            is_bool_dtype(df[column])
            or is_object_dtype(df[column])
            or is_string_dtype(df[column])
            or isinstance(df[column].dtype, pd.CategoricalDtype)
        ):
            categorical_columns.append(column_name)
    return categorical_columns


def _is_time_series(parsed_datetime_columns: dict[str, pd.Series]) -> bool:
    for parsed in parsed_datetime_columns.values():
        cleaned = parsed.dropna().sort_values().drop_duplicates()
        if cleaned.shape[0] <= MIN_TIME_SERIES_UNIQUE_TIMESTAMPS:
            continue
        try:
            cleaned.sort_values()
        except Exception:
            continue
        return True
    return False


def _has_ml_target_candidate(df: pd.DataFrame, numeric_columns: list[str]) -> bool:
    return any(
        not _looks_like_identifier(column_name, df[column_name])
        for column_name in numeric_columns
    )


def detect_dataset_type(df: pd.DataFrame) -> dict[str, Any]:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("detect_dataset_type expects a pandas DataFrame.")

    datetime_columns, parsed_datetime_columns = _detect_datetime_columns(df)
    excluded_columns = set(datetime_columns)
    numeric_columns = _detect_numeric_columns(df, excluded=excluded_columns)
    categorical_columns = _detect_categorical_columns(df, excluded=excluded_columns | set(numeric_columns))

    has_numeric_target_candidate = _has_ml_target_candidate(df, numeric_columns)
    mixed_feature_types = bool(categorical_columns or datetime_columns or len(numeric_columns) > 1)

    return {
        "has_datetime": bool(datetime_columns),
        "datetime_columns": datetime_columns,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "row_count": int(len(df)),
        "is_time_series": _is_time_series(parsed_datetime_columns),
        "is_ml_ready": bool(has_numeric_target_candidate and mixed_feature_types),
    }
