from __future__ import annotations

import re
from typing import Any

import pandas as pd
from pandas.api.types import is_numeric_dtype

from backend.services.target_detector import coerce_datetime_series


def _normalize_column_name(column_name: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(column_name or "").strip().lower())


def _normalize_join_series(series: pd.Series) -> pd.Series:
    parsed_datetime = coerce_datetime_series(series, column_name=str(series.name or ""))
    if parsed_datetime is not None:
        return parsed_datetime.dt.strftime("%Y-%m-%dT%H:%M:%S").fillna("")

    if is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce")
        return numeric.round(10).astype("string").fillna("")

    return series.astype("string").str.strip().str.lower().fillna("")


def detect_common_columns(left_df: pd.DataFrame, right_df: pd.DataFrame) -> list[dict[str, Any]]:
    left_lookup = {_normalize_column_name(column): str(column) for column in left_df.columns}
    right_lookup = {_normalize_column_name(column): str(column) for column in right_df.columns}
    suggestions: list[dict[str, Any]] = []

    for normalized_name, left_column in left_lookup.items():
        right_column = right_lookup.get(normalized_name)
        if not right_column:
            continue

        left_values = set(_normalize_join_series(left_df[left_column]).head(200).tolist())
        right_values = set(_normalize_join_series(right_df[right_column]).head(200).tolist())
        overlap = len((left_values & right_values) - {""})
        compatibility_score = overlap / max(1, min(len(left_values), len(right_values)))
        suggestions.append(
            {
                "left_column": left_column,
                "right_column": right_column,
                "compatibility_score": round(compatibility_score, 4),
                "reason": "Normalized column names match across both datasets.",
            }
        )

    suggestions.sort(key=lambda item: (-float(item["compatibility_score"]), item["left_column"], item["right_column"]))
    return suggestions


def join_datasets(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    *,
    left_on: str,
    right_on: str,
    how: str = "inner",
    suffixes: tuple[str, str] = ("_left", "_right"),
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if left_on not in left_df.columns:
        raise ValueError(f"Join column '{left_on}' was not found in the left dataset.")
    if right_on not in right_df.columns:
        raise ValueError(f"Join column '{right_on}' was not found in the right dataset.")

    normalized_left = left_df.copy(deep=True)
    normalized_right = right_df.copy(deep=True)
    left_key = "__aidssist_join_left__"
    right_key = "__aidssist_join_right__"
    normalized_left[left_key] = _normalize_join_series(normalized_left[left_on])
    normalized_right[right_key] = _normalize_join_series(normalized_right[right_on])

    merged = normalized_left.merge(
        normalized_right,
        how=str(how or "inner").strip().lower(),
        left_on=left_key,
        right_on=right_key,
        suffixes=suffixes,
        indicator=True,
    )
    report = {
        "join_type": str(how or "inner").strip().lower(),
        "left_on": left_on,
        "right_on": right_on,
        "matched_rows": int((merged["_merge"] == "both").sum()),
        "left_only_rows": int((merged["_merge"] == "left_only").sum()),
        "right_only_rows": int((merged["_merge"] == "right_only").sum()),
        "row_count": int(len(merged)),
        "column_count": int(len(merged.columns) - 3),
    }

    merged = merged.drop(columns=[left_key, right_key, "_merge"]).reset_index(drop=True)
    return merged, report


def cross_filter_dataset(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    *,
    source_column: str,
    target_column: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if source_column not in source_df.columns:
        raise ValueError(f"Source column '{source_column}' was not found.")
    if target_column not in target_df.columns:
        raise ValueError(f"Target column '{target_column}' was not found.")

    source_values = set(_normalize_join_series(source_df[source_column]).dropna().tolist())
    filtered = target_df.copy(deep=True)
    target_values = _normalize_join_series(filtered[target_column])
    filtered = filtered.loc[target_values.isin(source_values)].copy().reset_index(drop=True)

    report = {
        "source_column": source_column,
        "target_column": target_column,
        "matched_distinct_values": len(source_values),
        "retained_rows": int(len(filtered)),
        "dropped_rows": int(len(target_df) - len(filtered)),
    }
    return filtered, report


__all__ = [
    "cross_filter_dataset",
    "detect_common_columns",
    "join_datasets",
]
