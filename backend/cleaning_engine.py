from __future__ import annotations

from dataclasses import field

from backend.compat import dataclass

import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
)


@dataclass(slots=True)
class CleaningEngineResult:
    cleaned_df: pd.DataFrame
    outliers: dict[str, int]
    issues: list[str]
    quality_score: float
    cleaning_report: dict[str, object]
    actions: list[str] = field(default_factory=list)


def _looks_like_text(series: pd.Series) -> bool:
    return bool(
        is_object_dtype(series)
        or is_string_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
    )


def _resolve_fill_value(
    series: pd.Series,
    *,
    numeric_fill_strategy: str = "median",
    text_fill_value: str = "Unknown",
):
    if _looks_like_text(series):
        return text_fill_value

    if is_datetime64_any_dtype(series):
        non_null = series.dropna()
        return non_null.median() if not non_null.empty else pd.NaT

    if is_bool_dtype(series):
        non_null = series.dropna()
        if non_null.empty:
            return pd.NA
        mode = non_null.mode(dropna=True)
        return mode.iloc[0] if not mode.empty else non_null.iloc[0]

    non_null = series.dropna()
    if non_null.empty:
        return pd.NA

    if numeric_fill_strategy == "mean" and is_numeric_dtype(series):
        return non_null.mean()
    return non_null.median()


def _handle_missing_with_strategy(
    df: pd.DataFrame,
    *,
    numeric_fill_strategy: str = "median",
    text_fill_value: str = "Unknown",
    drop_columns_over: float = 0.5,
) -> tuple[pd.DataFrame, list[str]]:
    working_df = df.copy()
    actions: list[str] = []

    for column_name in list(working_df.columns):
        series = working_df[column_name]
        missing_ratio = float(series.isna().mean()) if len(working_df) else 0.0

        if missing_ratio > drop_columns_over:
            working_df = working_df.drop(columns=[column_name])
            actions.append(
                f"Dropped '{column_name}' because {missing_ratio:.0%} of values were missing."
            )
            continue

        if not series.isna().any():
            continue

        fill_value = _resolve_fill_value(
            series,
            numeric_fill_strategy=numeric_fill_strategy,
            text_fill_value=text_fill_value,
        )
        working_df[column_name] = series.fillna(fill_value)
        if _looks_like_text(series):
            actions.append(
                f"Filled missing text values in '{column_name}' with '{text_fill_value}'."
            )
        elif is_datetime64_any_dtype(series):
            actions.append(
                f"Filled missing datetime values in '{column_name}' with the median timestamp."
            )
        else:
            actions.append(
                f"Filled missing numeric values in '{column_name}' with the {numeric_fill_strategy}."
            )

    return working_df, actions


def convert_types(df: pd.DataFrame) -> pd.DataFrame:
    converted_df = df.copy()
    for column_name in converted_df.columns:
        lowered_name = str(column_name).lower()
        if "date" in lowered_name or "time" in lowered_name:
            try:
                converted_df[column_name] = pd.to_datetime(
                    converted_df[column_name],
                    errors="coerce",
                    format="mixed",
                )
            except Exception:
                pass
    return converted_df


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    handled_df, _ = _handle_missing_with_strategy(df)
    return handled_df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates()


def detect_outliers(df: pd.DataFrame) -> dict[str, int]:
    outliers: dict[str, int] = {}

    for column_name in df.select_dtypes(include=["number"]).columns:
        series = pd.to_numeric(df[column_name], errors="coerce").dropna()
        if series.empty:
            outliers[str(column_name)] = 0
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        outliers[str(column_name)] = int(
            df[
                (pd.to_numeric(df[column_name], errors="coerce") < q1 - 1.5 * iqr)
                | (pd.to_numeric(df[column_name], errors="coerce") > q3 + 1.5 * iqr)
            ].shape[0]
        )

    return outliers


def validate_data(df: pd.DataFrame) -> list[str]:
    issues: list[str] = []

    if df.empty:
        issues.append("Empty dataset")

    if int(df.isna().sum().sum()) > 0:
        issues.append("Missing values remain")

    return issues


def compute_quality_score(df: pd.DataFrame) -> float:
    missing_penalty = float(df.isna().mean().mean()) if len(df.columns) else 0.0
    duplicate_penalty = float(df.duplicated().mean()) if len(df) else 0.0
    score = max(0.0, 1 - (missing_penalty + duplicate_penalty))
    return round(score, 2)


def build_cleaning_report(
    original_df: pd.DataFrame,
    cleaned_df: pd.DataFrame,
    *,
    outliers: dict[str, int],
    issues: list[str],
    quality_score: float,
    actions: list[str] | None = None,
) -> dict[str, object]:
    actions = list(actions or [])
    original_missing = int(original_df.isna().sum().sum())
    cleaned_missing = int(cleaned_df.isna().sum().sum())
    original_duplicates = int(original_df.duplicated().sum()) if len(original_df) else 0
    cleaned_duplicates = int(cleaned_df.duplicated().sum()) if len(cleaned_df) else 0

    common_columns = [column for column in original_df.columns if column in cleaned_df.columns]
    type_conversions: dict[str, dict[str, str]] = {}
    for column_name in common_columns:
        previous_type = str(original_df[column_name].dtype)
        current_type = str(cleaned_df[column_name].dtype)
        if previous_type != current_type:
            type_conversions[str(column_name)] = {
                "from": previous_type,
                "to": current_type,
            }

    return {
        "quality_score": round(float(quality_score), 2),
        "missing_handled": max(0, original_missing - cleaned_missing),
        "duplicates_removed": max(0, original_duplicates - cleaned_duplicates),
        "outliers_detected": int(sum(outliers.values())),
        "outlier_columns": {str(key): int(value) for key, value in outliers.items()},
        "issues": [str(issue) for issue in issues if str(issue).strip()],
        "actions": actions,
        "columns_dropped": [
            str(column_name)
            for column_name in original_df.columns
            if column_name not in cleaned_df.columns
        ],
        "type_conversions": type_conversions,
        "before": {
            "row_count": int(len(original_df)),
            "column_count": int(len(original_df.columns)),
            "missing_cells": original_missing,
            "duplicate_rows": original_duplicates,
        },
        "after": {
            "row_count": int(len(cleaned_df)),
            "column_count": int(len(cleaned_df.columns)),
            "missing_cells": cleaned_missing,
            "duplicate_rows": cleaned_duplicates,
        },
    }


def execute_cleaning_engine(
    df: pd.DataFrame,
    *,
    numeric_fill_strategy: str = "median",
    text_fill_value: str = "Unknown",
    drop_columns_over: float = 0.5,
) -> CleaningEngineResult:
    original_df = df.copy()
    actions: list[str] = []

    converted_df = convert_types(df)
    for column_name in converted_df.columns:
        if column_name not in original_df.columns:
            continue
        if str(original_df[column_name].dtype) != str(converted_df[column_name].dtype):
            actions.append(f"Converted '{column_name}' to {converted_df[column_name].dtype}.")

    handled_df, missing_actions = _handle_missing_with_strategy(
        converted_df,
        numeric_fill_strategy=numeric_fill_strategy,
        text_fill_value=text_fill_value,
        drop_columns_over=drop_columns_over,
    )
    actions.extend(missing_actions)

    duplicate_count = int(handled_df.duplicated().sum()) if len(handled_df) else 0
    deduped_df = remove_duplicates(handled_df).reset_index(drop=True)
    if duplicate_count:
        actions.append(f"Removed {duplicate_count:,} duplicate rows.")

    outliers = detect_outliers(deduped_df)
    issues = validate_data(deduped_df)
    quality_score = compute_quality_score(deduped_df)
    cleaning_report = build_cleaning_report(
        original_df,
        deduped_df,
        outliers=outliers,
        issues=issues,
        quality_score=quality_score,
        actions=actions,
    )

    return CleaningEngineResult(
        cleaned_df=deduped_df,
        outliers=outliers,
        issues=issues,
        quality_score=quality_score,
        cleaning_report=cleaning_report,
        actions=actions,
    )


def clean_data(df: pd.DataFrame) -> dict[str, object]:
    result = execute_cleaning_engine(df)
    return {
        "cleaned_df": result.cleaned_df,
        "outliers": result.outliers,
        "issues": result.issues,
        "quality_score": result.quality_score,
    }
