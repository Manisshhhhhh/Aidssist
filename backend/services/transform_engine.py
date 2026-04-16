from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype

from backend.services.target_detector import coerce_datetime_series


@dataclass(slots=True)
class FilterCondition:
    column: str
    operator: str
    value: Any | None = None
    value_to: Any | None = None


@dataclass(slots=True)
class SortInstruction:
    column: str
    ascending: bool = True


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy(deep=True)
    normalized.columns = [str(column) for column in normalized.columns]
    return normalized


def _ensure_columns_exist(df: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    resolved = [str(column) for column in columns if str(column or "").strip()]
    missing = [column for column in resolved if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")
    return resolved


def _coerce_scalar(value: Any, series: pd.Series) -> Any:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if is_numeric_dtype(series):
        return pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]

    datetime_series = coerce_datetime_series(series, column_name=str(series.name or ""))
    if is_datetime64_any_dtype(series) or datetime_series is not None:
        return pd.to_datetime(value, errors="coerce", format="mixed")

    return str(value)


def _series_for_comparison(series: pd.Series) -> pd.Series:
    datetime_series = coerce_datetime_series(series, column_name=str(series.name or ""))
    if datetime_series is not None:
        return datetime_series
    if is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    return series.astype("string")


def apply_filters(
    df: pd.DataFrame,
    conditions: list[FilterCondition],
    *,
    combine_with: str = "and",
) -> pd.DataFrame:
    transformed = _normalize_columns(df)
    if not conditions:
        return transformed

    combine_mode = str(combine_with or "and").strip().lower()
    if combine_mode not in {"and", "or"}:
        raise ValueError("combine_with must be either 'and' or 'or'.")

    combined_mask = pd.Series(True, index=transformed.index) if combine_mode == "and" else pd.Series(False, index=transformed.index)

    for condition in conditions:
        column = str(condition.column or "").strip()
        if column not in transformed.columns:
            raise ValueError(f"Column '{column}' was not found.")

        comparable_series = _series_for_comparison(transformed[column])
        operator = str(condition.operator or "eq").strip().lower()
        value = _coerce_scalar(condition.value, comparable_series)
        value_to = _coerce_scalar(condition.value_to, comparable_series)

        if operator == "eq":
            mask = comparable_series == value
        elif operator == "ne":
            mask = comparable_series != value
        elif operator == "gt":
            mask = comparable_series > value
        elif operator == "gte":
            mask = comparable_series >= value
        elif operator == "lt":
            mask = comparable_series < value
        elif operator == "lte":
            mask = comparable_series <= value
        elif operator == "contains":
            mask = comparable_series.astype("string").str.contains(str(condition.value or ""), case=False, na=False)
        elif operator == "starts_with":
            mask = comparable_series.astype("string").str.startswith(str(condition.value or ""), na=False)
        elif operator == "ends_with":
            mask = comparable_series.astype("string").str.endswith(str(condition.value or ""), na=False)
        elif operator == "in":
            raw_values = condition.value if isinstance(condition.value, list) else str(condition.value or "").split(",")
            normalized_values = [_coerce_scalar(item, comparable_series) for item in raw_values]
            mask = comparable_series.isin(normalized_values)
        elif operator == "not_in":
            raw_values = condition.value if isinstance(condition.value, list) else str(condition.value or "").split(",")
            normalized_values = [_coerce_scalar(item, comparable_series) for item in raw_values]
            mask = ~comparable_series.isin(normalized_values)
        elif operator == "between":
            mask = comparable_series.between(value, value_to, inclusive="both")
        elif operator == "is_null":
            mask = comparable_series.isna()
        elif operator == "not_null":
            mask = comparable_series.notna()
        else:
            raise ValueError(f"Unsupported filter operator '{operator}'.")

        mask = mask.fillna(False)
        combined_mask = combined_mask & mask if combine_mode == "and" else combined_mask | mask

    return transformed.loc[combined_mask].copy().reset_index(drop=True)


def sort_dataframe(df: pd.DataFrame, instructions: list[SortInstruction]) -> pd.DataFrame:
    transformed = _normalize_columns(df)
    if not instructions:
        return transformed
    columns = _ensure_columns_exist(transformed, [instruction.column for instruction in instructions])
    ascending = [bool(instruction.ascending) for instruction in instructions]
    return transformed.sort_values(by=columns, ascending=ascending, kind="mergesort").reset_index(drop=True)


def select_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    transformed = _normalize_columns(df)
    selected = _ensure_columns_exist(transformed, columns)
    return transformed[selected].copy()


def rename_columns(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    transformed = _normalize_columns(df)
    clean_mapping: dict[str, str] = {}
    for source, target in dict(mapping or {}).items():
        source_name = str(source or "").strip()
        target_name = str(target or "").strip()
        if not source_name or not target_name:
            continue
        if source_name not in transformed.columns:
            raise ValueError(f"Column '{source_name}' was not found.")
        clean_mapping[source_name] = target_name
    if not clean_mapping:
        return transformed
    return transformed.rename(columns=clean_mapping)


def handle_missing_values(
    df: pd.DataFrame,
    *,
    strategy: str,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    transformed = _normalize_columns(df)
    selected_columns = _ensure_columns_exist(transformed, columns or list(transformed.columns))
    strategy_name = str(strategy or "").strip().lower()

    if strategy_name == "drop":
        return transformed.dropna(subset=selected_columns).reset_index(drop=True)

    for column in selected_columns:
        series = transformed[column]
        if strategy_name == "mean":
            if is_numeric_dtype(series):
                transformed[column] = series.fillna(pd.to_numeric(series, errors="coerce").mean())
        elif strategy_name == "median":
            if is_numeric_dtype(series):
                transformed[column] = series.fillna(pd.to_numeric(series, errors="coerce").median())
        elif strategy_name == "mode":
            mode = series.mode(dropna=True)
            if not mode.empty:
                transformed[column] = series.fillna(mode.iloc[0])
        elif strategy_name == "ffill":
            transformed[column] = series.ffill()
        else:
            raise ValueError(f"Unsupported missing value strategy '{strategy_name}'.")

    return transformed


def convert_column_types(df: pd.DataFrame, conversions: dict[str, str]) -> pd.DataFrame:
    transformed = _normalize_columns(df)
    for column, target_type in dict(conversions or {}).items():
        column_name = str(column or "").strip()
        if column_name not in transformed.columns:
            raise ValueError(f"Column '{column_name}' was not found.")

        normalized_target = str(target_type or "").strip().lower()
        if normalized_target == "numeric":
            transformed[column_name] = pd.to_numeric(transformed[column_name], errors="coerce")
        elif normalized_target == "datetime":
            transformed[column_name] = pd.to_datetime(transformed[column_name], errors="coerce", format="mixed")
        elif normalized_target in {"string", "text"}:
            transformed[column_name] = transformed[column_name].astype("string")
        elif normalized_target == "category":
            transformed[column_name] = transformed[column_name].astype("category")
        else:
            raise ValueError(f"Unsupported target type '{normalized_target}'.")

    return transformed


def remove_duplicates(
    df: pd.DataFrame,
    *,
    subset: list[str] | None = None,
    keep: str = "first",
) -> pd.DataFrame:
    transformed = _normalize_columns(df)
    resolved_subset = _ensure_columns_exist(transformed, subset or []) if subset else None
    return transformed.drop_duplicates(subset=resolved_subset, keep=keep).reset_index(drop=True)


def handle_outliers(
    df: pd.DataFrame,
    *,
    columns: list[str] | None = None,
    method: str = "iqr",
    action: str = "clip",
    z_threshold: float = 3.0,
    whisker_width: float = 1.5,
) -> pd.DataFrame:
    transformed = _normalize_columns(df)
    method_name = str(method or "iqr").strip().lower()
    action_name = str(action or "clip").strip().lower()
    selected_columns = _ensure_columns_exist(transformed, columns or list(transformed.select_dtypes(include=["number"]).columns))
    numeric_columns = [column for column in selected_columns if is_numeric_dtype(transformed[column])]
    row_mask_to_remove = pd.Series(False, index=transformed.index)

    for column in numeric_columns:
        numeric_series = pd.to_numeric(transformed[column], errors="coerce")
        non_null = numeric_series.dropna()
        if non_null.empty:
            continue

        if method_name == "zscore":
            std = float(non_null.std(ddof=0))
            if std <= 0 or np.isnan(std):
                continue
            mean = float(non_null.mean())
            lower_bound = mean - (float(z_threshold) * std)
            upper_bound = mean + (float(z_threshold) * std)
            outlier_mask = (numeric_series < lower_bound) | (numeric_series > upper_bound)
        elif method_name == "iqr":
            q1 = float(non_null.quantile(0.25))
            q3 = float(non_null.quantile(0.75))
            iqr = q3 - q1
            if iqr <= 0 or np.isnan(iqr):
                continue
            lower_bound = q1 - (float(whisker_width) * iqr)
            upper_bound = q3 + (float(whisker_width) * iqr)
            outlier_mask = (numeric_series < lower_bound) | (numeric_series > upper_bound)
        else:
            raise ValueError(f"Unsupported outlier method '{method_name}'.")

        outlier_mask = outlier_mask.fillna(False)
        if action_name == "remove":
            row_mask_to_remove = row_mask_to_remove | outlier_mask
        elif action_name == "clip":
            transformed.loc[numeric_series < lower_bound, column] = lower_bound
            transformed.loc[numeric_series > upper_bound, column] = upper_bound
        else:
            raise ValueError(f"Unsupported outlier action '{action_name}'.")

    if action_name == "remove":
        transformed = transformed.loc[~row_mask_to_remove].copy()
    return transformed.reset_index(drop=True)


def apply_operation(df: pd.DataFrame, operation: str, **kwargs: Any) -> pd.DataFrame:
    normalized_operation = str(operation or "").strip().lower()
    if normalized_operation == "filter":
        return apply_filters(df, kwargs.get("conditions", []), combine_with=kwargs.get("combine_with", "and"))
    if normalized_operation == "sort":
        return sort_dataframe(df, kwargs.get("instructions", []))
    if normalized_operation == "select_columns":
        return select_columns(df, kwargs.get("columns", []))
    if normalized_operation == "rename_columns":
        return rename_columns(df, kwargs.get("mapping", {}))
    if normalized_operation == "missing_values":
        return handle_missing_values(df, strategy=kwargs.get("strategy", "mode"), columns=kwargs.get("columns"))
    if normalized_operation == "convert_types":
        return convert_column_types(df, kwargs.get("conversions", {}))
    if normalized_operation == "remove_duplicates":
        return remove_duplicates(df, subset=kwargs.get("subset"), keep=kwargs.get("keep", "first"))
    if normalized_operation == "outliers":
        return handle_outliers(
            df,
            columns=kwargs.get("columns"),
            method=kwargs.get("method", "iqr"),
            action=kwargs.get("action", "clip"),
            z_threshold=float(kwargs.get("z_threshold", 3.0)),
            whisker_width=float(kwargs.get("whisker_width", 1.5)),
        )
    raise ValueError(f"Unsupported operation '{operation}'.")


def apply_pipeline(df: pd.DataFrame, operations: list[dict[str, Any]]) -> pd.DataFrame:
    current = _normalize_columns(df)
    for item in operations:
        current = apply_operation(current, item.get("operation", ""), **dict(item.get("parameters") or {}))
    return current


__all__ = [
    "FilterCondition",
    "SortInstruction",
    "apply_filters",
    "apply_operation",
    "apply_pipeline",
    "convert_column_types",
    "handle_missing_values",
    "handle_outliers",
    "remove_duplicates",
    "rename_columns",
    "select_columns",
    "sort_dataframe",
]
