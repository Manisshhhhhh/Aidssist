from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype


SUPPORTED_TIME_FILTERS = (
    "current_month",
    "last_month",
    "last_quarter",
    "last_year",
    "custom_range",
)

_TIME_TOKENS = ("date", "time", "month", "day", "year", "week", "quarter")


def _parse_datetime_series(series: pd.Series) -> pd.Series | None:
    if is_datetime64_any_dtype(series):
        parsed = pd.to_datetime(series, errors="coerce")
    else:
        parsed = pd.to_datetime(series.astype("string"), errors="coerce", format="mixed")
    if parsed.dropna().empty:
        return None
    success_ratio = float(parsed.notna().sum() / max(series.notna().sum(), 1))
    if success_ratio < 0.6:
        return None
    return parsed


def _coerce_timestamp(value: Any) -> pd.Timestamp | None:
    if value in (None, ""):
        return None
    parsed = pd.to_datetime(value, errors="coerce", format="mixed")
    if pd.isna(parsed):
        return None
    return pd.Timestamp(parsed)


def detect_time_column(df: pd.DataFrame) -> str | None:
    ranked: list[tuple[int, str]] = []
    for index, column in enumerate(df.columns):
        column_name = str(column)
        parsed = _parse_datetime_series(df[column])
        if parsed is None:
            continue
        lowered = column_name.lower()
        score = sum(4 for token in _TIME_TOKENS if token in lowered) + max(0, 5 - index)
        ranked.append((score, column_name))
    if not ranked:
        return None
    ranked.sort(reverse=True)
    return ranked[0][1]


def build_time_filter_options() -> list[str]:
    return list(SUPPORTED_TIME_FILTERS)


def _resolve_time_bounds(
    filter_type: str,
    *,
    anchor: pd.Timestamp,
    custom_range: dict[str, Any] | None = None,
    start_date: Any | None = None,
    end_date: Any | None = None,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    if filter_type == "current_month":
        start = anchor.replace(day=1)
        end = anchor + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        return start, end

    if filter_type == "last_month":
        current_month_start = anchor.replace(day=1)
        end = current_month_start - pd.Timedelta(microseconds=1)
        start = (current_month_start - pd.offsets.MonthBegin(1)).normalize()
        return start, end

    if filter_type == "last_quarter":
        quarter = ((anchor.month - 1) // 3) + 1
        current_quarter_start = pd.Timestamp(datetime(anchor.year, (quarter - 1) * 3 + 1, 1))
        end = current_quarter_start - pd.Timedelta(microseconds=1)
        start = (current_quarter_start - pd.offsets.QuarterBegin(startingMonth=1)).normalize()
        return start, end

    if filter_type == "last_year":
        end = anchor + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        start = (anchor - pd.DateOffset(years=1) + pd.Timedelta(days=1)).normalize()
        return start, end

    if filter_type == "custom_range":
        custom_range = dict(custom_range or {})
        resolved_start = _coerce_timestamp(start_date or custom_range.get("start_date") or custom_range.get("start"))
        resolved_end = _coerce_timestamp(end_date or custom_range.get("end_date") or custom_range.get("end"))
        if resolved_start is None or resolved_end is None:
            raise ValueError("Custom range requires both start_date and end_date.")
        if resolved_end < resolved_start:
            raise ValueError("Custom range end_date must be on or after start_date.")
        return resolved_start.normalize(), resolved_end.normalize() + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)

    raise ValueError(
        f"Unsupported time filter '{filter_type}'. Supported filters: {', '.join(SUPPORTED_TIME_FILTERS)}."
    )


def apply_time_filter(
    df: pd.DataFrame,
    filter_type: str,
    *,
    time_column: str | None = None,
    reference_time: Any | None = None,
    custom_range: dict[str, Any] | None = None,
    start_date: Any | None = None,
    end_date: Any | None = None,
) -> pd.DataFrame:
    normalized_filter = str(filter_type or "").strip().lower()
    if normalized_filter not in SUPPORTED_TIME_FILTERS:
        raise ValueError(
            f"Unsupported time filter '{filter_type}'. Supported filters: {', '.join(SUPPORTED_TIME_FILTERS)}."
        )

    resolved_time_column = time_column or detect_time_column(df)
    if not resolved_time_column or resolved_time_column not in df.columns:
        raise ValueError("No valid date column found for time filtering.")

    parsed_dates = _parse_datetime_series(df[resolved_time_column])
    if parsed_dates is None:
        raise ValueError(f"Column '{resolved_time_column}' could not be parsed safely as datetimes.")

    filtered = df.copy()
    filtered[resolved_time_column] = parsed_dates
    filtered = filtered.dropna(subset=[resolved_time_column]).sort_values(resolved_time_column).reset_index(drop=True)
    if filtered.empty:
        return filtered

    if normalized_filter == "custom_range":
        anchor = filtered[resolved_time_column].max().normalize()
    else:
        anchor = pd.Timestamp(reference_time) if reference_time is not None else pd.Timestamp(filtered[resolved_time_column].max())
        anchor = anchor.normalize()

    start, end = _resolve_time_bounds(
        normalized_filter,
        anchor=anchor,
        custom_range=custom_range,
        start_date=start_date,
        end_date=end_date,
    )
    mask = (filtered[resolved_time_column] >= start) & (filtered[resolved_time_column] <= end)
    return filtered.loc[mask].reset_index(drop=True)


def filter_by_time(
    df: pd.DataFrame,
    filter_type: str,
    *,
    time_column: str | None = None,
    reference_time: Any | None = None,
    custom_range: dict[str, Any] | None = None,
    start_date: Any | None = None,
    end_date: Any | None = None,
) -> pd.DataFrame:
    return apply_time_filter(
        df,
        filter_type,
        time_column=time_column,
        reference_time=reference_time,
        custom_range=custom_range,
        start_date=start_date,
        end_date=end_date,
    )


__all__ = [
    "SUPPORTED_TIME_FILTERS",
    "apply_time_filter",
    "build_time_filter_options",
    "detect_time_column",
    "filter_by_time",
]
