from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from pandas.api.types import is_numeric_dtype

from backend.dashboard_helpers import infer_datetime_columns
from backend.services.profiling_engine import DatasetProfileReport, build_profile_report


MAX_BAR_CATEGORIES = 20
MAX_SCATTER_POINTS = 1000


@dataclass(slots=True)
class ChartRecommendation:
    chart_type: str
    x: str | None
    y: str | None
    aggregation: str | None
    title: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "chart_type": self.chart_type,
            "x": self.x,
            "y": self.y,
            "aggregation": self.aggregation,
            "title": self.title,
            "reason": self.reason,
        }


@dataclass(slots=True)
class ChartPayload:
    chart_type: str
    title: str
    data: pd.DataFrame
    x: str
    y: str | None = None
    aggregation: str | None = None
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "chart_type": self.chart_type,
            "title": self.title,
            "data": self.data.to_dict(orient="records"),
            "x": self.x,
            "y": self.y,
            "aggregation": self.aggregation,
            "reason": self.reason,
        }


def _resolve_profile(df: pd.DataFrame, profile: DatasetProfileReport | None) -> DatasetProfileReport:
    return profile if profile is not None else build_profile_report(df)


def suggest_chart(
    df: pd.DataFrame,
    *,
    profile: DatasetProfileReport | None = None,
    chart_type: str = "auto",
    x: str | None = None,
    y: str | None = None,
    aggregation: str = "sum",
) -> ChartRecommendation:
    normalized = df.copy(deep=True)
    normalized.columns = [str(column) for column in normalized.columns]
    resolved_profile = _resolve_profile(normalized, profile)
    normalized_chart_type = str(chart_type or "auto").strip().lower()

    if normalized_chart_type != "auto":
        title = f"{normalized_chart_type.title()} view"
        return ChartRecommendation(
            chart_type=normalized_chart_type,
            x=x or (resolved_profile.datetime_columns[0] if resolved_profile.datetime_columns else (resolved_profile.categorical_columns[0] if resolved_profile.categorical_columns else resolved_profile.numeric_columns[0] if resolved_profile.numeric_columns else None)),
            y=y,
            aggregation=aggregation if normalized_chart_type in {"line", "bar"} else None,
            title=title,
            reason="Using the chart controls selected in the studio.",
        )

    if resolved_profile.datetime_columns and resolved_profile.numeric_columns:
        return ChartRecommendation(
            chart_type="line",
            x=resolved_profile.datetime_columns[0],
            y=resolved_profile.numeric_columns[0],
            aggregation=aggregation,
            title=f"{resolved_profile.numeric_columns[0]} over time",
            reason="A datetime axis plus numeric measure makes a line chart the best default.",
        )

    low_cardinality_categories = []
    for column in resolved_profile.categorical_columns:
        unique_count = int(normalized[column].astype("string").nunique(dropna=True))
        if unique_count <= MAX_BAR_CATEGORIES:
            low_cardinality_categories.append(column)

    if low_cardinality_categories and resolved_profile.numeric_columns:
        return ChartRecommendation(
            chart_type="bar",
            x=low_cardinality_categories[0],
            y=resolved_profile.numeric_columns[0],
            aggregation=aggregation,
            title=f"{resolved_profile.numeric_columns[0]} by {low_cardinality_categories[0]}",
            reason="A categorical dimension with a numeric measure is best summarized as a bar chart.",
        )

    if len(resolved_profile.numeric_columns) >= 2:
        return ChartRecommendation(
            chart_type="scatter",
            x=resolved_profile.numeric_columns[0],
            y=resolved_profile.numeric_columns[1],
            aggregation=None,
            title=f"{resolved_profile.numeric_columns[1]} vs {resolved_profile.numeric_columns[0]}",
            reason="Two numeric measures are best explored as a scatter plot.",
        )

    if resolved_profile.numeric_columns:
        return ChartRecommendation(
            chart_type="histogram",
            x=resolved_profile.numeric_columns[0],
            y=None,
            aggregation=None,
            title=f"Distribution of {resolved_profile.numeric_columns[0]}",
            reason="A single numeric measure is best explored through its distribution.",
        )

    fallback_x = resolved_profile.categorical_columns[0] if resolved_profile.categorical_columns else normalized.columns[0]
    return ChartRecommendation(
        chart_type="bar",
        x=fallback_x,
        y=None,
        aggregation="count",
        title=f"Count by {fallback_x}",
        reason="Counting categorical values gives the clearest default summary here.",
    )


def _aggregate(series: pd.Series, aggregation: str) -> pd.Series:
    normalized_aggregation = str(aggregation or "sum").strip().lower()
    if normalized_aggregation == "mean":
        return series.mean()
    if normalized_aggregation == "median":
        return series.median()
    if normalized_aggregation == "count":
        return series.count()
    return series.sum()


def build_chart_payload(
    df: pd.DataFrame,
    *,
    profile: DatasetProfileReport | None = None,
    chart_type: str = "auto",
    x: str | None = None,
    y: str | None = None,
    aggregation: str = "sum",
) -> ChartPayload | None:
    normalized = df.copy(deep=True)
    normalized.columns = [str(column) for column in normalized.columns]
    if normalized.empty or len(normalized.columns) == 0:
        return None

    recommendation = suggest_chart(
        normalized,
        profile=profile,
        chart_type=chart_type,
        x=x,
        y=y,
        aggregation=aggregation,
    )
    actual_type = recommendation.chart_type
    x_column = recommendation.x
    y_column = recommendation.y
    resolved_profile = _resolve_profile(normalized, profile)

    if x_column is None or x_column not in normalized.columns:
        return None

    if actual_type == "histogram":
        histogram_frame = normalized[[x_column]].copy()
        histogram_frame[x_column] = pd.to_numeric(histogram_frame[x_column], errors="coerce")
        histogram_frame = histogram_frame.dropna().reset_index(drop=True)
        if histogram_frame.empty:
            return None
        return ChartPayload(
            chart_type="histogram",
            title=recommendation.title,
            data=histogram_frame,
            x=x_column,
            y=None,
            aggregation=None,
            reason=recommendation.reason,
        )

    if actual_type == "scatter":
        if y_column is None or y_column not in normalized.columns:
            return None
        scatter_frame = normalized[[x_column, y_column]].copy()
        scatter_frame[x_column] = pd.to_numeric(scatter_frame[x_column], errors="coerce")
        scatter_frame[y_column] = pd.to_numeric(scatter_frame[y_column], errors="coerce")
        scatter_frame = scatter_frame.dropna().head(MAX_SCATTER_POINTS).reset_index(drop=True)
        if scatter_frame.empty:
            return None
        return ChartPayload(
            chart_type="scatter",
            title=recommendation.title,
            data=scatter_frame,
            x=x_column,
            y=y_column,
            aggregation=None,
            reason=recommendation.reason,
        )

    datetime_columns = infer_datetime_columns(normalized)
    if x_column in datetime_columns:
        time_frame = normalized[[x_column] + ([y_column] if y_column else [])].copy()
        time_frame[x_column] = pd.to_datetime(time_frame[x_column], errors="coerce", format="mixed")
        time_frame = time_frame.dropna(subset=[x_column])
        if y_column and y_column in time_frame.columns:
            time_frame[y_column] = pd.to_numeric(time_frame[y_column], errors="coerce")
            time_frame = time_frame.dropna(subset=[y_column])
            grouped = (
                time_frame.assign(**{x_column: time_frame[x_column].dt.floor("D")})
                .groupby(x_column, dropna=False)[y_column]
                .agg(lambda values: _aggregate(values, aggregation))
                .reset_index()
                .sort_values(x_column)
                .reset_index(drop=True)
            )
        else:
            grouped = (
                time_frame.assign(**{x_column: time_frame[x_column].dt.floor("D")})
                .groupby(x_column, dropna=False)
                .size()
                .reset_index(name="count")
                .sort_values(x_column)
                .reset_index(drop=True)
            )
            y_column = "count"
        if grouped.empty:
            return None
        return ChartPayload(
            chart_type="line" if actual_type == "auto" else actual_type,
            title=recommendation.title,
            data=grouped,
            x=x_column,
            y=y_column,
            aggregation=aggregation,
            reason=recommendation.reason,
        )

    category_frame = normalized[[x_column] + ([y_column] if y_column else [])].copy()
    category_frame[x_column] = category_frame[x_column].astype("string").fillna("Unknown")
    if y_column and y_column in category_frame.columns and is_numeric_dtype(category_frame[y_column]):
        grouped = (
            category_frame.groupby(x_column, dropna=False)[y_column]
            .agg(lambda values: _aggregate(values, aggregation))
            .reset_index()
            .sort_values(y_column, ascending=False)
            .head(MAX_BAR_CATEGORIES)
            .reset_index(drop=True)
        )
    elif y_column and y_column in category_frame.columns:
        grouped = (
            category_frame.groupby(x_column, dropna=False)[y_column]
            .count()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
            .head(MAX_BAR_CATEGORIES)
            .reset_index(drop=True)
        )
        y_column = "count"
    else:
        grouped = (
            category_frame.groupby(x_column, dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
            .head(MAX_BAR_CATEGORIES)
            .reset_index(drop=True)
        )
        y_column = "count"

    if grouped.empty:
        return None

    payload_chart_type = "bar" if actual_type == "auto" else actual_type
    return ChartPayload(
        chart_type=payload_chart_type,
        title=recommendation.title,
        data=grouped,
        x=x_column,
        y=y_column,
        aggregation=aggregation if payload_chart_type in {"bar", "line"} else None,
        reason=recommendation.reason,
    )


__all__ = [
    "ChartPayload",
    "ChartRecommendation",
    "build_chart_payload",
    "suggest_chart",
]
