from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype

from backend.dataset_understanding import analyze_dataset
from backend.dashboard_helpers import infer_datetime_columns
from backend.services.target_detector import detect_target_column


MAX_SAMPLE_VALUES = 5


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    if float(denominator or 0.0) <= 0.0:
        return 0.0
    return float(numerator) / float(denominator)


def _sample_values(series: pd.Series) -> list[str]:
    samples: list[str] = []
    for value in series.dropna().head(50).tolist():
        text = str(value).strip()
        if not text or text in samples:
            continue
        samples.append(text)
        if len(samples) >= MAX_SAMPLE_VALUES:
            break
    return samples


def _numeric_like_series(series: pd.Series) -> bool:
    if is_numeric_dtype(series):
        return False
    parsed = pd.to_numeric(series, errors="coerce")
    non_null = int(series.notna().sum())
    if non_null == 0:
        return False
    return _safe_ratio(int(parsed.notna().sum()), non_null) >= 0.75


def _semantic_type(series: pd.Series, *, column_name: str, datetime_columns: dict[str, pd.Series]) -> str:
    if column_name in datetime_columns or is_datetime64_any_dtype(series):
        return "datetime"
    if is_bool_dtype(series):
        return "boolean"
    if is_numeric_dtype(series):
        return "numeric"
    if _numeric_like_series(series):
        return "numeric_text"
    return "categorical"


def _outlier_count(series: pd.Series) -> int:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return 0
    q1 = float(numeric.quantile(0.25))
    q3 = float(numeric.quantile(0.75))
    iqr = q3 - q1
    if iqr <= 0 or np.isnan(iqr):
        return 0
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    return int(((numeric < lower_bound) | (numeric > upper_bound)).sum())


def _build_column_suggestions(
    series: pd.Series,
    *,
    column_name: str,
    semantic_type: str,
    missing_ratio: float,
    unique_count: int,
    row_count: int,
    datetime_columns: dict[str, pd.Series],
) -> list[str]:
    suggestions: list[str] = []
    if missing_ratio >= 0.2:
        suggestions.append(f"Handle missing values in {column_name}")
    if semantic_type == "numeric_text":
        suggestions.append(f"Convert {column_name} to numeric")
    if column_name in datetime_columns and not is_datetime64_any_dtype(series):
        suggestions.append(f"Convert {column_name} to datetime")
    if semantic_type == "numeric" and _outlier_count(series) > max(3, int(row_count * 0.02)):
        suggestions.append(f"Review outliers in {column_name}")
    if unique_count <= 1 and row_count > 0:
        suggestions.append(f"Drop constant column {column_name}")
    return suggestions


@dataclass(slots=True)
class ColumnProfile:
    name: str
    dtype: str
    semantic_type: str
    missing_count: int
    missing_pct: float
    unique_count: int
    outlier_count: int = 0
    sample_values: list[str] = field(default_factory=list)
    suggested_fixes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "semantic_type": self.semantic_type,
            "missing_count": self.missing_count,
            "missing_pct": round(self.missing_pct, 4),
            "unique_count": self.unique_count,
            "outlier_count": self.outlier_count,
            "sample_values": list(self.sample_values),
            "suggested_fixes": list(self.suggested_fixes),
        }


@dataclass(slots=True)
class DatasetProfileReport:
    dataset_name: str
    row_count: int
    column_count: int
    missing_cell_count: int
    duplicate_row_count: int
    quality_score: float
    dataset_type: str
    domain: str
    is_time_series: bool
    target_column: str | None
    target_type: str | None
    datetime_columns: list[str] = field(default_factory=list)
    numeric_columns: list[str] = field(default_factory=list)
    categorical_columns: list[str] = field(default_factory=list)
    column_profiles: list[ColumnProfile] = field(default_factory=list)
    suggested_fixes: list[str] = field(default_factory=list)
    insights: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "missing_cell_count": self.missing_cell_count,
            "duplicate_row_count": self.duplicate_row_count,
            "quality_score": round(self.quality_score, 2),
            "dataset_type": self.dataset_type,
            "domain": self.domain,
            "is_time_series": self.is_time_series,
            "target_column": self.target_column,
            "target_type": self.target_type,
            "datetime_columns": list(self.datetime_columns),
            "numeric_columns": list(self.numeric_columns),
            "categorical_columns": list(self.categorical_columns),
            "column_profiles": [profile.to_dict() for profile in self.column_profiles],
            "suggested_fixes": list(self.suggested_fixes),
            "insights": list(self.insights),
        }


def build_profile_report(df: pd.DataFrame, dataset_name: str = "dataset") -> DatasetProfileReport:
    normalized = df.copy(deep=True)
    normalized.columns = [str(column) for column in normalized.columns]
    row_count = int(len(normalized))
    column_count = int(len(normalized.columns))
    missing_cell_count = int(normalized.isna().sum().sum())
    duplicate_row_count = int(normalized.duplicated().sum())
    datetime_columns = infer_datetime_columns(normalized)
    dataset_context = analyze_dataset(normalized)
    target_details = detect_target_column(normalized)

    column_profiles: list[ColumnProfile] = []
    suggested_fixes: list[str] = []
    constant_columns = 0
    outlier_columns = 0

    for column in normalized.columns:
        series = normalized[column]
        semantic_type = _semantic_type(series, column_name=column, datetime_columns=datetime_columns)
        missing_count = int(series.isna().sum())
        missing_pct = _safe_ratio(missing_count, row_count)
        unique_count = int(series.nunique(dropna=True))
        outlier_count = _outlier_count(series) if semantic_type == "numeric" else 0
        if unique_count <= 1 and row_count > 0:
            constant_columns += 1
        if outlier_count > 0:
            outlier_columns += 1

        profile = ColumnProfile(
            name=column,
            dtype=str(series.dtype),
            semantic_type=semantic_type,
            missing_count=missing_count,
            missing_pct=missing_pct,
            unique_count=unique_count,
            outlier_count=outlier_count,
            sample_values=_sample_values(series),
            suggested_fixes=_build_column_suggestions(
                series,
                column_name=column,
                semantic_type=semantic_type,
                missing_ratio=missing_pct,
                unique_count=unique_count,
                row_count=row_count,
                datetime_columns=datetime_columns,
            ),
        )
        suggested_fixes.extend(profile.suggested_fixes)
        column_profiles.append(profile)

    total_cells = max(1, row_count * max(column_count, 1))
    missing_ratio = _safe_ratio(missing_cell_count, total_cells)
    duplicate_ratio = _safe_ratio(duplicate_row_count, row_count)
    constant_ratio = _safe_ratio(constant_columns, column_count)
    type_issue_count = sum(1 for profile in column_profiles if profile.semantic_type in {"numeric_text"})
    type_issue_ratio = _safe_ratio(type_issue_count, column_count)
    outlier_ratio = _safe_ratio(outlier_columns, max(1, len([profile for profile in column_profiles if profile.semantic_type == "numeric"])))

    quality_score = 10.0
    quality_score -= min(3.5, missing_ratio * 14.0)
    quality_score -= min(2.5, duplicate_ratio * 12.0)
    quality_score -= min(1.0, constant_ratio * 6.0)
    quality_score -= min(1.5, type_issue_ratio * 8.0)
    quality_score -= min(1.5, outlier_ratio * 2.5)
    quality_score = max(0.0, min(10.0, round(quality_score, 2)))

    deduped_suggested_fixes: list[str] = []
    for suggestion in suggested_fixes:
        if suggestion and suggestion not in deduped_suggested_fixes:
            deduped_suggested_fixes.append(suggestion)

    if duplicate_row_count > 0:
        deduped_suggested_fixes.append("Remove duplicate rows")
    if missing_cell_count > 0 and not any("Handle missing values" in suggestion for suggestion in deduped_suggested_fixes):
        deduped_suggested_fixes.append("Review missing values across key columns")

    insights: list[str] = []
    if row_count > 0 and column_count > 0:
        insights.append(f"{dataset_name} contains {row_count:,} rows across {column_count:,} columns.")
    if missing_cell_count > 0:
        insights.append(f"There are {missing_cell_count:,} missing cells that may need cleaning.")
    if duplicate_row_count > 0:
        insights.append(f"{duplicate_row_count:,} duplicate rows were detected.")
    if target_details.get("target"):
        insights.append(f"The strongest ML target candidate is {target_details['target']}.")
    if dataset_context.get("is_time_series"):
        insights.append("The dataset looks forecast-ready because it contains a usable time axis.")
    if not insights:
        insights.append("The dataset is structurally clean and ready for analysis.")

    numeric_columns = [profile.name for profile in column_profiles if profile.semantic_type == "numeric"]
    categorical_columns = [profile.name for profile in column_profiles if profile.semantic_type not in {"numeric", "datetime"}]

    return DatasetProfileReport(
        dataset_name=dataset_name,
        row_count=row_count,
        column_count=column_count,
        missing_cell_count=missing_cell_count,
        duplicate_row_count=duplicate_row_count,
        quality_score=quality_score,
        dataset_type=str(dataset_context.get("dataset_type") or dataset_context.get("domain") or "generic"),
        domain=str(dataset_context.get("domain") or "generic"),
        is_time_series=bool(dataset_context.get("is_time_series")),
        target_column=str(target_details.get("target") or "") or None,
        target_type=str(target_details.get("type") or "") or None,
        datetime_columns=list(datetime_columns.keys()),
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        column_profiles=column_profiles,
        suggested_fixes=deduped_suggested_fixes[:12],
        insights=insights[:8],
    )


__all__ = [
    "ColumnProfile",
    "DatasetProfileReport",
    "build_profile_report",
]
