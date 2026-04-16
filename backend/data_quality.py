from __future__ import annotations

from dataclasses import asdict
from typing import Any, Literal

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype

from backend.compat import dataclass
from backend.cleaning_engine import (
    build_cleaning_report,
    compute_quality_score as compute_cleaning_quality_score,
    detect_outliers as detect_cleaning_outliers,
    execute_cleaning_engine,
    validate_data as validate_cleaning_data,
)


MISSING_WARNING_RATIO = 0.4
MIXED_TYPE_MIN_RATIO = 0.2
MIXED_TYPE_MAX_RATIO = 0.95
TYPE_COERCE_SUCCESS_RATIO = 0.85
OUTLIER_WARNING_RATIO = 0.1
MIN_OUTLIER_SAMPLE = 8
INVALID_WARNING_RATIO = 0.15
MIN_ZSCORE_SAMPLE = 12
ZSCORE_THRESHOLD = 3.0
MAX_ANOMALY_SAMPLES = 25
_DATETIME_HINTS = ("date", "time", "day", "week", "month", "quarter", "year", "timestamp")
_BOOLEAN_TRUE_VALUES = {"true", "t", "yes", "y", "1"}
_BOOLEAN_FALSE_VALUES = {"false", "f", "no", "n", "0"}
_BOOLEAN_VALUES = _BOOLEAN_TRUE_VALUES | _BOOLEAN_FALSE_VALUES


@dataclass(slots=True)
class ValidationFinding:
    severity: Literal["warning", "error"]
    category: str
    message: str
    columns: list[str]
    suggested_fix: str | None = None


@dataclass(slots=True)
class CleaningOptions:
    parse_dates: bool = True
    coerce_numeric_text: bool = True
    trim_strings: bool = True
    drop_duplicates: bool = True
    fill_numeric_nulls: Literal["none", "mean", "median"] = "none"
    fill_text_nulls: Literal["none", "missing"] = "none"
    drop_null_rows_over: float = 1.0
    drop_null_columns_over: float = 1.0


@dataclass(slots=True)
class CleaningResult:
    dataframe: pd.DataFrame
    actions: list[str]
    report: dict[str, object] | None = None


def _safe_series_name(column_name: object) -> str:
    return str(column_name)


def _round_or_none(value: Any, digits: int = 6) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(parsed):
        return None
    return round(parsed, digits)


def _safe_percent(numerator: int | float, denominator: int | float) -> float:
    try:
        numerator_value = float(numerator)
        denominator_value = float(denominator)
    except (TypeError, ValueError):
        return 0.0
    if denominator_value <= 0:
        return 0.0
    return round((numerator_value / denominator_value) * 100, 2)


def _empty_invalid_summary() -> dict[str, float | int]:
    return {"count": 0, "percent": 0.0}


def _empty_outlier_summary(method: str = "iqr") -> dict[str, Any]:
    return {
        "method": method,
        "count": 0,
        "percent": 0.0,
        "lower_bound": None,
        "upper_bound": None,
        "z_threshold": None,
        "max_abs_zscore": None,
        "sample_indices": [],
    }


def _empty_profile() -> dict[str, Any]:
    return {
        "row_count": 0,
        "column_count": 0,
        "missing_percent": {},
        "unique_counts": {},
        "outliers": {},
        "anomalies": {},
        "invalid_values": {},
        "inconsistent_types": {},
        "column_types": {},
        "column_classification": {},
        "columns": {},
        "summary": {
            "numeric_columns": [],
            "categorical_columns": [],
            "datetime_columns": [],
            "boolean_columns": [],
            "columns_with_missing_values": [],
            "columns_with_invalid_values": [],
            "mixed_type_columns": [],
            "anomaly_columns": [],
        },
        "overall_missing_percent": 0.0,
        "invalid_summary": {"total_invalid_values": 0, "overall_invalid_percent": 0.0},
        "anomaly_summary": {"total_anomalies": 0, "overall_anomaly_percent": 0.0},
    }


def _normalize_text_series(series: pd.Series) -> pd.Series:
    if series.empty:
        return pd.Series(dtype="string")

    non_null = series[series.notna()]
    if non_null.empty:
        return pd.Series(dtype="string")

    try:
        normalized = non_null.astype("string").str.strip()
    except Exception:
        normalized = pd.Series(
            [str(value).strip() for value in non_null.tolist()],
            index=non_null.index,
            dtype="string",
        )

    normalized = normalized.replace({"": pd.NA})
    return normalized.dropna()


def _safe_numeric_conversion(series: pd.Series) -> pd.Series:
    if series.empty or is_bool_dtype(series):
        return pd.Series(np.nan, index=series.index, dtype="float64")

    try:
        if is_numeric_dtype(series):
            converted = pd.to_numeric(series, errors="coerce")
        else:
            normalized = series.astype("string").str.strip().replace({"": pd.NA}).str.replace(",", "", regex=False)
            converted = pd.to_numeric(normalized, errors="coerce")
    except Exception:
        return pd.Series(np.nan, index=series.index, dtype="float64")

    converted_series = pd.Series(converted, index=series.index, dtype="float64")
    finite_mask = np.isfinite(converted_series.to_numpy(dtype="float64", na_value=np.nan))
    return converted_series.where(finite_mask, np.nan)


def _safe_datetime_conversion(series: pd.Series) -> pd.Series:
    if series.empty:
        return pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")

    try:
        if is_datetime64_any_dtype(series):
            converted = pd.to_datetime(series, errors="coerce")
        else:
            normalized = series.astype("string").str.strip().replace({"": pd.NA})
            try:
                converted = pd.to_datetime(normalized, errors="coerce", format="mixed")
            except Exception:
                converted = pd.to_datetime(normalized, errors="coerce")
    except Exception:
        return pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")

    return pd.Series(converted, index=series.index)


def _boolean_valid_mask(series: pd.Series) -> pd.Series:
    text = _normalize_text_series(series)
    if text.empty:
        return pd.Series(dtype="bool")
    return text.str.lower().isin(_BOOLEAN_VALUES)


def _has_datetime_hint(column_name: object) -> bool:
    lowered_name = _safe_series_name(column_name).strip().lower()
    return any(token in lowered_name for token in _DATETIME_HINTS)


def _infer_column_classification(series: pd.Series, column_name: object) -> dict[str, Any]:
    non_null = series[series.notna()]
    observed_types = sorted({type(value).__name__ for value in non_null.tolist()})

    if non_null.empty:
        return {
            "classification": "categorical",
            "numeric_ratio": 0.0,
            "datetime_ratio": 0.0,
            "boolean_ratio": 0.0,
            "observed_types": observed_types,
        }

    if is_bool_dtype(series):
        return {
            "classification": "boolean",
            "numeric_ratio": 0.0,
            "datetime_ratio": 0.0,
            "boolean_ratio": 1.0,
            "observed_types": observed_types,
        }

    if is_datetime64_any_dtype(series):
        return {
            "classification": "datetime",
            "numeric_ratio": 0.0,
            "datetime_ratio": 1.0,
            "boolean_ratio": 0.0,
            "observed_types": observed_types,
        }

    if is_numeric_dtype(series):
        return {
            "classification": "numeric",
            "numeric_ratio": 1.0,
            "datetime_ratio": 0.0,
            "boolean_ratio": 0.0,
            "observed_types": observed_types,
        }

    normalized_text = _normalize_text_series(series)
    if normalized_text.empty:
        return {
            "classification": "categorical",
            "numeric_ratio": 0.0,
            "datetime_ratio": 0.0,
            "boolean_ratio": 0.0,
            "observed_types": observed_types,
        }

    numeric_ratio = float(
        pd.to_numeric(normalized_text.str.replace(",", "", regex=False), errors="coerce").notna().mean()
    )
    datetime_ratio = float(_safe_datetime_conversion(normalized_text).notna().mean())
    normalized_boolean = normalized_text.str.lower()
    boolean_ratio = float(normalized_boolean.isin(_BOOLEAN_VALUES).mean())

    classification = "categorical"
    if boolean_ratio >= TYPE_COERCE_SUCCESS_RATIO and int(normalized_boolean[normalized_boolean.isin(_BOOLEAN_VALUES)].nunique()) <= 2:
        classification = "boolean"
    elif datetime_ratio >= TYPE_COERCE_SUCCESS_RATIO and (datetime_ratio > numeric_ratio or _has_datetime_hint(column_name)):
        classification = "datetime"
    elif numeric_ratio >= TYPE_COERCE_SUCCESS_RATIO:
        classification = "numeric"

    return {
        "classification": classification,
        "numeric_ratio": round(numeric_ratio, 4),
        "datetime_ratio": round(datetime_ratio, 4),
        "boolean_ratio": round(boolean_ratio, 4),
        "observed_types": observed_types,
    }


def _infer_invalid_values(
    series: pd.Series,
    classification: str,
    inference: dict[str, Any] | None = None,
) -> dict[str, float | int]:
    inference = dict(inference or {})
    non_null_count = int(series.notna().sum())
    if non_null_count == 0:
        return _empty_invalid_summary()

    valid_count = non_null_count
    if classification == "numeric":
        valid_count = int(_safe_numeric_conversion(series).notna().sum())
    elif classification == "datetime":
        valid_count = int(_safe_datetime_conversion(series).notna().sum())
    elif classification == "boolean" and not is_bool_dtype(series):
        valid_count = int(_boolean_valid_mask(series).sum())
    elif classification == "categorical":
        numeric_ratio = float(inference.get("numeric_ratio") or 0.0)
        datetime_ratio = float(inference.get("datetime_ratio") or 0.0)
        boolean_ratio = float(inference.get("boolean_ratio") or 0.0)
        if numeric_ratio >= MIXED_TYPE_MIN_RATIO and numeric_ratio >= max(datetime_ratio, boolean_ratio):
            valid_count = int(_safe_numeric_conversion(series).notna().sum())
        elif datetime_ratio >= MIXED_TYPE_MIN_RATIO and datetime_ratio >= boolean_ratio:
            valid_count = int(_safe_datetime_conversion(series).notna().sum())
        elif boolean_ratio >= MIXED_TYPE_MIN_RATIO:
            valid_count = int(_boolean_valid_mask(series).sum())

    invalid_count = max(non_null_count - valid_count, 0)
    return {
        "count": invalid_count,
        "percent": _safe_percent(invalid_count, non_null_count),
    }


def _infer_inconsistent_type_details(
    series: pd.Series,
    classification: str,
    inference: dict[str, Any] | None = None,
    invalid_values: dict[str, float | int] | None = None,
) -> dict[str, float | int | list[str] | None]:
    inference = dict(inference or {})
    invalid_values = dict(invalid_values or {})
    non_null_count = int(series.notna().sum())
    invalid_count = int(invalid_values.get("count") or 0)
    observed_types = list(inference.get("observed_types") or [])
    numeric_ratio = float(inference.get("numeric_ratio") or 0.0)
    datetime_ratio = float(inference.get("datetime_ratio") or 0.0)
    boolean_ratio = float(inference.get("boolean_ratio") or 0.0)

    issue = "consistent"
    if classification == "numeric" and invalid_count:
        issue = "mixed_numeric_text"
    elif classification == "datetime" and invalid_count:
        issue = "mixed_datetime_text"
    elif classification == "boolean" and invalid_count:
        issue = "mixed_boolean_text"
    elif classification == "categorical":
        if MIXED_TYPE_MIN_RATIO < numeric_ratio < MIXED_TYPE_MAX_RATIO:
            issue = "mixed_numeric_text"
        elif MIXED_TYPE_MIN_RATIO < datetime_ratio < MIXED_TYPE_MAX_RATIO:
            issue = "mixed_datetime_text"
        elif MIXED_TYPE_MIN_RATIO < boolean_ratio < MIXED_TYPE_MAX_RATIO:
            issue = "mixed_boolean_text"
        elif len(observed_types) > 1:
            issue = "mixed_python_types"

    valid_count = max(non_null_count - invalid_count, 0)
    return {
        "issue": issue,
        "numeric_ratio": round(numeric_ratio, 4),
        "datetime_ratio": round(datetime_ratio, 4),
        "boolean_ratio": round(boolean_ratio, 4),
        "observed_types": observed_types,
        "valid_count": valid_count,
        "valid_percent": _safe_percent(valid_count, non_null_count),
        "invalid_count": invalid_count,
        "invalid_percent": _safe_percent(invalid_count, non_null_count),
    }


def _detect_numeric_outliers(series: pd.Series) -> dict[str, Any]:
    if is_bool_dtype(series):
        return _empty_outlier_summary()

    numeric_series = _safe_numeric_conversion(series).dropna()
    if numeric_series.shape[0] < MIN_OUTLIER_SAMPLE:
        return _empty_outlier_summary()

    method = "iqr"
    iqr_mask = pd.Series(False, index=numeric_series.index)
    lower_bound: float | None = None
    upper_bound: float | None = None

    try:
        q1 = float(numeric_series.quantile(0.25))
        q3 = float(numeric_series.quantile(0.75))
        iqr = float(q3 - q1)
    except Exception:
        iqr = 0.0

    if np.isfinite(iqr) and iqr > 0:
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        iqr_mask = (numeric_series < lower_bound) | (numeric_series > upper_bound)

    z_mask = pd.Series(False, index=numeric_series.index)
    z_threshold: float | None = None
    max_abs_zscore: float | None = None
    if numeric_series.shape[0] >= MIN_ZSCORE_SAMPLE:
        std_value = float(numeric_series.std(ddof=0))
        if np.isfinite(std_value) and std_value > 0:
            z_scores = (numeric_series - float(numeric_series.mean())) / std_value
            absolute_z_scores = z_scores.abs()
            z_threshold = ZSCORE_THRESHOLD
            max_abs_zscore = float(absolute_z_scores.max()) if not absolute_z_scores.empty else None
            z_mask = absolute_z_scores >= ZSCORE_THRESHOLD
            method = "iqr_zscore" if lower_bound is not None else "zscore"

    outlier_mask = iqr_mask | z_mask
    if lower_bound is None and upper_bound is None and z_threshold is None:
        return _empty_outlier_summary()

    return {
        "method": method,
        "count": int(outlier_mask.sum()),
        "percent": _safe_percent(int(outlier_mask.sum()), int(numeric_series.shape[0])),
        "lower_bound": _round_or_none(lower_bound),
        "upper_bound": _round_or_none(upper_bound),
        "z_threshold": _round_or_none(z_threshold, digits=2),
        "max_abs_zscore": _round_or_none(max_abs_zscore, digits=4),
        "sample_indices": [str(index) for index in numeric_series.index[outlier_mask].tolist()[:MAX_ANOMALY_SAMPLES]],
    }


def profile_data(df: pd.DataFrame | None) -> dict:
    if df is None or not isinstance(df, pd.DataFrame):
        return _empty_profile()

    missing_percent: dict[str, float] = {}
    unique_counts: dict[str, int] = {}
    outliers: dict[str, dict[str, Any]] = {}
    invalid_values: dict[str, dict[str, float | int]] = {}
    inconsistent_types: dict[str, dict[str, float | int | list[str] | None]] = {}
    column_types: dict[str, str] = {}
    column_classification: dict[str, str] = {}
    columns: dict[str, dict[str, Any]] = {}
    summary = {
        "numeric_columns": [],
        "categorical_columns": [],
        "datetime_columns": [],
        "boolean_columns": [],
        "columns_with_missing_values": [],
        "columns_with_invalid_values": [],
        "mixed_type_columns": [],
        "anomaly_columns": [],
    }
    total_invalid_values = 0
    total_numeric_observations = 0

    for column_name in df.columns:
        series = df[column_name]
        resolved_name = _safe_series_name(column_name)
        missing_count = int(series.isna().sum())
        missing_percent[resolved_name] = _safe_percent(missing_count, len(df))
        unique_counts[resolved_name] = int(series.nunique(dropna=True))
        column_types[resolved_name] = str(series.dtype)

        inference = _infer_column_classification(series, column_name)
        classification = str(inference.get("classification") or "categorical")
        column_classification[resolved_name] = classification
        summary[f"{classification}_columns"].append(resolved_name)

        invalid_summary = _infer_invalid_values(series, classification, inference=inference)
        invalid_values[resolved_name] = invalid_summary
        total_invalid_values += int(invalid_summary.get("count") or 0)
        if int(invalid_summary.get("count") or 0):
            summary["columns_with_invalid_values"].append(resolved_name)

        consistency = _infer_inconsistent_type_details(
            series,
            classification,
            inference=inference,
            invalid_values=invalid_summary,
        )
        if str(consistency.get("issue") or "consistent") != "consistent":
            inconsistent_types[resolved_name] = consistency
            summary["mixed_type_columns"].append(resolved_name)

        if missing_count:
            summary["columns_with_missing_values"].append(resolved_name)

        anomaly_summary = _empty_outlier_summary()
        if classification == "numeric":
            numeric_non_null = int(_safe_numeric_conversion(series).notna().sum())
            total_numeric_observations += numeric_non_null
            anomaly_summary = _detect_numeric_outliers(series)
            outliers[resolved_name] = anomaly_summary
            if int(anomaly_summary.get("count") or 0):
                summary["anomaly_columns"].append(resolved_name)

        columns[resolved_name] = {
            "dtype": column_types[resolved_name],
            "classification": classification,
            "missing_count": missing_count,
            "missing_percent": missing_percent[resolved_name],
            "non_null_count": int(series.notna().sum()),
            "unique_values": unique_counts[resolved_name],
            "invalid_values": invalid_summary,
            "data_type_consistency": consistency,
            "anomaly_summary": anomaly_summary,
        }

    overall_missing_percent = 0.0
    if len(df) and len(df.columns):
        overall_missing_percent = _safe_percent(int(df.isna().sum().sum()), len(df) * len(df.columns))

    total_non_null_cells = int(df.notna().sum().sum()) if len(df.columns) else 0
    total_anomalies = sum(int((details or {}).get("count") or 0) for details in outliers.values())

    return {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "missing_percent": missing_percent,
        "unique_counts": unique_counts,
        "outliers": outliers,
        "anomalies": outliers,
        "invalid_values": invalid_values,
        "inconsistent_types": inconsistent_types,
        "column_types": column_types,
        "column_classification": column_classification,
        "columns": columns,
        "summary": summary,
        "overall_missing_percent": overall_missing_percent,
        "invalid_summary": {
            "total_invalid_values": total_invalid_values,
            "overall_invalid_percent": _safe_percent(total_invalid_values, total_non_null_cells),
        },
        "anomaly_summary": {
            "total_anomalies": total_anomalies,
            "overall_anomaly_percent": _safe_percent(total_anomalies, total_numeric_observations),
        },
    }


def _compute_data_quality_ratio(df_profile: dict | None) -> float:
    profile = dict(df_profile or {})
    row_count = int(profile.get("row_count") or 0)
    column_count = int(profile.get("column_count") or 0)
    if row_count == 0 or column_count == 0:
        return 0.0

    overall_missing_ratio = float(profile.get("overall_missing_percent") or 0.0) / 100.0
    invalid_ratio = float(dict(profile.get("invalid_summary") or {}).get("overall_invalid_percent") or 0.0) / 100.0
    anomaly_ratio = float(dict(profile.get("anomaly_summary") or {}).get("overall_anomaly_percent") or 0.0) / 100.0
    inconsistent_ratio = len(dict(profile.get("inconsistent_types") or {})) / max(column_count, 1)

    completeness_score = max(0.0, 1.0 - overall_missing_ratio)
    validity_score = max(0.0, 1.0 - min(1.0, invalid_ratio * 1.5))
    anomaly_score = max(0.0, 1.0 - min(1.0, anomaly_ratio * 1.25))
    consistency_score = max(0.0, 1.0 - inconsistent_ratio)

    score = (
        (0.35 * completeness_score)
        + (0.35 * validity_score)
        + (0.2 * anomaly_score)
        + (0.1 * consistency_score)
    )

    if row_count < 12:
        score *= 0.9
    elif row_count < 30:
        score *= 0.95

    return round(max(0.0, min(1.0, score)), 2)


def generate_data_warnings(df_profile: dict | None) -> list[str]:
    profile = dict(df_profile or {})
    warnings: list[str] = []

    for column_name, missing_value in dict(profile.get("missing_percent") or {}).items():
        try:
            resolved_missing = float(missing_value)
        except (TypeError, ValueError):
            continue
        if resolved_missing >= 40.0:
            warnings.append(f"{resolved_missing:.0f}% missing in column {column_name}")

    for column_name, invalid_details in dict(profile.get("invalid_values") or {}).items():
        invalid_count = int((invalid_details or {}).get("count") or 0)
        invalid_percent = float((invalid_details or {}).get("percent") or 0.0)
        if invalid_count and invalid_percent >= (INVALID_WARNING_RATIO * 100):
            warnings.append(
                f"Column {column_name} contains {invalid_count} invalid values ({invalid_percent:.0f}% of non-null rows)"
            )

    for column_name, details in dict(profile.get("inconsistent_types") or {}).items():
        issue = str((details or {}).get("issue") or "")
        if issue == "mixed_numeric_text":
            warnings.append(f"Column {column_name} has mixed numeric and text values")
        elif issue == "mixed_datetime_text":
            warnings.append(f"Column {column_name} has mixed date and text values")
        elif issue == "mixed_boolean_text":
            warnings.append(f"Column {column_name} has mixed boolean-like and text values")
        else:
            warnings.append(f"Column {column_name} has mixed types")

    for column_name, details in dict(profile.get("anomalies") or {}).items():
        anomaly_count = int((details or {}).get("count") or 0)
        anomaly_percent = float((details or {}).get("percent") or 0.0)
        if anomaly_count and anomaly_percent >= (OUTLIER_WARNING_RATIO * 100):
            warnings.append(
                f"Column {column_name} contains {anomaly_count} potential anomalies ({anomaly_percent:.0f}% of numeric rows)"
            )

    quality_ratio = _compute_data_quality_ratio(profile)
    if quality_ratio < 0.65:
        warnings.append(
            f"Data quality score is {quality_ratio:.2f}, so ML predictions should be treated cautiously."
        )

    row_count = int(profile.get("row_count") or 0)
    if 0 < row_count < 12:
        warnings.append("Dataset is small, so model reliability may be limited")

    return list(dict.fromkeys(warnings))


def compute_data_quality(df_profile: dict | None) -> float:
    return round(_compute_data_quality_ratio(df_profile) * 10.0, 2)


def build_data_quality_report(df: pd.DataFrame | None) -> dict:
    profile = profile_data(df)
    quality_ratio = _compute_data_quality_ratio(profile)
    warnings = generate_data_warnings(profile)
    return {
        "data_profile": profile,
        "data_quality_score": quality_ratio,
        "warnings": warnings,
        "anomalies": dict(profile.get("anomalies") or {}),
        "profile": profile,
        "score": round(quality_ratio * 10.0, 2),
        "issues": warnings,
    }


def finding_to_dict(finding: ValidationFinding) -> dict:
    return asdict(finding)


def has_blocking_findings(findings: list[ValidationFinding]) -> bool:
    return any(finding.severity == "error" for finding in findings)


def summarize_findings(findings: list[ValidationFinding]) -> tuple[int, int]:
    warning_count = sum(1 for finding in findings if finding.severity == "warning")
    error_count = sum(1 for finding in findings if finding.severity == "error")
    return warning_count, error_count


def create_load_failure_finding(error_message: str) -> ValidationFinding:
    return ValidationFinding(
        severity="error",
        category="source_load_failed",
        message=str(error_message),
        columns=[],
        suggested_fix="Check the connection details or source file and try again.",
    )


def _build_missing_findings(df: pd.DataFrame) -> list[ValidationFinding]:
    findings: list[ValidationFinding] = []

    for column_name in df.columns:
        missing_ratio = float(df[column_name].isna().mean()) if len(df) else 0.0
        if missing_ratio == 1.0:
            findings.append(
                ValidationFinding(
                    severity="error",
                    category="all_values_missing",
                    message=f"Column '{column_name}' is entirely empty.",
                    columns=[str(column_name)],
                    suggested_fix="Drop the empty column or replace it with a populated source field.",
                )
            )
            continue

        if missing_ratio >= MISSING_WARNING_RATIO:
            findings.append(
                ValidationFinding(
                    severity="warning",
                    category="high_missingness",
                    message=(
                        f"Column '{column_name}' has {missing_ratio:.0%} missing values, "
                        "which may make downstream analysis unreliable."
                    ),
                    columns=[str(column_name)],
                    suggested_fix="Impute the column or drop rows/columns with high missingness.",
                )
            )

    return findings


def _infer_mixed_type_findings(df: pd.DataFrame) -> list[ValidationFinding]:
    findings: list[ValidationFinding] = []

    for column_name in df.columns:
        series = df[column_name]
        if is_numeric_dtype(series) or is_datetime64_any_dtype(series) or is_bool_dtype(series):
            continue

        non_null = series.dropna().astype("string").str.strip()
        non_null = non_null[non_null != ""]
        if non_null.empty:
            continue

        numeric_ratio = float(pd.to_numeric(non_null, errors="coerce").notna().mean())
        datetime_ratio = float(pd.to_datetime(non_null, errors="coerce", format="mixed").notna().mean())

        if MIXED_TYPE_MIN_RATIO < numeric_ratio < MIXED_TYPE_MAX_RATIO:
            findings.append(
                ValidationFinding(
                    severity="warning",
                    category="mixed_numeric_text",
                    message=(
                        f"Column '{column_name}' mixes numeric-looking values with text, "
                        "so calculations may behave inconsistently."
                    ),
                    columns=[str(column_name)],
                    suggested_fix="Coerce numeric text or split the column into separate fields.",
                )
            )
            continue

        if MIXED_TYPE_MIN_RATIO < datetime_ratio < MIXED_TYPE_MAX_RATIO:
            findings.append(
                ValidationFinding(
                    severity="warning",
                    category="mixed_datetime_text",
                    message=(
                        f"Column '{column_name}' mixes datetime-like values with plain text, "
                        "so date operations may be unreliable."
                    ),
                    columns=[str(column_name)],
                    suggested_fix="Enable date parsing or clean inconsistent date values.",
                )
            )

    return findings


def validate_dataframe(df: pd.DataFrame) -> list[ValidationFinding]:
    findings: list[ValidationFinding] = []

    if df is None:
        return [
            ValidationFinding(
                severity="error",
                category="missing_dataset",
                message="No dataset is loaded.",
                columns=[],
                suggested_fix="Load a source before running validation.",
            )
        ]

    if df.empty:
        findings.append(
            ValidationFinding(
                severity="error",
                category="empty_dataset",
                message="The loaded dataset has no rows.",
                columns=[],
                suggested_fix="Choose a source that returns at least one row.",
            )
        )

    if len(df.columns) == 0:
        findings.append(
            ValidationFinding(
                severity="error",
                category="missing_columns",
                message="The loaded dataset has no columns.",
                columns=[],
                suggested_fix="Choose a valid table, query, or file with tabular data.",
            )
        )

    duplicate_count = int(df.duplicated().sum()) if len(df) else 0
    if duplicate_count:
        findings.append(
            ValidationFinding(
                severity="warning",
                category="duplicate_rows",
                message=f"The dataset contains {duplicate_count:,} duplicate rows.",
                columns=[],
                suggested_fix="Enable duplicate removal before analysis.",
            )
        )

    if len(df) < 2:
        findings.append(
            ValidationFinding(
                severity="warning",
                category="small_dataset",
                message="The dataset has fewer than two rows, so trend analysis may be limited.",
                columns=[],
                suggested_fix="Use a larger dataset when possible.",
            )
        )

    findings.extend(_build_missing_findings(df))
    findings.extend(_infer_mixed_type_findings(df))
    return findings


def _trim_strings(df: pd.DataFrame) -> list[str]:
    actions: list[str] = []
    for column_name in df.columns:
        series = df[column_name]
        if is_numeric_dtype(series) or is_datetime64_any_dtype(series) or is_bool_dtype(series):
            continue
        trimmed = series.astype("string").str.strip()
        if not trimmed.equals(series.astype("string")):
            df[column_name] = trimmed.replace({"": pd.NA})
            actions.append(f"Trimmed whitespace in '{column_name}'.")
    return actions


def _coerce_datetime_series(series: pd.Series) -> pd.Series | None:
    non_null = series.dropna()
    if non_null.empty or is_numeric_dtype(series) or is_bool_dtype(series):
        return None

    parsed = pd.to_datetime(series.astype("string"), errors="coerce", format="mixed")
    success_ratio = float(parsed.dropna().shape[0] / non_null.shape[0])
    if success_ratio >= TYPE_COERCE_SUCCESS_RATIO:
        return parsed
    return None


def _coerce_numeric_series(series: pd.Series) -> pd.Series | None:
    non_null = series.dropna()
    if non_null.empty or is_bool_dtype(series) or is_datetime64_any_dtype(series) or is_numeric_dtype(series):
        return None

    normalized = series.astype("string").str.replace(",", "", regex=False)
    coerced = pd.to_numeric(normalized, errors="coerce")
    success_ratio = float(coerced.dropna().shape[0] / non_null.shape[0])
    if success_ratio >= TYPE_COERCE_SUCCESS_RATIO:
        return coerced
    return None


def apply_cleaning_plan(df: pd.DataFrame, options: CleaningOptions) -> CleaningResult:
    original_df = df.copy()
    cleaned_df = df.copy()
    actions: list[str] = []

    if options.trim_strings:
        actions.extend(_trim_strings(cleaned_df))

    if options.parse_dates:
        for column_name in cleaned_df.columns:
            parsed_series = _coerce_datetime_series(cleaned_df[column_name])
            if parsed_series is not None:
                cleaned_df[column_name] = parsed_series
                actions.append(f"Parsed '{column_name}' as datetime.")

    if options.coerce_numeric_text:
        for column_name in cleaned_df.columns:
            coerced_series = _coerce_numeric_series(cleaned_df[column_name])
            if coerced_series is not None:
                cleaned_df[column_name] = coerced_series
                actions.append(f"Coerced '{column_name}' from text to numeric.")

    drop_null_columns_over = 0.5
    if options.drop_null_columns_over < 1.0:
        drop_null_columns_over = min(drop_null_columns_over, float(options.drop_null_columns_over))

    numeric_fill_strategy = "median"
    if options.fill_numeric_nulls == "mean":
        numeric_fill_strategy = "mean"

    text_fill_value = "Unknown"
    if options.fill_text_nulls == "missing":
        text_fill_value = "Missing"

    engine_result = execute_cleaning_engine(
        cleaned_df,
        numeric_fill_strategy=numeric_fill_strategy,
        text_fill_value=text_fill_value,
        drop_columns_over=drop_null_columns_over,
    )
    cleaned_df = engine_result.cleaned_df
    actions.extend(engine_result.actions)

    if options.drop_null_columns_over < 0.5:
        column_missing = cleaned_df.isna().mean()
        columns_to_drop = column_missing[column_missing >= options.drop_null_columns_over].index.tolist()
        if columns_to_drop:
            cleaned_df = cleaned_df.drop(columns=columns_to_drop)
            actions.append(
                "Dropped null-heavy columns: " + ", ".join(str(column_name) for column_name in columns_to_drop) + "."
            )

    if options.drop_null_rows_over < 1.0 and not cleaned_df.empty:
        row_missing_ratio = cleaned_df.isna().mean(axis=1)
        rows_to_drop = int((row_missing_ratio >= options.drop_null_rows_over).sum())
        if rows_to_drop:
            cleaned_df = cleaned_df.loc[row_missing_ratio < options.drop_null_rows_over].reset_index(drop=True)
            actions.append(f"Dropped {rows_to_drop:,} null-heavy rows.")

    cleaned_df = cleaned_df.reset_index(drop=True)
    report = build_cleaning_report(
        original_df,
        cleaned_df,
        outliers=detect_cleaning_outliers(cleaned_df),
        issues=validate_cleaning_data(cleaned_df),
        quality_score=compute_cleaning_quality_score(cleaned_df),
        actions=actions,
    )

    return CleaningResult(
        dataframe=cleaned_df,
        actions=actions,
        report=report,
    )
