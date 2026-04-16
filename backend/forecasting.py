from __future__ import annotations

import json
import math
import pickle
import re
from dataclasses import asdict, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge

from backend.aidssist_runtime.serialization import deserialize_result, serialize_result
from backend.aidssist_runtime.storage import build_object_key, get_object_store
from backend.compat import dataclass
from backend.data_quality import build_data_quality_report
from backend.question_engine import build_question_payload
from backend.services.data_intelligence import detect_dataset_type
from backend.services.decision_engine import build_decision_layer, build_forecast_recommendations
from backend.services.failure_logging import get_failure_patterns, log_failure
from backend.services.learning_engine import get_learning_patterns
from backend.services.limitations import build_limitations
from backend.services.model_quality import build_explanation, evaluate_model_with_warnings, interpret_model_quality
from backend.services.result_consistency import build_forecast_consistency, build_reproducibility_metadata
from backend.services.trust_layer import build_risk_statement
from backend.time_filter_service import apply_time_filter, build_time_filter_options
from backend.workflow_store import WorkflowStore


SUPPORTED_FREQUENCIES = ("auto", "D", "W", "M", "Q")
SUPPORTED_HORIZONS = ("next_week", "next_month", "next_quarter", "next_year")
STANDARDIZED_FORECAST_WINDOWS = ("current_month", "last_month", "next_week", "next_month", "next_quarter", "next_year")
SUPPORTED_MODEL_STRATEGIES = ("hybrid", "explainable", "accuracy")
SUPPORTED_TRAINING_MODES = ("auto", "local", "background")
DEFAULT_CONFIDENCE_LEVEL = 0.95
FORECAST_MAPPING_SAMPLE_ROWS = 5000
IDENTIFIER_HINT_TOKENS = ("id", "key", "code", "sku", "zip", "postal", "account", "row", "index")
DATETIME_HINT_TOKENS = ("date", "time", "timestamp", "month", "week", "day", "year", "period")
STRICT_DATE_DETECTION_SUCCESS_RATIO = 0.85
TIME_SERIES_ELIGIBILITY_SUCCESS_RATIO = 0.7
MIN_DATETIME_UNIQUE_VALUES = 3
NON_TIME_SERIES_SUGGESTIONS = [
    "Use analysis mode instead",
    "Upload time-based dataset",
    "Try correlation analysis",
    "Try clustering or segmentation analysis",
    "Explore feature importance",
    "Run classification on labeled outcomes",
]
MIN_HISTORY_POINTS = {
    "D": 30,
    "W": 16,
    "M": 12,
    "Q": 8,
}
HORIZON_PERIOD_MAP = {
    "D": {"next_week": 7, "next_month": 30, "next_quarter": 90, "next_year": 365},
    "W": {"next_week": 1, "next_month": 4, "next_quarter": 13, "next_year": 52},
    "M": {"next_month": 1, "next_quarter": 3, "next_year": 12},
    "Q": {"next_quarter": 1, "next_year": 4},
}
FREQUENCY_LABELS = {
    "auto": "Auto detect",
    "D": "Daily",
    "W": "Weekly",
    "M": "Monthly",
    "Q": "Quarterly",
}
HORIZON_LABELS = {
    "next_week": "Next week",
    "next_month": "Next month",
    "next_quarter": "Next quarter",
    "next_year": "Next year",
}


class ForecastError(RuntimeError):
    """Raised when forecasting cannot be completed safely."""


@dataclass(slots=True)
class ForecastConfig:
    date_column: str = ""
    target_column: str = ""
    driver_columns: list[str] = field(default_factory=list)
    aggregation_frequency: str = "auto"
    horizon: str = "next_month"
    model_strategy: str = "hybrid"
    training_mode: str = "auto"


@dataclass(slots=True)
class ForecastValidationResult:
    errors: list[str]
    warnings: list[str]
    resolved_frequency: str | None
    history_points: int
    minimum_history_points: int
    compatible_horizons: list[str]
    date_start: str | None
    date_end: str | None

    @property
    def is_valid(self) -> bool:
        return not self.errors


@dataclass(slots=True)
class PreparedForecastData:
    dataframe: pd.DataFrame
    resolved_frequency: str
    horizon_periods: int
    future_dates: pd.DatetimeIndex


def _forecast_fix_suggestion(error_message: str) -> str:
    normalized = str(error_message or "").strip()
    suggestion_map = {
        "We couldn't detect a time column. Please select one.": "Add a parseable date column such as order_date, date, or timestamp.",
        "No valid time column detected. Forecasting requires time-based data.": "Use analysis mode instead, upload a time-based dataset, or add a reliable date column.",
        "No valid time column detected": "Use analysis mode instead, upload a time-based dataset, or add a reliable date column.",
        "We couldn't detect a primary KPI. Please select one numeric column.": "Choose one numeric metric to forecast, such as sales or revenue.",
        "We couldn't infer a reliable forecast frequency from this dataset. Try a manual override in Advanced options.": "Pick Daily, Weekly, Monthly, or Quarterly manually in Advanced options.",
        "No valid date column found.": "Add a parseable date column such as order_date, date, or timestamp.",
        "No valid numeric target column found.": "Choose one numeric metric to forecast, such as sales or revenue.",
        "Aidssist could not infer a safe forecast frequency.": "Pick Daily, Weekly, Monthly, or Quarterly manually in Advanced options.",
    }
    if "minimum" in normalized.lower() or "history" in normalized.lower():
        return "Add more historical rows or choose a coarser forecast frequency before retrying."
    return suggestion_map.get(
        normalized,
        "Review the forecast mapping and ensure the dataset has clean time-series inputs.",
    )


def _user_facing_forecast_error(error_message: str) -> str:
    normalized = str(error_message or "").strip()
    if normalized == "No valid time column detected. Forecasting requires time-based data.":
        return "We couldn't detect a time column. Please select one."
    if normalized == "No valid time column detected":
        return "We couldn't detect a time column. Please select one."
    if normalized in {"No valid date column found.", "Choose a date column for forecasting."}:
        return "We couldn't detect a time column. Please select one."
    if normalized in {
        "No valid numeric target column found.",
        "Choose one primary target metric to forecast.",
    }:
        return "We couldn't detect a primary KPI. Please select one numeric column."
    if "could not be parsed safely as dates" in normalized:
        column_match = re.search(r"Column '(.+?)'", normalized)
        column_name = column_match.group(1) if column_match else "that column"
        return f"We couldn't parse '{column_name}' as a time column. Please select another one."
    if "does not contain usable numeric target values" in normalized:
        column_match = re.search(r"Column '(.+?)'", normalized)
        column_name = column_match.group(1) if column_match else "that column"
        return f"'{column_name}' doesn't contain enough numeric values to forecast. Please choose another KPI."
    if "Aidssist could not infer a safe forecast frequency" in normalized:
        return "We couldn't infer a reliable forecast frequency from this dataset. Try a manual override in Advanced options."
    return normalized


def forecast_config_to_dict(config: ForecastConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["driver_columns"] = list(config.driver_columns)
    return payload


def forecast_config_from_dict(payload: dict[str, Any] | None) -> ForecastConfig:
    payload = dict(payload or {})
    driver_columns = payload.get("driver_columns") or []
    if not isinstance(driver_columns, list):
        driver_columns = list(driver_columns)
    return ForecastConfig(
        date_column=str(payload.get("date_column") or ""),
        target_column=str(payload.get("target_column") or ""),
        driver_columns=[str(column) for column in driver_columns if str(column or "").strip()],
        aggregation_frequency=str(payload.get("aggregation_frequency") or "auto"),
        horizon=str(payload.get("horizon") or ""),
        model_strategy=str(payload.get("model_strategy") or ""),
        training_mode=str(payload.get("training_mode") or ""),
    )


def build_forecast_config_signature(config: ForecastConfig | dict[str, Any] | None) -> str:
    if isinstance(config, ForecastConfig):
        payload = forecast_config_to_dict(config)
    else:
        payload = forecast_config_to_dict(forecast_config_from_dict(config))
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def get_forecast_horizon_options() -> list[dict[str, str]]:
    return [
        {"value": horizon, "label": HORIZON_LABELS[horizon]}
        for horizon in SUPPORTED_HORIZONS
    ]


def _normalize_token(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _has_datetime_name_hint(column_name: object | None) -> bool:
    normalized = _normalize_token(column_name)
    return any(token in normalized for token in DATETIME_HINT_TOKENS)


def _infer_datetime_series(
    series: pd.Series,
    *,
    column_name: object | None = None,
    min_success_ratio: float = TIME_SERIES_ELIGIBILITY_SUCCESS_RATIO,
) -> pd.Series | None:
    has_datetime_hint = _has_datetime_name_hint(column_name)
    if is_numeric_dtype(series) and not has_datetime_hint:
        return None

    if is_datetime64_any_dtype(series):
        parsed = pd.to_datetime(series, errors="coerce")
    else:
        parsed = pd.to_datetime(series.astype("string"), errors="coerce", format="mixed")
    if parsed.dropna().empty:
        return None

    non_null = series.dropna()
    if non_null.empty:
        return None
    if not has_datetime_hint:
        normalized_values = non_null.astype("string").str.strip()
        if normalized_values.empty or float(normalized_values.str.len().median()) < 6:
            return None
    success_ratio = float(parsed.dropna().shape[0] / max(non_null.shape[0], 1))
    if success_ratio < float(min_success_ratio):
        return None
    if parsed.dropna().nunique() < min(MIN_DATETIME_UNIQUE_VALUES, len(non_null)):
        return None
    return parsed


def _sample_dataframe_for_mapping(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) <= FORECAST_MAPPING_SAMPLE_ROWS:
        return df

    sampled_positions = np.linspace(0, len(df) - 1, num=FORECAST_MAPPING_SAMPLE_ROWS, dtype=int)
    return df.iloc[np.unique(sampled_positions)]


def detect_date_column(df: pd.DataFrame) -> str | None:
    mapping_df = _sample_dataframe_for_mapping(df)
    for column_name in _rank_date_columns(mapping_df):
        parsed = _infer_datetime_series(
            mapping_df[column_name],
            column_name=column_name,
            min_success_ratio=STRICT_DATE_DETECTION_SUCCESS_RATIO,
        )
        if parsed is not None:
            return str(column_name)
    return None


def is_time_series_dataset(df: pd.DataFrame) -> tuple[bool, str | None]:
    detected_column = detect_date_column(df)
    if detected_column:
        return True, detected_column

    for column in df.columns:
        column_name = str(column)
        try:
            parsed = _infer_datetime_series(
                df[column],
                column_name=column_name,
                min_success_ratio=TIME_SERIES_ELIGIBILITY_SUCCESS_RATIO,
            )
        except Exception:
            continue
        if parsed is None:
            continue
        if int(parsed.notna().sum()) > int(TIME_SERIES_ELIGIBILITY_SUCCESS_RATIO * len(df)):
            return True, column_name
    return False, None


def build_forecast_eligibility(df: pd.DataFrame) -> dict[str, Any]:
    allowed, detected_time_column = is_time_series_dataset(df)
    if allowed:
        if detected_time_column:
            reason = f"Detected '{detected_time_column}' as the time column for forecasting."
        else:
            reason = "Detected a usable time column for forecasting."
        suggestions: list[str] = []
    else:
        reason = "No valid time column detected"
        suggestions = list(NON_TIME_SERIES_SUGGESTIONS)
    return {
        "allowed": bool(allowed),
        "reason": reason,
        "detected_time_column": detected_time_column,
        "suggestions": suggestions,
    }


def _candidate_confidence(candidate_scores: list[tuple[float, str]]) -> tuple[float, str]:
    if not candidate_scores:
        return 0.0, "low"

    top_score = float(candidate_scores[0][0])
    second_score = float(candidate_scores[1][0]) if len(candidate_scores) > 1 else 0.0
    confidence_score = 0.58
    if top_score >= 8:
        confidence_score += 0.12
    if top_score >= 12:
        confidence_score += 0.10
    if len(candidate_scores) == 1 or top_score - second_score >= 3.0:
        confidence_score += 0.12
    elif top_score - second_score >= 1.5:
        confidence_score += 0.06
    confidence_score = max(0.0, min(confidence_score, 0.96))
    if confidence_score >= 0.82:
        return round(confidence_score, 2), "high"
    if confidence_score >= 0.62:
        return round(confidence_score, 2), "medium"
    return round(confidence_score, 2), "low"


def _score_date_columns(df: pd.DataFrame) -> list[tuple[float, str]]:
    ranked: list[tuple[float, str]] = []
    for index, column in enumerate(df.columns):
        column_name = str(column)
        parsed = _infer_datetime_series(df[column], column_name=column_name)
        if parsed is None:
            continue

        normalized = _normalize_token(column_name)
        non_null = df[column].dropna()
        parsed_non_null = parsed.dropna()
        unique_ratio = float(parsed_non_null.nunique() / max(len(parsed_non_null), 1))
        is_ordered = bool(parsed_non_null.is_monotonic_increasing or parsed_non_null.is_monotonic_decreasing)
        success_ratio = float(parsed_non_null.shape[0] / max(non_null.shape[0], 1))

        score = 2.0 + (success_ratio * 6.0)
        if "date" in normalized:
            score += 5.0
        if "time" in normalized or "timestamp" in normalized:
            score += 3.5
        if any(token in normalized for token in ("month", "day", "year", "week")):
            score += 2.0
        if unique_ratio >= 0.8:
            score += 1.5
        if is_ordered:
            score += 1.0
        score += max(0, 5 - index) * 0.35
        ranked.append((round(score, 4), column_name))

    ranked.sort(key=lambda item: (-item[0], item[1]))
    return ranked


def _rank_date_columns(df: pd.DataFrame) -> list[str]:
    return [column_name for _, column_name in _score_date_columns(df)]


def _looks_like_identifier(column_name: str, numeric_series: pd.Series) -> bool:
    normalized = _normalize_token(column_name)
    if any(token in normalized for token in IDENTIFIER_HINT_TOKENS):
        return True

    clean = pd.to_numeric(numeric_series, errors="coerce").dropna()
    if clean.shape[0] < 4:
        return False
    unique_ratio = float(clean.nunique() / max(len(clean), 1))
    diffs = clean.diff().dropna()
    near_sequence = bool(not diffs.empty and float((diffs.abs() <= 1.0).mean()) >= 0.85)
    integer_like = bool((clean % 1).abs().max() < 1e-9)
    monotonic = bool(clean.is_monotonic_increasing or clean.is_monotonic_decreasing)
    return unique_ratio >= 0.98 and near_sequence and integer_like and monotonic


def _score_target_columns(
    df: pd.DataFrame,
    excluded_columns: set[str] | None = None,
) -> list[tuple[float, str]]:
    excluded_columns = {str(column) for column in (excluded_columns or set())}
    ranked: list[tuple[float, str]] = []
    for index, column in enumerate(df.columns):
        column_name = str(column)
        if column_name in excluded_columns:
            continue

        series = pd.to_numeric(df[column], errors="coerce")
        clean = series.dropna()
        if clean.empty:
            continue

        normalized = _normalize_token(column_name)
        variability_ratio = float(min(clean.nunique(), 25) / 25.0)
        non_null_ratio = float(clean.shape[0] / max(df[column].dropna().shape[0], 1))
        score = 1.5 + (non_null_ratio * 3.0) + (variability_ratio * 2.0)

        if any(token in normalized for token in ("revenue", "sales", "profit", "orders", "income", "gmv")):
            score += 6.0
        if any(token in normalized for token in ("cost", "expense", "margin", "customers", "quantity", "units", "volume")):
            score += 3.0
        if any(token in normalized for token in ("rate", "ratio", "pct", "percent")):
            score += 1.0
        if _looks_like_identifier(column_name, clean):
            score -= 8.0
        if float(clean.std(ddof=0) or 0.0) > 0.0:
            score += 1.0
        score += max(0, 5 - index) * 0.35
        ranked.append((round(score, 4), column_name))

    ranked.sort(key=lambda item: (-item[0], item[1]))
    return ranked


def _rank_target_columns(df: pd.DataFrame, excluded_columns: set[str] | None = None) -> list[str]:
    return [column_name for _, column_name in _score_target_columns(df, excluded_columns=excluded_columns)]


def auto_detect_time_column(df: pd.DataFrame) -> str:
    detected = detect_date_column(df)
    return detected or ""


def auto_detect_kpi(df: pd.DataFrame) -> str:
    sampled_df = _sample_dataframe_for_mapping(df)
    ranked = _rank_target_columns(sampled_df, excluded_columns=set(_rank_date_columns(sampled_df)))
    return ranked[0] if ranked else ""


def _suggest_default_horizon(resolved_frequency: str | None, history_points: int) -> str:
    compatible_horizons = _compatible_horizons_for_frequency(resolved_frequency)
    if not compatible_horizons:
        return "next_month"

    recommended_horizon = compatible_horizons[0]
    max_periods = max(1, history_points // 4)
    for horizon in compatible_horizons:
        horizon_periods = HORIZON_PERIOD_MAP.get(str(resolved_frequency or ""), {}).get(horizon, 0)
        if horizon_periods <= max_periods:
            recommended_horizon = horizon
        else:
            break
    return recommended_horizon


def build_auto_forecast_config(
    df: pd.DataFrame,
    config: ForecastConfig | dict[str, Any] | None = None,
) -> dict[str, Any]:
    supplied_config = config if isinstance(config, ForecastConfig) else forecast_config_from_dict(config)
    forecast_eligibility = build_forecast_eligibility(df)
    sampled_df = _sample_dataframe_for_mapping(df)
    date_candidate_scores = _score_date_columns(sampled_df)
    date_candidates = [column_name for _, column_name in date_candidate_scores]
    target_candidate_scores = _score_target_columns(sampled_df, excluded_columns=set(date_candidates))
    target_candidates = [column_name for _, column_name in target_candidate_scores]
    detected_time_column = str(forecast_eligibility.get("detected_time_column") or "")

    resolved_config = ForecastConfig(
        date_column=supplied_config.date_column or detected_time_column or (date_candidates[0] if date_candidates else ""),
        target_column=supplied_config.target_column or (target_candidates[0] if target_candidates else ""),
        driver_columns=list(supplied_config.driver_columns or []),
        aggregation_frequency=str(supplied_config.aggregation_frequency or "auto"),
        horizon=str(supplied_config.horizon or ""),
        model_strategy=str(supplied_config.model_strategy or "hybrid"),
        training_mode=str(supplied_config.training_mode or "auto"),
    )

    parsed_dates = None
    if resolved_config.date_column and resolved_config.date_column in df.columns:
        parsed_dates = _infer_datetime_series(df[resolved_config.date_column], column_name=resolved_config.date_column)
    resolved_frequency = _resolve_frequency(resolved_config, parsed_dates) if parsed_dates is not None else None

    history_points = int(len(df))
    if (
        parsed_dates is not None
        and resolved_config.target_column
        and resolved_config.target_column in df.columns
        and resolved_frequency is not None
    ):
        target_series = pd.to_numeric(df[resolved_config.target_column], errors="coerce")
        valid_mask = parsed_dates.notna() & target_series.notna()
        cleaned = pd.DataFrame({"_date": parsed_dates[valid_mask], "_target": target_series[valid_mask]})
        if not cleaned.empty:
            history_points = int(
                cleaned.groupby(pd.Grouper(key="_date", freq=_to_pandas_frequency(resolved_frequency)))["_target"]
                .sum()
                .dropna()
                .shape[0]
            )

    resolved_horizon = resolved_config.horizon or _suggest_default_horizon(resolved_frequency, history_points)
    resolved_config.horizon = resolved_horizon or "next_month"

    date_confidence_score, date_confidence_label = _candidate_confidence(date_candidate_scores)
    target_confidence_score, target_confidence_label = _candidate_confidence(target_candidate_scores)
    if resolved_config.date_column and resolved_config.target_column:
        overall_score = min(date_confidence_score, target_confidence_score)
        if resolved_frequency is not None:
            overall_score = min(0.99, overall_score + 0.08)
        minimum_history = _minimum_history_for_frequency(resolved_frequency)
        if minimum_history and history_points >= minimum_history:
            overall_score = min(0.99, overall_score + 0.05)
    else:
        overall_score = 0.0
    if overall_score >= 0.82:
        overall_confidence = "high"
    elif overall_score >= 0.62:
        overall_confidence = "medium"
    else:
        overall_confidence = "low"

    return {
        "date_column": resolved_config.date_column or None,
        "target": resolved_config.target_column or None,
        "frequency": FREQUENCY_LABELS.get(resolved_frequency, "Auto detect").lower() if resolved_frequency else None,
        "frequency_code": resolved_frequency or None,
        "data_points": int(history_points),
        "horizon": resolved_config.horizon,
        "horizon_label": HORIZON_LABELS.get(resolved_config.horizon, resolved_config.horizon.replace("_", " ").title()),
        "model_strategy": resolved_config.model_strategy,
        "training_mode": resolved_config.training_mode,
        "confidence": overall_confidence,
        "confidence_score": round(overall_score, 2),
        "date_confidence": {"label": date_confidence_label, "score": date_confidence_score},
        "target_confidence": {"label": target_confidence_label, "score": target_confidence_score},
        "forecast_allowed": bool(forecast_eligibility.get("allowed")),
    }


def suggest_forecast_mapping(df: pd.DataFrame) -> dict[str, Any]:
    mapping_df = _sample_dataframe_for_mapping(df)
    date_candidate_scores = _score_date_columns(mapping_df)
    date_candidates = [column_name for _, column_name in date_candidate_scores]
    target_candidate_scores = _score_target_columns(mapping_df, excluded_columns=set(date_candidates))
    target_candidates = [column_name for _, column_name in target_candidate_scores]
    suggested_date = date_candidates[0] if date_candidates else ""
    suggested_target = target_candidates[0] if target_candidates else ""

    driver_candidates: list[str] = []
    excluded_driver_columns = set(date_candidates)
    if suggested_target:
        excluded_driver_columns.add(suggested_target)

    for column in mapping_df.columns:
        column_name = str(column)
        if column_name in excluded_driver_columns:
            continue
        series = mapping_df[column]
        if is_numeric_dtype(series) or series.astype("string").nunique(dropna=True) <= 30:
            driver_candidates.append(column_name)

    suggested_drivers = driver_candidates[:4]
    suggested_frequency = "auto"
    history_points = int(len(mapping_df))
    if suggested_date:
        parsed = _infer_datetime_series(mapping_df[suggested_date], column_name=suggested_date)
        if parsed is not None:
            inferred = infer_frequency_from_dates(parsed)
            if inferred is not None:
                suggested_frequency = inferred
                if suggested_target and suggested_target in mapping_df.columns:
                    target_series = pd.to_numeric(mapping_df[suggested_target], errors="coerce")
                    valid_mask = parsed.notna() & target_series.notna()
                    cleaned = pd.DataFrame({"_date": parsed[valid_mask], "_target": target_series[valid_mask]})
                    if not cleaned.empty:
                        history_points = int(
                            cleaned.groupby(pd.Grouper(key="_date", freq=_to_pandas_frequency(inferred)))["_target"]
                            .sum()
                            .dropna()
                            .shape[0]
                        )

    suggested_horizon = _suggest_default_horizon(
        suggested_frequency if suggested_frequency in HORIZON_PERIOD_MAP else None,
        history_points,
    )
    date_confidence_score, date_confidence_label = _candidate_confidence(date_candidate_scores)
    target_confidence_score, target_confidence_label = _candidate_confidence(target_candidate_scores)

    return {
        "date_column": suggested_date,
        "target_column": suggested_target,
        "driver_columns": suggested_drivers,
        "aggregation_frequency": suggested_frequency,
        "horizon": suggested_horizon,
        "model_strategy": "hybrid",
        "training_mode": "auto",
        "date_candidates": date_candidates,
        "target_candidates": target_candidates,
        "driver_candidates": driver_candidates,
        "date_confidence": {"label": date_confidence_label, "score": date_confidence_score},
        "target_confidence": {"label": target_confidence_label, "score": target_confidence_score},
    }


def get_forecast_mapping_options(df: pd.DataFrame) -> dict[str, Any]:
    suggestions = suggest_forecast_mapping(df)
    return {
        "date_columns": suggestions["date_candidates"],
        "target_columns": suggestions["target_candidates"],
        "driver_columns": suggestions["driver_candidates"],
        "frequency_options": [
            {"value": frequency, "label": FREQUENCY_LABELS[frequency]}
            for frequency in SUPPORTED_FREQUENCIES
        ],
        "horizon_options": get_forecast_horizon_options(),
        "model_strategy_options": list(SUPPORTED_MODEL_STRATEGIES),
        "training_mode_options": list(SUPPORTED_TRAINING_MODES),
        "suggestions": suggestions,
    }


def infer_frequency_from_dates(date_series: pd.Series) -> str | None:
    cleaned = pd.to_datetime(date_series, errors="coerce").dropna().sort_values().drop_duplicates()
    if cleaned.shape[0] < 3:
        return None

    diffs = cleaned.diff().dropna()
    if diffs.empty:
        return None
    median_days = float(diffs.dt.total_seconds().median() / 86400)

    if median_days <= 1.5:
        return "D"
    if median_days <= 10:
        return "W"
    if median_days <= 40:
        return "M"
    if median_days <= 120:
        return "Q"
    return None


def _to_pandas_frequency(resolved_frequency: str) -> str:
    if resolved_frequency == "M":
        return "ME"
    if resolved_frequency == "Q":
        return "QE"
    return resolved_frequency


def _resolve_frequency(config: ForecastConfig, parsed_dates: pd.Series) -> str | None:
    explicit_frequency = str(config.aggregation_frequency or "auto")
    if explicit_frequency != "auto":
        return explicit_frequency if explicit_frequency in SUPPORTED_FREQUENCIES else None
    return infer_frequency_from_dates(parsed_dates)


def _minimum_history_for_frequency(resolved_frequency: str | None) -> int:
    return MIN_HISTORY_POINTS.get(str(resolved_frequency or ""), 0)


def _compatible_horizons_for_frequency(resolved_frequency: str | None) -> list[str]:
    return list(HORIZON_PERIOD_MAP.get(str(resolved_frequency or ""), {}).keys())


def _resolve_horizon_periods(resolved_frequency: str, horizon: str) -> int:
    periods = HORIZON_PERIOD_MAP.get(resolved_frequency, {}).get(horizon)
    if periods is None:
        raise ForecastError(
            f"Horizon '{HORIZON_LABELS.get(horizon, horizon)}' is not compatible with {FREQUENCY_LABELS.get(resolved_frequency, resolved_frequency)} data."
        )
    return int(periods)


def _resolve_config_with_suggestions(df: pd.DataFrame, config: ForecastConfig) -> ForecastConfig:
    suggestions = suggest_forecast_mapping(df)
    return ForecastConfig(
        date_column=config.date_column or str(suggestions.get("date_column") or ""),
        target_column=config.target_column or str(suggestions.get("target_column") or ""),
        driver_columns=list(config.driver_columns or suggestions.get("driver_columns") or []),
        aggregation_frequency=(
            config.aggregation_frequency
            if str(config.aggregation_frequency or "auto") != "auto"
            else str(suggestions.get("aggregation_frequency") or "auto")
        ),
        horizon=str(config.horizon or suggestions.get("horizon") or "next_month"),
        model_strategy=str(config.model_strategy or suggestions.get("model_strategy") or "hybrid"),
        training_mode=str(config.training_mode or suggestions.get("training_mode") or "auto"),
    )


def _assert_valid_forecast_time_column(df: pd.DataFrame) -> None:
    dataset_profile = detect_dataset_type(df)
    if not dataset_profile.get("has_datetime"):
        raise ValueError("No valid time column detected. Forecasting requires time-based data.")


def prepare_time_series(
    df: pd.DataFrame,
    *,
    date_column: str | None = None,
    target_column: str | None = None,
    aggregation_frequency: str = "auto",
) -> pd.DataFrame:
    resolved_date_column = date_column or (_rank_date_columns(df)[0] if _rank_date_columns(df) else "")
    if not resolved_date_column or resolved_date_column not in df.columns:
        raise ForecastError("No valid date column found.")

    parsed_dates = _infer_datetime_series(df[resolved_date_column], column_name=resolved_date_column)
    if parsed_dates is None:
        raise ForecastError("No valid date column found.")

    excluded_columns = {resolved_date_column}
    ranked_targets = _rank_target_columns(df, excluded_columns=excluded_columns)
    resolved_target_column = target_column or (ranked_targets[0] if ranked_targets else "")
    if not resolved_target_column or resolved_target_column not in df.columns:
        raise ForecastError("No valid numeric target column found.")

    target_values = pd.to_numeric(df[resolved_target_column], errors="coerce")
    if target_values.dropna().empty:
        raise ForecastError(f"Column '{resolved_target_column}' does not contain usable numeric target values.")

    resolved_frequency = (
        str(aggregation_frequency or "auto")
        if str(aggregation_frequency or "auto") != "auto"
        else infer_frequency_from_dates(parsed_dates)
    )
    if resolved_frequency is None or resolved_frequency not in HORIZON_PERIOD_MAP:
        raise ForecastError("Aidssist could not infer a safe forecast frequency.")

    working_df = pd.DataFrame(
        {
            "_date": parsed_dates,
            "_target": target_values,
        }
    )
    driver_columns = [
        str(column)
        for column in df.columns
        if str(column) not in {resolved_date_column, resolved_target_column}
    ]
    for driver_column in driver_columns:
        working_df[driver_column] = df[driver_column]

    working_df = working_df.dropna(subset=["_date", "_target"])
    if working_df.empty:
        raise ForecastError("No usable rows remain after parsing the selected date and target columns.")

    observed_index = (
        working_df.set_index("_date")
        .groupby(pd.Grouper(freq=_to_pandas_frequency(resolved_frequency)))
        .size()
    )
    observed_index = observed_index[observed_index > 0].index

    aggregation_map: dict[str, Any] = {"_target": "sum"}
    for driver_column in driver_columns:
        series = working_df[driver_column]
        aggregation_map[driver_column] = (
            _aggregate_numeric_driver(driver_column)
            if is_numeric_dtype(series)
            else (lambda values: values.dropna().astype("string").iloc[-1] if not values.dropna().empty else pd.NA)
        )

    prepared = (
        working_df.set_index("_date")
        .groupby(pd.Grouper(freq=_to_pandas_frequency(resolved_frequency)))
        .agg(aggregation_map)
        .rename(columns={"_target": "target"})
        .sort_index()
    )
    if prepared.empty:
        raise ForecastError("No usable historical rows remain after aggregation.")

    full_index = pd.date_range(
        start=observed_index.min(),
        end=observed_index.max(),
        freq=_to_pandas_frequency(resolved_frequency),
    )
    missing_timestamps = int(len(full_index.difference(observed_index)))
    prepared = prepared.reindex(full_index)

    prepared["target"] = prepared["target"].interpolate(limit_direction="both")
    prepared["target"] = prepared["target"].ffill().bfill()

    for driver_column in driver_columns:
        if driver_column not in prepared.columns:
            continue
        series = prepared[driver_column]
        if is_numeric_dtype(series):
            prepared[driver_column] = pd.to_numeric(series, errors="coerce").interpolate(limit_direction="both")
            prepared[driver_column] = prepared[driver_column].ffill().bfill().fillna(0.0)
        else:
            prepared[driver_column] = series.astype("string").replace({"<NA>": pd.NA})
            prepared[driver_column] = prepared[driver_column].ffill().bfill().fillna("")

    prepared = prepared.reset_index().rename(columns={"index": "date"}).reset_index(drop=True)
    prepared.attrs["time_column"] = resolved_date_column
    prepared.attrs["target_column"] = resolved_target_column
    prepared.attrs["resolved_frequency"] = resolved_frequency
    prepared.attrs["missing_timestamps"] = missing_timestamps
    prepared.attrs["driver_columns"] = driver_columns
    return prepared


def validate_time_series(
    df: pd.DataFrame,
    *,
    date_column: str | None = None,
    target_column: str | None = None,
    aggregation_frequency: str = "auto",
) -> dict[str, Any]:
    try:
        cleaned_df = prepare_time_series(
            df,
            date_column=date_column,
            target_column=target_column,
            aggregation_frequency=aggregation_frequency,
        )
        resolved_frequency = str(cleaned_df.attrs.get("resolved_frequency") or "")
        minimum_points = _minimum_history_for_frequency(resolved_frequency)
        data_points = int(cleaned_df.shape[0])
        warnings: list[str] = []
        is_valid = data_points >= minimum_points
        if not is_valid:
            warnings.append(
                f"Forecasting works best with at least {minimum_points} {FREQUENCY_LABELS.get(resolved_frequency, resolved_frequency).lower()} data points."
            )
        return {
            "valid": is_valid,
            "cleaned_df": cleaned_df,
            "time_column": cleaned_df.attrs.get("time_column"),
            "target_column": cleaned_df.attrs.get("target_column"),
            "frequency": resolved_frequency,
            "data_points": data_points,
            "minimum_data_points": minimum_points,
            "filled_missing_timestamps": int(cleaned_df.attrs.get("missing_timestamps") or 0),
            "warnings": warnings,
            "error": None if is_valid else {"message": warnings[0], "suggestion": _forecast_fix_suggestion(warnings[0])},
        }
    except ForecastError as error:
        error_message = _user_facing_forecast_error(str(error))
        return {
            "valid": False,
            "cleaned_df": None,
            "time_column": date_column,
            "target_column": target_column,
            "frequency": None,
            "data_points": int(len(df)),
            "minimum_data_points": 0,
            "filled_missing_timestamps": 0,
            "warnings": [],
            "error": {
                "message": error_message,
                "suggestion": _forecast_fix_suggestion(error_message),
            },
        }


def build_forecast_features(df: pd.DataFrame) -> pd.DataFrame:
    featured = df.copy()
    if "date" not in featured.columns:
        if isinstance(featured.index, pd.DatetimeIndex):
            featured = featured.reset_index().rename(columns={"index": "date"})
        else:
            featured["date"] = pd.RangeIndex(start=0, stop=len(featured))

    if "target" not in featured.columns:
        numeric_candidates = featured.select_dtypes(include="number").columns.tolist()
        if not numeric_candidates:
            raise ForecastError("Forecast features require at least one numeric target-like column.")
        featured = featured.rename(columns={str(numeric_candidates[0]): "target"})

    featured["target"] = pd.to_numeric(featured["target"], errors="coerce")
    featured["lag_1"] = featured["target"].shift(1)
    featured["lag_2"] = featured["target"].shift(2)
    featured["lag_3"] = featured["target"].shift(3)
    featured["lag_7"] = featured["target"].shift(7)
    featured["rolling_mean_3"] = featured["target"].shift(1).rolling(3).mean()
    featured["rolling_mean_7"] = featured["target"].shift(1).rolling(7).mean()
    featured["rolling_std_3"] = featured["target"].shift(1).rolling(3).std().fillna(0.0)
    featured["rolling_std_7"] = featured["target"].shift(1).rolling(7).std().fillna(0.0)
    featured["rolling_min_7"] = featured["target"].shift(1).rolling(7).min()
    featured["rolling_max_7"] = featured["target"].shift(1).rolling(7).max()
    featured["time_index"] = np.arange(len(featured))
    if pd.api.types.is_datetime64_any_dtype(featured["date"]):
        featured["month"] = featured["date"].dt.month
        featured["quarter"] = featured["date"].dt.quarter
        featured["weekofyear"] = featured["date"].dt.isocalendar().week.astype(int)
        featured["dayofweek"] = featured["date"].dt.dayofweek
    else:
        featured["month"] = 0
        featured["quarter"] = 0
        featured["weekofyear"] = 0
        featured["dayofweek"] = 0
    return featured


def validate_forecast_config(df: pd.DataFrame, config: ForecastConfig) -> ForecastValidationResult:
    errors: list[str] = []
    warnings: list[str] = []

    if not config.date_column:
        errors.append("We couldn't detect a time column. Please select one.")
    elif config.date_column not in df.columns:
        errors.append(f"Date column '{config.date_column}' was not found in the dataset.")

    if not config.target_column:
        errors.append("We couldn't detect a primary KPI. Please select one numeric column.")
    elif config.target_column not in df.columns:
        errors.append(f"Target column '{config.target_column}' was not found in the dataset.")

    for driver_column in config.driver_columns:
        if driver_column not in df.columns:
            errors.append(f"Driver column '{driver_column}' was not found in the dataset.")

    if config.horizon not in SUPPORTED_HORIZONS:
        errors.append("Choose a supported forecast horizon.")
    if config.aggregation_frequency not in SUPPORTED_FREQUENCIES:
        errors.append("Choose a supported aggregation frequency.")
    if config.model_strategy not in SUPPORTED_MODEL_STRATEGIES:
        errors.append("Choose a supported model strategy.")
    if config.training_mode not in SUPPORTED_TRAINING_MODES:
        errors.append("Choose a supported training mode.")

    parsed_dates = None
    if config.date_column and config.date_column in df.columns:
        parsed_dates = _infer_datetime_series(df[config.date_column], column_name=config.date_column)
        if parsed_dates is None:
            errors.append(f"We couldn't parse '{config.date_column}' as a time column. Please select another one.")

    target_series = None
    if config.target_column and config.target_column in df.columns:
        target_series = pd.to_numeric(df[config.target_column], errors="coerce")
        if target_series.dropna().empty:
            errors.append(f"'{config.target_column}' doesn't contain enough numeric values to forecast. Please choose another KPI.")

    resolved_frequency = None
    history_points = 0
    date_start = None
    date_end = None
    compatible_horizons: list[str] = []
    minimum_history_points = 0

    if parsed_dates is not None and target_series is not None and not errors:
        resolved_frequency = _resolve_frequency(config, parsed_dates)
        if resolved_frequency is None:
            errors.append(
                "We couldn't infer a reliable forecast frequency from this dataset. Try a manual override in Advanced options."
            )
        else:
            compatible_horizons = _compatible_horizons_for_frequency(resolved_frequency)
            minimum_history_points = _minimum_history_for_frequency(resolved_frequency)
            if config.horizon not in compatible_horizons:
                errors.append(
                    f"{HORIZON_LABELS.get(config.horizon, config.horizon)} is not compatible with {FREQUENCY_LABELS[resolved_frequency]} data."
                )

            valid_mask = parsed_dates.notna() & target_series.notna()
            cleaned = pd.DataFrame(
                {
                    "_date": parsed_dates[valid_mask],
                    "_target": target_series[valid_mask],
                }
            )
            if cleaned.empty:
                errors.append("No usable rows remain after parsing the selected date and target columns.")
            else:
                grouped = cleaned.groupby(
                    pd.Grouper(key="_date", freq=_to_pandas_frequency(resolved_frequency))
                )["_target"].sum().dropna()
                history_points = int(grouped.shape[0])
                if not grouped.empty:
                    date_start = str(grouped.index.min().date())
                    date_end = str(grouped.index.max().date())

                if history_points < minimum_history_points:
                    errors.append(
                        f"Aidssist needs at least {minimum_history_points} historical {FREQUENCY_LABELS[resolved_frequency].lower()} points, but only found {history_points}."
                    )
                elif history_points < minimum_history_points + 4:
                    warnings.append(
                        f"The history is just above the minimum for {FREQUENCY_LABELS[resolved_frequency].lower()} forecasting. Confidence may be lower."
                    )

    return ForecastValidationResult(
        errors=errors,
        warnings=warnings,
        resolved_frequency=resolved_frequency,
        history_points=history_points,
        minimum_history_points=minimum_history_points,
        compatible_horizons=compatible_horizons,
        date_start=date_start,
        date_end=date_end,
    )


def _build_forecast_data_score(validation: ForecastValidationResult) -> dict[str, Any]:
    warning_count = len(validation.warnings)
    blocking_count = len(validation.errors)
    score = 100 - (warning_count * 8) - (blocking_count * 35)
    score = max(0, min(100, score))
    if score >= 85:
        band = "strong"
    elif score >= 70:
        band = "usable"
    elif score >= 50:
        band = "watch"
    else:
        band = "weak"
    return {
        "score": int(score),
        "band": band,
        "warning_count": warning_count,
        "blocking_issue_count": blocking_count,
    }


def _build_forecast_pipeline_trace(
    *,
    config: ForecastConfig,
    validation: ForecastValidationResult,
    chosen_model_name: str,
    recommendations: list[dict[str, Any]],
    decision_layer: dict[str, Any] | None = None,
    error: str | None = None,
    data_quality: dict[str, Any] | None = None,
    model_metrics: dict[str, Any] | None = None,
    model_quality: str | None = None,
    risk: str | None = None,
    explanation: dict[str, Any] | None = None,
    limitations: list[str] | None = None,
    logged_failure_count: int = 0,
    inconsistency_detected: bool = False,
    result_hash: str | None = None,
    dataset_fingerprint: str | None = None,
) -> list[dict[str, Any]]:
    query_label = f"Forecast {config.target_column or 'target'} for {config.horizon}"
    data_score = _build_forecast_data_score(validation)
    return [
        {
            "stage": "user_query",
            "title": "User Query",
            "status": "completed",
            "detail": {"query": query_label},
        },
        {
            "stage": "intent_detection",
            "title": "Intent Detection",
            "status": "completed",
            "detail": {"legacy_intent": "forecast", "analysis_intent": "prediction"},
        },
        {
            "stage": "contract_execution",
            "title": "Contract Execution",
            "status": "completed",
            "detail": {
                "date_column": config.date_column,
                "target_column": config.target_column,
                "driver_columns": list(config.driver_columns),
            },
        },
        {
            "stage": "forecast_ml",
            "title": "Forecast / ML (integrated)",
            "status": "completed",
            "detail": {
                "chosen_model": chosen_model_name,
                "training_mode": config.training_mode,
                "aggregation_frequency": config.aggregation_frequency,
            },
        },
        {
            "stage": "validation_data_score",
            "title": "Validation + Data Score",
            "status": "failed" if validation.errors else "completed",
            "detail": {
                "data_score": data_score,
                "data_quality": dict(data_quality or {}),
                "warnings": list(validation.warnings),
                "blocking_errors": list(validation.errors),
            },
        },
        {
            "stage": "execution",
            "title": "Execution",
            "status": "failed" if error else "completed",
            "detail": {"error": error},
        },
        {
            "stage": "model_evaluation",
            "title": "Model Evaluation",
            "status": "completed" if any(value is not None for value in dict(model_metrics or {}).values()) else "skipped",
            "detail": {"metrics": dict(model_metrics or {"mae": None, "r2": None})},
        },
        {
            "stage": "explainability",
            "title": "Explainability",
            "status": "completed" if list((explanation or {}).get("top_features") or []) else "skipped",
            "detail": {"top_features": list((explanation or {}).get("top_features") or [])},
        },
        {
            "stage": "trust_layer",
            "title": "Trust Layer",
            "status": "completed",
            "detail": {
                "model_quality": str(model_quality or ""),
                "risk": str(risk or ""),
                "data_quality_score": float((data_quality or {}).get("score") or 0.0),
            },
        },
        {
            "stage": "decision_engine",
            "title": "Decision Engine",
            "status": "completed" if ((decision_layer or {}).get("decisions") or []) else "skipped",
            "detail": {
                "decision_count": len(((decision_layer or {}).get("decisions") or [])),
                "decision_confidence": str((decision_layer or {}).get("decision_confidence") or "low"),
                "top_decision": (((decision_layer or {}).get("top_decision") or {}).get("action")),
            },
        },
        {
            "stage": "learning_engine",
            "title": "Learning Engine",
            "status": "completed" if ((decision_layer or {}).get("learning_insights") or {}) else "skipped",
            "detail": {
                "pattern_count": len((((decision_layer or {}).get("learning_insights") or {}).get("patterns") or [])),
                "confidence_adjustment": (((decision_layer or {}).get("learning_insights") or {}).get("confidence_adjustment")),
                "risk_adjustment": (((decision_layer or {}).get("learning_insights") or {}).get("risk_adjustment")),
            },
        },
        {
            "stage": "insight_decisions",
            "title": "Insight + Decisions",
            "status": "completed" if recommendations else "pending",
            "detail": {"recommendation_count": len(recommendations)},
        },
        {
            "stage": "failure_logging",
            "title": "Failure Logging",
            "status": "completed" if logged_failure_count or error else "skipped",
            "detail": {"logged_failures": int(logged_failure_count or 0)},
        },
        {
            "stage": "consistency_check",
            "title": "Consistency Check",
            "status": "completed",
            "detail": {
                "result_hash": str(result_hash or ""),
                "inconsistency_detected": bool(inconsistency_detected),
                "dataset_fingerprint": str(dataset_fingerprint or ""),
                "limitations": list(limitations or []),
            },
        },
        {
            "stage": "caching",
            "title": "Caching",
            "status": "pending",
            "detail": {"status": "pending"},
        },
        {
            "stage": "memory_update",
            "title": "Memory Update",
            "status": "pending",
            "detail": {"status": "pending"},
        },
    ]


def _aggregate_numeric_driver(column_name: str) -> str:
    normalized = _normalize_token(column_name)
    if any(token in normalized for token in ("discount", "margin", "rate", "ratio", "percent", "price")):
        return "mean"
    return "sum"


def _build_prepared_forecast_data(df: pd.DataFrame, config: ForecastConfig) -> PreparedForecastData:
    validation = validate_forecast_config(df, config)
    if not validation.is_valid or validation.resolved_frequency is None:
        raise ForecastError(" ".join(validation.errors or ["Forecast configuration is invalid."]))

    prepared = prepare_time_series(
        df,
        date_column=config.date_column,
        target_column=config.target_column,
        aggregation_frequency=validation.resolved_frequency,
    )

    if prepared.empty:
        raise ForecastError("No usable historical rows remain after aggregation.")

    resolved_frequency = str(prepared.attrs.get("resolved_frequency") or validation.resolved_frequency)
    horizon_periods = _resolve_horizon_periods(resolved_frequency, config.horizon)
    future_dates = pd.date_range(
        start=prepared["date"].iloc[-1],
        periods=horizon_periods + 1,
        freq=_to_pandas_frequency(resolved_frequency),
    )[1:]

    return PreparedForecastData(
        dataframe=prepared,
        resolved_frequency=resolved_frequency,
        horizon_periods=horizon_periods,
        future_dates=future_dates,
    )


def _add_lag_features(frame: pd.DataFrame) -> pd.DataFrame:
    return build_forecast_features(frame)


def _build_feature_matrix(frame: pd.DataFrame, driver_columns: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    featured = _add_lag_features(frame)
    feature_columns = [
        "lag_1",
        "lag_2",
        "lag_3",
        "lag_7",
        "rolling_mean_3",
        "rolling_mean_7",
        "rolling_std_3",
        "rolling_std_7",
        "rolling_min_7",
        "rolling_max_7",
        "time_index",
        "month",
        "quarter",
        "weekofyear",
        "dayofweek",
    ]

    encoded_parts = [featured.loc[:, feature_columns]]
    for driver_column in driver_columns:
        series = featured[driver_column]
        if is_numeric_dtype(series):
            encoded_parts.append(pd.DataFrame({f"driver__{driver_column}": pd.to_numeric(series, errors="coerce")}))
        else:
            dummies = pd.get_dummies(series.astype("string"), prefix=f"driver__{driver_column}", dtype=float)
            encoded_parts.append(dummies)

    X = pd.concat(encoded_parts, axis=1)
    y = featured["target"]
    valid_mask = X.notna().all(axis=1) & y.notna()
    return X.loc[valid_mask].astype(float), y.loc[valid_mask].astype(float)


def _future_driver_defaults(frame: pd.DataFrame, driver_columns: list[str]) -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    for driver_column in driver_columns:
        series = frame[driver_column].dropna()
        if series.empty:
            defaults[driver_column] = 0.0 if is_numeric_dtype(frame[driver_column]) else ""
            continue
        if is_numeric_dtype(frame[driver_column]):
            defaults[driver_column] = float(series.tail(min(3, len(series))).mean())
        else:
            defaults[driver_column] = str(series.astype("string").iloc[-1])
    return defaults


def _build_future_feature_row(
    history_frame: pd.DataFrame,
    next_date: pd.Timestamp,
    driver_columns: list[str],
    driver_values: dict[str, Any],
) -> pd.DataFrame:
    future_row = {"date": next_date, "target": np.nan}
    for driver_column in driver_columns:
        future_row[driver_column] = driver_values.get(driver_column)
    combined = pd.concat([history_frame, pd.DataFrame([future_row])], ignore_index=True)
    featured = _add_lag_features(combined)

    encoded_parts = [featured.loc[:, [
        "lag_1",
        "lag_2",
        "lag_3",
        "lag_7",
        "rolling_mean_3",
        "rolling_mean_7",
        "rolling_std_3",
        "rolling_std_7",
        "rolling_min_7",
        "rolling_max_7",
        "time_index",
        "month",
        "quarter",
        "weekofyear",
        "dayofweek",
    ]]]
    for driver_column in driver_columns:
        series = featured[driver_column]
        if is_numeric_dtype(series):
            encoded_parts.append(pd.DataFrame({f"driver__{driver_column}": pd.to_numeric(series, errors="coerce")}))
        else:
            encoded_parts.append(pd.get_dummies(series.astype("string"), prefix=f"driver__{driver_column}", dtype=float))

    feature_row = pd.concat(encoded_parts, axis=1).iloc[[-1]].astype(float)
    return feature_row


class ForecastModelAdapter:
    name = "base"

    def evaluate(self, frame: pd.DataFrame, driver_columns: list[str], min_history_points: int) -> dict[str, Any]:
        raise NotImplementedError

    def forecast(self, frame: pd.DataFrame, prepared: PreparedForecastData, driver_columns: list[str]) -> dict[str, Any]:
        raise NotImplementedError


class NaiveLastValueAdapter(ForecastModelAdapter):
    name = "naive_last_value"

    def evaluate(self, frame: pd.DataFrame, driver_columns: list[str], min_history_points: int) -> dict[str, Any]:
        del driver_columns
        if len(frame) < min_history_points:
            raise ForecastError("Not enough history for naive baseline evaluation.")
        actual = frame["target"].iloc[1:].astype(float).tolist()
        predicted = frame["target"].shift(1).dropna().astype(float).tolist()
        return _build_metric_payload(actual, predicted)

    def forecast(self, frame: pd.DataFrame, prepared: PreparedForecastData, driver_columns: list[str]) -> dict[str, Any]:
        del prepared, driver_columns
        last_value = float(frame["target"].iloc[-1])
        predictions = [last_value for _ in range(prepared.horizon_periods)]
        return {"predictions": predictions, "model_state": {"type": self.name}}


class MovingAverageAdapter(ForecastModelAdapter):
    name = "moving_average"

    def evaluate(self, frame: pd.DataFrame, driver_columns: list[str], min_history_points: int) -> dict[str, Any]:
        del driver_columns
        if len(frame) < max(min_history_points, 4):
            raise ForecastError("Not enough history for moving-average evaluation.")
        actual: list[float] = []
        predicted: list[float] = []
        for index in range(3, len(frame)):
            actual.append(float(frame["target"].iloc[index]))
            predicted.append(float(frame["target"].iloc[max(0, index - 3):index].mean()))
        return _build_metric_payload(actual, predicted)

    def forecast(self, frame: pd.DataFrame, prepared: PreparedForecastData, driver_columns: list[str]) -> dict[str, Any]:
        del driver_columns
        rolling_value = float(frame["target"].tail(min(3, len(frame))).mean())
        predictions = [rolling_value for _ in range(prepared.horizon_periods)]
        return {"predictions": predictions, "model_state": {"type": self.name}}


class LinearRegressionForecastAdapter(ForecastModelAdapter):
    name = "linear_regression"

    def evaluate(self, frame: pd.DataFrame, driver_columns: list[str], min_history_points: int) -> dict[str, Any]:
        if len(frame) < max(min_history_points, 10):
            raise ForecastError("Not enough history for linear regression forecasting.")

        actual: list[float] = []
        predicted: list[float] = []
        starting_index = max(8, min_history_points - 1)

        for index in range(starting_index, len(frame)):
            train_frame = frame.iloc[:index].copy()
            X_train, y_train = _build_feature_matrix(train_frame, driver_columns)
            if len(X_train) < 6:
                continue

            model = LinearRegression()
            model.fit(X_train, y_train)

            next_date = pd.Timestamp(frame["date"].iloc[index])
            driver_values = {
                driver_column: frame.iloc[index][driver_column]
                for driver_column in driver_columns
            }
            feature_row = _build_future_feature_row(train_frame, next_date, driver_columns, driver_values)
            feature_row = feature_row.reindex(columns=X_train.columns, fill_value=0.0)
            predicted.append(float(model.predict(feature_row)[0]))
            actual.append(float(frame["target"].iloc[index]))

        if not actual:
            raise ForecastError("Linear regression could not build enough walk-forward folds.")
        return _build_metric_payload(actual, predicted)

    def forecast(self, frame: pd.DataFrame, prepared: PreparedForecastData, driver_columns: list[str]) -> dict[str, Any]:
        X_train, y_train = _build_feature_matrix(frame, driver_columns)
        if len(X_train) < 6:
            raise ForecastError("Linear regression does not have enough training rows.")

        model = LinearRegression()
        model.fit(X_train, y_train)

        defaults = _future_driver_defaults(frame, driver_columns)
        history = frame.copy()
        predictions: list[float] = []
        for next_date in prepared.future_dates:
            feature_row = _build_future_feature_row(history, next_date, driver_columns, defaults)
            feature_row = feature_row.reindex(columns=X_train.columns, fill_value=0.0)
            next_value = float(model.predict(feature_row)[0])
            predictions.append(next_value)
            appended = {"date": next_date, "target": next_value}
            for driver_column in driver_columns:
                appended[driver_column] = defaults.get(driver_column)
            history = pd.concat([history, pd.DataFrame([appended])], ignore_index=True)

        feature_names = list(X_train.columns)
        coefficients = [float(value) for value in model.coef_]
        return {
            "predictions": predictions,
            "feature_names": feature_names,
            "feature_importances": coefficients,
            "model_state": {
                "type": self.name,
                "feature_names": feature_names,
                "model": model,
            },
        }


class RidgeForecastAdapter(ForecastModelAdapter):
    name = "ridge_regression"

    def evaluate(self, frame: pd.DataFrame, driver_columns: list[str], min_history_points: int) -> dict[str, Any]:
        if len(frame) < max(min_history_points, 12):
            raise ForecastError("Not enough history for explainable regression.")

        actual: list[float] = []
        predicted: list[float] = []
        starting_index = max(8, min_history_points - 1)

        for index in range(starting_index, len(frame)):
            train_frame = frame.iloc[:index].copy()
            X_train, y_train = _build_feature_matrix(train_frame, driver_columns)
            if len(X_train) < 6:
                continue

            model = Ridge(alpha=1.0)
            model.fit(X_train, y_train)

            next_date = pd.Timestamp(frame["date"].iloc[index])
            driver_values = {
                driver_column: frame.iloc[index][driver_column]
                for driver_column in driver_columns
            }
            feature_row = _build_future_feature_row(train_frame, next_date, driver_columns, driver_values)
            feature_row = feature_row.reindex(columns=X_train.columns, fill_value=0.0)
            predicted.append(float(model.predict(feature_row)[0]))
            actual.append(float(frame["target"].iloc[index]))

        if not actual:
            raise ForecastError("Explainable regression could not build enough walk-forward folds.")
        return _build_metric_payload(actual, predicted)

    def forecast(self, frame: pd.DataFrame, prepared: PreparedForecastData, driver_columns: list[str]) -> dict[str, Any]:
        X_train, y_train = _build_feature_matrix(frame, driver_columns)
        if len(X_train) < 6:
            raise ForecastError("Explainable regression does not have enough training rows.")

        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)

        defaults = _future_driver_defaults(frame, driver_columns)
        history = frame.copy()
        predictions: list[float] = []
        for next_date in prepared.future_dates:
            feature_row = _build_future_feature_row(history, next_date, driver_columns, defaults)
            feature_row = feature_row.reindex(columns=X_train.columns, fill_value=0.0)
            next_value = float(model.predict(feature_row)[0])
            predictions.append(next_value)
            appended = {"date": next_date, "target": next_value}
            for driver_column in driver_columns:
                appended[driver_column] = defaults.get(driver_column)
            history = pd.concat([history, pd.DataFrame([appended])], ignore_index=True)

        feature_names = list(X_train.columns)
        coefficients = [float(value) for value in model.coef_]
        return {
            "predictions": predictions,
            "feature_names": feature_names,
            "feature_importances": coefficients,
            "model_state": {
                "type": self.name,
                "feature_names": feature_names,
                "model": model,
            },
        }


class RandomForestForecastAdapter(ForecastModelAdapter):
    name = "random_forest"

    def evaluate(self, frame: pd.DataFrame, driver_columns: list[str], min_history_points: int) -> dict[str, Any]:
        if len(frame) < max(min_history_points, 18):
            raise ForecastError("Not enough history for the stronger tree-based model.")

        actual: list[float] = []
        predicted: list[float] = []
        starting_index = max(10, min_history_points - 1)

        for index in range(starting_index, len(frame)):
            train_frame = frame.iloc[:index].copy()
            X_train, y_train = _build_feature_matrix(train_frame, driver_columns)
            if len(X_train) < 8:
                continue

            model = RandomForestRegressor(
                n_estimators=160,
                random_state=42,
                min_samples_leaf=2,
            )
            model.fit(X_train, y_train)

            next_date = pd.Timestamp(frame["date"].iloc[index])
            driver_values = {
                driver_column: frame.iloc[index][driver_column]
                for driver_column in driver_columns
            }
            feature_row = _build_future_feature_row(train_frame, next_date, driver_columns, driver_values)
            feature_row = feature_row.reindex(columns=X_train.columns, fill_value=0.0)
            predicted.append(float(model.predict(feature_row)[0]))
            actual.append(float(frame["target"].iloc[index]))

        if not actual:
            raise ForecastError("The stronger tree-based model could not build enough walk-forward folds.")
        return _build_metric_payload(actual, predicted)

    def forecast(self, frame: pd.DataFrame, prepared: PreparedForecastData, driver_columns: list[str]) -> dict[str, Any]:
        X_train, y_train = _build_feature_matrix(frame, driver_columns)
        if len(X_train) < 8:
            raise ForecastError("The stronger tree-based model does not have enough training rows.")

        model = RandomForestRegressor(
            n_estimators=160,
            random_state=42,
            min_samples_leaf=2,
        )
        model.fit(X_train, y_train)
        defaults = _future_driver_defaults(frame, driver_columns)
        history = frame.copy()
        predictions: list[float] = []
        for next_date in prepared.future_dates:
            feature_row = _build_future_feature_row(history, next_date, driver_columns, defaults)
            feature_row = feature_row.reindex(columns=X_train.columns, fill_value=0.0)
            next_value = float(model.predict(feature_row)[0])
            predictions.append(next_value)
            appended = {"date": next_date, "target": next_value}
            for driver_column in driver_columns:
                appended[driver_column] = defaults.get(driver_column)
            history = pd.concat([history, pd.DataFrame([appended])], ignore_index=True)

        feature_names = list(X_train.columns)
        importances = [float(value) for value in model.feature_importances_]
        return {
            "predictions": predictions,
            "feature_names": feature_names,
            "feature_importances": importances,
            "model_state": {
                "type": self.name,
                "feature_names": feature_names,
                "model": model,
            },
        }


def _build_metric_payload(actual: list[float], predicted: list[float]) -> dict[str, float]:
    actual_array = np.asarray(actual, dtype=float)
    predicted_array = np.asarray(predicted, dtype=float)
    residuals = actual_array - predicted_array
    model_metrics, _ = evaluate_model_with_warnings(actual_array, predicted_array)
    mae = float(model_metrics.get("mae") or np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(np.square(residuals))))
    safe_denominator = np.where(np.abs(actual_array) < 1e-9, 1.0, np.abs(actual_array))
    mape = float(np.mean(np.abs(residuals) / safe_denominator) * 100)
    directional_accuracy = float(np.mean(np.sign(actual_array) == np.sign(predicted_array)) * 100)
    return {
        "mae": mae,
        "r2": model_metrics.get("r2"),
        "rmse": rmse,
        "mape": mape,
        "directional_accuracy": directional_accuracy,
        "fold_count": int(len(actual_array)),
        "score": mape + (rmse * 0.05),
        "residual_std": float(np.std(residuals)) if residuals.size else 0.0,
    }


def _select_model(
    config: ForecastConfig,
    evaluations: dict[str, dict[str, Any]],
) -> str:
    ranked = sorted(evaluations.items(), key=lambda item: item[1]["score"])
    if not ranked:
        raise ForecastError("No forecast model could be evaluated safely.")

    best_name, best_metrics = ranked[0]
    if config.model_strategy == "accuracy":
        return best_name

    if config.model_strategy == "explainable":
        explainable_candidates = [
            name
            for name in ("linear_regression", "ridge_regression", "moving_average", "naive_last_value")
            if name in evaluations
        ]
        return min(explainable_candidates, key=lambda name: evaluations[name]["score"])

    if best_name == "random_forest":
        for explainable_name in ("linear_regression", "ridge_regression"):
            if explainable_name in evaluations:
                explainable_score = evaluations[explainable_name]["score"]
                if explainable_score <= best_metrics["score"] * 1.05:
                    return explainable_name

    return best_name


def _build_confidence_bounds(predictions: list[float], residual_std: float) -> tuple[list[float], list[float]]:
    if residual_std <= 0:
        residual_std = max(abs(predictions[-1]) * 0.03 if predictions else 1.0, 1.0)

    lower: list[float] = []
    upper: list[float] = []
    for index, prediction in enumerate(predictions, start=1):
        interval = 1.96 * residual_std * math.sqrt(index)
        lower.append(float(prediction - interval))
        upper.append(float(prediction + interval))
    return lower, upper


def _season_length_for_frequency(resolved_frequency: str) -> int | None:
    return {
        "D": 7,
        "W": 4,
        "M": 12,
        "Q": 4,
    }.get(resolved_frequency)


def _season_bucket(date_value: pd.Timestamp, resolved_frequency: str) -> int:
    if resolved_frequency == "D":
        return int(date_value.dayofweek)
    if resolved_frequency == "W":
        return int(date_value.isocalendar().week % 4)
    if resolved_frequency == "M":
        return int(date_value.month)
    if resolved_frequency == "Q":
        return int(date_value.quarter)
    return 0


def _estimate_seasonal_adjustment(
    frame: pd.DataFrame,
    future_dates: pd.DatetimeIndex,
    resolved_frequency: str,
) -> tuple[list[float], dict[str, Any]]:
    season_length = _season_length_for_frequency(resolved_frequency)
    if season_length is None or len(frame) < season_length * 2:
        return [0.0 for _ in future_dates], {"applied": False, "method": None, "season_length": season_length}

    rolling_trend = frame["target"].rolling(window=season_length, min_periods=max(2, season_length // 2)).mean()
    seasonal_source = (frame["target"] - rolling_trend).fillna(0.0)
    seasonal_buckets: dict[int, list[float]] = {}
    for date_value, seasonal_value in zip(frame["date"], seasonal_source):
        seasonal_buckets.setdefault(_season_bucket(pd.Timestamp(date_value), resolved_frequency), []).append(float(seasonal_value))

    seasonal_profile = {
        bucket: float(np.mean(values))
        for bucket, values in seasonal_buckets.items()
        if values
    }
    if not seasonal_profile:
        return [0.0 for _ in future_dates], {"applied": False, "method": None, "season_length": season_length}

    adjustments = [
        float(seasonal_profile.get(_season_bucket(pd.Timestamp(date_value), resolved_frequency), 0.0))
        for date_value in future_dates
    ]
    return adjustments, {
        "applied": True,
        "method": "cyclical_mean_decomposition",
        "season_length": season_length,
    }


def _clone_prepared_with_horizon(prepared: PreparedForecastData, periods: int) -> PreparedForecastData:
    future_dates = pd.date_range(
        start=prepared.dataframe["date"].iloc[-1],
        periods=periods + 1,
        freq=_to_pandas_frequency(prepared.resolved_frequency),
    )[1:]
    return PreparedForecastData(
        dataframe=prepared.dataframe.copy(),
        resolved_frequency=prepared.resolved_frequency,
        horizon_periods=periods,
        future_dates=future_dates,
    )


def _build_standardized_forecast_payload(
    *,
    history_frame: pd.DataFrame,
    resolved_frequency: str,
    future_dates: pd.DatetimeIndex,
    predictions: list[float],
    lower_bounds: list[float],
    upper_bounds: list[float],
) -> dict[str, list[dict[str, Any]]]:
    slices: dict[str, list[dict[str, Any]]] = {}
    history_slice_frame = history_frame.loc[:, ["date", "target"]].copy() if {"date", "target"}.issubset(history_frame.columns) else pd.DataFrame(columns=["date", "target"])
    if not history_slice_frame.empty:
        history_slice_frame["date"] = pd.to_datetime(history_slice_frame["date"], errors="coerce", format="mixed")
        history_slice_frame["target"] = pd.to_numeric(history_slice_frame["target"], errors="coerce")
        history_slice_frame = history_slice_frame.dropna(subset=["date", "target"]).sort_values("date").reset_index(drop=True)

    for filter_name in ("current_month", "last_month"):
        if history_slice_frame.empty:
            slices[filter_name] = []
            continue
        try:
            filtered_history = apply_time_filter(history_slice_frame, filter_name, time_column="date")
        except ValueError:
            filtered_history = pd.DataFrame(columns=["date", "target"])
        slices[filter_name] = [
            {
                "date": pd.Timestamp(row["date"]).isoformat(),
                "value": float(row["target"]),
                "lower_bound": None,
                "upper_bound": None,
            }
            for _, row in filtered_history.iterrows()
        ]

    horizon_lengths = HORIZON_PERIOD_MAP.get(resolved_frequency, {})
    for horizon_name in SUPPORTED_HORIZONS:
        length = int(horizon_lengths.get(horizon_name, 0) or 0)
        if length <= 0:
            slices[horizon_name] = []
            continue
        rows: list[dict[str, Any]] = []
        for date_value, prediction, lower_bound, upper_bound in zip(
            future_dates[:length],
            predictions[:length],
            lower_bounds[:length],
            upper_bounds[:length],
        ):
            rows.append(
                {
                    "date": pd.Timestamp(date_value).isoformat(),
                    "value": float(prediction),
                    "lower_bound": float(lower_bound),
                    "upper_bound": float(upper_bound),
                }
            )
        slices[horizon_name] = rows
    return slices


def _build_forecast_confidence_payload(uncertainty_ratio: float, validation: ForecastValidationResult) -> dict[str, Any]:
    score = max(0.0, min(1.0, 1.0 - uncertainty_ratio - (len(validation.warnings) * 0.08)))
    if score >= 0.75:
        label = "high"
    elif score >= 0.5:
        label = "medium"
    else:
        label = "low"
    return {
        "score": round(score, 4),
        "label": label,
        "warnings": list(validation.warnings),
    }


def _empty_forecast_slices() -> dict[str, list[dict[str, Any]]]:
    return {horizon_name: [] for horizon_name in STANDARDIZED_FORECAST_WINDOWS}


def _aggregate_driver_importance(
    feature_names: list[str],
    raw_importance: list[float],
    driver_columns: list[str],
    frame: pd.DataFrame,
) -> list[dict[str, Any]]:
    if not feature_names or not raw_importance:
        return _fallback_driver_importance(frame, driver_columns)

    grouped: dict[str, float] = {}
    for feature_name, importance in zip(feature_names, raw_importance):
        if not feature_name.startswith("driver__"):
            continue
        normalized_feature = feature_name.replace("driver__", "", 1)
        driver_column = next(
            (
                candidate
                for candidate in driver_columns
                if normalized_feature == candidate or normalized_feature.startswith(f"{candidate}_")
            ),
            normalized_feature,
        )
        grouped[driver_column] = grouped.get(driver_column, 0.0) + abs(float(importance))

    ranked = sorted(grouped.items(), key=lambda item: item[1], reverse=True)
    if not ranked:
        return _fallback_driver_importance(frame, driver_columns)

    return [
        {
            "driver": driver_name,
            "importance": round(score, 6),
            "direction_hint": _driver_direction_hint(frame, driver_name),
        }
        for driver_name, score in ranked[:5]
    ]


def _fallback_driver_importance(frame: pd.DataFrame, driver_columns: list[str]) -> list[dict[str, Any]]:
    rankings: list[dict[str, Any]] = []
    for driver_column in driver_columns:
        series = frame[driver_column]
        if is_numeric_dtype(series):
            correlation = float(pd.Series(frame["target"]).corr(pd.to_numeric(series, errors="coerce")))
            if math.isnan(correlation):
                correlation = 0.0
            rankings.append(
                {
                    "driver": driver_column,
                    "importance": round(abs(correlation), 6),
                    "direction_hint": "positive" if correlation >= 0 else "negative",
                }
            )
        else:
            rankings.append(
                {
                    "driver": driver_column,
                    "importance": 0.1,
                    "direction_hint": "mixed",
                }
            )

    rankings.sort(key=lambda item: item["importance"], reverse=True)
    return rankings[:5]


def _driver_direction_hint(frame: pd.DataFrame, driver_column: str) -> str:
    series = frame[driver_column]
    if is_numeric_dtype(series):
        correlation = pd.Series(frame["target"]).corr(pd.to_numeric(series, errors="coerce"))
        if pd.isna(correlation):
            return "mixed"
        return "positive" if correlation >= 0 else "negative"
    return "mixed"


def _recent_delta(frame: pd.DataFrame, column_name: str) -> float:
    numeric_series = pd.to_numeric(frame[column_name], errors="coerce").dropna()
    if numeric_series.shape[0] < 2:
        return 0.0
    baseline = float(numeric_series.tail(min(4, len(numeric_series))).head(-1).mean())
    latest = float(numeric_series.iloc[-1])
    if abs(baseline) < 1e-9:
        return latest
    return (latest - baseline) / abs(baseline)


def _build_recommendations(
    frame: pd.DataFrame,
    predictions: list[float],
    driver_importance: list[dict[str, Any]],
    trend_status: str,
    uncertainty_ratio: float,
) -> list[dict[str, Any]]:
    current_baseline = float(frame["target"].tail(min(4, len(frame))).mean())
    forecast_average = float(np.mean(predictions)) if predictions else current_baseline
    delta_ratio = 0.0
    if abs(current_baseline) > 1e-9:
        delta_ratio = (forecast_average - current_baseline) / abs(current_baseline)

    recommendations: list[dict[str, Any]] = []
    top_driver_names = [item["driver"] for item in driver_importance[:3]]
    driver_tokens = {_normalize_token(driver_name): driver_name for driver_name in top_driver_names}

    def add_recommendation(category: str, priority: int, title: str, action: str, rationale: str, confidence: str):
        recommendations.append(
            {
                "category": category,
                "priority": priority,
                "title": title,
                "recommended_action": action,
                "rationale": rationale,
                "impact_direction": "increase_profit" if delta_ratio >= 0 else "protect_revenue",
                "confidence": confidence,
            }
        )

    confidence = "high" if uncertainty_ratio < 0.12 else "medium" if uncertainty_ratio < 0.25 else "low"

    if delta_ratio < -0.05:
        add_recommendation(
            "demand_recovery_focus",
            1,
            "Prepare a revenue-protection plan",
            "Focus next-period actions on protecting demand in the weakest segments and review the most volatile revenue drivers first.",
            f"The forecast points to a {delta_ratio:.1%} decline versus the recent baseline and the trend is classified as {trend_status}.",
            confidence,
        )
    else:
        add_recommendation(
            "growth_capture",
            1,
            "Lean into the strongest growth pockets",
            "Increase investment in the strongest-performing channels, regions, or offers while the forecast remains above the recent baseline.",
            f"The forecast points to a {delta_ratio:.1%} improvement versus the recent baseline.",
            confidence,
        )

    if any(token in driver_tokens for token in ("discount", "price", "pricing")):
        driver_name = next(driver_tokens[token] for token in driver_tokens if token in {"discount", "price", "pricing"})
        add_recommendation(
            "pricing_discount_review",
            2,
            "Review pricing and discount pressure",
            "Tighten discount guardrails and test smaller incentives in low-margin segments before widening promotions.",
            f"'{driver_name}' is one of the strongest forecast drivers and can materially move the selected KPI.",
            confidence,
        )

    if any(token in driver_tokens for token in ("region", "channel", "segment", "market")):
        driver_name = next(driver_tokens[token] for token in driver_tokens if token in {"region", "channel", "segment", "market"})
        add_recommendation(
            "channel_region_investment_shift",
            2,
            "Rebalance channel and regional investment",
            "Shift more attention and budget toward the best-converting territories or channels and reduce exposure where forecast momentum is weak.",
            f"'{driver_name}' appears among the highest-impact business drivers in the forecast.",
            confidence,
        )

    if any(token in driver_tokens for token in ("cost", "expense", "margin", "profit")):
        driver_name = next(driver_tokens[token] for token in driver_tokens if token in {"cost", "expense", "margin", "profit"})
        add_recommendation(
            "cost_control_review",
            3,
            "Review cost structure before scaling",
            "Audit the cost or margin driver before expanding spend so the next-period gains translate into real profit.",
            f"'{driver_name}' is materially linked to the forecasted result and affects unit economics.",
            confidence,
        )

    if any(token in driver_tokens for token in ("customer", "retention", "churn", "subscriber", "users")):
        driver_name = next(driver_tokens[token] for token in driver_tokens if token in {"customer", "retention", "churn", "subscriber", "users"})
        add_recommendation(
            "retention_demand_recovery",
            3,
            "Prioritize retention and repeat demand",
            "Use account outreach, lifecycle nudges, or loyalty offers to stabilize repeat demand before chasing pure acquisition volume.",
            f"'{driver_name}' is one of the most influential demand-side drivers in the forecast.",
            confidence,
        )

    if any(token in driver_tokens for token in ("inventory", "stock", "supply")):
        driver_name = next(driver_tokens[token] for token in driver_tokens if token in {"inventory", "stock", "supply"})
        add_recommendation(
            "inventory_supply_caution",
            3,
            "Align inventory with the forecast",
            "Tighten inventory or supply planning around the projected demand path to avoid overstock or missed revenue.",
            f"'{driver_name}' is materially connected to the forecast trajectory.",
            confidence,
        )

    if uncertainty_ratio >= 0.25:
        add_recommendation(
            "confidence_risk_review",
            4,
            "Treat the forecast as high-variance",
            "Use shorter review cycles and scenario checkpoints before committing to larger commercial or cost decisions.",
            f"The uncertainty band is wide relative to the forecast baseline ({uncertainty_ratio:.0%} of the forecast level).",
            "medium",
        )

    unique_recommendations = []
    seen_titles: set[str] = set()
    for recommendation in sorted(recommendations, key=lambda item: item["priority"]):
        if recommendation["title"] in seen_titles:
            continue
        seen_titles.add(recommendation["title"])
        unique_recommendations.append(recommendation)

    return unique_recommendations[:5]


def _classify_trend(frame: pd.DataFrame, predictions: list[float], uncertainty_ratio: float) -> str:
    current_baseline = float(frame["target"].tail(min(4, len(frame))).mean())
    forecast_average = float(np.mean(predictions)) if predictions else current_baseline
    if abs(current_baseline) < 1e-9:
        delta_ratio = 0.0
    else:
        delta_ratio = (forecast_average - current_baseline) / abs(current_baseline)

    if uncertainty_ratio >= 0.25:
        return "volatility"
    if delta_ratio >= 0.08:
        return "growth"
    if delta_ratio <= -0.08:
        return "decline"
    if delta_ratio > 0 and _recent_delta(frame, "target") < -0.05:
        return "recovery_risk"
    return "stable"


def _build_summary(
    config: ForecastConfig,
    frame: pd.DataFrame,
    predictions: list[float],
    trend_status: str,
    chosen_model: str,
    resolved_frequency: str,
    uncertainty_ratio: float,
) -> str:
    current_baseline = float(frame["target"].tail(min(4, len(frame))).mean())
    forecast_average = float(np.mean(predictions)) if predictions else current_baseline
    delta_ratio = 0.0 if abs(current_baseline) < 1e-9 else (forecast_average - current_baseline) / abs(current_baseline)
    return (
        f"{config.target_column} is forecast to move {delta_ratio:.1%} over the "
        f"{HORIZON_LABELS.get(config.horizon, config.horizon).lower()} window. "
        f"The selected {FREQUENCY_LABELS[resolved_frequency].lower()} model is {chosen_model.replace('_', ' ')}, "
        f"the trend is classified as {trend_status.replace('_', ' ')}, and forecast uncertainty is approximately {uncertainty_ratio:.0%} of the forecast level."
    )


def _build_chart_records(frame: pd.DataFrame, future_dates: pd.DatetimeIndex, predictions: list[float], lower: list[float], upper: list[float]) -> list[dict[str, Any]]:
    history_records = [
        {
            "date": pd.Timestamp(row["date"]).isoformat(),
            "value": float(row["target"]),
            "lower_bound": None,
            "upper_bound": None,
            "series": "Historical",
        }
        for _, row in frame.iterrows()
    ]
    forecast_records = [
        {
            "date": pd.Timestamp(date_value).isoformat(),
            "value": float(prediction),
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound),
            "series": "Forecast",
        }
        for date_value, prediction, lower_bound, upper_bound in zip(future_dates, predictions, lower, upper)
    ]
    return history_records + forecast_records


def _build_forecast_dashboard(
    *,
    config: ForecastConfig,
    chart_records: list[dict[str, Any]],
    driver_importance: list[dict[str, Any]],
    current_baseline: float,
    forecast_average: float,
    active_filter: str | None = None,
) -> dict[str, Any]:
    pie_total = max(abs(current_baseline), 1.0) + max(abs(forecast_average), 1.0)
    pie_rows = [
        {
            "segment": "Historical baseline",
            "value": float(max(abs(current_baseline), 0.0)),
            "share": round(float(max(abs(current_baseline), 0.0) / pie_total), 4),
        },
        {
            "segment": "Forecast average",
            "value": float(max(abs(forecast_average), 0.0)),
            "share": round(float(max(abs(forecast_average), 0.0) / pie_total), 4),
        },
    ]

    return {
        "charts": [
            {
                "type": "line",
                "purpose": "trend",
                "title": f"{config.target_column} Trend",
                "x": "date",
                "y": "value",
                "series_key": "series",
                "time_column": "date",
                "rows": chart_records,
                "layout": {"x": 0, "y": 0, "w": 8, "h": 4},
                "drilldown": {"enabled": True, "field": "series"},
            },
            {
                "type": "bar",
                "purpose": "comparison",
                "title": "Forecast Drivers",
                "x": "driver",
                "y": "importance",
                "rows": driver_importance,
                "layout": {"x": 8, "y": 0, "w": 4, "h": 2},
                "drilldown": {"enabled": True, "field": "driver"},
            },
            {
                "type": "pie",
                "purpose": "distribution",
                "title": "Baseline vs Forecast Mix",
                "x": "segment",
                "y": "value",
                "rows": pie_rows,
                "layout": {"x": 8, "y": 2, "w": 4, "h": 2},
                "drilldown": {"enabled": True, "field": "segment"},
            },
        ],
        "filters": build_time_filter_options(),
        "kpis": [
            {"metric": "current_baseline", "value": float(current_baseline)},
            {"metric": "forecast_average", "value": float(forecast_average)},
            {"metric": "driver_count", "value": int(len(driver_importance))},
        ],
        "layout": {"type": "grid", "columns": 12, "row_height": 120, "mode": "tableau_like"},
        "drilldown_ready": True,
        "time_column": "date",
        "applied_time_filter": active_filter,
        "active_filter": active_filter,
        "visualization_type": "line",
    }


def _build_trend_fallback_output(
    *,
    df: pd.DataFrame,
    config: ForecastConfig,
    error_message: str,
    validation: ForecastValidationResult | None = None,
    workflow_context: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    validation_payload = validate_time_series(
        df,
        date_column=config.date_column or None,
        target_column=config.target_column or None,
        aggregation_frequency=(validation.resolved_frequency if validation and validation.resolved_frequency else config.aggregation_frequency),
    )
    frame = validation_payload.get("cleaned_df")
    if not isinstance(frame, pd.DataFrame) or frame.empty or len(frame) < 2:
        return None

    resolved_frequency = str(validation_payload.get("frequency") or validation.resolved_frequency or "")
    if resolved_frequency not in HORIZON_PERIOD_MAP:
        return None

    compatible_horizons = HORIZON_PERIOD_MAP.get(resolved_frequency, {})
    fallback_horizon = config.horizon if config.horizon in compatible_horizons else (next(iter(compatible_horizons)) if compatible_horizons else "")
    horizon_periods = int(compatible_horizons.get(fallback_horizon, 0) or 0)
    if horizon_periods <= 0:
        return None

    rolling_window = max(2, min(7, len(frame)))
    target_series = pd.to_numeric(frame["target"], errors="coerce").ffill().bfill()
    rolling_mean = target_series.rolling(window=rolling_window, min_periods=1).mean()
    deltas = rolling_mean.diff().dropna()
    slope = float(deltas.tail(min(3, len(deltas))).mean()) if not deltas.empty else 0.0

    future_dates = pd.date_range(
        start=frame["date"].iloc[-1],
        periods=horizon_periods + 1,
        freq=_to_pandas_frequency(resolved_frequency),
    )[1:]
    last_value = float(rolling_mean.iloc[-1])
    predictions = [max(last_value + (slope * (index + 1)), 0.0) for index in range(horizon_periods)]
    residual_std = float(target_series.diff().dropna().std() or target_series.std() or max(abs(slope), 1.0) * 0.15)
    lower_bounds, upper_bounds = _build_confidence_bounds(predictions, residual_std)
    chart_records = _build_chart_records(frame, future_dates, predictions, lower_bounds, upper_bounds)
    standardized_forecast = _build_standardized_forecast_payload(
        history_frame=frame,
        resolved_frequency=resolved_frequency,
        future_dates=future_dates,
        predictions=predictions,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
    )
    uncertainty_ratio = 0.0
    if predictions:
        uncertainty_ratio = float(
            np.mean(
                [
                    (upper - lower) / max(abs(prediction), 1.0)
                    for prediction, lower, upper in zip(predictions, lower_bounds, upper_bounds)
                ]
            )
        )
    trend_status = _classify_trend(frame, predictions, uncertainty_ratio)
    current_baseline = float(target_series.tail(min(4, len(target_series))).mean())
    forecast_average = float(np.mean(predictions)) if predictions else current_baseline
    active_filter = str((workflow_context or {}).get("time_filter") or "").strip().lower() or None
    dashboard = _build_forecast_dashboard(
        config=config,
        chart_records=chart_records,
        driver_importance=[],
        current_baseline=current_baseline,
        forecast_average=forecast_average,
        active_filter=active_filter,
    )
    confidence_score = max(0.2, min(0.7, 1.0 - uncertainty_ratio - 0.15))
    confidence_payload = {
        "score": round(confidence_score, 4),
        "label": "medium" if confidence_score >= 0.5 else "low",
        "warnings": list(dict.fromkeys([error_message] + list(validation_payload.get("warnings") or []) + list((validation or ForecastValidationResult([], [], None, 0, 0, [], None, None)).warnings))),
    }
    forecast_table = pd.DataFrame(
        {
            "date": future_dates,
            "forecast": predictions,
            "lower_bound": lower_bounds,
            "upper_bound": upper_bounds,
        }
    )
    fallback_validation = validation or ForecastValidationResult(
        errors=[],
        warnings=list(validation_payload.get("warnings") or []),
        resolved_frequency=resolved_frequency,
        history_points=int(len(frame)),
        minimum_history_points=int(validation_payload.get("minimum_data_points") or _minimum_history_for_frequency(resolved_frequency)),
        compatible_horizons=list(compatible_horizons.keys()),
        date_start=str(pd.Timestamp(frame["date"].min()).date()),
        date_end=str(pd.Timestamp(frame["date"].max()).date()),
    )
    data_score = _build_forecast_data_score(fallback_validation)
    data_quality = build_data_quality_report(frame)
    model_metrics = {"mae": None, "r2": None}
    model_quality = "moderate" if len(frame) >= max(4, fallback_validation.minimum_history_points // 2) else "weak"
    risk = build_risk_statement(data_quality, model_quality)
    user_facing_error = _user_facing_forecast_error(error_message)
    summary = f"Forecast fallback used rolling trend analysis because {user_facing_error}"
    decision_layer = build_decision_layer(
        {
            "forecast_table": forecast_table,
            "history_baseline": current_baseline,
            "delta_ratio": 0.0 if abs(current_baseline) < 1e-9 else (forecast_average - current_baseline) / abs(current_baseline),
            "target_column": config.target_column,
            "trend_status": trend_status,
            "uncertainty_ratio": uncertainty_ratio,
            "config": forecast_config_to_dict(config),
        },
        [summary],
        plan={
            "intent": "prediction",
            "analysis_type": "time_series",
            "analysis_route": "forecasting",
            "target_column": config.target_column,
            "metric_column": config.target_column,
            "datetime_column": config.date_column,
        },
        model_quality=model_quality,
        data_quality=data_quality,
        reproducibility={"result_hash": "", "consistent_with_prior_runs": True, "prior_hash_count": 0, "consistency_validated": False},
        risk=risk,
        warnings=list(confidence_payload["warnings"]),
        learning_patterns={},
    )
    recommendations = build_forecast_recommendations(decision_layer)
    query_label = f"Forecast {config.target_column or 'target'} for {config.horizon}"
    question_payload = build_question_payload(
        df,
        source_fingerprint=str((workflow_context or {}).get("source_fingerprint") or ""),
        recent_queries=[query_label],
    )
    forecast_metadata = {
        "time_column": validation_payload.get("time_column") or config.date_column or None,
        "data_points": int(frame.shape[0]),
        "frequency": FREQUENCY_LABELS.get(resolved_frequency, resolved_frequency).lower(),
        "filled_missing_timestamps": int(validation_payload.get("filled_missing_timestamps") or 0),
    }
    auto_config = build_auto_forecast_config(df, config)
    forecast_eligibility = build_forecast_eligibility(df)
    pipeline_trace = _build_forecast_pipeline_trace(
        config=config,
        validation=fallback_validation,
        chosen_model_name="trend_analysis_fallback",
        recommendations=recommendations,
        decision_layer=decision_layer,
        error=user_facing_error,
        data_quality=data_quality,
        model_metrics=model_metrics,
        model_quality=model_quality,
        risk=risk,
        explanation={"top_features": [], "impact": []},
        limitations=[_forecast_fix_suggestion(error_message)],
        logged_failure_count=1,
        inconsistency_detected=False,
        result_hash=None,
        dataset_fingerprint=(workflow_context or {}).get("source_fingerprint"),
    )
    return {
        "status": "FALLBACK",
        "error": {
            "message": user_facing_error,
            "suggestion": _forecast_fix_suggestion(user_facing_error),
        },
        "error_message": user_facing_error,
        "summary": summary,
        "forecast": standardized_forecast,
        "time_series": chart_records,
        "confidence": confidence_payload,
        "auto_config": auto_config,
        "forecast_eligibility": forecast_eligibility,
        "forecast_metadata": forecast_metadata,
        "context": question_payload["context"],
        "suggestions": question_payload["suggestions"],
        "recommended_next_step": question_payload["recommended_next_step"],
        "suggested_questions": question_payload["suggested_questions"],
        "dashboard": dashboard,
        "result": forecast_table,
        "forecast_table": forecast_table,
        "comparison_table": pd.DataFrame(),
        "driver_importance_table": pd.DataFrame(),
        "chart_records": chart_records,
        "decision_layer": decision_layer,
        "recommendations": recommendations,
        "driver_importance": [],
        "evaluation_metrics": {},
        "baseline_metrics": {},
        "chosen_model": "trend_analysis_fallback",
        "trend_status": trend_status,
        "uncertainty_ratio": round(uncertainty_ratio, 6),
        "resolved_frequency": resolved_frequency,
        "history_points": int(frame.shape[0]),
        "history_start": str(pd.Timestamp(frame["date"].min()).date()),
        "history_end": str(pd.Timestamp(frame["date"].max()).date()),
        "horizon": fallback_horizon or config.horizon,
        "horizon_periods": horizon_periods,
        "config": forecast_config_to_dict(config),
        "data_quality": data_quality,
        "model_metrics": model_metrics,
        "explanation": {"top_features": [], "impact": []},
        "model_quality": model_quality,
        "risk": risk,
        "dataset_fingerprint": str((workflow_context or {}).get("source_fingerprint") or ""),
        "reproducibility": {"dataset_fingerprint": str((workflow_context or {}).get("source_fingerprint") or ""), "pipeline_trace_hash": "", "result_hash": "", "consistent_with_prior_runs": True, "prior_hash_count": 0, "consistency_validated": False},
        "result_hash": "",
        "inconsistency_detected": False,
        "limitations": [_forecast_fix_suggestion(error_message)],
        "data_score": data_score,
        "pipeline_trace": pipeline_trace,
        "cache_status": {"status": "pending"},
        "memory_update": {"status": "pending"},
        "failure_events": [],
        "failure_patterns": get_failure_patterns(),
        "trend_analysis": {
            "method": "rolling_trend_fallback",
            "rolling_window": rolling_window,
            "slope_per_period": round(slope, 6),
        },
        "active_filter": active_filter,
        "visualization_type": "line",
    }


def _build_forecast_error_output(
    *,
    df: pd.DataFrame,
    config: ForecastConfig,
    error_message: str,
    validation: ForecastValidationResult | None = None,
    workflow_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    suggestion = _forecast_fix_suggestion(error_message)
    user_facing_error = _user_facing_forecast_error(error_message)
    question_payload = build_question_payload(
        df,
        source_fingerprint=str((workflow_context or {}).get("source_fingerprint") or ""),
        recent_queries=[f"Forecast {config.target_column or 'target'} for {config.horizon}"],
    )
    fallback_output = _build_trend_fallback_output(
        df=df,
        config=config,
        error_message=error_message,
        validation=validation,
        workflow_context=workflow_context,
    )
    if fallback_output is not None:
        return fallback_output
    data_score = _build_forecast_data_score(validation or ForecastValidationResult([], [], None, 0, 0, [], None, None))
    pipeline_trace = _build_forecast_pipeline_trace(
        config=config,
        validation=validation or ForecastValidationResult([user_facing_error], [], None, 0, 0, [], None, None),
        chosen_model_name="unavailable",
        recommendations=[],
        error=user_facing_error,
        data_quality={},
        model_metrics={"mae": None, "r2": None},
        model_quality="weak",
        risk="high",
        explanation={"top_features": [], "impact": []},
        limitations=[suggestion],
        logged_failure_count=1,
        inconsistency_detected=False,
        result_hash=None,
        dataset_fingerprint=(workflow_context or {}).get("source_fingerprint"),
    )
    empty_table = pd.DataFrame(columns=["date", "forecast", "lower_bound", "upper_bound"])
    auto_config = build_auto_forecast_config(df, config)
    forecast_eligibility = build_forecast_eligibility(df)
    return {
        "status": "FAILED",
        "error": {
            "message": user_facing_error,
            "suggestion": suggestion,
        },
        "error_message": user_facing_error,
        "summary": f"Forecast failed: {user_facing_error}",
        "forecast": _empty_forecast_slices(),
        "time_series": [],
        "confidence": {"score": 0.0, "label": "low", "warnings": list((validation or ForecastValidationResult([], [], None, 0, 0, [], None, None)).warnings)},
        "auto_config": auto_config,
        "forecast_eligibility": forecast_eligibility,
        "forecast_metadata": {
            "time_column": config.date_column or None,
            "data_points": int(len(df)),
            "frequency": (validation.resolved_frequency if validation else None) or None,
        },
        "context": question_payload["context"],
        "suggestions": question_payload["suggestions"],
        "recommended_next_step": question_payload["recommended_next_step"],
        "suggested_questions": question_payload["suggested_questions"],
        "dashboard": {
            "charts": [],
            "filters": build_time_filter_options(),
            "kpis": [],
            "layout": {"type": "grid", "columns": 12, "row_height": 120, "mode": "tableau_like"},
            "drilldown_ready": False,
            "active_filter": str((workflow_context or {}).get("time_filter") or "").strip().lower() or None,
            "visualization_type": None,
        },
        "active_filter": str((workflow_context or {}).get("time_filter") or "").strip().lower() or None,
        "visualization_type": None,
        "result": empty_table,
        "forecast_table": empty_table,
        "comparison_table": pd.DataFrame(),
        "driver_importance_table": pd.DataFrame(),
        "chart_records": [],
        "decision_layer": build_decision_layer(
            {
                "forecast_table": empty_table,
                "history_baseline": 0.0,
                "delta_ratio": 0.0,
                "target_column": config.target_column,
                "trend_status": "stable",
                "uncertainty_ratio": 1.0,
                "config": forecast_config_to_dict(config),
            },
            [error_message],
            plan={
                "intent": "prediction",
                "analysis_type": "time_series",
                "analysis_route": "forecasting",
                "target_column": config.target_column,
                "metric_column": config.target_column,
                "datetime_column": config.date_column,
            },
            model_quality="weak",
            data_quality={},
            reproducibility={"result_hash": "", "consistent_with_prior_runs": True, "prior_hash_count": 0, "consistency_validated": False},
            risk="high",
            warnings=[user_facing_error],
            learning_patterns={},
        ),
        "recommendations": [],
        "driver_importance": [],
        "evaluation_metrics": {},
        "baseline_metrics": {},
        "chosen_model": None,
        "trend_status": "stable",
        "uncertainty_ratio": 1.0,
        "resolved_frequency": validation.resolved_frequency if validation else None,
        "history_points": int(validation.history_points) if validation else 0,
        "history_start": validation.date_start if validation else None,
        "history_end": validation.date_end if validation else None,
        "horizon": config.horizon,
        "horizon_periods": 0,
        "config": forecast_config_to_dict(config),
        "data_quality": {},
        "model_metrics": {"mae": None, "r2": None},
        "explanation": {"top_features": [], "impact": []},
        "model_quality": "weak",
        "risk": "high",
        "dataset_fingerprint": str((workflow_context or {}).get("source_fingerprint") or ""),
        "reproducibility": {"dataset_fingerprint": str((workflow_context or {}).get("source_fingerprint") or ""), "pipeline_trace_hash": "", "result_hash": "", "consistent_with_prior_runs": True, "prior_hash_count": 0, "consistency_validated": False},
        "result_hash": "",
        "inconsistency_detected": False,
        "limitations": [suggestion],
        "data_score": data_score,
        "pipeline_trace": pipeline_trace,
        "cache_status": {"status": "pending"},
        "memory_update": {"status": "pending"},
        "failure_events": [],
        "failure_patterns": get_failure_patterns(),
    }


def run_forecast_pipeline(
    df: pd.DataFrame,
    config: ForecastConfig | dict[str, Any],
    *,
    workflow_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    workflow_context = dict(workflow_context or {})
    config = config if isinstance(config, ForecastConfig) else forecast_config_from_dict(config)
    resolved_config = _resolve_config_with_suggestions(df, config)
    forecast_eligibility = build_forecast_eligibility(df)
    query_label = f"Forecast {resolved_config.target_column or 'target'} for {resolved_config.horizon}"
    try:
        _assert_valid_forecast_time_column(df)

        if not forecast_eligibility.get("allowed"):
            return _build_forecast_error_output(
                df=df,
                config=resolved_config,
                error_message=str(forecast_eligibility.get("reason") or "No valid time column detected"),
                validation=ForecastValidationResult(
                    errors=[str(forecast_eligibility.get("reason") or "No valid time column detected")],
                    warnings=[],
                    resolved_frequency=None,
                    history_points=0,
                    minimum_history_points=0,
                    compatible_horizons=[],
                    date_start=None,
                    date_end=None,
                ),
                workflow_context=workflow_context,
            )

        validation = validate_forecast_config(df, resolved_config)
        if not validation.is_valid:
            error_message = validation.errors[0] if validation.errors else "Forecast configuration is invalid."
            return _build_forecast_error_output(
                df=df,
                config=resolved_config,
                error_message=error_message,
                validation=validation,
                workflow_context=workflow_context,
            )

        prepared = _build_prepared_forecast_data(df, resolved_config)
        frame = prepared.dataframe.copy()
        failure_events: list[dict[str, Any]] = []
        driver_columns = [column for column in resolved_config.driver_columns if column in frame.columns]
        min_history_points = _minimum_history_for_frequency(prepared.resolved_frequency)

        adapters: list[ForecastModelAdapter] = [
            LinearRegressionForecastAdapter(),
            MovingAverageAdapter(),
            NaiveLastValueAdapter(),
        ]
        if resolved_config.model_strategy in {"hybrid", "accuracy", "explainable"}:
            adapters.append(RidgeForecastAdapter())
        if resolved_config.model_strategy in {"hybrid", "accuracy"}:
            adapters.append(RandomForestForecastAdapter())

        evaluations: dict[str, dict[str, Any]] = {}
        for adapter in adapters:
            try:
                evaluations[adapter.name] = adapter.evaluate(frame, driver_columns, min_history_points)
            except ForecastError as error:
                failure_events.append(
                    log_failure(
                        query_label,
                        error,
                        f"forecast_evaluate_{adapter.name}",
                        metadata={"target_column": resolved_config.target_column, "horizon": resolved_config.horizon},
                    )
                )
                continue

        chosen_model_name = _select_model(resolved_config, evaluations)
        adapter_map = {adapter.name: adapter for adapter in adapters}
        max_supported_periods = max(HORIZON_PERIOD_MAP.get(prepared.resolved_frequency, {resolved_config.horizon: prepared.horizon_periods}).values())
        full_prepared = _clone_prepared_with_horizon(prepared, max_supported_periods)
        forecast_payload = adapter_map[chosen_model_name].forecast(frame, full_prepared, driver_columns)
        full_predictions = [max(float(value), 0.0) for value in forecast_payload["predictions"]]
        seasonal_adjustments, seasonal_metadata = _estimate_seasonal_adjustment(
            frame,
            full_prepared.future_dates,
            prepared.resolved_frequency,
        )
        if seasonal_metadata.get("applied"):
            full_predictions = [
                max(float(prediction + adjustment), 0.0)
                for prediction, adjustment in zip(full_predictions, seasonal_adjustments)
            ]

        chosen_metrics = evaluations[chosen_model_name]
        full_lower_bounds, full_upper_bounds = _build_confidence_bounds(
            full_predictions,
            float(chosen_metrics.get("residual_std", 0.0)),
        )
        uncertainty_ratio = 0.0
        if full_predictions:
            uncertainty_ratio = float(
                np.mean(
                    [
                        (upper - lower) / max(abs(prediction), 1.0)
                        for prediction, lower, upper in zip(full_predictions, full_lower_bounds, full_upper_bounds)
                    ]
                )
            )

        selected_predictions = full_predictions[: prepared.horizon_periods]
        selected_lower_bounds = full_lower_bounds[: prepared.horizon_periods]
        selected_upper_bounds = full_upper_bounds[: prepared.horizon_periods]
        selected_future_dates = full_prepared.future_dates[: prepared.horizon_periods]

        trend_status = _classify_trend(frame, selected_predictions, uncertainty_ratio)
        driver_importance = _aggregate_driver_importance(
            forecast_payload.get("feature_names", []),
            forecast_payload.get("feature_importances", []),
            driver_columns,
            frame,
        )
        model_metrics = {
            "mae": chosen_metrics.get("mae"),
            "r2": chosen_metrics.get("r2"),
        }
        try:
            explanation = build_explanation(
                model=(forecast_payload.get("model_state") or {}).get("model"),
                feature_names=forecast_payload.get("feature_names"),
                raw_importance=forecast_payload.get("feature_importances"),
            )
        except Exception as error:
            failure_events.append(
                log_failure(
                    query_label,
                    error,
                    "forecast_explainability",
                    metadata={"chosen_model": chosen_model_name},
                )
            )
            explanation = {"top_features": [], "impact": []}
        summary = _build_summary(
            resolved_config,
            frame,
            selected_predictions,
            trend_status,
            chosen_model_name,
            prepared.resolved_frequency,
            uncertainty_ratio,
        )

        forecast_table = pd.DataFrame(
            {
                "date": selected_future_dates,
                "forecast": selected_predictions,
                "lower_bound": selected_lower_bounds,
                "upper_bound": selected_upper_bounds,
            }
        )
        comparison_table = pd.DataFrame(
            [
                {
                    "model": model_name,
                    "mae": round(metrics["mae"], 4),
                    "rmse": round(metrics["rmse"], 4),
                    "mape": round(metrics["mape"], 4),
                    "directional_accuracy": round(metrics["directional_accuracy"], 2),
                    "selected": model_name == chosen_model_name,
                }
                for model_name, metrics in sorted(evaluations.items(), key=lambda item: item[1]["score"])
            ]
        )
        driver_importance_table = pd.DataFrame(driver_importance)

        data_score = _build_forecast_data_score(validation)
        data_quality = build_data_quality_report(frame)
        model_quality = interpret_model_quality(model_metrics.get("mae"), model_metrics.get("r2"))
        risk = build_risk_statement(data_quality, model_quality)
        current_baseline = float(frame["target"].tail(min(4, len(frame))).mean())
        forecast_average = float(np.mean(selected_predictions)) if selected_predictions else current_baseline

        owns_store = False
        store = None
        try:
            if workflow_context.get("source_fingerprint"):
                try:
                    store = WorkflowStore()
                    owns_store = True
                except Exception:
                    store = None
            consistency = build_forecast_consistency(
                store=store,
                result=forecast_table.to_dict(orient="records"),
                source_fingerprint=workflow_context.get("source_fingerprint"),
                target_column=resolved_config.target_column,
                horizon=resolved_config.horizon,
            )
            limitations = build_limitations(
                query=query_label,
                result=forecast_table,
                df=frame,
                warnings=list(validation.warnings),
                data_score=data_score,
                data_quality=data_quality,
                model_metrics=model_metrics,
                model_quality=model_quality,
                risk=risk,
                explanation=explanation,
                inconsistency_detected=bool(consistency.get("inconsistency_detected")),
                analysis_type="time_series",
                use_llm=bool(workflow_context.get("enable_limitations_llm")),
                store=store,
                metadata={"target_column": resolved_config.target_column, "horizon": resolved_config.horizon},
            )

            delta_ratio = 0.0 if abs(current_baseline) < 1e-9 else (forecast_average - current_baseline) / abs(current_baseline)
            decision_insights = [summary]
            if driver_importance:
                decision_insights.append(
                    f"`{driver_importance[0]['driver']}` appears as one of the strongest visible forecast drivers."
                )
            if validation.warnings:
                decision_insights.append(str(validation.warnings[0]))
            learning_patterns = get_learning_patterns(
                store,
                workflow_context.get("source_fingerprint"),
            ) if store is not None else {}
            decision_layer = build_decision_layer(
                {
                    "forecast_table": forecast_table,
                    "history_baseline": current_baseline,
                    "delta_ratio": delta_ratio,
                    "target_column": resolved_config.target_column,
                    "trend_status": trend_status,
                    "uncertainty_ratio": uncertainty_ratio,
                    "config": forecast_config_to_dict(resolved_config),
                },
                decision_insights,
                plan={
                    "intent": "prediction",
                    "analysis_type": "time_series",
                    "analysis_route": "forecasting",
                    "target_column": resolved_config.target_column,
                    "metric_column": resolved_config.target_column,
                    "datetime_column": resolved_config.date_column,
                },
                model_quality=model_quality,
                data_quality=data_quality,
                reproducibility={
                    "result_hash": consistency.get("result_hash"),
                    "consistent_with_prior_runs": not bool(consistency.get("inconsistency_detected")),
                    "prior_hash_count": int(consistency.get("prior_hash_count") or 0),
                    "consistency_validated": bool(consistency.get("consistency_validated")),
                },
                risk=risk,
                warnings=list(validation.warnings),
                learning_patterns=learning_patterns,
            )
            recommendations = build_forecast_recommendations(decision_layer)
            chart_records = _build_chart_records(
                frame,
                full_prepared.future_dates,
                full_predictions,
                full_lower_bounds,
                full_upper_bounds,
            )
            question_payload = build_question_payload(
                df,
                source_fingerprint=str(workflow_context.get("source_fingerprint") or ""),
                recent_queries=[query_label],
            )
            dashboard = _build_forecast_dashboard(
                config=resolved_config,
                chart_records=chart_records,
                driver_importance=driver_importance,
                current_baseline=current_baseline,
                forecast_average=forecast_average,
                active_filter=str(workflow_context.get("time_filter") or "").strip().lower() or None,
            )
            confidence_payload = _build_forecast_confidence_payload(uncertainty_ratio, validation)
            forecast_metadata = {
                "time_column": resolved_config.date_column,
                "data_points": int(frame.shape[0]),
                "frequency": FREQUENCY_LABELS.get(prepared.resolved_frequency, prepared.resolved_frequency).lower(),
                "filled_missing_timestamps": int(prepared.dataframe.attrs.get("missing_timestamps") or 0),
            }
            auto_config = build_auto_forecast_config(df, resolved_config)
            forecast_eligibility = build_forecast_eligibility(df)
            standardized_forecast = _build_standardized_forecast_payload(
                history_frame=frame,
                resolved_frequency=prepared.resolved_frequency,
                future_dates=full_prepared.future_dates,
                predictions=full_predictions,
                lower_bounds=full_lower_bounds,
                upper_bounds=full_upper_bounds,
            )
            artifact_payload = {
                "config": forecast_config_to_dict(resolved_config),
                "workflow_context": workflow_context or {},
                "resolved_frequency": prepared.resolved_frequency,
                "history_points": int(frame.shape[0]),
                "chosen_model": chosen_model_name,
                "baseline_metrics": evaluations,
                "evaluation_metrics": chosen_metrics,
                "model_metrics": model_metrics,
                "explanation": explanation,
                "trend_status": trend_status,
                "uncertainty_ratio": uncertainty_ratio,
                "forecast_records": [
                    {
                        "date": pd.Timestamp(row["date"]).isoformat(),
                        "forecast": float(row["forecast"]),
                        "lower_bound": float(row["lower_bound"]),
                        "upper_bound": float(row["upper_bound"]),
                    }
                    for _, row in forecast_table.iterrows()
                ],
                "forecast": standardized_forecast,
                "time_series": chart_records,
                "confidence": confidence_payload,
                "forecast_metadata": forecast_metadata,
                "context": question_payload["context"],
                "suggestions": question_payload["suggestions"],
                "recommended_next_step": question_payload["recommended_next_step"],
                "dashboard": dashboard,
                "suggested_questions": question_payload["suggested_questions"],
                "seasonal_decomposition": seasonal_metadata,
                "active_filter": str(workflow_context.get("time_filter") or "").strip().lower() or None,
                "visualization_type": "line",
                "decision_layer": decision_layer,
                "recommendations": recommendations,
                "limitations": limitations,
                "driver_importance": driver_importance,
                "model_state": forecast_payload.get("model_state"),
                "data_quality": data_quality,
                "model_quality": model_quality,
                "risk": risk,
                "result_hash": consistency.get("result_hash"),
            }
            pipeline_trace = _build_forecast_pipeline_trace(
                config=resolved_config,
                validation=validation,
                chosen_model_name=chosen_model_name,
                recommendations=recommendations,
                decision_layer=decision_layer,
                error=None,
                data_quality=data_quality,
                model_metrics=model_metrics,
                model_quality=model_quality,
                risk=risk,
                explanation=explanation,
                limitations=limitations,
                logged_failure_count=len(failure_events),
                inconsistency_detected=bool(consistency.get("inconsistency_detected")),
                result_hash=consistency.get("result_hash"),
                dataset_fingerprint=workflow_context.get("source_fingerprint"),
            )
            reproducibility = build_reproducibility_metadata(
                source_fingerprint=workflow_context.get("source_fingerprint"),
                pipeline_trace=pipeline_trace,
                result_hash=consistency.get("result_hash"),
                consistency_payload=consistency,
            )
        finally:
            if owns_store and store is not None:
                store.close()

        return {
            "status": "PASSED",
            "error": None,
            "error_message": None,
            "summary": summary,
            "forecast": standardized_forecast,
            "time_series": chart_records,
            "confidence": confidence_payload,
            "auto_config": auto_config,
            "forecast_eligibility": forecast_eligibility,
            "forecast_metadata": forecast_metadata,
            "context": question_payload["context"],
            "suggestions": question_payload["suggestions"],
            "recommended_next_step": question_payload["recommended_next_step"],
            "suggested_questions": question_payload["suggested_questions"],
            "dashboard": dashboard,
            "active_filter": str(workflow_context.get("time_filter") or "").strip().lower() or None,
            "visualization_type": "line",
            "data_quality": data_quality,
            "model_metrics": model_metrics,
            "explanation": explanation,
            "model_quality": model_quality,
            "risk": risk,
            "dataset_fingerprint": str(workflow_context.get("source_fingerprint") or ""),
            "reproducibility": reproducibility,
            "result_hash": consistency.get("result_hash"),
            "inconsistency_detected": bool(consistency.get("inconsistency_detected")),
            "limitations": limitations,
            "data_score": data_score,
            "pipeline_trace": pipeline_trace,
            "cache_status": {"status": "pending"},
            "memory_update": {"status": "pending"},
            "failure_events": failure_events,
            "failure_patterns": get_failure_patterns(),
            "result": forecast_table,
            "forecast_table": forecast_table,
            "comparison_table": comparison_table,
            "driver_importance_table": driver_importance_table,
            "chart_records": chart_records,
            "decision_layer": decision_layer,
            "recommendations": recommendations,
            "driver_importance": driver_importance,
            "evaluation_metrics": chosen_metrics,
            "baseline_metrics": evaluations,
            "chosen_model": chosen_model_name,
            "trend_status": trend_status,
            "uncertainty_ratio": round(uncertainty_ratio, 6),
            "resolved_frequency": prepared.resolved_frequency,
            "history_points": int(frame.shape[0]),
            "history_start": str(frame["date"].min().date()),
            "history_end": str(frame["date"].max().date()),
            "horizon": resolved_config.horizon,
            "horizon_periods": prepared.horizon_periods,
            "config": forecast_config_to_dict(resolved_config),
            "artifact_payload": artifact_payload,
            "seasonal_decomposition": seasonal_metadata,
        }
    except ForecastError as error:
        return _build_forecast_error_output(
            df=df,
            config=resolved_config,
            error_message=str(error),
            validation=validate_forecast_config(df, resolved_config),
            workflow_context=workflow_context,
        )
    except ValueError as error:
        return _build_forecast_error_output(
            df=df,
            config=resolved_config,
            error_message=str(error),
            validation=validate_forecast_config(df, resolved_config),
            workflow_context=workflow_context,
        )
    except Exception as error:
        return _build_forecast_error_output(
            df=df,
            config=resolved_config,
            error_message=str(error) or "Forecast failed unexpectedly.",
            validation=validate_forecast_config(df, resolved_config),
            workflow_context=workflow_context,
        )


def serialize_forecast_output(output: dict[str, Any]) -> dict[str, Any]:
    serialized = dict(output)
    for key in ("result", "forecast_table", "comparison_table", "driver_importance_table"):
        serialized[key] = serialize_result(output.get(key))
    serialized.pop("artifact_payload", None)
    return serialized


def deserialize_forecast_output(payload: dict[str, Any]) -> dict[str, Any]:
    deserialized = dict(payload)
    for key in ("result", "forecast_table", "comparison_table", "driver_importance_table"):
        deserialized[key] = deserialize_result(payload.get(key))
    return deserialized


def persist_forecast_artifact(
    forecast_output: dict[str, Any],
    *,
    workflow_id: str | None,
    workflow_version: int | None,
    source_fingerprint: str,
    dataset_name: str,
) -> tuple[str, dict[str, Any]]:
    artifact_payload = dict(forecast_output.get("artifact_payload") or {})
    if str(forecast_output.get("status") or "").upper() != "PASSED" or not artifact_payload:
        raise ForecastError("Only successful forecast runs can be persisted as artifacts.")
    object_identifier = workflow_id or source_fingerprint or "adhoc"
    artifact_file_name = f"{Path(dataset_name).stem}_{forecast_output.get('chosen_model', 'forecast')}_{forecast_output.get('horizon', 'forecast')}.pkl"
    object_key = build_object_key("forecast_artifacts", object_identifier, artifact_file_name)
    artifact_bytes = pickle.dumps(artifact_payload)
    stored_key = get_object_store().put_bytes(object_key, artifact_bytes, content_type="application/octet-stream")

    metadata = {
        "workflow_id": workflow_id,
        "workflow_version": workflow_version,
        "source_fingerprint": source_fingerprint,
        "target_column": forecast_output.get("config", {}).get("target_column"),
        "horizon": forecast_output.get("horizon"),
        "model_name": forecast_output.get("chosen_model"),
        "status": forecast_output.get("status", "PASSED"),
        "artifact_key": stored_key,
        "evaluation_metrics": forecast_output.get("evaluation_metrics", {}),
        "result_hash": forecast_output.get("result_hash"),
        "model_quality": forecast_output.get("model_quality"),
        "risk": forecast_output.get("risk"),
        "summary": forecast_output.get("summary"),
        "decision_layer": forecast_output.get("decision_layer", {}),
        "recommendations": forecast_output.get("recommendations", []),
    }
    return stored_key, metadata
