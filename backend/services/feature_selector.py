from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

from backend.services.target_detector import (
    coerce_datetime_series,
    infer_target_type,
    is_id_like,
)


MAX_FEATURE_ROWS = 5000
MAX_SELECTED_FEATURES = 12


def _prepare_target(series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    target_type = infer_target_type(series)
    cleaned = series.copy()
    if target_type == "classification":
        encoded = cleaned.astype("category").cat.codes.replace(-1, np.nan)
        values = encoded.to_numpy(dtype=float)
    else:
        values = pd.to_numeric(cleaned, errors="coerce").to_numpy(dtype=float)
    valid_mask = np.isfinite(values)
    return values, valid_mask


def _sample_indices(df: pd.DataFrame) -> pd.Index:
    if len(df) <= MAX_FEATURE_ROWS:
        return df.index
    return df.sample(n=MAX_FEATURE_ROWS, random_state=42).sort_index().index


def _encode_feature(series: pd.Series) -> np.ndarray:
    datetime_values = coerce_datetime_series(series, column_name=str(series.name))
    if datetime_values is not None:
        ordinal = datetime_values.map(lambda value: value.toordinal() if pd.notna(value) else np.nan)
        return ordinal.to_numpy(dtype=float).reshape(-1, 1)

    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce")
        return numeric.to_numpy(dtype=float).reshape(-1, 1)

    values = series.astype("string").replace({"<NA>": None})
    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        encoded_missing_value=-1,
    )
    encoded = encoder.fit_transform(values.to_frame())
    encoded = np.where(encoded < 0, np.nan, encoded)
    return encoded.astype(float)


def _variance_score(series: pd.Series) -> float:
    datetime_values = coerce_datetime_series(series, column_name=str(series.name))
    if datetime_values is not None:
        numeric = datetime_values.map(lambda value: value.toordinal() if pd.notna(value) else np.nan)
    elif pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce")
    else:
        unique_count = int(series.nunique(dropna=True))
        total = int(series.notna().sum())
        if total == 0:
            return 0.0
        return min(1.0, unique_count / max(total, 1))

    cleaned = pd.to_numeric(numeric, errors="coerce").dropna()
    if cleaned.empty:
        return 0.0
    variance = float(cleaned.var())
    if not math.isfinite(variance) or variance <= 0:
        return 0.0
    return variance


def _correlation_score(feature: pd.Series, target: pd.Series, *, target_type: str) -> float:
    datetime_values = coerce_datetime_series(feature, column_name=str(feature.name))
    if datetime_values is not None:
        feature_values = datetime_values.map(lambda value: value.toordinal() if pd.notna(value) else np.nan)
    else:
        feature_values = pd.to_numeric(feature, errors="coerce")

    if target_type == "classification":
        target_values = target.astype("category").cat.codes.replace(-1, np.nan)
    else:
        target_values = pd.to_numeric(target, errors="coerce")

    paired = pd.DataFrame({"feature": feature_values, "target": target_values}).dropna()
    if paired.shape[0] < 3:
        return 0.0

    correlation = paired["feature"].corr(paired["target"], method="spearman")
    if correlation is None or not math.isfinite(float(correlation)):
        return 0.0
    return float(correlation)


def _mutual_information_score(feature: pd.Series, target: pd.Series, *, target_type: str) -> float:
    X = _encode_feature(feature)
    target_values, valid_target_mask = _prepare_target(target)
    if X.shape[0] != target_values.shape[0]:
        return 0.0

    feature_values = X.reshape(-1)
    valid_mask = valid_target_mask & np.isfinite(feature_values)
    if int(valid_mask.sum()) < 3:
        return 0.0

    filtered_X = feature_values[valid_mask].reshape(-1, 1)
    filtered_y = target_values[valid_mask]
    if target_type == "classification":
        discrete_y = filtered_y.astype(int)
        if np.unique(discrete_y).shape[0] < 2:
            return 0.0
        try:
            score = mutual_info_classif(filtered_X, discrete_y, discrete_features=False, random_state=42)
        except ValueError:
            return 0.0
    else:
        if np.unique(filtered_y).shape[0] < 2:
            return 0.0
        try:
            score = mutual_info_regression(filtered_X, filtered_y, discrete_features=False, random_state=42)
        except ValueError:
            return 0.0
    return float(score[0]) if len(score) else 0.0


def _normalize_metric(values: dict[str, float]) -> dict[str, float]:
    finite_values = [abs(float(value)) for value in values.values() if math.isfinite(float(value))]
    maximum = max(finite_values) if finite_values else 0.0
    if maximum <= 0:
        return {key: 0.0 for key in values}
    return {key: abs(float(value)) / maximum for key, value in values.items()}


def _candidate_columns(df: pd.DataFrame, target: str) -> list[str]:
    candidates: list[str] = []
    row_count = max(int(df.shape[0]), 1)
    for column in df.columns:
        column_name = str(column)
        if column_name == str(target):
            continue
        series = df[column]
        if is_id_like(series, column_name=column_name):
            continue
        if int(series.nunique(dropna=True)) <= 1:
            continue
        missing_ratio = float(series.isna().mean())
        if missing_ratio > 0.4:
            continue
        non_null = int(series.notna().sum())
        if non_null < max(3, int(row_count * 0.1)):
            continue
        candidates.append(column_name)
    return candidates


def select_features(df: pd.DataFrame, target: str) -> dict[str, Any]:
    if df is None or df.empty or target not in df.columns:
        return {"selected_features": [], "importance_scores": {}}

    sampled_df = df.loc[_sample_indices(df)].copy()
    candidates = _candidate_columns(sampled_df, target)
    if not candidates:
        return {"selected_features": [], "importance_scores": {}}

    target_type = infer_target_type(sampled_df[target])
    correlation_scores: dict[str, float] = {}
    mutual_information_scores: dict[str, float] = {}
    variance_scores: dict[str, float] = {}
    signed_scores: dict[str, float] = {}

    for column_name in candidates:
        feature = sampled_df[column_name]
        corr_score = _correlation_score(feature, sampled_df[target], target_type=target_type)
        mi_score = _mutual_information_score(feature, sampled_df[target], target_type=target_type)
        variance = _variance_score(feature)
        correlation_scores[column_name] = abs(corr_score)
        mutual_information_scores[column_name] = mi_score
        variance_scores[column_name] = variance
        signed_scores[column_name] = corr_score

    normalized_correlation = _normalize_metric(correlation_scores)
    normalized_mi = _normalize_metric(mutual_information_scores)
    normalized_variance = _normalize_metric(variance_scores)

    combined_scores: dict[str, float] = {}
    for column_name in candidates:
        combined_score = (
            (normalized_correlation.get(column_name, 0.0) * 0.5)
            + (normalized_mi.get(column_name, 0.0) * 0.4)
            + (normalized_variance.get(column_name, 0.0) * 0.1)
        )
        sign = -1.0 if float(signed_scores.get(column_name, 0.0)) < 0 else 1.0
        combined_scores[column_name] = round(float(combined_score) * sign, 6)

    ranked_features = sorted(
        combined_scores.items(),
        key=lambda item: (-abs(item[1]), item[0]),
    )
    selected_features = [
        column_name
        for column_name, score in ranked_features
        if abs(float(score)) > 0.0
    ][:MAX_SELECTED_FEATURES]

    if not selected_features and ranked_features:
        selected_features = [ranked_features[0][0]]

    importance_scores = {
        column_name: combined_scores[column_name]
        for column_name in selected_features
    }
    return {
        "selected_features": selected_features,
        "importance_scores": importance_scores,
    }


__all__ = ["select_features"]
