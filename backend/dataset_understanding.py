from __future__ import annotations

import re
from typing import Any

import pandas as pd


_DOMAIN_KEYWORDS = {
    "business": {
        "sales",
        "revenue",
        "profit",
        "margin",
        "customer",
        "product",
        "region",
        "channel",
        "order",
        "invoice",
        "segment",
        "cost",
        "quantity",
        "gmv",
        "discount",
    },
    "medical": {
        "patient",
        "diagnosis",
        "doctor",
        "hospital",
        "vital",
        "blood",
        "heart",
        "risk",
        "medication",
        "lab",
        "glucose",
        "dose",
        "symptom",
        "treatment",
        "disease",
        "pressure",
    },
    "finance": {
        "ticker",
        "asset",
        "portfolio",
        "price",
        "return",
        "volatility",
        "balance",
        "interest",
        "loan",
        "credit",
        "cash",
        "cashflow",
        "expense",
        "equity",
        "stock",
        "bond",
        "nav",
        "pnl",
        "yield",
    },
}

_TIME_HINTS = ("date", "time", "timestamp", "day", "month", "year", "week")
_METRIC_HINTS = ("revenue", "sales", "profit", "margin", "cases", "deaths", "price", "return", "score", "amount")


def _normalize_token(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _tokenize_text(value: object) -> list[str]:
    text = str(value or "").strip().lower().replace("-", " ").replace("_", " ")
    return [part for part in text.split() if part]


def _column_and_value_tokens(df: pd.DataFrame) -> set[str]:
    tokens: set[str] = set()
    for column in df.columns:
        tokens.update(_tokenize_text(column))

    sample = df.head(30)
    for column in sample.columns:
        series = sample[column].dropna()
        if series.empty or pd.api.types.is_numeric_dtype(series):
            continue
        for value in series.astype("string").head(20):
            tokens.update(_tokenize_text(value))
    return tokens


def _detect_time_columns(df: pd.DataFrame) -> list[str]:
    detected: list[str] = []
    for column in df.columns:
        series = df[column]
        column_name = str(column or "")
        normalized_name = column_name.lower()
        if pd.api.types.is_numeric_dtype(series) and not any(token in normalized_name for token in _TIME_HINTS):
            continue
        try:
            parsed = pd.to_datetime(series, errors="coerce", format="mixed")
        except Exception:
            continue
        non_null = series.dropna()
        if non_null.empty:
            continue
        if not any(token in normalized_name for token in _TIME_HINTS):
            normalized_values = non_null.astype("string").str.strip()
            if normalized_values.empty or float(normalized_values.str.len().median()) < 6:
                continue
        success_ratio = float(parsed.notna().sum() / max(len(non_null), 1))
        if success_ratio >= 0.7:
            detected.append(column_name)
    return detected


def _rank_primary_metrics(df: pd.DataFrame, excluded: set[str] | None = None) -> list[str]:
    excluded = {str(column) for column in (excluded or set())}
    ranked: list[tuple[int, str]] = []
    for index, column in enumerate(df.select_dtypes(include=["number"]).columns):
        column_name = str(column)
        if column_name in excluded:
            continue
        normalized = _normalize_token(column_name)
        score = 0
        if any(token in normalized for token in _METRIC_HINTS):
            score += 10
        score += max(0, 5 - index)
        ranked.append((score, column_name))
    ranked.sort(reverse=True)
    return [column_name for _, column_name in ranked]


def _categorical_features(df: pd.DataFrame, *, excluded: set[str] | None = None) -> list[str]:
    excluded = {str(column) for column in (excluded or set())}
    candidates: list[str] = []
    for column in df.columns:
        column_name = str(column)
        if column_name in excluded:
            continue
        series = df[column]
        if pd.api.types.is_numeric_dtype(series):
            continue
        unique_count = int(series.astype("string").nunique(dropna=True))
        if unique_count <= min(50, max(12, int(len(df) * 0.3) if len(df) else 0)):
            candidates.append(column_name)
    return candidates


def detect_domain(df: pd.DataFrame) -> str:
    observed_tokens = _column_and_value_tokens(df)
    scores = {
        domain: len(observed_tokens.intersection(keywords))
        for domain, keywords in _DOMAIN_KEYWORDS.items()
    }
    best_domain, best_score = max(scores.items(), key=lambda item: item[1], default=("generic", 0))
    if best_score <= 0:
        return "generic"
    return best_domain


def analyze_dataset(df: pd.DataFrame) -> dict[str, Any]:
    time_columns = _detect_time_columns(df)
    primary_metrics = _rank_primary_metrics(df, excluded=set(time_columns))
    categorical_features = _categorical_features(df, excluded=set(time_columns))
    domain = detect_domain(df)

    column_summary = []
    for column in df.columns:
        column_name = str(column)
        series = df[column]
        role = "metric" if column_name in primary_metrics else "dimension"
        if column_name in time_columns:
            role = "time"
        column_summary.append(
            {
                "name": column_name,
                "dtype": str(series.dtype),
                "non_null_count": int(series.notna().sum()),
                "missing_count": int(series.isna().sum()),
                "unique_count": int(series.nunique(dropna=True)),
                "role": role,
            }
        )

    return {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "column_summary": column_summary,
        "data_types": {str(column): str(dtype) for column, dtype in df.dtypes.astype(str).to_dict().items()},
        "missing_values": {str(column): int(df[column].isna().sum()) for column in df.columns},
        "domain": domain,
        "dataset_type": domain,
        "is_time_series": bool(time_columns and primary_metrics),
        "time_columns": time_columns,
        "primary_metrics": primary_metrics,
        "categorical_features": categorical_features,
    }
