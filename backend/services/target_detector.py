from __future__ import annotations

import math
import re
from typing import Any

import pandas as pd


TARGET_NAME_HINTS = (
    "sales",
    "revenue",
    "price",
    "score",
    "target",
    "label",
    "outcome",
    "result",
    "cases",
    "profit",
    "class",
    "status",
    "churn",
    "risk",
)
TARGET_HINT_WEIGHTS = {
    "sales": 3.8,
    "revenue": 3.7,
    "cases": 3.6,
    "price": 3.4,
    "score": 3.3,
    "target": 3.2,
    "label": 3.1,
    "outcome": 3.0,
    "result": 2.8,
    "profit": 2.7,
    "class": 2.6,
    "status": 2.5,
    "churn": 2.5,
    "risk": 2.4,
}
ID_NAME_HINTS = ("id", "uuid", "guid", "identifier", "index", "code")
DATETIME_NAME_HINTS = ("date", "time", "timestamp", "day", "month", "year", "week")
QUERY_TARGET_PATTERNS = (
    r"(?:predict|forecast|estimate|classify|model|improve|optimize|optimise)\s+(?P<column>[a-z0-9_ \-]+)",
    r"(?:what affects|drivers of|drive|impact|influence)\s+(?P<column>[a-z0-9_ \-]+)",
    r"target(?:\s+column)?\s*(?:is|=|:)?\s*(?P<column>[a-z0-9_ \-]+)",
)


def normalize_name(value: Any) -> str:
    return str(value or "").strip().lower()


def normalize_token(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", normalize_name(value))


def clean_query_term(value: str | None) -> str:
    cleaned = normalize_name(value)
    cleaned = re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", cleaned)
    cleaned = re.sub(r"^(the|a|an)\s+", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def coerce_datetime_series(
    series: pd.Series,
    *,
    column_name: str | None = None,
    min_success_ratio: float = 0.6,
) -> pd.Series | None:
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce", format="mixed")

    if pd.api.types.is_numeric_dtype(series) and not any(
        token in normalize_name(column_name) for token in DATETIME_NAME_HINTS
    ):
        return None

    parsed = pd.to_datetime(series, errors="coerce", format="mixed")
    non_null_count = int(series.notna().sum())
    if non_null_count == 0:
        return None

    success_ratio = float(parsed.notna().sum()) / float(non_null_count)
    if success_ratio >= float(min_success_ratio):
        return parsed

    return None


def is_datetime_like(series: pd.Series, *, column_name: str | None = None) -> bool:
    return coerce_datetime_series(series, column_name=column_name) is not None


def is_id_like(series: pd.Series, *, column_name: str | None = None) -> bool:
    column_label = normalize_name(column_name)
    non_null = series.dropna()
    unique_count = int(non_null.nunique(dropna=True))
    total = int(non_null.shape[0])
    uniqueness_ratio = (unique_count / total) if total else 0.0

    if any(
        column_label == hint
        or column_label.endswith(f"_{hint}")
        or column_label.startswith(f"{hint}_")
        for hint in ID_NAME_HINTS
    ):
        return True

    if (
        total >= 5
        and uniqueness_ratio >= 0.95
        and (
            column_label.endswith("id")
            or any(token in column_label for token in ID_NAME_HINTS)
        )
    ):
        return True

    if total >= 5 and uniqueness_ratio >= 0.95 and pd.api.types.is_integer_dtype(series):
        ordered = pd.to_numeric(non_null, errors="coerce").dropna().sort_values()
        if ordered.shape[0] >= 5:
            deltas = ordered.diff().dropna()
            if not deltas.empty and int(deltas.nunique()) == 1 and abs(float(deltas.iloc[0])) == 1.0:
                return True

    return False


def infer_target_type(series: pd.Series) -> str:
    cleaned = series.dropna()
    if cleaned.empty:
        return "regression" if pd.api.types.is_numeric_dtype(series) else "classification"

    if pd.api.types.is_bool_dtype(series):
        return "classification"

    unique_count = int(cleaned.nunique(dropna=True))
    row_count = int(cleaned.shape[0])
    uniqueness_ratio = unique_count / max(row_count, 1)
    numeric_like = pd.api.types.is_numeric_dtype(series)

    if not numeric_like:
        return "classification"

    if unique_count >= max(3, int(row_count * 0.8)):
        return "regression"

    class_threshold = max(8, min(20, int(math.sqrt(max(row_count, 1))) + 2))
    if unique_count <= class_threshold:
        return "classification"
    if uniqueness_ratio <= 0.05 and unique_count <= 25:
        return "classification"
    return "regression"


def _find_matching_column(term: str | None, df: pd.DataFrame) -> str | None:
    normalized_term = normalize_token(clean_query_term(term))
    if not normalized_term:
        return None

    exact_matches: list[str] = []
    contains_matches: list[str] = []
    reversed_contains_matches: list[str] = []

    for column in df.columns:
        column_name = str(column)
        normalized_column = normalize_token(column_name)
        if not normalized_column:
            continue
        if normalized_column == normalized_term:
            exact_matches.append(column_name)
        elif normalized_term in normalized_column:
            contains_matches.append(column_name)
        elif normalized_column in normalized_term:
            reversed_contains_matches.append(column_name)

    if exact_matches:
        return exact_matches[0]
    if contains_matches:
        return contains_matches[0]
    if reversed_contains_matches:
        return reversed_contains_matches[0]
    return None


def _target_from_query(df: pd.DataFrame, user_query: str | None) -> str | None:
    normalized_query = normalize_name(user_query)
    if not normalized_query:
        return None

    for column in df.columns:
        column_name = str(column)
        normalized_column = normalize_token(column_name)
        if normalized_column and normalized_column in normalize_token(normalized_query):
            if not is_id_like(df[column], column_name=column_name) and not is_datetime_like(
                df[column], column_name=column_name
            ):
                return column_name

    for pattern in QUERY_TARGET_PATTERNS:
        match = re.search(pattern, normalized_query)
        if not match:
            continue
        matched_column = _find_matching_column(match.group("column"), df)
        if matched_column is None:
            continue
        if is_id_like(df[matched_column], column_name=matched_column):
            continue
        if is_datetime_like(df[matched_column], column_name=matched_column):
            continue
        return matched_column

    return None


def _score_target_candidate(
    series: pd.Series,
    *,
    column_name: str,
    row_count: int,
    column_index: int,
) -> float:
    if is_id_like(series, column_name=column_name) or is_datetime_like(series, column_name=column_name):
        return float("-inf")

    score = 0.0
    column_label = normalize_name(column_name)
    non_null = int(series.notna().sum())
    if non_null == 0:
        return float("-inf")

    unique_count = int(series.nunique(dropna=True))
    unique_ratio = unique_count / max(non_null, 1)
    missing_ratio = 1.0 - (non_null / max(row_count, 1))

    hint_bonus = max(
        (
            float(weight)
            for hint, weight in TARGET_HINT_WEIGHTS.items()
            if hint in column_label
        ),
        default=0.0,
    )
    if hint_bonus:
        score += hint_bonus
    if pd.api.types.is_numeric_dtype(series):
        score += 1.75
        if infer_target_type(series) == "regression":
            score += 0.75
    else:
        score += 1.0
    if unique_count > 1:
        score += 0.5
    if 0.0 < unique_ratio < 0.98:
        score += 0.6
    if missing_ratio <= 0.15:
        score += 0.5
    elif missing_ratio <= 0.3:
        score += 0.2

    if column_label in {"target", "label", "outcome"}:
        score += 2.0
    if column_label.endswith(("target", "label", "score", "status")):
        score += 0.8
    score += max(0.0, 0.25 - (float(column_index) * 0.03))

    return score


def build_target_suggested_questions(target: str | None) -> list[str]:
    normalized_target = str(target or "").strip()
    if not normalized_target:
        return []
    return [
        f"What affects {normalized_target}?",
        f"Predict {normalized_target}",
        f"Improve {normalized_target}",
    ]


def detect_target_column(df: pd.DataFrame, user_query: str | None = None) -> dict[str, Any]:
    if df is None or df.empty or len(df.columns) == 0:
        return {"target": "", "confidence": 0.0, "type": "regression"}

    query_target = _target_from_query(df, user_query)
    if query_target:
        return {
            "target": query_target,
            "confidence": 0.98,
            "type": infer_target_type(df[query_target]),
        }

    row_count = int(df.shape[0])
    scored_candidates: list[tuple[float, str]] = []
    for column_index, column in enumerate(df.columns):
        column_name = str(column)
        score = _score_target_candidate(
            df[column],
            column_name=column_name,
            row_count=row_count,
            column_index=column_index,
        )
        if math.isfinite(score):
            scored_candidates.append((score, column_name))

    if not scored_candidates:
        return {"target": "", "confidence": 0.0, "type": "regression"}

    scored_candidates.sort(key=lambda item: (-item[0], item[1]))
    best_score, best_column = scored_candidates[0]
    score_margin = best_score - scored_candidates[1][0] if len(scored_candidates) > 1 else best_score
    base_confidence = 0.55 + min(max(best_score, 0.0), 6.0) * 0.05
    confidence = min(0.94, max(0.35, base_confidence + max(score_margin, 0.0) * 0.03))

    return {
        "target": best_column,
        "confidence": round(confidence, 4),
        "type": infer_target_type(df[best_column]),
    }


__all__ = [
    "build_target_suggested_questions",
    "clean_query_term",
    "coerce_datetime_series",
    "detect_target_column",
    "infer_target_type",
    "is_datetime_like",
    "is_id_like",
    "normalize_name",
    "normalize_token",
]
