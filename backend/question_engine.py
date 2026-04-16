from __future__ import annotations

import pandas as pd

from backend.dataset_understanding import detect_domain as detect_dataset_domain
from backend.suggestion_engine import build_suggestion_payload


def detect_domain(df: pd.DataFrame) -> str:
    return detect_dataset_domain(df)


def generate_suggested_questions(
    df: pd.DataFrame,
    *,
    domain: str | None = None,
    limit: int = 6,
    include_time_question: bool = True,
) -> list[str]:
    del domain, include_time_question
    payload = build_suggestion_payload(df, limit=limit)
    return [str(item) for item in payload.get("suggested_questions", []) if str(item).strip()]


def build_question_payload(
    df: pd.DataFrame,
    *,
    limit: int = 6,
    source_fingerprint: str | None = None,
    recent_queries: list[str] | None = None,
) -> dict[str, object]:
    return build_suggestion_payload(
        df,
        source_fingerprint=source_fingerprint,
        recent_queries=recent_queries,
        limit=limit,
    )
