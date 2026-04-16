from __future__ import annotations

from typing import Any

import pandas as pd

from backend.services.data_intelligence import detect_dataset_type


FORECAST_TOKENS = ("predict", "forecast", "future")
ML_TOKENS = ("predict", "prediction", "forecast", "future", "classify", "classification", "regression", "estimate", "probability", "score")


def _normalize_query(user_query: str | None) -> str:
    return str(user_query or "").strip().lower()


def _contains_any_token(user_query: str, tokens: tuple[str, ...]) -> bool:
    normalized_query = _normalize_query(user_query)
    return any(token in normalized_query for token in tokens)


def decide_analysis_mode(df: pd.DataFrame, user_query: str) -> dict[str, Any]:
    intelligence = detect_dataset_type(df)
    has_forecast_language = _contains_any_token(user_query, FORECAST_TOKENS)
    has_ml_language = _contains_any_token(user_query, ML_TOKENS)
    has_numeric_target_candidate = bool(intelligence.get("numeric_columns"))

    if intelligence["is_time_series"] and has_forecast_language:
        return {
            "mode": "forecast",
            "reason": "Detected a valid time-series structure and forecast language in the request.",
            "confidence": 0.95,
        }

    if has_ml_language and has_numeric_target_candidate:
        if not intelligence["is_time_series"] and has_forecast_language:
            reason = "Dataset has no valid time column, so the request is being rerouted to predictive modeling."
            confidence = 0.88
        elif intelligence["is_ml_ready"]:
            reason = "Detected numeric target candidates and enough feature variety for predictive modeling."
            confidence = 0.86
        else:
            reason = "Detected a predictive request with usable numeric columns, so the engine is avoiding invalid forecasting."
            confidence = 0.78
        return {
            "mode": "ml",
            "reason": reason,
            "confidence": confidence,
        }

    return {
        "mode": "analysis",
        "reason": "Dataset and query are better suited to aggregation or exploratory analysis than forecasting.",
        "confidence": 0.8,
    }
