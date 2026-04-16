from __future__ import annotations

from typing import Any

import pandas as pd

from backend.services.failure_logging import log_failure


def _coerce_text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [line.strip(" -*\t") for line in value.splitlines() if line.strip(" -*\t")]
    return [str(value).strip()]


def _rule_based_limitations(
    *,
    df: pd.DataFrame | None,
    warnings: list[str],
    data_score: dict[str, Any] | None,
    data_quality: dict[str, Any] | None,
    model_metrics: dict[str, Any] | None,
    model_quality: str | None,
    risk: str | None,
    explanation: dict[str, Any] | None,
    inconsistency_detected: bool,
    analysis_type: str | None,
) -> list[str]:
    limitations: list[str] = []
    resolved_model_quality = str(model_quality or "").strip().lower()
    has_model_signals = resolved_model_quality != "not_applicable"
    if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
        row_count = int(len(df))
        missing_by_column = df.isna().sum()
        for column_name, missing_count in missing_by_column.items():
            if row_count and missing_count / row_count >= 0.4:
                limitations.append(f"Data missing 40%+ in `{column_name}`.")
        if row_count < 12:
            limitations.append("Sample size is small, so prediction stability is limited.")
        if analysis_type == "time_series":
            datetime_columns = [
                str(column)
                for column in df.columns
                if pd.api.types.is_datetime64_any_dtype(df[column])
            ]
            if not datetime_columns:
                limitations.append("Time-based forecasting assumptions are weak because no reliable datetime column is available.")

    for warning in warnings[:3]:
        limitations.append(str(warning))

    score = int((data_score or {}).get("score") or 0)
    if score and score < 70:
        limitations.append(f"Data score is {score}/100, so the result should be treated cautiously.")

    mae = (model_metrics or {}).get("mae")
    r2 = (model_metrics or {}).get("r2")
    if has_model_signals and mae is None and r2 is None:
        limitations.append("Model evaluation is incomplete, so predictive reliability is harder to verify.")
    elif has_model_signals and r2 is not None and float(r2) < 0.3:
        limitations.append("Prediction confidence is low because model fit is weak.")

    data_quality_score = float((data_quality or {}).get("score") or 0.0)
    if data_quality_score and data_quality_score < 5.0:
        limitations.append("Data quality is poor enough that automated decisions may be unreliable.")

    if resolved_model_quality == "weak":
        limitations.append("The model quality is weak, so predictions should not be treated as hard commitments.")

    if str(risk or "").strip().lower().startswith("high"):
        limitations.append("Combined trust risk is high because data quality and model reliability do not reinforce each other.")

    if has_model_signals and not list((explanation or {}).get("top_features") or []):
        limitations.append("Feature explainability is unavailable for the current model.")

    if inconsistency_detected:
        limitations.append("Results differ from prior matching runs, which may indicate data drift or non-deterministic behavior.")

    return list(dict.fromkeys(limitations))


def build_limitations(
    *,
    query: str,
    result: Any,
    df: pd.DataFrame | None = None,
    warnings: list[str] | None = None,
    data_score: dict[str, Any] | None = None,
    data_quality: dict[str, Any] | None = None,
    model_metrics: dict[str, Any] | None = None,
    model_quality: str | None = None,
    risk: str | None = None,
    explanation: dict[str, Any] | None = None,
    inconsistency_detected: bool = False,
    analysis_type: str | None = None,
    use_llm: bool = True,
    llm_model: str | None = None,
    store=None,
    metadata: dict[str, Any] | None = None,
) -> list[str]:
    limitations = _rule_based_limitations(
        df=df,
        warnings=list(warnings or []),
        data_score=data_score,
        data_quality=data_quality,
        model_metrics=model_metrics,
        model_quality=model_quality,
        risk=risk,
        explanation=explanation,
        inconsistency_detected=inconsistency_detected,
        analysis_type=analysis_type,
    )

    if not use_llm:
        return limitations

    try:
        from backend import prompt_pipeline

        llm_text = prompt_pipeline.analyze_system_weaknesses(
            query,
            result,
            model=llm_model or prompt_pipeline.DEFAULT_GEMINI_MODEL,
        )
        limitations.extend(_coerce_text_list(llm_text))
    except Exception as error:
        log_failure(
            query,
            error,
            "limitations_enrichment",
            store=store,
            metadata=metadata,
        )

    return list(dict.fromkeys(limitations))
