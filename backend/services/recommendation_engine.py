from __future__ import annotations

import math
from typing import Any

import pandas as pd


def _coerce_insights(insights: Any) -> list[str]:
    if insights is None:
        return []
    if isinstance(insights, list):
        return [str(item).strip() for item in insights if str(item).strip()]
    if isinstance(insights, str):
        return [line.strip(" -*\t") for line in insights.splitlines() if line.strip(" -*\t")]
    return [str(insights).strip()]


def _dataframe_recommendations(result: pd.DataFrame, plan: dict[str, Any] | None) -> list[str]:
    if result.empty:
        return []

    plan = dict(plan or {})
    recommendations: list[str] = []
    group_column = str(plan.get("group_column") or "")
    numeric_columns = [str(column) for column in result.select_dtypes(include="number").columns.tolist()]
    if group_column and group_column in result.columns and numeric_columns:
        metric_column = numeric_columns[0]
        ranked = result.sort_values(metric_column, ascending=False).reset_index(drop=True)
        top_row = ranked.iloc[0]
        recommendations.append(
            f"Increase focus on {top_row[group_column]} because it currently leads {metric_column}."
        )
        if len(ranked) > 1:
            bottom_row = ranked.iloc[-1]
            recommendations.append(
                f"Investigate {bottom_row[group_column]} because it is the weakest segment on {metric_column}."
            )
        return recommendations

    if {"predicted_value", "prediction_target"}.issubset({str(column) for column in result.columns}):
        target_name = str(result.iloc[0]["prediction_target"])
        trend = pd.to_numeric(result["predicted_value"], errors="coerce").diff().dropna()
        if not trend.empty and float(trend.mean()) > 0:
            recommendations.append(f"Prepare capacity for {target_name} because the near-term trajectory is increasing.")
        else:
            recommendations.append(f"Review the {target_name} plan before committing because the near-term trajectory is not clearly improving.")
        return recommendations

    if numeric_columns:
        metric_column = numeric_columns[0]
        top_value = pd.to_numeric(result[metric_column], errors="coerce").dropna()
        if not top_value.empty:
            recommendations.append(f"Prioritize the drivers behind {metric_column} because they dominate the current output.")
    return recommendations


def _dict_recommendations(result: dict[str, Any]) -> list[str]:
    recommendations: list[str] = []
    missing_by_column = dict(result.get("missing_by_column") or {})
    if missing_by_column:
        top_missing_column = max(missing_by_column.items(), key=lambda item: item[1])[0]
        recommendations.append(f"Repair {top_missing_column} first because it is the largest missing-data bottleneck.")
    if int(result.get("duplicate_rows") or 0) > 0:
        recommendations.append("Remove duplicate rows before publishing decisions from this dataset.")
    return recommendations


def _generate_analysis_recommendations(
    result: Any,
    insights: Any,
    *,
    plan: dict[str, Any] | None = None,
    warnings: list[str] | None = None,
    model_quality: str | None = None,
    risk: str | None = None,
) -> list[str]:
    recommendations: list[str] = []

    if isinstance(result, pd.DataFrame):
        recommendations.extend(_dataframe_recommendations(result, plan))
    elif isinstance(result, dict):
        recommendations.extend(_dict_recommendations(result))

    for warning in list(warnings or []):
        lowered_warning = str(warning).lower()
        if "missing" in lowered_warning:
            recommendations.append("Address high-missingness columns before scaling this result into operations.")
            break

    for insight in _coerce_insights(insights):
        lowered_insight = insight.lower()
        if any(token in lowered_insight for token in ("drop", "decline", "down", "weakest")):
            recommendations.append("Investigate the reported decline immediately and validate the affected segment in the source data.")
            break
        if any(token in lowered_insight for token in ("lead", "highest", "top", "growth")):
            recommendations.append("Replicate the strongest segment's operating pattern in adjacent segments.")
            break

    resolved_model_quality = str(model_quality or "").strip().lower()
    if resolved_model_quality == "weak":
        recommendations.append("Keep a human review gate in place because the current model quality is weak.")
    elif resolved_model_quality == "moderate":
        recommendations.append("Validate this output on a recent holdout slice before changing production targets.")

    if str(risk or "").strip().lower().startswith("high"):
        recommendations.append("Treat this result as directional only until data reliability improves.")
    elif str(risk or "").strip().lower().startswith("medium"):
        recommendations.append("Roll out any action in stages so assumptions can be monitored safely.")

    if not recommendations:
        recommendations.append("Document the current logic and rerun it after the next data refresh before making operational changes.")

    return list(dict.fromkeys(item for item in recommendations if str(item).strip()))[:5]


def _safe_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _detect_spike(series: pd.Series) -> bool:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.shape[0] < 6:
        return False
    delta = numeric.diff().dropna()
    if delta.empty:
        return False
    threshold = float(delta.std(ddof=0) or 0.0) * 2.5
    if threshold <= 0:
        return False
    return bool((delta.abs() > threshold).any())


def _generate_ml_recommendations(
    df: pd.DataFrame,
    target: str,
    insights: Any,
    feature_importance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    recommendations: list[str] = []
    opportunities: list[str] = []
    warnings: list[str] = []
    importance_scores = {
        str(feature): _safe_float(score)
        for feature, score in dict(feature_importance or {}).items()
        if str(feature).strip()
    }

    ranked_features = sorted(
        (
            (feature, score)
            for feature, score in importance_scores.items()
            if score is not None
        ),
        key=lambda item: (-abs(float(item[1])), item[0]),
    )

    for feature, score in ranked_features[:3]:
        if score is None:
            continue
        if score > 0:
            recommendations.append(f"Increase focus on {feature} because it is positively associated with {target}.")
            opportunities.append(f"Leverage {feature} to improve {target}.")
        elif score < 0:
            recommendations.append(f"Reduce or stabilize {feature} because it appears to drag down {target}.")
            opportunities.append(f"Fix the negative pressure from {feature} to protect {target}.")
        else:
            recommendations.append(f"Monitor {feature} closely because it is a meaningful driver of {target}.")

    if target in df.columns and _detect_spike(df[target]):
        warnings.append(f"Sudden spikes were detected in {target}.")
        recommendations.append(f"Investigate abrupt movement in {target} before scaling a decision.")

    missing_columns = [
        str(column)
        for column in df.columns
        if float(df[column].isna().mean()) > 0.0
    ]
    if missing_columns:
        warnings.append("Missing data is present in the dataset.")
        top_missing = max(
            missing_columns,
            key=lambda column: float(df[column].isna().mean()),
        )
        recommendations.append(f"Repair missing values in {top_missing} to improve model reliability for {target}.")

    for insight in _coerce_insights(insights):
        lowered = insight.lower()
        if "drop" in lowered or "decline" in lowered:
            recommendations.append(f"Address the reported decline drivers before expecting {target} to improve.")
            break

    if not opportunities and ranked_features:
        opportunities.append(f"Use the strongest drivers of {target} as the first optimization levers.")

    if warnings and len(warnings) >= 2:
        risk = "high"
    elif warnings:
        risk = "medium"
    else:
        risk = "low"

    if not recommendations:
        recommendations.append(f"Collect more signal for {target} before automating decisions from this dataset.")

    return {
        "recommendations": list(dict.fromkeys(recommendations))[:5],
        "risk": risk,
        "opportunities": list(dict.fromkeys(opportunities))[:5],
    }


def generate_recommendations(
    result: Any,
    insights_or_target: Any,
    *args,
    plan: dict[str, Any] | None = None,
    warnings: list[str] | None = None,
    model_quality: str | None = None,
    risk: str | None = None,
) -> Any:
    if isinstance(result, pd.DataFrame) and isinstance(insights_or_target, str) and args:
        return _generate_ml_recommendations(
            result,
            str(insights_or_target),
            args[0],
            args[1] if len(args) > 1 else None,
        )

    return _generate_analysis_recommendations(
        result,
        insights_or_target,
        plan=plan,
        warnings=warnings,
        model_quality=model_quality,
        risk=risk,
    )
