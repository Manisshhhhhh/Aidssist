from __future__ import annotations

import math
import re
from typing import Any

import pandas as pd

from backend.data_quality import build_data_quality_report
from backend.question_engine import build_question_payload
from backend.services.decision_engine import (
    build_decision_layer,
    derive_recommendations_from_decision_layer,
    ensure_decision_layer_defaults,
)
from backend.services.ml_schema_validator import validate_ml_output
from backend.services.model_quality import interpret_model_quality
from backend.services.recommendation_engine import generate_recommendations
from backend.services.result_consistency import build_reproducibility_metadata
from backend.services.trust_layer import assess_risk, build_risk_statement


ANALYSIS_INTENTS = (
    "data_cleaning",
    "analysis",
    "visualization",
    "prediction",
    "comparison",
    "root_cause",
)
ANALYSIS_TOOLS = ("SQL", "PYTHON", "EXCEL", "BI")
ANALYSIS_MODES = ("ad-hoc", "dashboard", "prediction")
_COST_ESTIMATES = ("low", "medium", "high")

_INTENT_KEYWORDS = {
    "data_cleaning": (
        "clean",
        "cleanup",
        "missing",
        "null",
        "fill",
        "dedupe",
        "duplicate",
        "invalid",
        "fix type",
        "standardize",
        "normalize",
        "trim",
    ),
    "visualization": (
        "chart",
        "plot",
        "graph",
        "visual",
        "visualize",
        "dashboard",
        "histogram",
        "scatter",
        "bar chart",
        "line chart",
        "heatmap",
    ),
    "prediction": (
        "predict",
        "prediction",
        "forecast",
        "estimate",
        "regression",
        "classify",
        "probability",
    ),
    "comparison": (
        "compare",
        "comparison",
        "versus",
        "vs",
        "difference",
        "better",
        "worse",
        "highest",
        "lowest",
        "top",
        "bottom",
    ),
    "root_cause": (
        "why",
        "root cause",
        "driver",
        "reason",
        "caused",
        "drop",
        "decline",
        "spike",
        "issue",
    ),
}


def _normalize_text(value: str | None) -> str:
    return str(value or "").strip().lower()


def _normalize_column_token(value: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", "", _normalize_text(value))


def _clean_column_query_term(term: str | None) -> str:
    cleaned = _normalize_text(term)
    cleaned = re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", cleaned)
    cleaned = re.sub(r"^(the|a|an)\s+", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def _find_matching_column(term: str | None, df: pd.DataFrame) -> str | None:
    normalized_term = _normalize_column_token(_clean_column_query_term(term))
    if not normalized_term:
        return None

    exact_matches: list[str] = []
    contains_matches: list[str] = []
    reversed_contains_matches: list[str] = []

    for column in df.columns:
        column_name = str(column)
        normalized_column = _normalize_column_token(column_name)
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


def classify_analysis_intent(query: str) -> str:
    normalized_query = _normalize_text(query)
    if not normalized_query:
        return "analysis"

    if any(keyword in normalized_query for keyword in _INTENT_KEYWORDS["prediction"]):
        return "prediction"
    if any(keyword in normalized_query for keyword in _INTENT_KEYWORDS["root_cause"]):
        return "root_cause"
    if any(keyword in normalized_query for keyword in _INTENT_KEYWORDS["comparison"]):
        return "comparison"
    if any(keyword in normalized_query for keyword in _INTENT_KEYWORDS["visualization"]):
        return "visualization"
    if any(keyword in normalized_query for keyword in _INTENT_KEYWORDS["data_cleaning"]):
        return "data_cleaning"
    return "analysis"


def _select_metric_column(query: str, df: pd.DataFrame) -> str | None:
    lowered_query = _normalize_text(query)
    query_token = _normalize_column_token(lowered_query)
    numeric_columns = [str(column) for column in df.select_dtypes(include="number").columns]

    for column in numeric_columns:
        normalized_column = _normalize_column_token(column)
        if normalized_column and normalized_column in query_token:
            return column

    metric_patterns = (
        r"(?:average|avg|mean|sum|total|median|max|min|predict|forecast|plot|chart|analyze|analyse|compare|distribution of)\s+(?P<column>.+?)\s+(?:by|for|across|over|vs|versus)\b",
        r"(?:average|avg|mean|sum|total|median|max|min|predict|forecast|plot|chart|analyze|analyse|compare|distribution of)\s+(?P<column>[a-z0-9_ \-]+)\b",
    )
    for pattern in metric_patterns:
        match = re.search(pattern, lowered_query)
        if not match:
            continue
        matched_column = _find_matching_column(match.group("column"), df)
        if matched_column is not None:
            return matched_column

    return numeric_columns[0] if numeric_columns else None


def _select_group_column(query: str, df: pd.DataFrame, *, excluded: set[str] | None = None) -> str | None:
    excluded = excluded or set()
    lowered_query = _normalize_text(query)
    query_token = _normalize_column_token(lowered_query)
    candidate_columns = [str(column) for column in df.columns if str(column) not in excluded]

    for column in candidate_columns:
        normalized_column = _normalize_column_token(column)
        if normalized_column and normalized_column in query_token:
            return column

    by_match = re.search(r"\bby\s+(?P<column>[a-z0-9_ \-]+)\b", lowered_query)
    if by_match:
        matched_column = _find_matching_column(by_match.group("column"), df)
        if matched_column is not None and matched_column not in excluded:
            return matched_column

    categorical = [
        str(column)
        for column in df.select_dtypes(exclude="number").columns
        if str(column) not in excluded
    ]
    if categorical:
        return categorical[0]

    for column in candidate_columns:
        if column not in excluded:
            return column
    return None


def _select_datetime_column(df: pd.DataFrame) -> str | None:
    for column in df.columns:
        series = df[column]
        if pd.api.types.is_datetime64_any_dtype(series):
            return str(column)

    for column in df.columns:
        column_name = str(column).lower()
        series = df[column]
        if not pd.api.types.is_object_dtype(series):
            continue
        if not any(token in column_name for token in ("date", "time", "day", "month", "year")):
            continue
        converted = pd.to_datetime(series, errors="coerce")
        if converted.notna().sum() >= max(3, int(len(series) * 0.6)):
            return str(column)
    return None


def _infer_required_columns(query: str, df: pd.DataFrame) -> list[str]:
    query_token = _normalize_column_token(query)
    required: list[str] = []
    for column in df.columns:
        column_name = str(column)
        normalized_column = _normalize_column_token(column_name)
        if normalized_column and normalized_column in query_token:
            required.append(column_name)
    return required


def select_analysis_type(query: str, df: pd.DataFrame) -> str:
    lowered_query = _normalize_text(query)
    time_column_detected = _select_datetime_column(df) is not None
    metric_column = _select_metric_column(query, df)
    group_column = _select_group_column(
        query,
        df,
        excluded={metric_column} if metric_column else set(),
    )
    prediction_needed = classify_analysis_intent(query) == "prediction"
    categorical_query = bool(
        group_column
        and metric_column
        and any(token in lowered_query for token in (" by ", "compare", "top", "bottom", "region", "segment", "category", "product"))
    )

    if time_column_detected and prediction_needed:
        return "time_series"
    if categorical_query:
        return "aggregation"
    if prediction_needed:
        return "ml"
    return "general"


def build_analysis_plan(query: str, df: pd.DataFrame, intent: str | None = None) -> dict[str, Any]:
    resolved_intent = intent or classify_analysis_intent(query)
    metric_column = _select_metric_column(query, df)
    group_column = _select_group_column(
        query,
        df,
        excluded={metric_column} if metric_column else set(),
    )
    datetime_column = _select_datetime_column(df)
    target_column = metric_column if resolved_intent == "prediction" else None
    required_columns = _infer_required_columns(query, df)
    analysis_type = select_analysis_type(query, df)
    analysis_route = {
        "time_series": "forecasting.py",
        "ml": "ml_pipeline",
        "aggregation": "pandas_analysis",
        "general": "pandas_analysis",
    }.get(analysis_type, "pandas_analysis")

    transformations: list[str] = []
    if df.isna().sum().sum():
        transformations.append("Assess null density before trusting the output.")
    if metric_column:
        transformations.append(f"Coerce `{metric_column}` to numeric when calculations require it.")
    if resolved_intent == "prediction" and datetime_column:
        transformations.append(f"Convert `{datetime_column}` into an ordered time feature for local modeling.")
    if resolved_intent == "data_cleaning":
        transformations.extend(
            (
                "Profile missing values by column.",
                "Identify duplicate rows before applying any fill strategy.",
                "Preserve a preview of the cleaned dataset for user review.",
            )
        )

    method = "llm_assisted"
    if analysis_type == "time_series":
        method = "forecasting_pipeline"
    elif resolved_intent == "prediction" and target_column:
        method = "deterministic_prediction"
    elif analysis_type == "aggregation":
        method = "deterministic_aggregation"
    elif resolved_intent in {"data_cleaning", "analysis", "comparison", "visualization", "root_cause"}:
        method = "deterministic" if (metric_column or group_column or resolved_intent == "data_cleaning") else "llm_assisted"

    return {
        "intent": resolved_intent,
        "analysis_type": analysis_type,
        "analysis_route": analysis_route,
        "required_columns": required_columns,
        "metric_column": metric_column,
        "group_column": group_column,
        "target_column": target_column,
        "datetime_column": datetime_column,
        "transformations": transformations,
        "method": method,
        "steps": [
            "Validate dataframe health and required columns.",
            "Profile data quality and choose the safest analysis route.",
            "Apply safe coercions for the selected columns.",
            "Execute the chosen analysis path and verify the result.",
        ],
    }


def validate_analysis_request(query: str, df: pd.DataFrame, plan: dict[str, Any]) -> dict[str, Any]:
    normalized_query = _normalize_text(query)
    warnings: list[str] = []
    blocking_errors: list[str] = []

    if not isinstance(df, pd.DataFrame):
        return {"warnings": [], "blocking_errors": ["The analysis runtime did not receive a pandas dataframe."]}

    row_count = int(len(df))
    column_count = int(len(df.columns))
    if row_count == 0 or column_count == 0:
        blocking_errors.append("The dataset is empty, so no reliable analysis can be executed.")
        return {"warnings": warnings, "blocking_errors": blocking_errors}

    data_quality_report = build_data_quality_report(df)
    df_profile = dict(data_quality_report.get("data_profile") or {})
    data_quality_score = float(data_quality_report.get("score") or 0.0)
    normalized_quality_score = float(data_quality_report.get("data_quality_score") or 0.0)
    warnings.extend(list(data_quality_report.get("warnings") or []))

    missing_by_column = df.isna().sum()
    missing_total = int(missing_by_column.sum())
    if missing_total:
        warnings.append(f"The dataset still contains {missing_total:,} missing values.")
        for column_name, missing_count in missing_by_column.items():
            if row_count and missing_count / row_count >= 0.4:
                warnings.append(
                    f"Column `{column_name}` has high missingness ({missing_count:,}/{row_count:,} rows), which can distort the result."
                )

    duplicate_rows = int(df.duplicated().sum())
    if duplicate_rows:
        warnings.append(f"The dataset contains {duplicate_rows:,} duplicate rows that may bias comparisons.")

    if row_count < 5:
        warnings.append("The dataset is very small, so the output should be treated as directional only.")

    metric_column = plan.get("metric_column")
    if metric_column:
        coerced_metric = pd.to_numeric(df[metric_column], errors="coerce")
        usable_metric_rows = int(coerced_metric.notna().sum())
        if usable_metric_rows == 0:
            blocking_errors.append(
                f"Column `{metric_column}` does not contain usable numeric values for this request."
            )
        elif usable_metric_rows < max(3, int(row_count * 0.5)):
            warnings.append(
                f"Column `{metric_column}` has limited numeric coverage ({usable_metric_rows:,}/{row_count:,} usable rows)."
            )

    if plan.get("intent") == "prediction":
        target_column = plan.get("target_column")
        if not target_column:
            blocking_errors.append("Prediction requires a clear numeric target column.")
        elif row_count < 12:
            blocking_errors.append("Prediction needs at least 12 rows for a minimally stable model.")
        if plan.get("datetime_column") is None and row_count < 24:
            warnings.append("No reliable date column was detected, so prediction will fall back to row-order trend modeling.")
        if plan.get("analysis_type") == "time_series" and plan.get("datetime_column") is None:
            blocking_errors.append("A time-series request needs a valid date column before forecasting can run safely.")
        invalid_total = int(dict(df_profile.get("invalid_summary") or {}).get("total_invalid_values") or 0)
        anomaly_total = int(dict(df_profile.get("anomaly_summary") or {}).get("total_anomalies") or 0)
        if normalized_quality_score < 0.75:
            warnings.append(
                f"Prediction reliability is reduced because the data quality score is {normalized_quality_score:.2f}."
            )
        if invalid_total:
            warnings.append(
                f"Prediction inputs include {invalid_total:,} values that do not match the inferred column types."
            )
        if anomaly_total:
            warnings.append(
                f"Prediction inputs still contain {anomaly_total:,} anomalous numeric observations."
            )

    if any(token in normalized_query for token in ("next month", "next quarter", "next year", "forecast")) and plan.get("datetime_column") is None:
        warnings.append("Forecast-style language was detected without a reliable date column.")

    return {
        "warnings": list(dict.fromkeys(warnings)),
        "blocking_errors": list(dict.fromkeys(blocking_errors)),
        "data_profile": df_profile,
        "data_quality_score": normalized_quality_score,
        "anomalies": dict(data_quality_report.get("anomalies") or {}),
        "data_quality": data_quality_report,
    }


def _build_top_rows_code(row_count: int) -> str:
    safe_count = max(1, int(row_count))
    return f"result = df.head({safe_count}).copy()"


def _build_cleaning_audit_code() -> str:
    return "\n".join(
        (
            "import pandas as pd",
            "",
            "deduped_df = df.drop_duplicates().copy()",
            "missing_counts = df.isna().sum().sort_values(ascending=False)",
            "duplicate_rows = int(df.duplicated().sum())",
            "dtype_summary = df.dtypes.astype(str).reset_index()",
            "dtype_summary.columns = ['column', 'dtype']",
            "preview = deduped_df.head(20).where(pd.notna(deduped_df.head(20)), None)",
            "result = {",
            "    'row_count': int(len(df)),",
            "    'column_count': int(len(df.columns)),",
            "    'missing_by_column': missing_counts[missing_counts > 0].to_dict(),",
            "    'duplicate_rows': duplicate_rows,",
            "    'dtypes': dtype_summary.to_dict(orient='records'),",
            "    'cleaned_preview': preview.to_dict(orient='records'),",
            "}",
        )
    )


def _build_group_metric_code(metric_column: str, group_column: str, *, aggregation: str = "mean") -> str:
    if aggregation == "mean":
        label = f"average_{_normalize_column_token(metric_column) or 'value'}"
    else:
        label = f"{aggregation}_{_normalize_column_token(metric_column) or 'value'}"
    return "\n".join(
        (
            "import pandas as pd",
            "",
            f"metric_column = {metric_column!r}",
            f"group_column = {group_column!r}",
            "",
            "metric_values = pd.to_numeric(df[metric_column], errors='coerce')",
            "analysis_df = df.assign(_aidssist_metric_value=metric_values).dropna(subset=[group_column, '_aidssist_metric_value'])",
            "if analysis_df.empty:",
            '    raise ValueError("No valid rows remain after coercing the analysis columns.")',
            "result = (",
            "    analysis_df.groupby(group_column, dropna=False)['_aidssist_metric_value']",
            f"    .{aggregation}()",
            "    .sort_values(ascending=False)",
            f"    .reset_index(name={label!r})",
            ")",
        )
    )


def _build_prediction_code(target_column: str, datetime_column: str | None) -> str:
    feature_block = [
        f"target_column = {target_column!r}",
        f"datetime_column = {datetime_column!r}",
        "analysis_df = df.copy()",
        "analysis_df[target_column] = pd.to_numeric(analysis_df[target_column], errors='coerce')",
        "analysis_df = analysis_df.dropna(subset=[target_column]).reset_index(drop=True)",
        "if len(analysis_df) < 12:",
        '    raise ValueError("Prediction needs at least 12 usable rows after coercion.")',
    ]
    if datetime_column:
        feature_block.extend(
            (
                "analysis_df[datetime_column] = pd.to_datetime(analysis_df[datetime_column], errors='coerce')",
                "analysis_df = analysis_df.dropna(subset=[datetime_column]).sort_values(datetime_column).reset_index(drop=True)",
                "feature_values = analysis_df[datetime_column].map(pd.Timestamp.toordinal).to_numpy().reshape(-1, 1)",
                "step_source = analysis_df[datetime_column].diff().dropna()",
                "step_days = int(max(1, step_source.dt.days.median())) if not step_source.empty else 1",
                "future_anchor = analysis_df[datetime_column].iloc[-1]",
                "future_index = [future_anchor + pd.Timedelta(days=step_days * offset) for offset in range(1, 4)]",
                "future_features = pd.Series(future_index).map(pd.Timestamp.toordinal).to_numpy().reshape(-1, 1)",
            )
        )
    else:
        feature_block.extend(
            (
                "feature_values = analysis_df.index.to_numpy().reshape(-1, 1)",
                "future_index = list(range(len(analysis_df), len(analysis_df) + 3))",
                "future_features = pd.Series(future_index).to_numpy().reshape(-1, 1)",
            )
        )

    feature_block.extend(
        (
            "from sklearn.linear_model import LinearRegression",
            "model = LinearRegression()",
            "model.fit(feature_values, analysis_df[target_column].to_numpy())",
            "future_prediction = model.predict(future_features)",
            "result = pd.DataFrame({",
            "    'prediction_step': [1, 2, 3],",
            "    'prediction_target': [target_column] * 3,",
            "    'predicted_value': [float(value) for value in future_prediction],",
            "})",
        )
    )
    return "\n".join(("import pandas as pd", "", *feature_block))


def build_deterministic_analysis_code(query: str, df: pd.DataFrame, plan: dict[str, Any]) -> dict[str, str] | None:
    intent = plan.get("intent")
    normalized_query = _normalize_text(query)
    required_columns = set(plan.get("required_columns") or [])

    if intent == "data_cleaning":
        return {"label": "data-cleaning audit", "code": _build_cleaning_audit_code()}

    top_rows_match = re.search(r"\btop\s+(\d+)\s+rows?\b", normalized_query)
    if top_rows_match:
        row_count = int(top_rows_match.group(1))
        return {"label": f"top-{row_count}-rows shortcut", "code": _build_top_rows_code(row_count)}

    metric_column = plan.get("metric_column")
    group_column = plan.get("group_column")
    target_column = plan.get("target_column")
    datetime_column = plan.get("datetime_column")

    if intent == "prediction" and target_column:
        return {
            "label": f"local-prediction shortcut ({target_column})",
            "code": _build_prediction_code(target_column, datetime_column),
        }

    explicit_metric = bool(metric_column and metric_column in required_columns)
    explicit_group = bool(group_column and group_column in required_columns)
    if metric_column and group_column and metric_column != group_column and (explicit_metric or explicit_group):
        aggregation = "mean"
        if any(token in normalized_query for token in ("sum", "total")):
            aggregation = "sum"
        elif "median" in normalized_query:
            aggregation = "median"
        elif "min" in normalized_query:
            aggregation = "min"
        elif "max" in normalized_query or "highest" in normalized_query:
            aggregation = "max"

        return {
            "label": f"grouped-{aggregation} shortcut ({metric_column} by {group_column})",
            "code": _build_group_metric_code(metric_column, group_column, aggregation=aggregation),
        }

    return None


def validate_analysis_output(result: Any) -> list[str]:
    warnings: list[str] = []
    if result is None:
        warnings.append("The execution completed without producing a result object.")
        return warnings

    if isinstance(result, pd.DataFrame):
        if result.empty:
            warnings.append("The analysis result is empty, so there is no substantive output to interpret.")
        elif result.isna().all(axis=None):
            warnings.append("The analysis result contains only missing values.")
    elif isinstance(result, dict):
        if not result:
            warnings.append("The analysis result is empty.")
    elif isinstance(result, (list, tuple)) and not result:
        warnings.append("The analysis result is empty.")
    elif isinstance(result, float) and math.isnan(result):
        warnings.append("The analysis result is NaN.")
    return warnings


def _result_primary_metric(result: Any) -> tuple[str | None, Any]:
    if isinstance(result, pd.DataFrame) and not result.empty:
        numeric_columns = result.select_dtypes(include="number").columns.tolist()
        if numeric_columns:
            column = str(numeric_columns[0])
            return column, result.iloc[0][column]
        return str(result.columns[0]), result.iloc[0][0]
    if isinstance(result, dict):
        for key, value in result.items():
            return str(key), value
    if isinstance(result, (int, float, str)):
        return "value", result
    return None, None


def summarize_analysis_result(
    *,
    query: str,
    result: Any,
    intent: str,
    warnings: list[str],
    method: str,
) -> str:
    metric_name, metric_value = _result_primary_metric(result)
    warning_clause = ""
    if warnings:
        warning_clause = f" Data quality caveats remain: {warnings[0]}"

    if isinstance(result, pd.DataFrame):
        base = (
            f"The {intent.replace('_', ' ')} request for '{query}' returned "
            f"{len(result):,} rows across {len(result.columns):,} columns."
        )
        if metric_name is not None:
            base += f" The leading output is `{metric_name}` = {metric_value}."
        return base + warning_clause

    if isinstance(result, dict):
        keys = ", ".join(list(result.keys())[:4]) or "no keys"
        return (
            f"The {intent.replace('_', ' ')} request for '{query}' returned a structured result with keys: {keys}. "
            f"Execution used the {method.replace('_', ' ')} path."
            f"{warning_clause}"
        )

    if metric_name is not None:
        return (
            f"The {intent.replace('_', ' ')} request for '{query}' produced `{metric_name}` = {metric_value}. "
            f"Execution used the {method.replace('_', ' ')} path."
            f"{warning_clause}"
        )

    return (
        f"The {intent.replace('_', ' ')} request for '{query}' completed on the {method.replace('_', ' ')} path."
        f"{warning_clause}"
    )


def generate_analysis_insights(
    *,
    result: Any,
    intent: str,
    plan: dict[str, Any],
    warnings: list[str],
) -> list[str]:
    insights: list[str] = []

    if isinstance(result, pd.DataFrame) and not result.empty:
        numeric_columns = result.select_dtypes(include="number").columns.tolist()
        if plan.get("group_column") and numeric_columns and plan["group_column"] in result.columns:
            best_row = result.iloc[0]
            insights.append(
                f"`{best_row[plan['group_column']]}` leads the visible output on `{numeric_columns[0]}` with {best_row[numeric_columns[0]]}."
            )
        elif numeric_columns:
            column = str(numeric_columns[0])
            insights.append(
                f"The returned table highlights `{column}` first, with a leading value of {result.iloc[0][column]}."
            )
        insights.append(f"The result shape is {len(result):,} rows by {len(result.columns):,} columns, which is suitable for downstream review.")
    elif isinstance(result, dict) and result:
        first_key = next(iter(result))
        insights.append(f"The output includes `{first_key}`, which indicates the pipeline returned a structured decision payload instead of raw rows.")
        if "missing_by_column" in result and result["missing_by_column"]:
            insights.append("Missingness remains concentrated in specific columns, so cleaning actions should be prioritized before deeper modeling.")
    else:
        metric_name, metric_value = _result_primary_metric(result)
        if metric_name is not None:
            insights.append(f"The primary returned value is `{metric_name}` = {metric_value}.")

    if intent == "prediction":
        insights.append("The prediction path uses a local scikit-learn trend model, so it is best for fast directional guidance rather than long-horizon certainty.")
    elif intent == "root_cause":
        insights.append("Root-cause style prompts work best as hypothesis generation; any causal claim should still be validated against the raw source data.")
    elif intent == "data_cleaning":
        insights.append("Cleaning outputs are designed to expose reliability gaps first so the user can decide whether to transform the dataset or continue with caution.")

    if warnings:
        insights.append(f"Data quality is a live factor here: {warnings[0]}")

    return insights[:3]


def generate_analysis_recommendations(
    *,
    intent: str,
    plan: dict[str, Any],
    warnings: list[str],
    result: Any,
    insights: list[str] | None = None,
    model_quality: str | None = None,
    risk: str | None = None,
) -> list[str]:
    recommendations = generate_recommendations(
        result,
        insights or [],
        plan=plan,
        warnings=warnings,
        model_quality=model_quality,
        risk=risk,
    )

    if intent == "data_cleaning":
        recommendations.append("Persist a cleaned derived dataset so downstream analysis is reproducible and easier to audit.")
    elif intent == "root_cause":
        recommendations.append("Validate the suspected driver with a targeted slice of raw source data before acting on it.")

    return list(dict.fromkeys(recommendations))[:5]


def _normalize_model_metrics(value: Any) -> dict[str, float | None]:
    payload = value if isinstance(value, dict) else {}
    normalized: dict[str, float | None] = {"mae": None, "r2": None, "accuracy": None}
    for key in ("mae", "r2", "accuracy"):
        raw_value = payload.get(key)
        if raw_value is None:
            continue
        try:
            normalized[key] = float(raw_value)
        except (TypeError, ValueError):
            normalized[key] = None
    return normalized


def _normalize_explanation(value: Any) -> dict[str, list[Any]]:
    payload = value if isinstance(value, dict) else {}
    top_features = [str(item) for item in payload.get("top_features", []) if str(item).strip()]
    impact: list[float] = []
    for item in payload.get("impact", []):
        try:
            impact.append(float(item))
        except (TypeError, ValueError):
            continue
    aligned_length = min(len(top_features), len(impact)) if impact else len(top_features)
    if aligned_length and impact:
        top_features = top_features[:aligned_length]
        impact = impact[:aligned_length]
    elif not impact:
        top_features = top_features[:]
    return {"top_features": top_features, "impact": impact}


def _normalize_ml_intelligence(value: Any) -> dict[str, Any]:
    payload = value if isinstance(value, dict) else {}
    if not payload:
        return {}

    if "error" in payload and "fallback" in payload:
        return {
            "error": str(payload.get("error") or "").strip() or "ML output validation failed.",
            "fallback": str(payload.get("fallback") or "").strip() or "analysis_mode",
        }

    def _safe_float(item: Any, *, default: float = 0.0) -> float:
        try:
            parsed = float(item)
        except (TypeError, ValueError):
            return default
        if not math.isfinite(parsed):
            return default
        return parsed

    def _clip_score(item: Any) -> float:
        parsed = _safe_float(item)
        if parsed > 1.0 and parsed <= 10.0:
            parsed = parsed / 10.0
        return round(max(0.0, min(parsed, 1.0)), 2)

    features = [str(item).strip() for item in payload.get("features", []) if str(item).strip()]
    feature_importance = {
        str(feature).strip(): round(abs(_safe_float(score)), 2)
        for feature, score in dict(payload.get("feature_importance") or payload.get("importance_scores") or {}).items()
        if str(feature).strip()
    }
    feature_importance = {
        feature: score
        for feature, score in feature_importance.items()
        if feature in features and score > 0.0
    }
    normalized = {
        "target": str(payload.get("target") or "").strip(),
        "problem_type": str(payload.get("problem_type") or payload.get("target_type") or payload.get("type") or "").strip(),
        "features": features,
        "metrics": {
            "mae": round(_safe_float(dict(payload.get("metrics") or {}).get("mae")), 2),
            "r2": round(_safe_float(dict(payload.get("metrics") or {}).get("r2")), 2),
        },
        "top_features": [
            str(item).strip()
            for item in payload.get("top_features", [])
            if str(item).strip() and str(item).strip() in feature_importance
        ],
        "feature_importance": feature_importance,
        "predictions_sample": list(payload.get("predictions_sample") or payload.get("predictions") or [])[:5],
        "data_quality_score": _clip_score(payload.get("data_quality_score")),
        "confidence": _clip_score(payload.get("confidence")),
        "warnings": [str(item).strip() for item in payload.get("warnings", []) if str(item).strip()],
        "recommendations": [str(item).strip() for item in payload.get("recommendations", []) if str(item).strip()],
    }
    try:
        validate_ml_output(normalized)
        return normalized
    except Exception as error:
        return {
            "error": str(error),
            "fallback": "analysis_mode",
        }


def _normalize_data_quality(value: Any) -> dict[str, Any]:
    payload = value if isinstance(value, dict) else {}
    raw_score = payload.get("score")
    raw_ratio = payload.get("data_quality_score")

    def _safe_float(item: Any) -> float | None:
        try:
            parsed = float(item)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(parsed):
            return None
        return parsed

    normalized_score = _safe_float(raw_score)
    normalized_ratio = _safe_float(raw_ratio)
    if normalized_score is None and normalized_ratio is not None:
        normalized_score = normalized_ratio * 10.0 if normalized_ratio <= 1.0 else normalized_ratio
    if normalized_ratio is None and normalized_score is not None:
        normalized_ratio = normalized_score / 10.0 if normalized_score > 1.0 else normalized_score

    issues = [str(item).strip() for item in payload.get("issues", []) if str(item).strip()]
    warnings = [str(item).strip() for item in payload.get("warnings", []) if str(item).strip()] or issues[:]
    profile = payload.get("data_profile")
    if not isinstance(profile, dict):
        profile = payload.get("profile")
    if not isinstance(profile, dict):
        profile = {}
    anomalies = payload.get("anomalies")
    if not isinstance(anomalies, dict):
        anomalies = {}
    return {
        "score": round(max(0.0, min(normalized_score or 0.0, 10.0)), 2),
        "data_quality_score": round(max(0.0, min(normalized_ratio or 0.0, 1.0)), 2),
        "issues": issues,
        "warnings": warnings,
        "profile": profile,
        "data_profile": profile,
        "anomalies": anomalies,
    }


def _normalize_cleaning_report(value: Any) -> dict[str, Any]:
    payload = value if isinstance(value, dict) else {}
    before = payload.get("before") if isinstance(payload.get("before"), dict) else {}
    after = payload.get("after") if isinstance(payload.get("after"), dict) else {}
    raw_type_conversions = payload.get("type_conversions") if isinstance(payload.get("type_conversions"), dict) else {}

    def _as_float(item: Any, default: float = 0.0) -> float:
        try:
            return float(item)
        except (TypeError, ValueError):
            return default

    def _as_int(item: Any, default: int = 0) -> int:
        try:
            return int(item)
        except (TypeError, ValueError):
            return default

    return {
        "quality_score": round(_as_float(payload.get("quality_score")), 2),
        "missing_handled": _as_int(payload.get("missing_handled")),
        "duplicates_removed": _as_int(payload.get("duplicates_removed")),
        "outliers_detected": _as_int(payload.get("outliers_detected")),
        "outlier_columns": {
            str(column_name): _as_int(count)
            for column_name, count in dict(payload.get("outlier_columns") or {}).items()
        },
        "issues": [str(item).strip() for item in payload.get("issues", []) if str(item).strip()],
        "actions": [str(item).strip() for item in payload.get("actions", []) if str(item).strip()],
        "columns_dropped": [str(item).strip() for item in payload.get("columns_dropped", []) if str(item).strip()],
        "type_conversions": {
            str(column_name): {
                "from": str(dict(details or {}).get("from") or ""),
                "to": str(dict(details or {}).get("to") or ""),
            }
            for column_name, details in raw_type_conversions.items()
        },
        "before": {
            "row_count": _as_int(before.get("row_count")),
            "column_count": _as_int(before.get("column_count")),
            "missing_cells": _as_int(before.get("missing_cells")),
            "duplicate_rows": _as_int(before.get("duplicate_rows")),
        },
        "after": {
            "row_count": _as_int(after.get("row_count")),
            "column_count": _as_int(after.get("column_count")),
            "missing_cells": _as_int(after.get("missing_cells")),
            "duplicate_rows": _as_int(after.get("duplicate_rows")),
        },
    }


def _normalize_reproducibility(value: Any) -> dict[str, Any]:
    payload = value if isinstance(value, dict) else {}
    return {
        "dataset_fingerprint": str(payload.get("dataset_fingerprint") or ""),
        "pipeline_trace_hash": str(payload.get("pipeline_trace_hash") or ""),
        "result_hash": str(payload.get("result_hash") or ""),
        "consistent_with_prior_runs": bool(
            payload.get("consistent_with_prior_runs")
            if payload.get("consistent_with_prior_runs") is not None
            else True
        ),
        "prior_hash_count": int(payload.get("prior_hash_count") or 0),
        "consistency_validated": bool(payload.get("consistency_validated")),
    }


def _default_cost_for_tool(tool: str | None) -> str:
    normalized_tool = str(tool or "").strip().upper()
    if normalized_tool == "BI":
        return "medium"
    if normalized_tool == "LLM":
        return "high"
    return "low"


def _infer_tool_used(
    *,
    value: Any,
    intent: str | None,
    execution_plan: list[dict[str, Any]] | None,
    excel_analysis: dict[str, Any] | None,
    dashboard: dict[str, Any] | None,
) -> str:
    normalized = str(value or "").strip().upper()
    if normalized in ANALYSIS_TOOLS:
        return normalized
    execution_steps = list(execution_plan or [])
    step_tools = [str(step.get("tool") or "").strip().upper() for step in execution_steps]
    if "PYTHON" in step_tools:
        return "PYTHON"
    if "SQL" in step_tools:
        return "SQL"
    if "EXCEL" in step_tools:
        return "EXCEL"
    if "BI" in step_tools:
        return "BI"
    if dashboard:
        return "BI"
    if excel_analysis:
        return "EXCEL"
    if str(intent or "").strip().lower() == "visualization" and dashboard:
        return "BI"
    return "PYTHON"


def _normalize_analysis_mode(value: Any, *, intent: str | None, tool_used: str) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in ANALYSIS_MODES:
        return normalized
    if str(intent or "").strip().lower() == "prediction":
        return "prediction"
    if tool_used == "BI" or str(intent or "").strip().lower() == "visualization":
        return "dashboard"
    return "ad-hoc"


def _normalize_execution_step(value: Any, *, step_number: int) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None

    tool = str(value.get("tool") or "PYTHON").strip().upper()
    if tool not in ANALYSIS_TOOLS:
        tool = "PYTHON"

    task = str(value.get("task") or "").strip()
    if not task:
        task = {
            "SQL": "Use SQL-style reasoning for this step.",
            "EXCEL": "Build an Excel-style summary for this step.",
            "PYTHON": "Run Python analysis for this step.",
            "BI": "Build a BI dashboard artifact for this step.",
        }.get(tool, "Run Python analysis for this step.")

    payload = {
        "step": max(1, int(value.get("step") or step_number)),
        "tool": tool,
        "task": task,
    }

    query = str(value.get("query") or "").strip()
    if query:
        payload["query"] = query

    depends_on = []
    for item in list(value.get("depends_on") or []):
        try:
            dependency = int(item)
        except (TypeError, ValueError):
            continue
        if dependency > 0 and dependency not in depends_on:
            depends_on.append(dependency)
    if depends_on:
        payload["depends_on"] = depends_on

    if value.get("uses_context") is not None:
        payload["uses_context"] = bool(value.get("uses_context"))

    sql_plan = value.get("sql_plan")
    if sql_plan is not None:
        payload["sql_plan"] = str(sql_plan).strip() or None

    python_steps = [
        str(item).strip()
        for item in value.get("python_steps", [])
        if str(item).strip()
    ]
    if python_steps:
        payload["python_steps"] = python_steps

    excel_logic = value.get("excel_logic")
    if isinstance(excel_logic, dict) and excel_logic:
        payload["excel_logic"] = excel_logic

    fallback_reason = value.get("fallback_reason")
    if fallback_reason is not None:
        payload["fallback_reason"] = str(fallback_reason).strip() or None

    cost_estimate = str(value.get("cost_estimate") or "").strip().lower()
    payload["cost_estimate"] = cost_estimate if cost_estimate in _COST_ESTIMATES else _default_cost_for_tool(tool)

    return payload


def _legacy_execution_plan_to_steps(value: dict[str, Any]) -> list[dict[str, Any]]:
    payload = dict(value or {})
    steps: list[dict[str, Any]] = []
    sql_plan = payload.get("sql_plan")
    if str(sql_plan or "").strip():
        steps.append(
            {
                "step": len(steps) + 1,
                "tool": "SQL",
                "task": str(sql_plan).strip(),
                "sql_plan": str(sql_plan).strip(),
                "cost_estimate": _default_cost_for_tool("SQL"),
            }
        )

    python_steps = [
        str(item).strip()
        for item in payload.get("python_steps", [])
        if str(item).strip()
    ]
    if python_steps:
        steps.append(
            {
                "step": len(steps) + 1,
                "tool": "PYTHON",
                "task": " -> ".join(python_steps),
                "python_steps": python_steps,
                "cost_estimate": _default_cost_for_tool("PYTHON"),
            }
        )

    excel_logic = payload.get("excel_logic")
    fallback_reason = payload.get("fallback_reason")
    if isinstance(excel_logic, dict) and excel_logic:
        steps.append(
            {
                "step": len(steps) + 1,
                "tool": "EXCEL",
                "task": "Build an Excel-style analyst summary.",
                "excel_logic": excel_logic,
                "fallback_reason": str(fallback_reason).strip() if str(fallback_reason or "").strip() else None,
                "cost_estimate": _default_cost_for_tool("EXCEL"),
            }
        )
    elif str(fallback_reason or "").strip():
        steps.append(
            {
                "step": len(steps) + 1,
                "tool": "EXCEL",
                "task": "Fallback to an Excel-style analyst summary.",
                "fallback_reason": str(fallback_reason).strip(),
                "cost_estimate": _default_cost_for_tool("EXCEL"),
            }
        )
    return steps


def _normalize_execution_plan(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        normalized_steps = [
            _normalize_execution_step(item, step_number=index)
            for index, item in enumerate(value, start=1)
        ]
        normalized_steps = [item for item in normalized_steps if item]
    elif isinstance(value, dict):
        normalized_steps = _legacy_execution_plan_to_steps(value)
    else:
        normalized_steps = []

    for index, item in enumerate(normalized_steps, start=1):
        item["step"] = index
    return normalized_steps


def _default_execution_plan(*, tool_used: str, analysis_mode: str, intent: str | None) -> list[dict[str, Any]]:
    task = {
        "SQL": "Use SQL-style reasoning to answer the request.",
        "EXCEL": "Build an Excel-style summary or pivot for the request.",
        "PYTHON": "Run Python analysis for the request.",
        "BI": "Build a dashboard-oriented BI output for the request.",
    }.get(tool_used, "Run Python analysis for the request.")
    if analysis_mode == "prediction" or str(intent or "").strip().lower() == "prediction":
        task = "Run Python prediction or forecasting for the request." if tool_used == "PYTHON" else task
    return [{"step": 1, "tool": tool_used, "task": task, "cost_estimate": _default_cost_for_tool(tool_used)}]


def _normalize_execution_trace(value: Any, *, execution_plan: list[dict[str, Any]]) -> list[dict[str, Any]]:
    execution_steps = list(execution_plan or [])
    plan_lookup = {
        int(step.get("step") or index): dict(step)
        for index, step in enumerate(execution_steps, start=1)
    }

    normalized_trace: list[dict[str, Any]] = []
    for index, item in enumerate(value if isinstance(value, list) else [], start=1):
        if not isinstance(item, dict):
            continue
        step_number = max(1, int(item.get("step") or index))
        step_definition = dict(plan_lookup.get(step_number) or {})
        normalized_trace.append(
            {
                "step": step_number,
                "tool": str(item.get("tool") or step_definition.get("tool") or "").strip().upper() or None,
                "task": str(item.get("task") or step_definition.get("task") or "").strip() or None,
                "status": str(item.get("status") or "completed").strip().lower() or "completed",
                "execution_time_ms": max(0, int(item.get("execution_time_ms") or 0)),
                "cost_estimate": (
                    str(item.get("cost_estimate") or step_definition.get("cost_estimate") or _default_cost_for_tool(step_definition.get("tool"))).strip().lower()
                    if str(item.get("cost_estimate") or step_definition.get("cost_estimate") or "").strip()
                    else _default_cost_for_tool(step_definition.get("tool"))
                ),
                "warnings": [
                    str(warning).strip()
                    for warning in item.get("warnings", [])
                    if str(warning).strip()
                ],
                "error": str(item.get("error")).strip() if str(item.get("error") or "").strip() else None,
                "fallback_tool": (
                    str(item.get("fallback_tool")).strip().upper()
                    if str(item.get("fallback_tool") or "").strip()
                    else None
                ),
            }
        )

    if normalized_trace:
        return normalized_trace

    return [
        {
            "step": int(step.get("step") or index),
            "tool": str(step.get("tool") or "").strip().upper() or None,
            "task": str(step.get("task") or "").strip() or None,
            "status": "completed",
            "execution_time_ms": 0,
            "cost_estimate": str(step.get("cost_estimate") or "low").strip().lower() or "low",
            "warnings": [],
            "error": None,
            "fallback_tool": None,
        }
        for index, step in enumerate(execution_steps, start=1)
    ]


def _normalize_optimization(
    value: Any,
    *,
    execution_trace: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    payload = value if isinstance(value, dict) else {}
    trace = [dict(item) for item in list(execution_trace or []) if isinstance(item, dict)]
    execution_time_total = int(payload.get("execution_time_total") or sum(int(item.get("execution_time_ms") or 0) for item in trace))

    explicit_cost = str(payload.get("cost_estimate") or "").strip().lower()
    if explicit_cost not in _COST_ESTIMATES:
        trace_costs = [str(item.get("cost_estimate") or "low").strip().lower() for item in trace]
        if any(cost == "high" for cost in trace_costs):
            explicit_cost = "high"
        elif any(cost == "medium" for cost in trace_costs):
            explicit_cost = "medium"
        else:
            explicit_cost = "low"

    selected_plan_score = payload.get("selected_plan_score")
    try:
        normalized_score = round(max(0.0, min(1.0, float(selected_plan_score))), 4)
    except (TypeError, ValueError):
        normalized_score = 0.0

    constraints_applied = payload.get("constraints_applied")
    if not isinstance(constraints_applied, dict):
        constraints_applied = {}

    return {
        "execution_time_total": max(0, execution_time_total),
        "cost_estimate": explicit_cost,
        "optimized": bool(payload.get("optimized")),
        "parallel_execution": bool(payload.get("parallel_execution")),
        "plans_considered": max(1, int(payload.get("plans_considered") or 1)),
        "selected_plan_score": normalized_score,
        "constraints_applied": constraints_applied,
    }


def _normalize_excel_analysis(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict) or not value:
        return None
    pivot_table = value.get("pivot_table")
    if not isinstance(pivot_table, dict):
        pivot_table = {}
    aggregations = value.get("aggregations")
    if not isinstance(aggregations, dict):
        aggregations = {}
    summary = value.get("summary")
    if isinstance(summary, dict):
        normalized_summary: dict[str, Any] | str | None = summary
    elif summary is None:
        normalized_summary = None
    else:
        normalized_summary = str(summary)
    return {
        "pivot_table": pivot_table,
        "aggregations": aggregations,
        "summary": normalized_summary,
    }


def _normalize_dashboard(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict) or not value:
        return None
    charts = [item for item in value.get("charts", []) if isinstance(item, dict)]
    filters = [str(item) for item in value.get("filters", []) if str(item).strip()]
    kpis = [item for item in value.get("kpis", []) if isinstance(item, dict)]
    return {
        "charts": charts,
        "filters": filters,
        "kpis": kpis,
        "layout": dict(value.get("layout") or {}),
        "drilldown_ready": bool(value.get("drilldown_ready")),
        "time_column": value.get("time_column"),
        "applied_time_filter": value.get("applied_time_filter"),
        "active_filter": value.get("active_filter"),
        "visualization_type": value.get("visualization_type"),
    }


def _normalize_forecast_metadata(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict) or not value:
        return None
    return {
        "time_column": value.get("time_column"),
        "data_points": int(value.get("data_points") or 0),
        "frequency": str(value.get("frequency") or ""),
        "filled_missing_timestamps": int(value.get("filled_missing_timestamps") or 0),
    }


def _normalize_context(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict) or not value:
        return None
    return {
        "row_count": int(value.get("row_count") or 0),
        "column_count": int(value.get("column_count") or 0),
        "column_summary": [item for item in value.get("column_summary", []) if isinstance(item, dict)],
        "data_types": {str(key): str(item) for key, item in dict(value.get("data_types") or {}).items()},
        "missing_values": {str(key): int(item or 0) for key, item in dict(value.get("missing_values") or {}).items()},
        "domain": str(value.get("domain") or "generic"),
        "dataset_type": str(value.get("dataset_type") or value.get("domain") or "generic"),
        "is_time_series": bool(value.get("is_time_series")),
        "time_columns": [str(item) for item in value.get("time_columns", []) if str(item).strip()],
        "primary_metrics": [str(item) for item in value.get("primary_metrics", []) if str(item).strip()],
        "categorical_features": [str(item) for item in value.get("categorical_features", []) if str(item).strip()],
        "interaction_memory": {
            "dataset_type": str(((value.get("interaction_memory") or {}).get("dataset_type")) or "generic"),
            "queries": [str(item) for item in ((value.get("interaction_memory") or {}).get("queries") or []) if str(item).strip()],
            "successful_actions": [
                str(item)
                for item in ((value.get("interaction_memory") or {}).get("successful_actions") or [])
                if str(item).strip()
            ],
        },
    }


def _normalize_suggestions(value: Any) -> list[dict[str, Any]]:
    suggestions: list[dict[str, Any]] = []
    if not isinstance(value, list):
        return suggestions
    for item in value:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        prompt = str(item.get("prompt") or title).strip()
        if not title and not prompt:
            continue
        suggestions.append(
            {
                "title": title or prompt,
                "prompt": prompt or title,
                "action_type": str(item.get("action_type") or "analysis"),
                "category": str(item.get("category") or ""),
                "goal": str(item.get("goal") or ""),
                "rationale": str(item.get("rationale") or ""),
                "score": float(item.get("score") or 0.0),
                "rank": int(item.get("rank") or 0),
            }
        )
    return suggestions


def _normalize_contract_string(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _normalize_system_decision(value: Any) -> dict[str, Any]:
    payload = dict(value or {})
    selected_mode = str(
        payload.get("selected_mode")
        or payload.get("mode")
        or "analysis"
    ).strip().lower()
    if selected_mode not in {"forecast", "ml", "analysis"}:
        selected_mode = "analysis"

    confidence = payload.get("confidence")
    try:
        confidence_value = float(confidence)
    except (TypeError, ValueError):
        confidence_value = 0.0
    confidence_value = max(0.0, min(confidence_value, 1.0))

    return {
        "selected_mode": selected_mode,
        "reason": str(payload.get("reason") or "").strip(),
        "suggestion": str(payload.get("suggestion") or "").strip(),
        "confidence": confidence_value,
    }


def _normalize_decision_layer(
    value: Any,
    *,
    risk: str | None = None,
    recommendations: list[str] | None = None,
) -> dict[str, Any]:
    return ensure_decision_layer_defaults(
        value,
        risk=risk,
        recommendations=recommendations,
    )


def ensure_analysis_contract_defaults(contract: dict[str, Any] | None) -> dict[str, Any]:
    payload = dict(contract or {})
    payload.setdefault("intent", "analysis")
    payload.setdefault("code", "")
    payload.setdefault("result_summary", "")
    payload["insights"] = [str(item) for item in payload.get("insights", []) if str(item).strip()]
    recommendations = [str(item) for item in payload.get("recommendations", []) if str(item).strip()]
    payload["warnings"] = [str(item) for item in payload.get("warnings", []) if str(item).strip()]
    payload.setdefault("confidence", "1/10")
    payload["model_metrics"] = _normalize_model_metrics(payload.get("model_metrics"))
    payload["explanation"] = _normalize_explanation(payload.get("explanation"))
    payload["ml_intelligence"] = _normalize_ml_intelligence(payload.get("ml_intelligence"))
    payload["data_quality"] = _normalize_data_quality(payload.get("data_quality"))
    payload["cleaning_report"] = _normalize_cleaning_report(payload.get("cleaning_report"))
    payload["model_quality"] = str(payload.get("model_quality") or "weak")
    payload["risk"] = str(payload.get("risk") or "")
    payload["result_hash"] = str(payload.get("result_hash") or "")
    payload["dataset_fingerprint"] = str(payload.get("dataset_fingerprint") or "")
    payload["reproducibility"] = _normalize_reproducibility(payload.get("reproducibility"))
    payload["inconsistency_detected"] = bool(payload.get("inconsistency_detected"))
    payload["limitations"] = [str(item) for item in payload.get("limitations", []) if str(item).strip()]
    payload["execution_plan"] = _normalize_execution_plan(payload.get("execution_plan"))
    payload["excel_analysis"] = _normalize_excel_analysis(payload.get("excel_analysis"))
    payload["dashboard"] = _normalize_dashboard(payload.get("dashboard"))
    payload["forecast_metadata"] = _normalize_forecast_metadata(payload.get("forecast_metadata"))
    payload["system_decision"] = _normalize_system_decision(payload.get("system_decision"))
    payload["context"] = _normalize_context(payload.get("context"))
    payload["suggestions"] = _normalize_suggestions(payload.get("suggestions"))
    payload["recommended_next_step"] = _normalize_contract_string(payload.get("recommended_next_step"))
    payload["suggested_questions"] = [str(item) for item in payload.get("suggested_questions", []) if str(item).strip()]
    if not payload["suggested_questions"] and payload["suggestions"]:
        payload["suggested_questions"] = [
            str(item.get("prompt") or item.get("title") or "")
            for item in payload["suggestions"]
            if str(item.get("prompt") or item.get("title") or "").strip()
        ]
    if not payload["recommended_next_step"] and payload["suggestions"]:
        payload["recommended_next_step"] = _normalize_contract_string(
            payload["suggestions"][0].get("prompt") or payload["suggestions"][0].get("title")
        )
    payload["active_filter"] = _normalize_contract_string(payload.get("active_filter"))
    payload["visualization_type"] = _normalize_contract_string(payload.get("visualization_type"))
    payload["tool_used"] = _infer_tool_used(
        value=payload.get("tool_used"),
        intent=payload.get("intent"),
        execution_plan=payload.get("execution_plan"),
        excel_analysis=payload.get("excel_analysis"),
        dashboard=payload.get("dashboard"),
    )
    payload["analysis_mode"] = _normalize_analysis_mode(
        payload.get("analysis_mode"),
        intent=payload.get("intent"),
        tool_used=payload["tool_used"],
    )
    if not payload["execution_plan"]:
        payload["execution_plan"] = _default_execution_plan(
            tool_used=payload["tool_used"],
            analysis_mode=payload["analysis_mode"],
            intent=payload.get("intent"),
        )
    payload["execution_trace"] = _normalize_execution_trace(
        payload.get("execution_trace"),
        execution_plan=payload["execution_plan"],
    )
    payload["optimization"] = _normalize_optimization(
        payload.get("optimization"),
        execution_trace=payload["execution_trace"],
    )
    payload["decision_layer"] = _normalize_decision_layer(
        payload.get("decision_layer"),
        risk=payload.get("risk"),
        recommendations=(recommendations + list(payload["ml_intelligence"].get("recommendations") or [])),
    )
    payload["recommendations"] = (
        derive_recommendations_from_decision_layer(payload["decision_layer"])
        or (recommendations + list(payload["ml_intelligence"].get("recommendations") or []))
    )
    return payload


def build_analysis_contract(
    *,
    query: str,
    df: pd.DataFrame,
    result: Any,
    executed_code: str | None,
    plan: dict[str, Any],
    preflight: dict[str, Any],
    method: str,
    model_metrics: dict[str, float | None] | None = None,
    explanation: dict[str, list[Any]] | None = None,
    data_quality: dict[str, Any] | None = None,
    cleaning_report: dict[str, Any] | None = None,
    model_quality: str | None = None,
    risk: str | None = None,
    result_hash: str | None = None,
    dataset_fingerprint: str | None = None,
    reproducibility: dict[str, Any] | None = None,
    inconsistency_detected: bool = False,
    limitations: list[str] | None = None,
    insights: list[str] | None = None,
    learning_patterns: dict[str, Any] | None = None,
    ml_intelligence: dict[str, Any] | None = None,
    tool_used: str | None = None,
    analysis_mode: str | None = None,
    execution_plan: Any = None,
    execution_trace: Any = None,
    optimization: dict[str, Any] | None = None,
    excel_analysis: dict[str, Any] | None = None,
    dashboard: dict[str, Any] | None = None,
    forecast_metadata: dict[str, Any] | None = None,
    system_decision: dict[str, Any] | None = None,
    context: dict[str, Any] | None = None,
    suggestions: list[dict[str, Any]] | None = None,
    recommended_next_step: str | None = None,
    suggested_questions: list[str] | None = None,
    active_filter: str | None = None,
    visualization_type: str | None = None,
) -> dict[str, Any]:
    warnings = list(preflight.get("warnings") or [])
    warnings.extend(validate_analysis_output(result))
    deduped_warnings = list(dict.fromkeys(warnings))
    intent = str(plan.get("intent") or classify_analysis_intent(query))
    resolved_model_metrics = model_metrics or {"mae": None, "r2": None}
    resolved_explanation = explanation or {"top_features": [], "impact": []}
    data_quality = _normalize_data_quality(
        data_quality
        or preflight.get("data_quality")
        or build_data_quality_report(df)
    )
    resolved_model_quality = str(
        model_quality
        or interpret_model_quality(resolved_model_metrics.get("mae"), resolved_model_metrics.get("r2"))
    )
    resolved_risk = str(risk or build_risk_statement(data_quality, resolved_model_quality))
    resolved_insights = [
        str(item).strip()
        for item in (
            insights
            or generate_analysis_insights(
                result=result,
                intent=intent,
                plan=plan,
                warnings=deduped_warnings,
            )
        )
        if str(item).strip()
    ]
    resolved_reproducibility = _normalize_reproducibility(
        reproducibility
        or build_reproducibility_metadata(
            source_fingerprint=dataset_fingerprint,
            pipeline_trace=[],
            result_hash=result_hash,
            consistency_payload={
                "result_hash": result_hash,
                "inconsistency_detected": inconsistency_detected,
            },
        )
    )
    decision_layer = build_decision_layer(
        result=result,
        insights=resolved_insights,
        plan=plan,
        model_quality=resolved_model_quality,
        data_quality=data_quality,
        reproducibility=resolved_reproducibility,
        risk=resolved_risk,
        warnings=deduped_warnings,
        learning_patterns=learning_patterns,
        seed_recommendations=list(dict(ml_intelligence or {}).get("recommendations") or []),
    )
    resolved_recommendations = derive_recommendations_from_decision_layer(decision_layer)
    question_payload = build_question_payload(
        df,
        source_fingerprint=dataset_fingerprint,
        recent_queries=[query],
    )
    confidence_score = 8 if method.startswith("deterministic") else 6
    confidence_score -= min(4, len(deduped_warnings))
    if len(df) < 25:
        confidence_score -= 1
    if preflight.get("blocking_errors"):
        confidence_score = 1
    if float(data_quality.get("score") or 0.0) < 5.0:
        confidence_score -= 2
    if resolved_model_quality == "weak":
        confidence_score -= 2
    elif resolved_model_quality == "moderate":
        confidence_score -= 1
    confidence_score = max(1, min(10, confidence_score))

    normalized_execution_plan = _normalize_execution_plan(execution_plan)
    normalized_execution_trace = _normalize_execution_trace(
        execution_trace,
        execution_plan=normalized_execution_plan,
    )

    return ensure_analysis_contract_defaults({
        "intent": intent,
        "code": str(executed_code or ""),
        "result_summary": summarize_analysis_result(
            query=query,
            result=result,
            intent=intent,
            warnings=deduped_warnings,
            method=method,
        ),
        "insights": resolved_insights,
        "recommendations": resolved_recommendations,
        "confidence": f"{confidence_score}/10",
        "warnings": deduped_warnings,
        "data_quality": data_quality,
        "cleaning_report": _normalize_cleaning_report(cleaning_report),
        "model_metrics": resolved_model_metrics,
        "explanation": resolved_explanation,
        "ml_intelligence": _normalize_ml_intelligence(ml_intelligence),
        "model_quality": resolved_model_quality,
        "risk": resolved_risk,
        "decision_layer": decision_layer,
        "result_hash": str(result_hash or ""),
        "dataset_fingerprint": str(dataset_fingerprint or resolved_reproducibility.get("dataset_fingerprint") or ""),
        "reproducibility": resolved_reproducibility,
        "inconsistency_detected": bool(inconsistency_detected),
        "limitations": list(limitations or []),
        "tool_used": str(tool_used or plan.get("tool_used") or "PYTHON"),
        "analysis_mode": str(analysis_mode or plan.get("analysis_mode") or ("prediction" if intent == "prediction" else "ad-hoc")),
        "execution_plan": normalized_execution_plan,
        "execution_trace": normalized_execution_trace,
        "optimization": _normalize_optimization(
            optimization,
            execution_trace=normalized_execution_trace,
        ),
        "excel_analysis": _normalize_excel_analysis(excel_analysis),
        "dashboard": _normalize_dashboard(dashboard),
        "forecast_metadata": _normalize_forecast_metadata(forecast_metadata),
        "system_decision": _normalize_system_decision(system_decision),
        "context": _normalize_context(context or question_payload.get("context")),
        "suggestions": _normalize_suggestions(suggestions or question_payload.get("suggestions")),
        "recommended_next_step": _normalize_contract_string(recommended_next_step or question_payload.get("recommended_next_step")),
        "suggested_questions": [
            str(item)
            for item in list(suggested_questions or question_payload.get("suggested_questions") or [])
            if str(item).strip()
        ],
        "active_filter": _normalize_contract_string(active_filter),
        "visualization_type": _normalize_contract_string(visualization_type),
    })
