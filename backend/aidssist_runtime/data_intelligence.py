from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import requests

from .ai_data_store import AIDataStore
from .config import get_settings
from .dataset_session import (
    get_catalog_path,
    get_duckdb_path,
    get_insights_path,
    get_schema_path,
    sanitize_relative_path,
    write_session_file,
)
from .ingestion import load_dataset_dataframe
from .llm_gateway import LLMUnavailableError, llm_is_configured, request_json_completion
from .logging_utils import get_logger
from .schema_detection import build_schema_payload, quote_ident
from .storage import get_object_store
from backend.workflow_store import WorkflowStore


LOGGER = get_logger(__name__)
FORBIDDEN_SQL_TOKENS = (
    "insert ",
    "update ",
    "delete ",
    "drop ",
    "alter ",
    "create ",
    "attach ",
    "detach ",
    "copy ",
    "pragma ",
    "call ",
)


def _serialize_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            return str(value)
    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def _serialize_frame(frame: pd.DataFrame, *, limit: int = 20) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    preview = frame.head(limit).copy()
    preview = preview.where(pd.notna(preview), None)
    return [
        {str(key): _serialize_value(value) for key, value in row.items()}
        for row in preview.to_dict(orient="records")
    ]


def _sanitize_table_name(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", Path(value or "table").stem).strip("_").lower()
    return normalized or "table"


def _unique_table_name(base_name: str, existing: set[str]) -> str:
    candidate = base_name
    counter = 2
    while candidate in existing:
        candidate = f"{base_name}_{counter}"
        counter += 1
    existing.add(candidate)
    return candidate


def _date_expression(column_name: str) -> str:
    quoted = quote_ident(column_name)
    return f"TRY_CAST({quoted} AS TIMESTAMP)"


def _pick_primary_table(schema: dict[str, Any]) -> dict[str, Any] | None:
    tables = list(schema.get("tables") or [])
    if not tables:
        return None
    return max(tables, key=lambda item: (int(item.get("row_count") or 0), int(item.get("column_count") or 0)))


def _categorical_columns(table: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        column
        for column in table.get("columns", [])
        if str(column.get("semantic_type") or "") in {"categorical", "text"}
    ]


def _numeric_columns(table: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        column
        for column in table.get("columns", [])
        if str(column.get("semantic_type") or "") in {"int", "float"}
    ]


def _date_columns(table: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        column
        for column in table.get("columns", [])
        if str(column.get("semantic_type") or "") == "date"
    ]


def _execute_dataframe(connection: duckdb.DuckDBPyConnection, sql: str) -> pd.DataFrame:
    return connection.execute(sql).df()


def _safe_preview_sql(table_name: str, limit: int = 8) -> str:
    return f"SELECT * FROM {quote_ident(table_name)} LIMIT {int(limit)}"


def _build_catalog(asset_id: str, *, force_refresh: bool = False) -> dict[str, Any]:
    with WorkflowStore() as store:
        asset = store.get_asset(asset_id)
        if asset is None:
            raise ValueError(f"Asset '{asset_id}' was not found.")
        datasets = store.list_asset_datasets(asset_id)
        if not datasets:
            raise ValueError("This asset does not include any tabular datasets yet.")

    session_id = asset.asset_id
    warehouse_path = get_duckdb_path(session_id)
    if warehouse_path.exists() and force_refresh:
        warehouse_path.unlink()

    object_store = get_object_store()
    connection = duckdb.connect(str(warehouse_path))
    existing_names: set[str] = set()
    catalog_tables: list[dict[str, Any]] = []
    try:
        for dataset in datasets:
            safe_relative = sanitize_relative_path(dataset.dataset_name or f"{dataset.dataset_id}.csv")
            source_bytes = object_store.get_bytes(dataset.object_key)
            source_path = write_session_file(session_id, safe_relative, source_bytes)
            base_table_name = _sanitize_table_name(safe_relative)
            table_name = _unique_table_name(base_table_name, existing_names)

            if dataset.source_kind == "csv":
                connection.execute(
                    f"""
                    CREATE OR REPLACE TABLE {quote_ident(table_name)} AS
                    SELECT * FROM read_csv_auto(?, sample_size=-1, ignore_errors=true, union_by_name=true)
                    """,
                    [str(source_path)],
                )
            else:
                frame = load_dataset_dataframe(dataset)
                connection.register(f"{table_name}__frame", frame)
                connection.execute(
                    f"CREATE OR REPLACE TABLE {quote_ident(table_name)} AS SELECT * FROM {quote_ident(f'{table_name}__frame')}"
                )
                connection.unregister(f"{table_name}__frame")

            row_count = int(connection.execute(f"SELECT COUNT(*) FROM {quote_ident(table_name)}").fetchone()[0])
            preview_frame = connection.execute(_safe_preview_sql(table_name)).df()
            catalog_tables.append(
                {
                    "dataset_id": dataset.dataset_id,
                    "source_name": dataset.dataset_name,
                    "source_kind": dataset.source_kind,
                    "table_name": table_name,
                    "source_path": str(source_path),
                    "row_count": row_count,
                    "preview_rows": _serialize_frame(preview_frame, limit=8),
                }
            )
    finally:
        connection.close()

    catalog = {"asset_id": asset_id, "session_id": session_id, "tables": catalog_tables}
    get_catalog_path(session_id).write_text(json.dumps(catalog, indent=2, default=str))
    return catalog


def _time_series_summary(
    connection: duckdb.DuckDBPyConnection,
    table_name: str,
    date_column: str,
    metric_column: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    trend_frame = _execute_dataframe(
        connection,
        f"""
        SELECT DATE_TRUNC('month', {_date_expression(date_column)}) AS period,
               SUM(COALESCE({quote_ident(metric_column)}, 0)) AS value
        FROM {quote_ident(table_name)}
        WHERE {_date_expression(date_column)} IS NOT NULL
        GROUP BY 1
        ORDER BY 1
        LIMIT 36
        """,
    )
    if trend_frame.empty:
        return [], [], []

    trend_frame["period"] = pd.to_datetime(trend_frame["period"], errors="coerce")
    trend_frame = trend_frame.dropna(subset=["period"])
    trend_rows = _serialize_frame(trend_frame.rename(columns={"period": "date"}), limit=36)
    insights: list[dict[str, Any]] = []
    anomalies: list[dict[str, Any]] = []
    recommendations: list[dict[str, Any]] = []

    if len(trend_frame.index) >= 2:
        latest = float(trend_frame.iloc[-1]["value"])
        previous = float(trend_frame.iloc[-2]["value"])
        delta_pct = ((latest - previous) / abs(previous) * 100) if previous else 0.0
        insights.append(
            {
                "kind": "trend",
                "title": f"{metric_column} changed {delta_pct:+.1f}% month over month",
                "narrative": f"{metric_column} moved from {previous:,.2f} to {latest:,.2f} in the latest monthly period.",
                "confidence": "high",
                "chart": {
                    "type": "line",
                    "title": f"{metric_column} by month",
                    "x": "date",
                    "y": "value",
                    "rows": trend_rows,
                },
            }
        )

    if len(trend_frame.index) >= 4:
        rolling_mean = float(trend_frame["value"].mean())
        rolling_std = float(trend_frame["value"].std() or 0.0)
        if rolling_std > 0:
            trend_frame["z_score"] = (trend_frame["value"] - rolling_mean) / rolling_std
            anomaly_rows = trend_frame.loc[trend_frame["z_score"].abs() >= 2.0].copy()
            for _, row in anomaly_rows.tail(2).iterrows():
                anomalies.append(
                    {
                        "kind": "anomaly",
                        "title": f"Spike detected on {row['period'].date()}",
                        "narrative": f"{metric_column} reached {float(row['value']):,.2f}, which is materially outside the recent monthly baseline.",
                        "confidence": "medium",
                        "chart": {
                            "type": "line",
                            "title": f"{metric_column} anomaly monitor",
                            "x": "date",
                            "y": "value",
                            "rows": trend_rows,
                        },
                    }
                )
            if anomaly_rows.shape[0]:
                recommendations.append(
                    {
                        "title": f"Investigate the {metric_column.lower()} anomaly window",
                        "body": "Audit campaigns, operational events, or data-quality breaks around the detected spike or dip before sharing decisions broadly.",
                        "priority": "high",
                    }
                )
    return insights, anomalies, recommendations


def _category_breakdown(
    connection: duckdb.DuckDBPyConnection,
    table_name: str,
    category_column: str,
    metric_column: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    frame = _execute_dataframe(
        connection,
        f"""
        SELECT {quote_ident(category_column)} AS category,
               SUM(COALESCE({quote_ident(metric_column)}, 0)) AS value
        FROM {quote_ident(table_name)}
        WHERE {quote_ident(category_column)} IS NOT NULL
        GROUP BY 1
        ORDER BY 2 DESC
        LIMIT 8
        """,
    )
    rows = _serialize_frame(frame)
    if not rows:
        return [], []
    top = rows[0]
    insight = {
        "kind": "breakdown",
        "title": f"{top['category']} leads {metric_column}",
        "narrative": f"{top['category']} is currently the top contributor for {metric_column}, based on the available grouped data.",
        "confidence": "medium",
        "chart": {
            "type": "bar",
            "title": f"Top {category_column} by {metric_column}",
            "x": "category",
            "y": "value",
            "rows": rows,
        },
    }
    recommendation = {
        "title": f"Double down on {top['category']}",
        "body": f"Use the strongest {category_column} segment as the first place to validate pricing, retention, or campaign expansion ideas.",
        "priority": "medium",
    }
    return [insight], [recommendation]


def _generate_rule_based_insights(
    connection: duckdb.DuckDBPyConnection,
    schema: dict[str, Any],
) -> dict[str, Any]:
    primary_table = _pick_primary_table(schema)
    if primary_table is None:
        return {
            "summary": "No insights are available because the asset does not contain tabular data yet.",
            "insights": [],
            "anomalies": [],
            "recommendations": [],
            "charts": [],
            "follow_up_questions": [],
        }

    table_name = str(primary_table["name"])
    numeric_columns = _numeric_columns(primary_table)
    date_columns = _date_columns(primary_table)
    categorical_columns = _categorical_columns(primary_table)

    insights: list[dict[str, Any]] = []
    anomalies: list[dict[str, Any]] = []
    recommendations: list[dict[str, Any]] = []

    if date_columns and numeric_columns:
        trend_insights, trend_anomalies, trend_recommendations = _time_series_summary(
            connection,
            table_name,
            str(date_columns[0]["name"]),
            str(numeric_columns[0]["name"]),
        )
        insights.extend(trend_insights)
        anomalies.extend(trend_anomalies)
        recommendations.extend(trend_recommendations)

    if categorical_columns and numeric_columns:
        breakdown_insights, breakdown_recommendations = _category_breakdown(
            connection,
            table_name,
            str(categorical_columns[0]["name"]),
            str(numeric_columns[0]["name"]),
        )
        insights.extend(breakdown_insights)
        recommendations.extend(breakdown_recommendations)

    charts = [
        dict(item["chart"])
        for item in [*insights, *anomalies]
        if isinstance(item.get("chart"), dict)
    ]
    summary = (
        insights[0]["narrative"]
        if insights
        else f"Aidssist profiled {primary_table['column_count']} columns across the {table_name} table."
    )
    follow_up_questions = [
        f"What is driving the trend in {numeric_columns[0]['name']}?" if numeric_columns else "What changed recently?",
        f"Which {categorical_columns[0]['name']} segments are most important?" if categorical_columns else "Which segments matter most?",
        "Where should we investigate anomalies first?",
    ]
    return {
        "summary": summary,
        "insights": insights[:4],
        "anomalies": anomalies[:3],
        "recommendations": recommendations[:4],
        "charts": charts[:4],
        "follow_up_questions": follow_up_questions,
    }


def _refine_insights_with_llm(schema: dict[str, Any], insight_payload: dict[str, Any]) -> dict[str, Any]:
    if not llm_is_configured():
        return insight_payload
    try:
        return request_json_completion(
            system_prompt=(
                "You are Aidssist's senior analytics copilot. Turn the structured profiling payload into "
                "clear JSON with keys summary, insights, anomalies, recommendations, charts, and follow_up_questions. "
                "Keep recommendations specific to the evidence and do not invent unsupported facts."
            ),
            user_prompt=json.dumps(
                {
                    "schema": schema,
                    "rule_based_payload": insight_payload,
                },
                default=str,
            ),
            temperature=0.1,
        )
    except Exception:
        LOGGER.warning("LLM refinement for insights failed; using rule-based output.", exc_info=True)
        return insight_payload


def prepare_asset_intelligence(asset_id: str, *, force_refresh: bool = False) -> dict[str, Any]:
    catalog = _build_catalog(asset_id, force_refresh=force_refresh)
    session_id = str(catalog.get("session_id") or asset_id)
    warehouse_path = get_duckdb_path(session_id)
    connection = duckdb.connect(str(warehouse_path))
    try:
        schema = build_schema_payload(
            connection,
            catalog,
            sample_rows=max(1000, get_settings().intelligence_sample_rows),
        )
        rule_based_insights = _generate_rule_based_insights(connection, schema)
        insights = _refine_insights_with_llm(schema, rule_based_insights)
        chat_context = {
            "dataset_type": schema.get("dataset_type"),
            "default_table": (_pick_primary_table(schema) or {}).get("name"),
            "tables": [
                {
                    "name": table.get("name"),
                    "columns": [column.get("name") for column in table.get("columns", [])],
                    "primary_keys": table.get("primary_keys", []),
                }
                for table in schema.get("tables", [])
            ],
            "relationships": list(schema.get("relationships") or []),
            "suggested_questions": list(insights.get("follow_up_questions") or []),
        }
    finally:
        connection.close()

    get_schema_path(session_id).write_text(json.dumps(schema, indent=2, default=str))
    get_insights_path(session_id).write_text(json.dumps(insights, indent=2, default=str))

    with WorkflowStore() as store:
        asset = store.get_asset(asset_id)
        if asset is None:
            raise ValueError(f"Asset '{asset_id}' was not found.")

    with AIDataStore() as intelligence_store:
        persisted = intelligence_store.upsert_asset_intelligence(
            workspace_id=asset.workspace_id,
            asset_id=asset.asset_id,
            session_id=session_id,
            status="ready",
            dataset_type=str(schema.get("dataset_type") or "") or None,
            catalog=catalog,
            schema=schema,
            insights=insights,
            chat_context=chat_context,
        )
    return {
        "asset_id": asset_id,
        "session_id": session_id,
        "catalog": persisted.catalog,
        "schema": persisted.schema,
        "insights": persisted.insights,
        "chat_context": persisted.chat_context,
        "dataset_type": persisted.dataset_type,
        "status": persisted.status,
    }


def get_asset_intelligence(asset_id: str, *, force_refresh: bool = False) -> dict[str, Any]:
    with AIDataStore() as store:
        cached = store.get_asset_intelligence(asset_id)
    if cached is not None and not force_refresh:
        return {
            "asset_id": asset_id,
            "session_id": cached.session_id,
            "catalog": cached.catalog,
            "schema": cached.schema,
            "insights": cached.insights,
            "chat_context": cached.chat_context,
            "dataset_type": cached.dataset_type,
            "status": cached.status,
        }
    return prepare_asset_intelligence(asset_id, force_refresh=force_refresh)


def _validate_sql(sql: str) -> str:
    normalized = str(sql or "").strip().rstrip(";")
    lowered = normalized.lower()
    if not normalized:
        raise ValueError("Aidssist could not generate a SQL statement for this question.")
    if not (lowered.startswith("select ") or lowered.startswith("with ")):
        raise ValueError("Only read-only SELECT queries are allowed in Ask Your Data.")
    if any(token in lowered for token in FORBIDDEN_SQL_TOKENS):
        raise ValueError("The generated SQL contained a forbidden statement.")
    return normalized


def _infer_chart_from_result(frame: pd.DataFrame) -> dict[str, Any] | None:
    if frame.empty or frame.shape[1] < 2:
        return None
    columns = list(frame.columns)
    first, second = columns[0], columns[1]
    if pd.api.types.is_datetime64_any_dtype(frame[first]):
        rows = _serialize_frame(frame.rename(columns={first: "date", second: "value"}), limit=40)
        return {"type": "line", "title": "Query result trend", "x": "date", "y": "value", "rows": rows}
    if pd.api.types.is_numeric_dtype(frame[second]):
        rows = _serialize_frame(frame.rename(columns={first: "label", second: "value"}), limit=20)
        return {"type": "bar", "title": "Query result breakdown", "x": "label", "y": "value", "rows": rows}
    return None


def _summarize_answer(question: str, frame: pd.DataFrame) -> str:
    del question
    if frame.empty:
        return "No rows matched the question with the current dataset."
    if frame.shape[0] == 1 and frame.shape[1] == 1:
        value = _serialize_value(frame.iloc[0, 0])
        return f"The result is {value}."
    columns = list(frame.columns)
    if frame.shape[1] >= 2 and pd.api.types.is_numeric_dtype(frame[columns[1]]):
        top_row = frame.iloc[0]
        return f"The leading result is {top_row[columns[0]]} at {float(top_row[columns[1]]):,.2f}."
    return f"Aidssist returned {frame.shape[0]} rows and {frame.shape[1]} columns."


def _fallback_sql(question: str, intelligence: dict[str, Any]) -> str:
    schema = dict(intelligence.get("schema") or {})
    table = _pick_primary_table(schema)
    if table is None:
        raise ValueError("No tables are available for Ask Your Data.")
    table_name = str(table["name"])
    numeric = _numeric_columns(table)
    categorical = _categorical_columns(table)
    dates = _date_columns(table)
    question_lower = str(question or "").lower()

    if "top" in question_lower and numeric and categorical:
        limit_match = re.search(r"\btop\s+(\d+)\b", question_lower)
        limit = int(limit_match.group(1)) if limit_match else 10
        return (
            f"SELECT {quote_ident(categorical[0]['name'])} AS label, "
            f"SUM({quote_ident(numeric[0]['name'])}) AS value "
            f"FROM {quote_ident(table_name)} "
            f"WHERE {quote_ident(categorical[0]['name'])} IS NOT NULL "
            f"GROUP BY 1 ORDER BY 2 DESC LIMIT {limit}"
        )
    if ("trend" in question_lower or "drop" in question_lower or "revenue" in question_lower) and numeric and dates:
        return (
            f"SELECT DATE_TRUNC('month', {_date_expression(str(dates[0]['name']))}) AS period, "
            f"SUM({quote_ident(numeric[0]['name'])}) AS value "
            f"FROM {quote_ident(table_name)} "
            f"WHERE {_date_expression(str(dates[0]['name']))} IS NOT NULL "
            f"GROUP BY 1 ORDER BY 1 LIMIT 24"
        )
    return f"SELECT * FROM {quote_ident(table_name)} LIMIT 20"


def _llm_sql_plan(question: str, intelligence: dict[str, Any]) -> dict[str, Any]:
    schema = intelligence.get("schema") or {}
    return request_json_completion(
        system_prompt=(
            "You are Aidssist's SQL planner. Return JSON with keys sql, answer_hint, and chart. "
            "Only produce a single DuckDB-compatible SELECT query over the provided tables. "
            "Never use DDL, DML, PRAGMA, COPY, ATTACH, or semicolon-separated statements."
        ),
        user_prompt=json.dumps(
            {
                "question": question,
                "schema": schema,
                "relationships": schema.get("relationships"),
                "dataset_type": intelligence.get("dataset_type"),
            },
            default=str,
        ),
        temperature=0.0,
        max_tokens=1400,
    )


def ask_asset_question(asset_id: str, question: str) -> dict[str, Any]:
    intelligence = get_asset_intelligence(asset_id)
    session_id = str(intelligence.get("session_id") or asset_id)
    warehouse_path = get_duckdb_path(session_id)
    if not warehouse_path.exists():
        intelligence = prepare_asset_intelligence(asset_id, force_refresh=True)

    plan: dict[str, Any] = {}
    sql = ""
    if llm_is_configured():
        try:
            plan = _llm_sql_plan(question, intelligence)
            sql = _validate_sql(str(plan.get("sql") or ""))
        except Exception:
            LOGGER.warning("LLM SQL planning failed; using deterministic fallback.", exc_info=True)
            sql = _validate_sql(_fallback_sql(question, intelligence))
    else:
        sql = _validate_sql(_fallback_sql(question, intelligence))

    connection = duckdb.connect(str(get_duckdb_path(session_id)))
    try:
        frame = connection.execute(sql).df()
    finally:
        connection.close()

    chart = dict(plan.get("chart") or {}) or _infer_chart_from_result(frame) or None
    answer = str(plan.get("answer_hint") or "").strip() or _summarize_answer(question, frame)
    rows = _serialize_frame(frame, limit=50)
    return {
        "question": question,
        "sql": sql,
        "answer": answer,
        "rows": rows,
        "columns": [str(column) for column in frame.columns.tolist()],
        "chart": chart,
    }
