from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import Integer, String, Text, create_engine, event, inspect, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from backend.compat import dataclass
from .config import APP_STATE_DIR, get_settings
from .logging_utils import get_logger
from .metrics import observe_db_query


LOGGER = get_logger(__name__)
_UNSET = object()


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _serialize_json(value: Any) -> str:
    return json.dumps(value, default=str, sort_keys=True)


def _normalize_email(email: str) -> str:
    return str(email or "").strip().lower()


def _resolve_database_url(db_path_or_url: str | Path | None) -> str:
    if db_path_or_url is None:
        configured_url = str(get_settings().database_url).strip()
        if configured_url:
            return configured_url
        APP_STATE_DIR.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{APP_STATE_DIR / 'workflow_store.sqlite3'}"

    if isinstance(db_path_or_url, Path):
        return f"sqlite:///{db_path_or_url}"

    raw_value = str(db_path_or_url).strip()
    if "://" in raw_value:
        return raw_value

    return f"sqlite:///{Path(raw_value)}"


class Base(DeclarativeBase):
    pass


class WorkflowModel(Base):
    __tablename__ = "workflows"

    workflow_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[str] = mapped_column(String(40), nullable=False)
    updated_at: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    latest_version: Mapped[int] = mapped_column(Integer, nullable=False)


class UserModel(Base):
    __tablename__ = "users"

    user_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    email: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
    display_name: Mapped[str] = mapped_column(String(255), nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    password_salt: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    updated_at: Mapped[str] = mapped_column(String(40), nullable=False, index=True)


class UserSessionModel(Base):
    __tablename__ = "user_sessions"

    session_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    token_hash: Mapped[str] = mapped_column(String(128), nullable=False, unique=True, index=True)
    created_at: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    expires_at: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    revoked_at: Mapped[str | None] = mapped_column(String(40), nullable=True, index=True)


class UserDatasetModel(Base):
    __tablename__ = "user_datasets"

    user_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    dataset_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    created_at: Mapped[str] = mapped_column(String(40), nullable=False, index=True)


class UserJobModel(Base):
    __tablename__ = "user_jobs"

    user_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    job_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    created_at: Mapped[str] = mapped_column(String(40), nullable=False, index=True)


class WorkspaceModel(Base):
    __tablename__ = "workspaces"

    workspace_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    updated_at: Mapped[str] = mapped_column(String(40), nullable=False, index=True)


class AssetModel(Base):
    __tablename__ = "assets"

    asset_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    workspace_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    asset_kind: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    primary_dataset_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    created_at: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    updated_at: Mapped[str] = mapped_column(String(40), nullable=False, index=True)


class AssetFileModel(Base):
    __tablename__ = "asset_files"

    asset_file_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    asset_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    dataset_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    file_name: Mapped[str] = mapped_column(String(255), nullable=False)
    media_type: Mapped[str] = mapped_column(String(255), nullable=False)
    file_kind: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    language: Mapped[str | None] = mapped_column(String(64), nullable=True)
    object_key: Mapped[str] = mapped_column(String(1024), nullable=False)
    checksum: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[str] = mapped_column(String(40), nullable=False, index=True)


class ChunkModel(Base):
    __tablename__ = "chunks"

    chunk_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    asset_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    asset_file_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    dataset_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    content_text: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    metadata_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    created_at: Mapped[str] = mapped_column(String(40), nullable=False, index=True)


class EmbeddingModel(Base):
    __tablename__ = "embeddings"

    embedding_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    chunk_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)
    model_name: Mapped[str] = mapped_column(String(255), nullable=False)
    dimension: Mapped[int] = mapped_column(Integer, nullable=False)
    vector_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[str] = mapped_column(String(40), nullable=False, index=True)


class DerivedDatasetModel(Base):
    __tablename__ = "derived_datasets"

    derived_dataset_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    workspace_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    asset_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    parent_dataset_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    dataset_name: Mapped[str] = mapped_column(String(255), nullable=False)
    dataset_key: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    source_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    object_key: Mapped[str] = mapped_column(String(1024), nullable=False)
    content_type: Mapped[str] = mapped_column(String(255), nullable=False)
    transform_prompt: Mapped[str | None] = mapped_column(Text, nullable=True)
    row_count: Mapped[int] = mapped_column(Integer, nullable=False)
    column_count: Mapped[int] = mapped_column(Integer, nullable=False)
    preview_columns_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    preview_rows_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    created_at: Mapped[str] = mapped_column(String(40), nullable=False, index=True)


class SolveRunModel(Base):
    __tablename__ = "solve_runs"

    run_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    workspace_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    user_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    asset_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    dataset_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    route: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    plan_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    retrieval_trace_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    retrieved_chunk_ids_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    final_output_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    final_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    packaged_output_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    result_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    queued_at: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    started_at: Mapped[str | None] = mapped_column(String(40), nullable=True)
    finished_at: Mapped[str | None] = mapped_column(String(40), nullable=True)
    queue_wait_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    elapsed_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)


class SolveStepModel(Base):
    __tablename__ = "solve_steps"

    step_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    step_index: Mapped[int] = mapped_column(Integer, nullable=False)
    stage: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    detail_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    created_at: Mapped[str] = mapped_column(String(40), nullable=False, index=True)


class ValidatorReportModel(Base):
    __tablename__ = "validator_reports"

    report_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    attempt_index: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    checks_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[str] = mapped_column(String(40), nullable=False, index=True)


class FeedbackEventModel(Base):
    __tablename__ = "feedback_events"

    feedback_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    chunk_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    metadata_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    created_at: Mapped[str] = mapped_column(String(40), nullable=False, index=True)


class FailureLogModel(Base):
    __tablename__ = "failure_logs"

    failure_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    stage: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    error_message: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    created_at: Mapped[str] = mapped_column(String(40), nullable=False, index=True)


class WorkflowVersionModel(Base):
    __tablename__ = "workflow_versions"

    workflow_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    version: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_config_json: Mapped[str] = mapped_column(Text, nullable=False)
    cleaning_options_json: Mapped[str] = mapped_column(Text, nullable=False)
    analysis_query: Mapped[str] = mapped_column(Text, nullable=False)
    forecast_config_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    chart_preferences_json: Mapped[str] = mapped_column(Text, nullable=False)
    export_settings_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[str] = mapped_column(String(40), nullable=False, index=True)


class WorkflowRunModel(Base):
    __tablename__ = "workflow_runs"

    run_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    workflow_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    workflow_version: Mapped[int | None] = mapped_column(Integer, nullable=True)
    workflow_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    source_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    source_label: Mapped[str] = mapped_column(String(255), nullable=False)
    validation_findings_json: Mapped[str] = mapped_column(Text, nullable=False)
    cleaning_actions_json: Mapped[str] = mapped_column(Text, nullable=False)
    generated_code: Mapped[str | None] = mapped_column(Text, nullable=True)
    final_status: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    export_artifacts_json: Mapped[str] = mapped_column(Text, nullable=False)
    analysis_query: Mapped[str] = mapped_column(Text, nullable=False)
    result_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    result_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    created_at: Mapped[str] = mapped_column(String(40), nullable=False, index=True)


class DatasetModel(Base):
    __tablename__ = "datasets"

    dataset_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    dataset_name: Mapped[str] = mapped_column(String(255), nullable=False)
    dataset_key: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    source_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    source_kind: Mapped[str] = mapped_column(String(32), nullable=False)
    source_label: Mapped[str] = mapped_column(String(255), nullable=False)
    object_key: Mapped[str] = mapped_column(String(1024), nullable=False)
    content_type: Mapped[str] = mapped_column(String(255), nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[str] = mapped_column(String(40), nullable=False, index=True)


class AnalysisJobModel(Base):
    __tablename__ = "analysis_jobs"

    job_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    dataset_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    intent: Mapped[str] = mapped_column(String(40), nullable=False, default="general")
    workflow_context_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    cache_key: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    queued_at: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    started_at: Mapped[str | None] = mapped_column(String(40), nullable=True)
    finished_at: Mapped[str | None] = mapped_column(String(40), nullable=True)
    queue_wait_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    elapsed_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    analysis_output_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    result_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    cache_hit: Mapped[int] = mapped_column(Integer, nullable=False, default=0)


class ForecastArtifactModel(Base):
    __tablename__ = "forecast_artifacts"

    artifact_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    workflow_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    workflow_version: Mapped[int | None] = mapped_column(Integer, nullable=True)
    workflow_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    source_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    source_label: Mapped[str] = mapped_column(String(255), nullable=False)
    target_column: Mapped[str] = mapped_column(String(255), nullable=False)
    horizon: Mapped[str] = mapped_column(String(64), nullable=False)
    model_name: Mapped[str] = mapped_column(String(128), nullable=False)
    training_mode: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    artifact_key: Mapped[str] = mapped_column(String(1024), nullable=False)
    forecast_config_json: Mapped[str] = mapped_column(Text, nullable=False)
    evaluation_metrics_json: Mapped[str] = mapped_column(Text, nullable=False)
    recommendation_payload_json: Mapped[str] = mapped_column(Text, nullable=False)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    result_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    created_at: Mapped[str] = mapped_column(String(40), nullable=False, index=True)


class DecisionHistoryModel(Base):
    __tablename__ = "decision_history"

    decision_history_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    job_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    forecast_artifact_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    source_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    decision_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    decision_json: Mapped[str] = mapped_column(Text, nullable=False)
    priority: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    decision_confidence: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    risk_level: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    result_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    outcome: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    updated_at: Mapped[str] = mapped_column(String(40), nullable=False, index=True)


def _attach_query_listeners(engine: Engine) -> None:
    if getattr(engine, "_aidssist_query_metrics_attached", False):
        return

    @event.listens_for(engine, "before_cursor_execute")
    def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
        del cursor, statement, parameters, executemany
        conn.info.setdefault("query_start_time", []).append(time.perf_counter())

    @event.listens_for(engine, "after_cursor_execute")
    def after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
        del cursor, parameters, context, executemany
        started = conn.info.get("query_start_time", []).pop(-1)
        duration = time.perf_counter() - started
        observe_db_query(duration, success=True)
        slow_query_ms = get_settings().slow_query_ms
        if duration * 1000 >= slow_query_ms:
            LOGGER.warning(
                "slow database query detected",
                extra={
                    "component": "database",
                    "duration_ms": round(duration * 1000, 2),
                    "endpoint": "sql",
                    "method": "EXECUTE",
                },
            )

    @event.listens_for(engine, "handle_error")
    def handle_error(exception_context):
        del exception_context
        observe_db_query(0.0, success=False)

    engine._aidssist_query_metrics_attached = True


@dataclass(slots=True)
class WorkflowDefinition:
    workflow_id: str
    name: str
    version: int
    source_config: dict
    cleaning_options: dict
    analysis_query: str
    forecast_config: dict
    chart_preferences: dict
    export_settings: dict
    created_at: str


@dataclass(slots=True)
class UserRecord:
    user_id: str
    email: str
    display_name: str
    created_at: str
    updated_at: str


@dataclass(slots=True)
class UserSessionRecord:
    session_id: str
    user_id: str
    token_hash: str
    created_at: str
    expires_at: str
    revoked_at: str | None


@dataclass(slots=True)
class WorkflowRunRecord:
    run_id: str
    workflow_id: str | None
    workflow_version: int | None
    workflow_name: str | None
    source_fingerprint: str
    source_label: str
    validation_findings: list[dict]
    cleaning_actions: list[str]
    generated_code: str | None
    final_status: str
    error_message: str | None
    export_artifacts: list[str]
    analysis_query: str
    result_summary: str | None
    result_hash: str | None
    created_at: str


@dataclass(slots=True)
class ForecastArtifactRecord:
    artifact_id: str
    workflow_id: str | None
    workflow_version: int | None
    workflow_name: str | None
    source_fingerprint: str
    source_label: str
    target_column: str
    horizon: str
    model_name: str
    training_mode: str
    status: str
    artifact_key: str
    forecast_config: dict
    evaluation_metrics: dict
    recommendation_payload: list[dict]
    summary: str | None
    result_hash: str | None
    created_at: str


@dataclass(slots=True)
class DecisionHistoryRecord:
    decision_history_id: str
    job_id: str | None
    forecast_artifact_id: str | None
    source_fingerprint: str
    query: str
    decision_id: str
    decision_json: dict[str, Any]
    priority: str
    decision_confidence: str
    risk_level: str
    result_hash: str | None
    outcome: str | None
    created_at: str
    updated_at: str


@dataclass(slots=True)
class DatasetRecord:
    dataset_id: str
    dataset_name: str
    dataset_key: str
    source_fingerprint: str
    source_kind: str
    source_label: str
    object_key: str
    content_type: str
    size_bytes: int
    created_at: str


@dataclass(slots=True)
class WorkspaceRecord:
    workspace_id: str
    user_id: str
    name: str
    description: str | None
    created_at: str
    updated_at: str


@dataclass(slots=True)
class AssetRecord:
    asset_id: str
    workspace_id: str
    title: str
    asset_kind: str
    primary_dataset_id: str | None
    created_at: str
    updated_at: str


@dataclass(slots=True)
class AssetFileRecord:
    asset_file_id: str
    asset_id: str
    dataset_id: str | None
    file_name: str
    media_type: str
    file_kind: str
    language: str | None
    object_key: str
    checksum: str
    size_bytes: int
    created_at: str


@dataclass(slots=True)
class ChunkRecord:
    chunk_id: str
    asset_id: str
    asset_file_id: str | None
    dataset_id: str | None
    chunk_index: int
    title: str
    content_text: str
    token_count: int
    metadata: dict[str, Any]
    created_at: str


@dataclass(slots=True)
class EmbeddingRecord:
    embedding_id: str
    chunk_id: str
    model_name: str
    dimension: int
    vector: list[float]
    created_at: str


@dataclass(slots=True)
class DerivedDatasetRecord:
    derived_dataset_id: str
    workspace_id: str
    asset_id: str | None
    parent_dataset_id: str | None
    dataset_name: str
    dataset_key: str
    source_fingerprint: str
    object_key: str
    content_type: str
    transform_prompt: str | None
    row_count: int
    column_count: int
    preview_columns: list[str]
    preview_rows: list[dict[str, Any]]
    created_at: str


@dataclass(slots=True)
class SolveRunRecord:
    run_id: str
    workspace_id: str
    user_id: str | None
    asset_id: str | None
    dataset_id: str | None
    query: str
    route: str
    status: str
    plan_text: str | None
    retrieval_trace: dict[str, Any]
    retrieved_chunk_ids: list[str]
    final_output: dict[str, Any] | None
    final_summary: str | None
    packaged_output: dict[str, Any] | None
    result_hash: str | None
    error_message: str | None
    queued_at: str
    started_at: str | None
    finished_at: str | None
    queue_wait_ms: int | None
    elapsed_ms: int | None


@dataclass(slots=True)
class SolveStepRecord:
    step_id: str
    run_id: str
    step_index: int
    stage: str
    status: str
    title: str
    detail: dict[str, Any]
    created_at: str


@dataclass(slots=True)
class ValidatorReportRecord:
    report_id: str
    run_id: str
    attempt_index: int
    status: str
    checks: list[dict[str, Any]]
    error_message: str | None
    created_at: str


@dataclass(slots=True)
class FeedbackEventRecord:
    feedback_id: str
    run_id: str
    chunk_id: str | None
    event_type: str
    score: int | None
    metadata: dict[str, Any]
    created_at: str


@dataclass(slots=True)
class FailureLogRecord:
    failure_id: str
    query: str
    stage: str
    error_message: str
    metadata: dict[str, Any]
    created_at: str


@dataclass(slots=True)
class AnalysisJobRecord:
    job_id: str
    dataset_id: str
    query: str
    status: str
    intent: str
    workflow_context: dict
    cache_key: str | None
    queued_at: str
    started_at: str | None
    finished_at: str | None
    queue_wait_ms: int | None
    elapsed_ms: int | None
    error_message: str | None
    analysis_output: dict | None
    result_summary: str | None
    cache_hit: bool


@dataclass(slots=True)
class AnalysisHistoryRecord:
    job_id: str
    dataset_id: str
    dataset_name: str
    query: str
    status: str
    intent: str
    queued_at: str
    finished_at: str | None
    result_summary: str | None
    error_message: str | None
    cache_hit: bool


class WorkflowStore:
    def __init__(self, db_path: str | Path | None = None):
        self.database_url = _resolve_database_url(db_path)
        settings = get_settings()
        engine_kwargs: dict[str, Any] = {"future": True, "pool_pre_ping": True}
        if self.database_url.startswith("sqlite"):
            engine_kwargs["connect_args"] = {"check_same_thread": False}
        else:
            engine_kwargs["pool_size"] = max(int(settings.db_pool_size), 1)
            engine_kwargs["max_overflow"] = max(int(settings.db_max_overflow), 0)

        self.engine = create_engine(self.database_url, **engine_kwargs)
        _attach_query_listeners(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine, expire_on_commit=False, future=True)
        self._initialize()

    def close(self) -> None:
        self.engine.dispose()

    def __enter__(self) -> "WorkflowStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        del exc_type, exc, tb
        self.close()

    def _initialize(self) -> None:
        Base.metadata.create_all(self.engine)
        self._ensure_optional_schema()
        if self.database_url.startswith("postgresql"):
            try:
                with self.engine.begin() as connection:
                    connection.exec_driver_sql("CREATE EXTENSION IF NOT EXISTS pg_stat_statements")
                    connection.exec_driver_sql("CREATE EXTENSION IF NOT EXISTS vector")
            except Exception:
                LOGGER.warning(
                    "database extensions could not be initialized",
                    extra={"component": "database"},
                )

    def _ensure_optional_schema(self) -> None:
        inspector = inspect(self.engine)
        existing_tables = set(inspector.get_table_names())
        if "workflow_versions" in existing_tables:
            existing_columns = {column["name"] for column in inspector.get_columns("workflow_versions")}
            if "forecast_config_json" not in existing_columns:
                with self.engine.begin() as connection:
                    connection.exec_driver_sql(
                        "ALTER TABLE workflow_versions ADD COLUMN forecast_config_json TEXT"
                    )
        optional_columns = {
            "workflow_runs": {"result_hash": "ALTER TABLE workflow_runs ADD COLUMN result_hash VARCHAR(64)"},
            "forecast_artifacts": {"result_hash": "ALTER TABLE forecast_artifacts ADD COLUMN result_hash VARCHAR(64)"},
            "solve_runs": {"result_hash": "ALTER TABLE solve_runs ADD COLUMN result_hash VARCHAR(64)"},
        }
        for table_name, column_map in optional_columns.items():
            if table_name not in existing_tables:
                continue
            existing_columns = {column["name"] for column in inspector.get_columns(table_name)}
            for column_name, statement in column_map.items():
                if column_name in existing_columns:
                    continue
                with self.engine.begin() as connection:
                    connection.exec_driver_sql(statement)

    def _session(self) -> Session:
        return self.SessionLocal()

    @staticmethod
    def _user_model_to_record(row: UserModel) -> UserRecord:
        return UserRecord(
            user_id=row.user_id,
            email=row.email,
            display_name=row.display_name,
            created_at=row.created_at,
            updated_at=row.updated_at,
        )

    @staticmethod
    def _workspace_model_to_record(row: WorkspaceModel) -> WorkspaceRecord:
        return WorkspaceRecord(
            workspace_id=row.workspace_id,
            user_id=row.user_id,
            name=row.name,
            description=row.description,
            created_at=row.created_at,
            updated_at=row.updated_at,
        )

    @staticmethod
    def _asset_model_to_record(row: AssetModel) -> AssetRecord:
        return AssetRecord(
            asset_id=row.asset_id,
            workspace_id=row.workspace_id,
            title=row.title,
            asset_kind=row.asset_kind,
            primary_dataset_id=row.primary_dataset_id,
            created_at=row.created_at,
            updated_at=row.updated_at,
        )

    @staticmethod
    def _asset_file_model_to_record(row: AssetFileModel) -> AssetFileRecord:
        return AssetFileRecord(
            asset_file_id=row.asset_file_id,
            asset_id=row.asset_id,
            dataset_id=row.dataset_id,
            file_name=row.file_name,
            media_type=row.media_type,
            file_kind=row.file_kind,
            language=row.language,
            object_key=row.object_key,
            checksum=row.checksum,
            size_bytes=row.size_bytes,
            created_at=row.created_at,
        )

    @staticmethod
    def _chunk_model_to_record(row: ChunkModel) -> ChunkRecord:
        return ChunkRecord(
            chunk_id=row.chunk_id,
            asset_id=row.asset_id,
            asset_file_id=row.asset_file_id,
            dataset_id=row.dataset_id,
            chunk_index=row.chunk_index,
            title=row.title,
            content_text=row.content_text,
            token_count=row.token_count,
            metadata=json.loads(row.metadata_json or "{}"),
            created_at=row.created_at,
        )

    @staticmethod
    def _embedding_model_to_record(row: EmbeddingModel) -> EmbeddingRecord:
        return EmbeddingRecord(
            embedding_id=row.embedding_id,
            chunk_id=row.chunk_id,
            model_name=row.model_name,
            dimension=row.dimension,
            vector=[float(value) for value in json.loads(row.vector_json or "[]")],
            created_at=row.created_at,
        )

    @staticmethod
    def _derived_dataset_model_to_record(row: DerivedDatasetModel) -> DerivedDatasetRecord:
        return DerivedDatasetRecord(
            derived_dataset_id=row.derived_dataset_id,
            workspace_id=row.workspace_id,
            asset_id=row.asset_id,
            parent_dataset_id=row.parent_dataset_id,
            dataset_name=row.dataset_name,
            dataset_key=row.dataset_key,
            source_fingerprint=row.source_fingerprint,
            object_key=row.object_key,
            content_type=row.content_type,
            transform_prompt=row.transform_prompt,
            row_count=row.row_count,
            column_count=row.column_count,
            preview_columns=[str(value) for value in json.loads(row.preview_columns_json or "[]")],
            preview_rows=list(json.loads(row.preview_rows_json or "[]")),
            created_at=row.created_at,
        )

    @staticmethod
    def _solve_run_model_to_record(row: SolveRunModel) -> SolveRunRecord:
        return SolveRunRecord(
            run_id=row.run_id,
            workspace_id=row.workspace_id,
            user_id=row.user_id,
            asset_id=row.asset_id,
            dataset_id=row.dataset_id,
            query=row.query,
            route=row.route,
            status=row.status,
            plan_text=row.plan_text,
            retrieval_trace=json.loads(row.retrieval_trace_json or "{}"),
            retrieved_chunk_ids=[str(value) for value in json.loads(row.retrieved_chunk_ids_json or "[]")],
            final_output=json.loads(row.final_output_json) if row.final_output_json else None,
            final_summary=row.final_summary,
            packaged_output=json.loads(row.packaged_output_json) if row.packaged_output_json else None,
            result_hash=row.result_hash,
            error_message=row.error_message,
            queued_at=row.queued_at,
            started_at=row.started_at,
            finished_at=row.finished_at,
            queue_wait_ms=row.queue_wait_ms,
            elapsed_ms=row.elapsed_ms,
        )

    @staticmethod
    def _solve_step_model_to_record(row: SolveStepModel) -> SolveStepRecord:
        return SolveStepRecord(
            step_id=row.step_id,
            run_id=row.run_id,
            step_index=row.step_index,
            stage=row.stage,
            status=row.status,
            title=row.title,
            detail=json.loads(row.detail_json or "{}"),
            created_at=row.created_at,
        )

    @staticmethod
    def _validator_report_model_to_record(row: ValidatorReportModel) -> ValidatorReportRecord:
        return ValidatorReportRecord(
            report_id=row.report_id,
            run_id=row.run_id,
            attempt_index=row.attempt_index,
            status=row.status,
            checks=list(json.loads(row.checks_json or "[]")),
            error_message=row.error_message,
            created_at=row.created_at,
        )

    @staticmethod
    def _feedback_event_model_to_record(row: FeedbackEventModel) -> FeedbackEventRecord:
        return FeedbackEventRecord(
            feedback_id=row.feedback_id,
            run_id=row.run_id,
            chunk_id=row.chunk_id,
            event_type=row.event_type,
            score=row.score,
            metadata=json.loads(row.metadata_json or "{}"),
            created_at=row.created_at,
        )

    @staticmethod
    def _failure_log_model_to_record(row: FailureLogModel) -> FailureLogRecord:
        return FailureLogRecord(
            failure_id=row.failure_id,
            query=row.query,
            stage=row.stage,
            error_message=row.error_message,
            metadata=json.loads(row.metadata_json or "{}"),
            created_at=row.created_at,
        )

    def create_user(
        self,
        *,
        email: str,
        display_name: str,
        password_hash: str,
        password_salt: str,
    ) -> UserRecord:
        normalized_email = _normalize_email(email)
        if not normalized_email:
            raise ValueError("Email is required.")

        now = _utc_now()
        user_id = uuid.uuid4().hex
        with self._session() as session, session.begin():
            existing_user = session.scalar(
                select(UserModel).where(UserModel.email == normalized_email)
            )
            if existing_user is not None:
                raise ValueError(f"User '{normalized_email}' already exists.")

            session.add(
                UserModel(
                    user_id=user_id,
                    email=normalized_email,
                    display_name=str(display_name or "").strip() or normalized_email.split("@")[0],
                    password_hash=password_hash,
                    password_salt=password_salt,
                    created_at=now,
                    updated_at=now,
                )
            )

        return self.get_user(user_id)  # type: ignore[return-value]

    def get_user(self, user_id: str) -> UserRecord | None:
        with self._session() as session:
            row = session.get(UserModel, user_id)
            if row is None:
                return None
            return self._user_model_to_record(row)

    def get_user_with_secret(self, email: str) -> tuple[UserRecord, str, str] | None:
        normalized_email = _normalize_email(email)
        with self._session() as session:
            row = session.scalar(
                select(UserModel).where(UserModel.email == normalized_email)
            )
            if row is None:
                return None
            return self._user_model_to_record(row), row.password_hash, row.password_salt

    def create_session(self, *, user_id: str, token_hash: str, expires_at: str) -> UserSessionRecord:
        record = UserSessionRecord(
            session_id=uuid.uuid4().hex,
            user_id=user_id,
            token_hash=token_hash,
            created_at=_utc_now(),
            expires_at=expires_at,
            revoked_at=None,
        )
        with self._session() as session, session.begin():
            session.add(
                UserSessionModel(
                    session_id=record.session_id,
                    user_id=record.user_id,
                    token_hash=record.token_hash,
                    created_at=record.created_at,
                    expires_at=record.expires_at,
                    revoked_at=record.revoked_at,
                )
            )
        return record

    def get_user_by_token_hash(self, token_hash: str) -> UserRecord | None:
        now = _utc_now()
        with self._session() as session:
            session_row = session.scalar(
                select(UserSessionModel).where(UserSessionModel.token_hash == token_hash)
            )
            if session_row is None or session_row.revoked_at is not None:
                return None
            if session_row.expires_at <= now:
                return None
            user_row = session.get(UserModel, session_row.user_id)
            if user_row is None:
                return None
            return self._user_model_to_record(user_row)

    def revoke_session(self, token_hash: str) -> bool:
        revoked_at = _utc_now()
        with self._session() as session, session.begin():
            row = session.scalar(
                select(UserSessionModel).where(UserSessionModel.token_hash == token_hash)
            )
            if row is None or row.revoked_at is not None:
                return False
            row.revoked_at = revoked_at
            session.add(row)
        return True

    def user_has_dataset_access(self, user_id: str, dataset_id: str) -> bool:
        with self._session() as session:
            row = session.get(
                UserDatasetModel,
                {"user_id": user_id, "dataset_id": dataset_id},
            )
            return row is not None

    def user_has_job_access(self, user_id: str, job_id: str) -> bool:
        with self._session() as session:
            row = session.get(
                UserJobModel,
                {"user_id": user_id, "job_id": job_id},
            )
            return row is not None

    def user_has_decision_access(self, user_id: str, decision_history_id: str) -> bool:
        with self._session() as session:
            row = session.scalar(
                select(DecisionHistoryModel)
                .join(UserJobModel, UserJobModel.job_id == DecisionHistoryModel.job_id)
                .where(
                    DecisionHistoryModel.decision_history_id == decision_history_id,
                    UserJobModel.user_id == user_id,
                )
            )
            return row is not None

    def create_workspace(
        self,
        *,
        user_id: str,
        name: str,
        description: str | None = None,
    ) -> WorkspaceRecord:
        record = WorkspaceRecord(
            workspace_id=uuid.uuid4().hex,
            user_id=user_id,
            name=str(name or "").strip() or "Untitled workspace",
            description=str(description).strip() if description else None,
            created_at=_utc_now(),
            updated_at=_utc_now(),
        )
        with self._session() as session, session.begin():
            session.add(
                WorkspaceModel(
                    workspace_id=record.workspace_id,
                    user_id=record.user_id,
                    name=record.name,
                    description=record.description,
                    created_at=record.created_at,
                    updated_at=record.updated_at,
                )
            )
        return record

    def user_has_workspace_access(self, user_id: str, workspace_id: str) -> bool:
        with self._session() as session:
            row = session.get(WorkspaceModel, workspace_id)
            return row is not None and row.user_id == user_id

    def list_user_workspaces(self, user_id: str, *, limit: int = 20) -> list[WorkspaceRecord]:
        with self._session() as session:
            rows = session.scalars(
                select(WorkspaceModel)
                .where(WorkspaceModel.user_id == user_id)
                .order_by(WorkspaceModel.updated_at.desc(), WorkspaceModel.name.asc())
                .limit(int(limit))
            ).all()
            return [self._workspace_model_to_record(row) for row in rows]

    def list_workspaces(self, *, limit: int = 100) -> list[WorkspaceRecord]:
        with self._session() as session:
            rows = session.scalars(
                select(WorkspaceModel)
                .order_by(WorkspaceModel.updated_at.desc(), WorkspaceModel.name.asc())
                .limit(int(limit))
            ).all()
            return [self._workspace_model_to_record(row) for row in rows]

    def get_workspace(self, workspace_id: str) -> WorkspaceRecord | None:
        with self._session() as session:
            row = session.get(WorkspaceModel, workspace_id)
            if row is None:
                return None
            return self._workspace_model_to_record(row)

    def create_asset(
        self,
        *,
        workspace_id: str,
        title: str,
        asset_kind: str,
        primary_dataset_id: str | None = None,
    ) -> AssetRecord:
        now = _utc_now()
        record = AssetRecord(
            asset_id=uuid.uuid4().hex,
            workspace_id=workspace_id,
            title=str(title or "").strip() or "Untitled asset",
            asset_kind=str(asset_kind or "mixed"),
            primary_dataset_id=primary_dataset_id,
            created_at=now,
            updated_at=now,
        )
        with self._session() as session, session.begin():
            session.add(
                AssetModel(
                    asset_id=record.asset_id,
                    workspace_id=record.workspace_id,
                    title=record.title,
                    asset_kind=record.asset_kind,
                    primary_dataset_id=record.primary_dataset_id,
                    created_at=record.created_at,
                    updated_at=record.updated_at,
                )
            )
        return record

    def update_asset_primary_dataset(self, asset_id: str, primary_dataset_id: str | None) -> AssetRecord | None:
        with self._session() as session, session.begin():
            row = session.get(AssetModel, asset_id)
            if row is None:
                return None
            row.primary_dataset_id = primary_dataset_id
            row.updated_at = _utc_now()
            session.add(row)
        return self.get_asset(asset_id)

    def update_asset_metadata(
        self,
        asset_id: str,
        *,
        asset_kind: str | None = None,
        primary_dataset_id: str | None | object = _UNSET,
        title: str | None | object = _UNSET,
    ) -> AssetRecord | None:
        with self._session() as session, session.begin():
            row = session.get(AssetModel, asset_id)
            if row is None:
                return None
            if asset_kind is not None:
                row.asset_kind = str(asset_kind or row.asset_kind)
            if primary_dataset_id is not _UNSET:
                row.primary_dataset_id = primary_dataset_id
            if title is not _UNSET and title is not None and str(title).strip():
                row.title = str(title).strip()
            row.updated_at = _utc_now()
            session.add(row)
        return self.get_asset(asset_id)

    def get_asset(self, asset_id: str) -> AssetRecord | None:
        with self._session() as session:
            row = session.get(AssetModel, asset_id)
            if row is None:
                return None
            return self._asset_model_to_record(row)

    def list_workspace_assets(self, workspace_id: str, *, limit: int = 50) -> list[AssetRecord]:
        with self._session() as session:
            rows = session.scalars(
                select(AssetModel)
                .where(AssetModel.workspace_id == workspace_id)
                .order_by(AssetModel.updated_at.desc(), AssetModel.created_at.desc())
                .limit(int(limit))
            ).all()
            return [self._asset_model_to_record(row) for row in rows]

    def create_asset_file(
        self,
        *,
        asset_id: str,
        dataset_id: str | None,
        file_name: str,
        media_type: str,
        file_kind: str,
        language: str | None,
        object_key: str,
        checksum: str,
        size_bytes: int,
    ) -> AssetFileRecord:
        record = AssetFileRecord(
            asset_file_id=uuid.uuid4().hex,
            asset_id=asset_id,
            dataset_id=dataset_id,
            file_name=str(file_name or ""),
            media_type=str(media_type or "application/octet-stream"),
            file_kind=str(file_kind or "binary"),
            language=str(language).strip() if language else None,
            object_key=str(object_key or ""),
            checksum=str(checksum or ""),
            size_bytes=int(size_bytes),
            created_at=_utc_now(),
        )
        with self._session() as session, session.begin():
            session.add(
                AssetFileModel(
                    asset_file_id=record.asset_file_id,
                    asset_id=record.asset_id,
                    dataset_id=record.dataset_id,
                    file_name=record.file_name,
                    media_type=record.media_type,
                    file_kind=record.file_kind,
                    language=record.language,
                    object_key=record.object_key,
                    checksum=record.checksum,
                    size_bytes=record.size_bytes,
                    created_at=record.created_at,
                )
            )
        return record

    def list_asset_files(self, asset_id: str) -> list[AssetFileRecord]:
        with self._session() as session:
            rows = session.scalars(
                select(AssetFileModel)
                .where(AssetFileModel.asset_id == asset_id)
                .order_by(AssetFileModel.created_at.asc(), AssetFileModel.file_name.asc())
            ).all()
            return [self._asset_file_model_to_record(row) for row in rows]

    def list_asset_datasets(self, asset_id: str) -> list[DatasetRecord]:
        with self._session() as session:
            rows = session.execute(
                select(DatasetModel)
                .join(AssetFileModel, AssetFileModel.dataset_id == DatasetModel.dataset_id)
                .where(AssetFileModel.asset_id == asset_id, AssetFileModel.dataset_id.is_not(None))
                .order_by(DatasetModel.created_at.desc())
            ).scalars().all()

        return [
            DatasetRecord(
                dataset_id=row.dataset_id,
                dataset_name=row.dataset_name,
                dataset_key=row.dataset_key,
                source_fingerprint=row.source_fingerprint,
                source_kind=row.source_kind,
                source_label=row.source_label,
                object_key=row.object_key,
                content_type=row.content_type,
                size_bytes=row.size_bytes,
                created_at=row.created_at,
            )
            for row in rows
        ]

    def get_asset_by_dataset_id(self, dataset_id: str) -> AssetRecord | None:
        with self._session() as session:
            row = session.execute(
                select(AssetModel)
                .join(AssetFileModel, AssetFileModel.asset_id == AssetModel.asset_id)
                .where(AssetFileModel.dataset_id == dataset_id)
                .order_by(AssetFileModel.created_at.desc())
            ).scalars().first()
            if row is None:
                return None
            return self._asset_model_to_record(row)

    def get_workspace_by_dataset_id(self, dataset_id: str) -> WorkspaceRecord | None:
        asset = self.get_asset_by_dataset_id(dataset_id)
        if asset is None:
            return None
        return self.get_workspace(asset.workspace_id)

    def record_chunk(
        self,
        *,
        asset_id: str,
        asset_file_id: str | None,
        dataset_id: str | None,
        chunk_index: int,
        title: str,
        content_text: str,
        token_count: int,
        metadata: dict[str, Any] | None = None,
    ) -> ChunkRecord:
        record = ChunkRecord(
            chunk_id=uuid.uuid4().hex,
            asset_id=asset_id,
            asset_file_id=asset_file_id,
            dataset_id=dataset_id,
            chunk_index=int(chunk_index),
            title=str(title or f"Chunk {chunk_index}"),
            content_text=str(content_text or ""),
            token_count=int(token_count),
            metadata=dict(metadata or {}),
            created_at=_utc_now(),
        )
        with self._session() as session, session.begin():
            session.add(
                ChunkModel(
                    chunk_id=record.chunk_id,
                    asset_id=record.asset_id,
                    asset_file_id=record.asset_file_id,
                    dataset_id=record.dataset_id,
                    chunk_index=record.chunk_index,
                    title=record.title,
                    content_text=record.content_text,
                    token_count=record.token_count,
                    metadata_json=_serialize_json(record.metadata),
                    created_at=record.created_at,
                )
            )
        return record

    def list_asset_chunks(self, asset_id: str, *, limit: int = 200) -> list[ChunkRecord]:
        with self._session() as session:
            rows = session.scalars(
                select(ChunkModel)
                .where(ChunkModel.asset_id == asset_id)
                .order_by(ChunkModel.chunk_index.asc(), ChunkModel.created_at.asc())
                .limit(int(limit))
            ).all()
            return [self._chunk_model_to_record(row) for row in rows]

    def list_workspace_chunks(self, workspace_id: str, *, limit: int = 500) -> list[ChunkRecord]:
        with self._session() as session:
            rows = session.execute(
                select(ChunkModel)
                .join(AssetModel, AssetModel.asset_id == ChunkModel.asset_id)
                .where(AssetModel.workspace_id == workspace_id)
                .order_by(ChunkModel.created_at.desc(), ChunkModel.chunk_index.asc())
                .limit(int(limit))
            ).scalars().all()
            return [self._chunk_model_to_record(row) for row in rows]

    def get_chunks(self, chunk_ids: list[str]) -> list[ChunkRecord]:
        if not chunk_ids:
            return []
        with self._session() as session:
            rows = session.scalars(
                select(ChunkModel).where(ChunkModel.chunk_id.in_(list(chunk_ids)))
            ).all()
            records = [self._chunk_model_to_record(row) for row in rows]
        order_lookup = {chunk_id: index for index, chunk_id in enumerate(chunk_ids)}
        return sorted(records, key=lambda record: order_lookup.get(record.chunk_id, len(order_lookup)))

    def upsert_embedding(
        self,
        *,
        chunk_id: str,
        model_name: str,
        vector: list[float],
    ) -> EmbeddingRecord:
        now = _utc_now()
        normalized_vector = [float(value) for value in vector]
        with self._session() as session, session.begin():
            row = session.scalar(select(EmbeddingModel).where(EmbeddingModel.chunk_id == chunk_id))
            if row is None:
                row = EmbeddingModel(
                    embedding_id=uuid.uuid4().hex,
                    chunk_id=chunk_id,
                    model_name=str(model_name or ""),
                    dimension=len(normalized_vector),
                    vector_json=_serialize_json(normalized_vector),
                    created_at=now,
                )
            else:
                row.model_name = str(model_name or row.model_name)
                row.dimension = len(normalized_vector)
                row.vector_json = _serialize_json(normalized_vector)
                row.created_at = now
            session.add(row)
        with self._session() as session:
            persisted = session.scalar(select(EmbeddingModel).where(EmbeddingModel.chunk_id == chunk_id))
            return self._embedding_model_to_record(persisted)  # type: ignore[arg-type]

    def list_embeddings_for_chunk_ids(self, chunk_ids: list[str]) -> list[EmbeddingRecord]:
        if not chunk_ids:
            return []
        with self._session() as session:
            rows = session.scalars(
                select(EmbeddingModel).where(EmbeddingModel.chunk_id.in_(list(chunk_ids)))
            ).all()
            return [self._embedding_model_to_record(row) for row in rows]

    def create_derived_dataset(
        self,
        *,
        workspace_id: str,
        asset_id: str | None,
        parent_dataset_id: str | None,
        dataset_name: str,
        dataset_key: str,
        source_fingerprint: str,
        object_key: str,
        content_type: str,
        transform_prompt: str | None,
        row_count: int,
        column_count: int,
        preview_columns: list[str],
        preview_rows: list[dict[str, Any]],
    ) -> DerivedDatasetRecord:
        record = DerivedDatasetRecord(
            derived_dataset_id=uuid.uuid4().hex,
            workspace_id=workspace_id,
            asset_id=asset_id,
            parent_dataset_id=parent_dataset_id,
            dataset_name=str(dataset_name or ""),
            dataset_key=str(dataset_key or ""),
            source_fingerprint=str(source_fingerprint or ""),
            object_key=str(object_key or ""),
            content_type=str(content_type or "text/csv"),
            transform_prompt=str(transform_prompt).strip() if transform_prompt else None,
            row_count=int(row_count),
            column_count=int(column_count),
            preview_columns=[str(value) for value in preview_columns],
            preview_rows=[dict(row) for row in preview_rows],
            created_at=_utc_now(),
        )
        with self._session() as session, session.begin():
            session.add(
                DerivedDatasetModel(
                    derived_dataset_id=record.derived_dataset_id,
                    workspace_id=record.workspace_id,
                    asset_id=record.asset_id,
                    parent_dataset_id=record.parent_dataset_id,
                    dataset_name=record.dataset_name,
                    dataset_key=record.dataset_key,
                    source_fingerprint=record.source_fingerprint,
                    object_key=record.object_key,
                    content_type=record.content_type,
                    transform_prompt=record.transform_prompt,
                    row_count=record.row_count,
                    column_count=record.column_count,
                    preview_columns_json=_serialize_json(record.preview_columns),
                    preview_rows_json=_serialize_json(record.preview_rows),
                    created_at=record.created_at,
                )
            )
        return record

    def get_derived_dataset(self, derived_dataset_id: str) -> DerivedDatasetRecord | None:
        with self._session() as session:
            row = session.get(DerivedDatasetModel, derived_dataset_id)
            if row is None:
                return None
            return self._derived_dataset_model_to_record(row)

    def list_workspace_derived_datasets(self, workspace_id: str, *, limit: int = 50) -> list[DerivedDatasetRecord]:
        with self._session() as session:
            rows = session.scalars(
                select(DerivedDatasetModel)
                .where(DerivedDatasetModel.workspace_id == workspace_id)
                .order_by(DerivedDatasetModel.created_at.desc())
                .limit(int(limit))
            ).all()
            return [self._derived_dataset_model_to_record(row) for row in rows]

    def create_solve_run(
        self,
        *,
        workspace_id: str,
        user_id: str | None,
        query: str,
        route: str,
        asset_id: str | None = None,
        dataset_id: str | None = None,
        status: str = "queued",
        plan_text: str | None = None,
        retrieval_trace: dict[str, Any] | None = None,
    ) -> SolveRunRecord:
        record = SolveRunRecord(
            run_id=uuid.uuid4().hex,
            workspace_id=workspace_id,
            user_id=user_id,
            asset_id=asset_id,
            dataset_id=dataset_id,
            query=str(query or ""),
            route=str(route or "general"),
            status=str(status or "queued"),
            plan_text=plan_text,
            retrieval_trace=dict(retrieval_trace or {}),
            retrieved_chunk_ids=[],
            final_output=None,
            final_summary=None,
            packaged_output=None,
            result_hash=None,
            error_message=None,
            queued_at=_utc_now(),
            started_at=None,
            finished_at=None,
            queue_wait_ms=None,
            elapsed_ms=None,
        )
        if status == "completed":
            record.started_at = record.queued_at
            record.finished_at = record.queued_at
            record.queue_wait_ms = 0
            record.elapsed_ms = 0
        with self._session() as session, session.begin():
            session.add(
                SolveRunModel(
                    run_id=record.run_id,
                    workspace_id=record.workspace_id,
                    user_id=record.user_id,
                    asset_id=record.asset_id,
                    dataset_id=record.dataset_id,
                    query=record.query,
                    route=record.route,
                    status=record.status,
                    plan_text=record.plan_text,
                    retrieval_trace_json=_serialize_json(record.retrieval_trace),
                    retrieved_chunk_ids_json=_serialize_json(record.retrieved_chunk_ids),
                    final_output_json=None,
                    final_summary=None,
                    packaged_output_json=None,
                    result_hash=None,
                    error_message=None,
                    queued_at=record.queued_at,
                    started_at=record.started_at,
                    finished_at=record.finished_at,
                    queue_wait_ms=record.queue_wait_ms,
                    elapsed_ms=record.elapsed_ms,
                )
            )
        return record

    def mark_solve_run_running(self, run_id: str) -> SolveRunRecord | None:
        started_at = _utc_now()
        with self._session() as session, session.begin():
            row = session.get(SolveRunModel, run_id)
            if row is None:
                return None
            row.status = "running"
            row.started_at = started_at
            row.queue_wait_ms = max(
                int((datetime.fromisoformat(started_at) - datetime.fromisoformat(row.queued_at)).total_seconds() * 1000),
                0,
            )
            session.add(row)
        return self.get_solve_run(run_id)

    def complete_solve_run(
        self,
        run_id: str,
        *,
        plan_text: str | None,
        retrieval_trace: dict[str, Any],
        retrieved_chunk_ids: list[str],
        final_output: dict[str, Any],
        final_summary: str | None,
        packaged_output: dict[str, Any] | None = None,
        result_hash: str | None = None,
    ) -> SolveRunRecord | None:
        finished_at = _utc_now()
        with self._session() as session, session.begin():
            row = session.get(SolveRunModel, run_id)
            if row is None:
                return None
            row.status = "completed"
            row.plan_text = plan_text
            row.retrieval_trace_json = _serialize_json(retrieval_trace or {})
            row.retrieved_chunk_ids_json = _serialize_json(retrieved_chunk_ids or [])
            row.final_output_json = _serialize_json(final_output or {})
            row.final_summary = final_summary
            row.packaged_output_json = _serialize_json(packaged_output or {})
            row.result_hash = str(result_hash or "") or None
            row.finished_at = finished_at
            if row.started_at:
                row.elapsed_ms = max(
                    int((datetime.fromisoformat(finished_at) - datetime.fromisoformat(row.started_at)).total_seconds() * 1000),
                    0,
                )
            session.add(row)
        return self.get_solve_run(run_id)

    def fail_solve_run(
        self,
        run_id: str,
        *,
        error_message: str,
        plan_text: str | None = None,
        retrieval_trace: dict[str, Any] | None = None,
        retrieved_chunk_ids: list[str] | None = None,
    ) -> SolveRunRecord | None:
        finished_at = _utc_now()
        with self._session() as session, session.begin():
            row = session.get(SolveRunModel, run_id)
            if row is None:
                return None
            row.status = "failed"
            row.error_message = str(error_message or "")
            row.result_hash = None
            if plan_text is not None:
                row.plan_text = plan_text
            if retrieval_trace is not None:
                row.retrieval_trace_json = _serialize_json(retrieval_trace)
            if retrieved_chunk_ids is not None:
                row.retrieved_chunk_ids_json = _serialize_json(retrieved_chunk_ids)
            row.finished_at = finished_at
            if row.started_at:
                row.elapsed_ms = max(
                    int((datetime.fromisoformat(finished_at) - datetime.fromisoformat(row.started_at)).total_seconds() * 1000),
                    0,
                )
            session.add(row)
        return self.get_solve_run(run_id)

    def get_solve_run(self, run_id: str) -> SolveRunRecord | None:
        with self._session() as session:
            row = session.get(SolveRunModel, run_id)
            if row is None:
                return None
            return self._solve_run_model_to_record(row)

    def list_workspace_solve_runs(
        self,
        workspace_id: str | None = None,
        *,
        limit: int = 50,
    ) -> list[SolveRunRecord]:
        with self._session() as session:
            query = select(SolveRunModel)
            if workspace_id is not None:
                query = query.where(SolveRunModel.workspace_id == workspace_id)
            rows = session.scalars(
                query.order_by(SolveRunModel.queued_at.desc()).limit(int(limit))
            ).all()
            return [self._solve_run_model_to_record(row) for row in rows]

    def record_solve_step(
        self,
        *,
        run_id: str,
        step_index: int,
        stage: str,
        status: str,
        title: str,
        detail: dict[str, Any] | None = None,
    ) -> SolveStepRecord:
        record = SolveStepRecord(
            step_id=uuid.uuid4().hex,
            run_id=run_id,
            step_index=int(step_index),
            stage=str(stage or "unknown"),
            status=str(status or "completed"),
            title=str(title or stage or "Step"),
            detail=dict(detail or {}),
            created_at=_utc_now(),
        )
        with self._session() as session, session.begin():
            session.add(
                SolveStepModel(
                    step_id=record.step_id,
                    run_id=record.run_id,
                    step_index=record.step_index,
                    stage=record.stage,
                    status=record.status,
                    title=record.title,
                    detail_json=_serialize_json(record.detail),
                    created_at=record.created_at,
                )
            )
        return record

    def list_solve_steps(self, run_id: str) -> list[SolveStepRecord]:
        with self._session() as session:
            rows = session.scalars(
                select(SolveStepModel)
                .where(SolveStepModel.run_id == run_id)
                .order_by(SolveStepModel.step_index.asc(), SolveStepModel.created_at.asc())
            ).all()
            return [self._solve_step_model_to_record(row) for row in rows]

    def record_validator_report(
        self,
        *,
        run_id: str,
        attempt_index: int,
        status: str,
        checks: list[dict[str, Any]],
        error_message: str | None = None,
    ) -> ValidatorReportRecord:
        record = ValidatorReportRecord(
            report_id=uuid.uuid4().hex,
            run_id=run_id,
            attempt_index=int(attempt_index),
            status=str(status or "unknown"),
            checks=[dict(item) for item in checks],
            error_message=str(error_message or "") if error_message else None,
            created_at=_utc_now(),
        )
        with self._session() as session, session.begin():
            session.add(
                ValidatorReportModel(
                    report_id=record.report_id,
                    run_id=record.run_id,
                    attempt_index=record.attempt_index,
                    status=record.status,
                    checks_json=_serialize_json(record.checks),
                    error_message=record.error_message,
                    created_at=record.created_at,
                )
            )
        return record

    def list_validator_reports(
        self,
        run_id: str | None = None,
        *,
        limit: int = 50,
    ) -> list[ValidatorReportRecord]:
        with self._session() as session:
            query = select(ValidatorReportModel)
            if run_id is not None:
                query = query.where(ValidatorReportModel.run_id == run_id)
            rows = session.scalars(
                query.order_by(ValidatorReportModel.created_at.desc()).limit(int(limit))
            ).all()
            return [self._validator_report_model_to_record(row) for row in rows]

    def record_feedback_event(
        self,
        *,
        run_id: str,
        chunk_id: str | None,
        event_type: str,
        score: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> FeedbackEventRecord:
        record = FeedbackEventRecord(
            feedback_id=uuid.uuid4().hex,
            run_id=run_id,
            chunk_id=chunk_id,
            event_type=str(event_type or "unknown"),
            score=score,
            metadata=dict(metadata or {}),
            created_at=_utc_now(),
        )
        with self._session() as session, session.begin():
            session.add(
                FeedbackEventModel(
                    feedback_id=record.feedback_id,
                    run_id=record.run_id,
                    chunk_id=record.chunk_id,
                    event_type=record.event_type,
                    score=record.score,
                    metadata_json=_serialize_json(record.metadata),
                    created_at=record.created_at,
                )
            )
        return record

    def list_feedback_events(
        self,
        *,
        run_id: str | None = None,
        limit: int = 100,
    ) -> list[FeedbackEventRecord]:
        with self._session() as session:
            query = select(FeedbackEventModel)
            if run_id is not None:
                query = query.where(FeedbackEventModel.run_id == run_id)
            rows = session.scalars(
                query.order_by(FeedbackEventModel.created_at.desc()).limit(int(limit))
            ).all()
            return [self._feedback_event_model_to_record(row) for row in rows]

    def record_failure_log(
        self,
        *,
        query: str,
        error: str,
        stage: str,
        metadata: dict[str, Any] | None = None,
    ) -> FailureLogRecord:
        record = FailureLogRecord(
            failure_id=uuid.uuid4().hex,
            query=str(query or ""),
            stage=str(stage or "unknown"),
            error_message=str(error or ""),
            metadata=dict(metadata or {}),
            created_at=_utc_now(),
        )
        with self._session() as session, session.begin():
            session.add(
                FailureLogModel(
                    failure_id=record.failure_id,
                    query=record.query,
                    stage=record.stage,
                    error_message=record.error_message,
                    metadata_json=_serialize_json(record.metadata),
                    created_at=record.created_at,
                )
            )
        return record

    def list_failure_logs(
        self,
        *,
        stage: str | None = None,
        limit: int = 100,
    ) -> list[FailureLogRecord]:
        with self._session() as session:
            query = select(FailureLogModel)
            if stage is not None:
                query = query.where(FailureLogModel.stage == stage)
            rows = session.scalars(
                query.order_by(FailureLogModel.created_at.desc()).limit(int(limit))
            ).all()
            return [self._failure_log_model_to_record(row) for row in rows]

    def list_matching_run_result_hashes(
        self,
        *,
        source_fingerprint: str,
        normalized_query: str,
        analysis_intent: str | None = None,
        limit: int = 100,
    ) -> list[str]:
        from backend.analysis_contract import classify_analysis_intent
        from backend.services.result_consistency import normalize_query

        with self._session() as session:
            rows = session.scalars(
                select(WorkflowRunModel)
                .where(WorkflowRunModel.source_fingerprint == source_fingerprint, WorkflowRunModel.result_hash.is_not(None))
                .order_by(WorkflowRunModel.created_at.desc())
                .limit(int(limit))
            ).all()
        hashes: list[str] = []
        for row in rows:
            if normalize_query(row.analysis_query) != normalized_query:
                continue
            if analysis_intent and classify_analysis_intent(row.analysis_query) != analysis_intent:
                continue
            if row.result_hash:
                hashes.append(str(row.result_hash))
        return hashes

    def list_matching_forecast_result_hashes(
        self,
        *,
        source_fingerprint: str,
        target_column: str,
        horizon: str,
        limit: int = 100,
    ) -> list[str]:
        with self._session() as session:
            rows = session.scalars(
                select(ForecastArtifactModel)
                .where(
                    ForecastArtifactModel.source_fingerprint == source_fingerprint,
                    ForecastArtifactModel.target_column == str(target_column or ""),
                    ForecastArtifactModel.horizon == str(horizon or ""),
                    ForecastArtifactModel.result_hash.is_not(None),
                )
                .order_by(ForecastArtifactModel.created_at.desc())
                .limit(int(limit))
            ).all()
        return [str(row.result_hash) for row in rows if row.result_hash]

    def list_matching_solve_result_hashes(
        self,
        *,
        source_fingerprint: str,
        normalized_query: str,
        route: str,
        limit: int = 100,
    ) -> list[str]:
        from backend.services.result_consistency import normalize_query

        with self._session() as session:
            rows = session.execute(
                select(SolveRunModel, DatasetModel)
                .join(DatasetModel, DatasetModel.dataset_id == SolveRunModel.dataset_id)
                .where(
                    SolveRunModel.route == str(route or "data"),
                    SolveRunModel.result_hash.is_not(None),
                    DatasetModel.source_fingerprint == source_fingerprint,
                )
                .order_by(SolveRunModel.queued_at.desc())
                .limit(int(limit))
            ).all()
        hashes: list[str] = []
        for solve_row, dataset_row in rows:
            del dataset_row
            if normalize_query(solve_row.query) != normalized_query:
                continue
            if solve_row.result_hash:
                hashes.append(str(solve_row.result_hash))
        return hashes

    def get_chunk_feedback_scores(self, chunk_ids: list[str]) -> dict[str, float]:
        if not chunk_ids:
            return {}
        with self._session() as session:
            rows = session.scalars(
                select(FeedbackEventModel)
                .where(FeedbackEventModel.chunk_id.in_(list(chunk_ids)), FeedbackEventModel.score.is_not(None))
            ).all()
        scores: dict[str, float] = {chunk_id: 0.0 for chunk_id in chunk_ids}
        for row in rows:
            if row.chunk_id is None:
                continue
            scores[row.chunk_id] = scores.get(row.chunk_id, 0.0) + float(row.score or 0)
        return scores

    def save_workflow(
        self,
        *,
        name: str,
        source_config: dict,
        cleaning_options: dict,
        analysis_query: str,
        forecast_config: dict | None,
        chart_preferences: dict,
        export_settings: dict,
        workflow_id: str | None = None,
    ) -> WorkflowDefinition:
        normalized_name = str(name or "").strip()
        if not normalized_name:
            raise ValueError("Workflow name is required.")

        now = _utc_now()
        with self._session() as session, session.begin():
            if workflow_id:
                workflow = session.get(WorkflowModel, workflow_id)
                if workflow is None:
                    raise ValueError(f"Workflow '{workflow_id}' does not exist.")
                version = int(workflow.latest_version) + 1
                workflow.name = normalized_name
                workflow.updated_at = now
                workflow.latest_version = version
            else:
                workflow_id = uuid.uuid4().hex
                version = 1
                session.add(
                    WorkflowModel(
                        workflow_id=workflow_id,
                        name=normalized_name,
                        created_at=now,
                        updated_at=now,
                        latest_version=version,
                    )
                )

            session.add(
                WorkflowVersionModel(
                    workflow_id=workflow_id,
                    version=version,
                    source_config_json=_serialize_json(source_config),
                    cleaning_options_json=_serialize_json(cleaning_options),
                    analysis_query=str(analysis_query or ""),
                    forecast_config_json=_serialize_json(forecast_config or {}),
                    chart_preferences_json=_serialize_json(chart_preferences),
                    export_settings_json=_serialize_json(export_settings),
                    created_at=now,
                )
            )

        return WorkflowDefinition(
            workflow_id=workflow_id,
            name=normalized_name,
            version=version,
            source_config=source_config,
            cleaning_options=cleaning_options,
            analysis_query=str(analysis_query or ""),
            forecast_config=dict(forecast_config or {}),
            chart_preferences=chart_preferences,
            export_settings=export_settings,
            created_at=now,
        )

    def list_workflows(self) -> list[WorkflowDefinition]:
        with self._session() as session:
            workflows = session.scalars(select(WorkflowModel).order_by(WorkflowModel.updated_at.desc(), WorkflowModel.name.asc())).all()
            definitions: list[WorkflowDefinition] = []
            for workflow in workflows:
                version_row = session.get(
                    WorkflowVersionModel,
                    {"workflow_id": workflow.workflow_id, "version": workflow.latest_version},
                )
                if version_row is not None:
                    definitions.append(self._workflow_version_to_definition(workflow, version_row))
            return definitions

    def list_workflow_versions(self, workflow_id: str) -> list[WorkflowDefinition]:
        with self._session() as session:
            workflow = session.get(WorkflowModel, workflow_id)
            if workflow is None:
                return []
            rows = session.scalars(
                select(WorkflowVersionModel)
                .where(WorkflowVersionModel.workflow_id == workflow_id)
                .order_by(WorkflowVersionModel.version.desc())
            ).all()
            return [self._workflow_version_to_definition(workflow, row) for row in rows]

    def get_workflow(self, workflow_id: str, version: int | None = None) -> WorkflowDefinition | None:
        with self._session() as session:
            workflow = session.get(WorkflowModel, workflow_id)
            if workflow is None:
                return None
            selected_version = int(version) if version is not None else int(workflow.latest_version)
            version_row = session.get(
                WorkflowVersionModel,
                {"workflow_id": workflow_id, "version": selected_version},
            )
            if version_row is None:
                return None
            return self._workflow_version_to_definition(workflow, version_row)

    def record_run(self, record: WorkflowRunRecord) -> WorkflowRunRecord:
        with self._session() as session, session.begin():
            session.add(
                WorkflowRunModel(
                    run_id=record.run_id,
                    workflow_id=record.workflow_id,
                    workflow_version=record.workflow_version,
                    workflow_name=record.workflow_name,
                    source_fingerprint=record.source_fingerprint,
                    source_label=record.source_label,
                    validation_findings_json=_serialize_json(record.validation_findings),
                    cleaning_actions_json=_serialize_json(record.cleaning_actions),
                    generated_code=record.generated_code,
                    final_status=record.final_status,
                    error_message=record.error_message,
                    export_artifacts_json=_serialize_json(record.export_artifacts),
                    analysis_query=record.analysis_query,
                    result_summary=record.result_summary,
                    result_hash=record.result_hash,
                    created_at=record.created_at,
                )
            )
        return record

    def list_runs(self, *, limit: int = 20, workflow_id: str | None = None) -> list[WorkflowRunRecord]:
        with self._session() as session:
            query = select(WorkflowRunModel)
            if workflow_id:
                query = query.where(WorkflowRunModel.workflow_id == workflow_id)
            query = query.order_by(WorkflowRunModel.created_at.desc()).limit(int(limit))
            rows = session.scalars(query).all()
            return [self._run_model_to_record(row) for row in rows]

    def record_forecast_artifact(self, record: ForecastArtifactRecord) -> ForecastArtifactRecord:
        with self._session() as session, session.begin():
            session.add(
                ForecastArtifactModel(
                    artifact_id=record.artifact_id,
                    workflow_id=record.workflow_id,
                    workflow_version=record.workflow_version,
                    workflow_name=record.workflow_name,
                    source_fingerprint=record.source_fingerprint,
                    source_label=record.source_label,
                    target_column=record.target_column,
                    horizon=record.horizon,
                    model_name=record.model_name,
                    training_mode=record.training_mode,
                    status=record.status,
                    artifact_key=record.artifact_key,
                    forecast_config_json=_serialize_json(record.forecast_config),
                    evaluation_metrics_json=_serialize_json(record.evaluation_metrics),
                    recommendation_payload_json=_serialize_json(record.recommendation_payload),
                    summary=record.summary,
                    result_hash=record.result_hash,
                    created_at=record.created_at,
                )
            )
        return record

    def list_forecast_artifacts(
        self,
        *,
        limit: int = 20,
        workflow_id: str | None = None,
        source_fingerprint: str | None = None,
    ) -> list[ForecastArtifactRecord]:
        with self._session() as session:
            query = select(ForecastArtifactModel)
            if workflow_id:
                query = query.where(ForecastArtifactModel.workflow_id == workflow_id)
            if source_fingerprint:
                query = query.where(ForecastArtifactModel.source_fingerprint == source_fingerprint)
            query = query.order_by(ForecastArtifactModel.created_at.desc()).limit(int(limit))
            rows = session.scalars(query).all()
            return [self._forecast_artifact_model_to_record(row) for row in rows]

    def record_decision_history(self, record: DecisionHistoryRecord) -> DecisionHistoryRecord:
        with self._session() as session, session.begin():
            session.add(
                DecisionHistoryModel(
                    decision_history_id=record.decision_history_id,
                    job_id=record.job_id,
                    forecast_artifact_id=record.forecast_artifact_id,
                    source_fingerprint=record.source_fingerprint,
                    query=record.query,
                    decision_id=record.decision_id,
                    decision_json=_serialize_json(record.decision_json),
                    priority=record.priority,
                    decision_confidence=record.decision_confidence,
                    risk_level=record.risk_level,
                    result_hash=record.result_hash,
                    outcome=record.outcome,
                    created_at=record.created_at,
                    updated_at=record.updated_at,
                )
            )
        return record

    def get_decision_history(
        self,
        decision_history_id: str,
        *,
        user_id: str | None = None,
    ) -> DecisionHistoryRecord | None:
        with self._session() as session:
            query = select(DecisionHistoryModel).where(
                DecisionHistoryModel.decision_history_id == decision_history_id
            )
            if user_id:
                query = query.join(UserJobModel, UserJobModel.job_id == DecisionHistoryModel.job_id).where(
                    UserJobModel.user_id == user_id
                )
            row = session.scalar(query)
            if row is None:
                return None
            return self._decision_history_model_to_record(row)

    def list_decision_history(
        self,
        *,
        limit: int = 20,
        user_id: str | None = None,
        job_id: str | None = None,
        forecast_artifact_id: str | None = None,
        source_fingerprint: str | None = None,
    ) -> list[DecisionHistoryRecord]:
        with self._session() as session:
            query = select(DecisionHistoryModel)
            if user_id:
                query = query.join(UserJobModel, UserJobModel.job_id == DecisionHistoryModel.job_id).where(
                    UserJobModel.user_id == user_id
                )
            if job_id:
                query = query.where(DecisionHistoryModel.job_id == job_id)
            if forecast_artifact_id:
                query = query.where(DecisionHistoryModel.forecast_artifact_id == forecast_artifact_id)
            if source_fingerprint:
                query = query.where(DecisionHistoryModel.source_fingerprint == source_fingerprint)
            query = query.order_by(DecisionHistoryModel.created_at.desc()).limit(int(limit))
            rows = session.scalars(query).all()
            return [self._decision_history_model_to_record(row) for row in rows]

    def update_decision_outcome(
        self,
        decision_history_id: str,
        outcome: str | None,
    ) -> DecisionHistoryRecord | None:
        updated_at = _utc_now()
        with self._session() as session, session.begin():
            row = session.get(DecisionHistoryModel, decision_history_id)
            if row is None:
                return None
            row.outcome = str(outcome).strip() if outcome is not None else None
            row.updated_at = updated_at
            session.add(row)
        return self.get_decision_history(decision_history_id)

    def create_dataset(
        self,
        *,
        dataset_name: str,
        dataset_key: str,
        source_fingerprint: str,
        source_kind: str,
        source_label: str,
        object_key: str,
        content_type: str,
        size_bytes: int,
        user_id: str | None = None,
    ) -> DatasetRecord:
        record = DatasetRecord(
            dataset_id=uuid.uuid4().hex,
            dataset_name=dataset_name,
            dataset_key=dataset_key,
            source_fingerprint=source_fingerprint,
            source_kind=source_kind,
            source_label=source_label,
            object_key=object_key,
            content_type=content_type,
            size_bytes=int(size_bytes),
            created_at=_utc_now(),
        )
        with self._session() as session, session.begin():
            session.add(
                DatasetModel(
                    dataset_id=record.dataset_id,
                    dataset_name=record.dataset_name,
                    dataset_key=record.dataset_key,
                    source_fingerprint=record.source_fingerprint,
                    source_kind=record.source_kind,
                    source_label=record.source_label,
                    object_key=record.object_key,
                    content_type=record.content_type,
                    size_bytes=record.size_bytes,
                    created_at=record.created_at,
                )
            )
            if user_id:
                session.add(
                    UserDatasetModel(
                        user_id=user_id,
                        dataset_id=record.dataset_id,
                        created_at=record.created_at,
                    )
                )
        return record

    def get_dataset(self, dataset_id: str) -> DatasetRecord | None:
        with self._session() as session:
            row = session.get(DatasetModel, dataset_id)
            if row is None:
                return None
            return DatasetRecord(
                dataset_id=row.dataset_id,
                dataset_name=row.dataset_name,
                dataset_key=row.dataset_key,
                source_fingerprint=row.source_fingerprint,
                source_kind=row.source_kind,
                source_label=row.source_label,
                object_key=row.object_key,
                content_type=row.content_type,
                size_bytes=row.size_bytes,
                created_at=row.created_at,
            )

    def create_job(
        self,
        *,
        dataset_id: str,
        query: str,
        intent: str,
        workflow_context: dict | None = None,
        cache_key: str | None = None,
        status: str = "queued",
        analysis_output: dict | None = None,
        result_summary: str | None = None,
        cache_hit: bool = False,
        user_id: str | None = None,
    ) -> AnalysisJobRecord:
        record = AnalysisJobRecord(
            job_id=uuid.uuid4().hex,
            dataset_id=dataset_id,
            query=str(query or ""),
            status=status,
            intent=str(intent or "general"),
            workflow_context=workflow_context or {},
            cache_key=cache_key,
            queued_at=_utc_now(),
            started_at=None,
            finished_at=None,
            queue_wait_ms=None,
            elapsed_ms=None,
            error_message=None,
            analysis_output=analysis_output,
            result_summary=result_summary,
            cache_hit=bool(cache_hit),
        )
        if status == "completed":
            record.started_at = record.queued_at
            record.finished_at = record.queued_at
            record.queue_wait_ms = 0
            record.elapsed_ms = 0

        with self._session() as session, session.begin():
            session.add(
                AnalysisJobModel(
                    job_id=record.job_id,
                    dataset_id=record.dataset_id,
                    query=record.query,
                    status=record.status,
                    intent=record.intent,
                    workflow_context_json=_serialize_json(record.workflow_context),
                    cache_key=record.cache_key,
                    queued_at=record.queued_at,
                    started_at=record.started_at,
                    finished_at=record.finished_at,
                    queue_wait_ms=record.queue_wait_ms,
                    elapsed_ms=record.elapsed_ms,
                    error_message=record.error_message,
                    analysis_output_json=_serialize_json(record.analysis_output) if record.analysis_output is not None else None,
                    result_summary=record.result_summary,
                    cache_hit=1 if record.cache_hit else 0,
                )
            )
            if user_id:
                session.add(
                    UserJobModel(
                        user_id=user_id,
                        job_id=record.job_id,
                        created_at=record.queued_at,
                    )
                )
        return record

    def mark_job_running(self, job_id: str) -> AnalysisJobRecord | None:
        started_at = _utc_now()
        with self._session() as session, session.begin():
            row = session.get(AnalysisJobModel, job_id)
            if row is None:
                return None
            row.status = "running"
            row.started_at = started_at
            row.queue_wait_ms = max(
                int((datetime.fromisoformat(started_at) - datetime.fromisoformat(row.queued_at)).total_seconds() * 1000),
                0,
            )
            session.add(row)
        return self.get_job(job_id)

    def complete_job(self, job_id: str, *, analysis_output: dict, result_summary: str | None = None, cache_hit: bool = False) -> AnalysisJobRecord | None:
        finished_at = _utc_now()
        with self._session() as session, session.begin():
            row = session.get(AnalysisJobModel, job_id)
            if row is None:
                return None
            row.status = "completed"
            row.finished_at = finished_at
            row.result_summary = result_summary
            row.analysis_output_json = _serialize_json(analysis_output)
            row.cache_hit = 1 if cache_hit else row.cache_hit
            if row.started_at:
                row.elapsed_ms = max(
                    int((datetime.fromisoformat(finished_at) - datetime.fromisoformat(row.started_at)).total_seconds() * 1000),
                    0,
                )
            session.add(row)
        return self.get_job(job_id)

    def fail_job(self, job_id: str, *, error_message: str) -> AnalysisJobRecord | None:
        finished_at = _utc_now()
        with self._session() as session, session.begin():
            row = session.get(AnalysisJobModel, job_id)
            if row is None:
                return None
            row.status = "failed"
            row.finished_at = finished_at
            row.error_message = str(error_message or "")
            if row.started_at:
                row.elapsed_ms = max(
                    int((datetime.fromisoformat(finished_at) - datetime.fromisoformat(row.started_at)).total_seconds() * 1000),
                    0,
                )
            session.add(row)
        return self.get_job(job_id)

    def get_job(self, job_id: str) -> AnalysisJobRecord | None:
        with self._session() as session:
            row = session.get(AnalysisJobModel, job_id)
            if row is None:
                return None
            return AnalysisJobRecord(
                job_id=row.job_id,
                dataset_id=row.dataset_id,
                query=row.query,
                status=row.status,
                intent=row.intent,
                workflow_context=json.loads(row.workflow_context_json or "{}"),
                cache_key=row.cache_key,
                queued_at=row.queued_at,
                started_at=row.started_at,
                finished_at=row.finished_at,
                queue_wait_ms=row.queue_wait_ms,
                elapsed_ms=row.elapsed_ms,
                error_message=row.error_message,
                analysis_output=json.loads(row.analysis_output_json) if row.analysis_output_json else None,
                result_summary=row.result_summary,
                cache_hit=bool(row.cache_hit),
            )

    def list_user_history(self, user_id: str, *, limit: int = 20) -> list[AnalysisHistoryRecord]:
        with self._session() as session:
            rows = session.execute(
                select(AnalysisJobModel, DatasetModel)
                .join(UserJobModel, UserJobModel.job_id == AnalysisJobModel.job_id)
                .join(DatasetModel, DatasetModel.dataset_id == AnalysisJobModel.dataset_id)
                .where(UserJobModel.user_id == user_id)
                .order_by(AnalysisJobModel.queued_at.desc())
                .limit(int(limit))
            ).all()

        history: list[AnalysisHistoryRecord] = []
        for job_row, dataset_row in rows:
            history.append(
                AnalysisHistoryRecord(
                    job_id=job_row.job_id,
                    dataset_id=job_row.dataset_id,
                    dataset_name=dataset_row.dataset_name,
                    query=job_row.query,
                    status=job_row.status,
                    intent=job_row.intent,
                    queued_at=job_row.queued_at,
                    finished_at=job_row.finished_at,
                    result_summary=job_row.result_summary,
                    error_message=job_row.error_message,
                    cache_hit=bool(job_row.cache_hit),
                )
            )
        return history

    def list_user_datasets(self, user_id: str, *, limit: int = 20) -> list[DatasetRecord]:
        with self._session() as session:
            rows = session.execute(
                select(DatasetModel)
                .join(UserDatasetModel, UserDatasetModel.dataset_id == DatasetModel.dataset_id)
                .where(UserDatasetModel.user_id == user_id)
                .order_by(DatasetModel.created_at.desc())
                .limit(int(limit))
            ).scalars().all()

        return [
            DatasetRecord(
                dataset_id=row.dataset_id,
                dataset_name=row.dataset_name,
                dataset_key=row.dataset_key,
                source_fingerprint=row.source_fingerprint,
                source_kind=row.source_kind,
                source_label=row.source_label,
                object_key=row.object_key,
                content_type=row.content_type,
                size_bytes=row.size_bytes,
                created_at=row.created_at,
            )
            for row in rows
        ]

    @staticmethod
    def build_run_record(
        *,
        workflow_id: str | None,
        workflow_version: int | None,
        workflow_name: str | None,
        source_fingerprint: str,
        source_label: str,
        validation_findings: list[dict],
        cleaning_actions: list[str],
        generated_code: str | None,
        final_status: str,
        error_message: str | None,
        export_artifacts: list[str],
        analysis_query: str,
        result_summary: str | None,
        result_hash: str | None = None,
    ) -> WorkflowRunRecord:
        return WorkflowRunRecord(
            run_id=uuid.uuid4().hex,
            workflow_id=workflow_id,
            workflow_version=workflow_version,
            workflow_name=workflow_name,
            source_fingerprint=source_fingerprint,
            source_label=source_label,
            validation_findings=validation_findings,
            cleaning_actions=cleaning_actions,
            generated_code=generated_code,
            final_status=final_status,
            error_message=error_message,
            export_artifacts=export_artifacts,
            analysis_query=str(analysis_query or ""),
            result_summary=result_summary,
            result_hash=str(result_hash or "") or None,
            created_at=_utc_now(),
        )

    @staticmethod
    def build_forecast_artifact_record(
        *,
        workflow_id: str | None,
        workflow_version: int | None,
        workflow_name: str | None,
        source_fingerprint: str,
        source_label: str,
        target_column: str,
        horizon: str,
        model_name: str,
        training_mode: str,
        status: str,
        artifact_key: str,
        forecast_config: dict,
        evaluation_metrics: dict,
        recommendation_payload: list[dict],
        summary: str | None,
        result_hash: str | None = None,
    ) -> ForecastArtifactRecord:
        return ForecastArtifactRecord(
            artifact_id=uuid.uuid4().hex,
            workflow_id=workflow_id,
            workflow_version=workflow_version,
            workflow_name=workflow_name,
            source_fingerprint=source_fingerprint,
            source_label=source_label,
            target_column=str(target_column or ""),
            horizon=str(horizon or ""),
            model_name=str(model_name or ""),
            training_mode=str(training_mode or "auto"),
            status=str(status or "UNKNOWN"),
            artifact_key=str(artifact_key or ""),
            forecast_config=dict(forecast_config or {}),
            evaluation_metrics=dict(evaluation_metrics or {}),
            recommendation_payload=list(recommendation_payload or []),
            summary=summary,
            result_hash=str(result_hash or "") or None,
            created_at=_utc_now(),
        )

    @staticmethod
    def build_decision_history_record(
        *,
        job_id: str | None,
        forecast_artifact_id: str | None,
        source_fingerprint: str,
        query: str,
        decision: dict[str, Any],
        decision_confidence: str,
        result_hash: str | None = None,
        outcome: str | None = None,
    ) -> DecisionHistoryRecord:
        timestamp = _utc_now()
        return DecisionHistoryRecord(
            decision_history_id=uuid.uuid4().hex,
            job_id=job_id,
            forecast_artifact_id=forecast_artifact_id,
            source_fingerprint=str(source_fingerprint or ""),
            query=str(query or ""),
            decision_id=str(decision.get("decision_id") or ""),
            decision_json=dict(decision or {}),
            priority=str(decision.get("priority") or "LOW"),
            decision_confidence=str(decision_confidence or decision.get("confidence") or "low"),
            risk_level=str(decision.get("risk_level") or "medium"),
            result_hash=str(result_hash or "") or None,
            outcome=str(outcome).strip() if outcome is not None else None,
            created_at=timestamp,
            updated_at=timestamp,
        )

    @staticmethod
    def _workflow_version_to_definition(
        workflow: WorkflowModel,
        version_row: WorkflowVersionModel,
    ) -> WorkflowDefinition:
        return WorkflowDefinition(
            workflow_id=workflow.workflow_id,
            name=workflow.name,
            version=int(version_row.version),
            source_config=json.loads(version_row.source_config_json),
            cleaning_options=json.loads(version_row.cleaning_options_json),
            analysis_query=version_row.analysis_query,
            forecast_config=json.loads(version_row.forecast_config_json or "{}"),
            chart_preferences=json.loads(version_row.chart_preferences_json),
            export_settings=json.loads(version_row.export_settings_json),
            created_at=version_row.created_at,
        )

    @staticmethod
    def _run_model_to_record(row: WorkflowRunModel) -> WorkflowRunRecord:
        return WorkflowRunRecord(
            run_id=row.run_id,
            workflow_id=row.workflow_id,
            workflow_version=row.workflow_version,
            workflow_name=row.workflow_name,
            source_fingerprint=row.source_fingerprint,
            source_label=row.source_label,
            validation_findings=json.loads(row.validation_findings_json),
            cleaning_actions=json.loads(row.cleaning_actions_json),
            generated_code=row.generated_code,
            final_status=row.final_status,
            error_message=row.error_message,
            export_artifacts=json.loads(row.export_artifacts_json),
            analysis_query=row.analysis_query,
            result_summary=row.result_summary,
            result_hash=row.result_hash,
            created_at=row.created_at,
        )

    @staticmethod
    def _forecast_artifact_model_to_record(row: ForecastArtifactModel) -> ForecastArtifactRecord:
        return ForecastArtifactRecord(
            artifact_id=row.artifact_id,
            workflow_id=row.workflow_id,
            workflow_version=row.workflow_version,
            workflow_name=row.workflow_name,
            source_fingerprint=row.source_fingerprint,
            source_label=row.source_label,
            target_column=row.target_column,
            horizon=row.horizon,
            model_name=row.model_name,
            training_mode=row.training_mode,
            status=row.status,
            artifact_key=row.artifact_key,
            forecast_config=json.loads(row.forecast_config_json or "{}"),
            evaluation_metrics=json.loads(row.evaluation_metrics_json or "{}"),
            recommendation_payload=json.loads(row.recommendation_payload_json or "[]"),
            summary=row.summary,
            result_hash=row.result_hash,
            created_at=row.created_at,
        )

    @staticmethod
    def _decision_history_model_to_record(row: DecisionHistoryModel) -> DecisionHistoryRecord:
        return DecisionHistoryRecord(
            decision_history_id=row.decision_history_id,
            job_id=row.job_id,
            forecast_artifact_id=row.forecast_artifact_id,
            source_fingerprint=row.source_fingerprint,
            query=row.query,
            decision_id=row.decision_id,
            decision_json=json.loads(row.decision_json or "{}"),
            priority=row.priority,
            decision_confidence=row.decision_confidence,
            risk_level=row.risk_level,
            result_hash=row.result_hash,
            outcome=row.outcome,
            created_at=row.created_at,
            updated_at=row.updated_at,
        )
