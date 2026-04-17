from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import String, Text, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from backend.compat import dataclass

from .config import APP_STATE_DIR, get_settings


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _serialize_json(value: Any) -> str:
    return json.dumps(value, default=str, sort_keys=True)


def _database_url() -> str:
    configured = str(get_settings().database_url or "").strip()
    if configured:
        return configured
    APP_STATE_DIR.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{APP_STATE_DIR / 'workflow_store.sqlite3'}"


class Base(DeclarativeBase):
    pass


class ImportJobModel(Base):
    __tablename__ = "ai_import_jobs"

    job_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    workspace_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    user_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    session_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    source_type: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    source_ref: Mapped[str] = mapped_column(Text, nullable=False)
    source_label: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    asset_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    celery_task_id: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    result_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    created_at: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    updated_at: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    completed_at: Mapped[str | None] = mapped_column(String(40), nullable=True, index=True)


class AssetIntelligenceModel(Base):
    __tablename__ = "asset_intelligence_snapshots"

    intelligence_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    workspace_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    asset_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)
    session_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    dataset_type: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    catalog_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    schema_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    insights_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    chat_context_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    created_at: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    updated_at: Mapped[str] = mapped_column(String(40), nullable=False, index=True)


@dataclass(slots=True)
class ImportJobRecord:
    job_id: str
    workspace_id: str
    user_id: str | None
    session_id: str
    source_type: str
    source_ref: str
    source_label: str
    status: str
    asset_id: str | None
    celery_task_id: str | None
    error_message: str | None
    result: dict[str, Any]
    created_at: str
    updated_at: str
    completed_at: str | None


@dataclass(slots=True)
class AssetIntelligenceRecord:
    intelligence_id: str
    workspace_id: str
    asset_id: str
    session_id: str
    status: str
    dataset_type: str | None
    catalog: dict[str, Any]
    schema: dict[str, Any]
    insights: dict[str, Any]
    chat_context: dict[str, Any]
    created_at: str
    updated_at: str


class AIDataStore:
    def __init__(self, db_path: str | Path | None = None):
        database_url = str(db_path).strip() if db_path is not None else _database_url()
        if db_path is not None and "://" not in database_url:
            database_url = f"sqlite:///{Path(database_url)}"

        engine_kwargs: dict[str, Any] = {"future": True, "pool_pre_ping": True}
        if database_url.startswith("sqlite"):
            engine_kwargs["connect_args"] = {"check_same_thread": False}

        self.engine = create_engine(database_url, **engine_kwargs)
        self.SessionLocal = sessionmaker(bind=self.engine, expire_on_commit=False, future=True)
        Base.metadata.create_all(self.engine)

    def close(self) -> None:
        self.engine.dispose()

    def __enter__(self) -> "AIDataStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        del exc_type, exc, tb
        self.close()

    def _session(self) -> Session:
        return self.SessionLocal()

    @staticmethod
    def _import_job_record(row: ImportJobModel) -> ImportJobRecord:
        return ImportJobRecord(
            job_id=row.job_id,
            workspace_id=row.workspace_id,
            user_id=row.user_id,
            session_id=row.session_id,
            source_type=row.source_type,
            source_ref=row.source_ref,
            source_label=row.source_label,
            status=row.status,
            asset_id=row.asset_id,
            celery_task_id=row.celery_task_id,
            error_message=row.error_message,
            result=dict(json.loads(row.result_json or "{}")),
            created_at=row.created_at,
            updated_at=row.updated_at,
            completed_at=row.completed_at,
        )

    @staticmethod
    def _intelligence_record(row: AssetIntelligenceModel) -> AssetIntelligenceRecord:
        return AssetIntelligenceRecord(
            intelligence_id=row.intelligence_id,
            workspace_id=row.workspace_id,
            asset_id=row.asset_id,
            session_id=row.session_id,
            status=row.status,
            dataset_type=row.dataset_type,
            catalog=dict(json.loads(row.catalog_json or "{}")),
            schema=dict(json.loads(row.schema_json or "{}")),
            insights=dict(json.loads(row.insights_json or "{}")),
            chat_context=dict(json.loads(row.chat_context_json or "{}")),
            created_at=row.created_at,
            updated_at=row.updated_at,
        )

    def create_import_job(
        self,
        *,
        workspace_id: str,
        user_id: str | None,
        source_type: str,
        source_ref: str,
        source_label: str,
        session_id: str | None = None,
        status: str = "queued",
    ) -> ImportJobRecord:
        now = _utc_now()
        record = ImportJobRecord(
            job_id=uuid.uuid4().hex,
            workspace_id=workspace_id,
            user_id=user_id,
            session_id=session_id or uuid.uuid4().hex,
            source_type=str(source_type or "unknown"),
            source_ref=str(source_ref or ""),
            source_label=str(source_label or source_ref or "External dataset"),
            status=str(status or "queued"),
            asset_id=None,
            celery_task_id=None,
            error_message=None,
            result={},
            created_at=now,
            updated_at=now,
            completed_at=None,
        )
        with self._session() as session, session.begin():
            session.add(
                ImportJobModel(
                    job_id=record.job_id,
                    workspace_id=record.workspace_id,
                    user_id=record.user_id,
                    session_id=record.session_id,
                    source_type=record.source_type,
                    source_ref=record.source_ref,
                    source_label=record.source_label,
                    status=record.status,
                    asset_id=record.asset_id,
                    celery_task_id=record.celery_task_id,
                    error_message=record.error_message,
                    result_json=_serialize_json(record.result),
                    created_at=record.created_at,
                    updated_at=record.updated_at,
                    completed_at=record.completed_at,
                )
            )
        return record

    def get_import_job(self, job_id: str) -> ImportJobRecord | None:
        with self._session() as session:
            row = session.get(ImportJobModel, job_id)
            return self._import_job_record(row) if row is not None else None

    def update_import_job(
        self,
        job_id: str,
        *,
        status: str | None = None,
        source_label: str | None = None,
        asset_id: str | None = None,
        celery_task_id: str | None = None,
        error_message: str | None = None,
        result: dict[str, Any] | None = None,
        completed: bool = False,
    ) -> ImportJobRecord | None:
        now = _utc_now()
        with self._session() as session, session.begin():
            row = session.get(ImportJobModel, job_id)
            if row is None:
                return None
            if status is not None:
                row.status = str(status or row.status)
            if source_label is not None:
                row.source_label = str(source_label or row.source_label)
            if asset_id is not None:
                row.asset_id = asset_id
            if celery_task_id is not None:
                row.celery_task_id = celery_task_id
            if error_message is not None:
                row.error_message = error_message
            if result is not None:
                row.result_json = _serialize_json(result)
            row.updated_at = now
            if completed:
                row.completed_at = now
            session.add(row)
        return self.get_import_job(job_id)

    def upsert_asset_intelligence(
        self,
        *,
        workspace_id: str,
        asset_id: str,
        session_id: str,
        status: str,
        dataset_type: str | None,
        catalog: dict[str, Any],
        schema: dict[str, Any],
        insights: dict[str, Any],
        chat_context: dict[str, Any],
    ) -> AssetIntelligenceRecord:
        now = _utc_now()
        with self._session() as session, session.begin():
            row = session.scalar(
                select(AssetIntelligenceModel).where(AssetIntelligenceModel.asset_id == asset_id)
            )
            if row is None:
                row = AssetIntelligenceModel(
                    intelligence_id=uuid.uuid4().hex,
                    workspace_id=workspace_id,
                    asset_id=asset_id,
                    session_id=session_id,
                    status=str(status or "ready"),
                    dataset_type=dataset_type,
                    catalog_json=_serialize_json(catalog),
                    schema_json=_serialize_json(schema),
                    insights_json=_serialize_json(insights),
                    chat_context_json=_serialize_json(chat_context),
                    created_at=now,
                    updated_at=now,
                )
            else:
                row.workspace_id = workspace_id
                row.session_id = session_id
                row.status = str(status or row.status)
                row.dataset_type = dataset_type
                row.catalog_json = _serialize_json(catalog)
                row.schema_json = _serialize_json(schema)
                row.insights_json = _serialize_json(insights)
                row.chat_context_json = _serialize_json(chat_context)
                row.updated_at = now
            session.add(row)

        persisted = self.get_asset_intelligence(asset_id)
        if persisted is None:
            raise ValueError(f"Asset intelligence for '{asset_id}' could not be persisted.")
        return persisted

    def get_asset_intelligence(self, asset_id: str) -> AssetIntelligenceRecord | None:
        with self._session() as session:
            row = session.scalar(
                select(AssetIntelligenceModel).where(AssetIntelligenceModel.asset_id == asset_id)
            )
            return self._intelligence_record(row) if row is not None else None
