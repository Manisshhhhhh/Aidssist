from __future__ import annotations

from typing import Any

from .ai_data_store import AIDataStore, ImportJobRecord
from .celery_app import celery_app
from .data_intelligence import prepare_asset_intelligence
from .dataset_session import write_session_file
from .external_imports import ImportedBinaryFile, download_kaggle_dataset, fetch_google_drive_selection
from .ingestion import build_asset_detail, ingest_workspace_files
from backend.workflow_store import WorkflowStore


def _normalize_import_files(imported_files: list[ImportedBinaryFile]) -> list[dict[str, Any]]:
    return [
        {
            "file_name": item.relative_path or item.file_name,
            "content_type": item.content_type,
            "content": item.content,
        }
        for item in imported_files
    ]


def _persist_session_sources(session_id: str, imported_files: list[ImportedBinaryFile]) -> None:
    for item in imported_files:
        write_session_file(session_id, item.relative_path or item.file_name, item.content)


def _finalize_import_job(job: ImportJobRecord, imported_files: list[ImportedBinaryFile]) -> dict[str, Any]:
    _persist_session_sources(job.session_id, imported_files)

    with WorkflowStore() as store:
        result = ingest_workspace_files(
            store=store,
            workspace_id=job.workspace_id,
            uploaded_files=_normalize_import_files(imported_files),
            user_id=job.user_id,
            title=job.source_label,
        )
        asset_id = result["asset"].asset_id
        intelligence = prepare_asset_intelligence(asset_id, force_refresh=True)
        asset_detail = build_asset_detail(store, asset_id)

    payload = {
        "asset_id": asset_id,
        "dataset_count": len(result.get("datasets") or []),
        "file_count": len(imported_files),
        "intelligence": intelligence,
        "asset": asset_detail,
    }
    with AIDataStore() as intelligence_store:
        intelligence_store.update_import_job(
            job.job_id,
            status="completed",
            asset_id=asset_id,
            result=payload,
            completed=True,
        )
    return payload


def import_google_drive_into_workspace(
    *,
    workspace_id: str,
    user_id: str | None,
    file_id: str,
    access_token: str,
) -> ImportJobRecord:
    with AIDataStore() as store:
        job = store.create_import_job(
            workspace_id=workspace_id,
            user_id=user_id,
            source_type="google_drive",
            source_ref=file_id,
            source_label="Google Drive import",
            status="processing",
        )
    imported_files = fetch_google_drive_selection(file_id, access_token)
    source_label = imported_files[0].relative_path.split("/")[0] if imported_files else "Google Drive import"
    with AIDataStore() as store:
        store.update_import_job(job.job_id, status="processing", source_label=source_label)
        refreshed = store.get_import_job(job.job_id)
    if refreshed is None:
        raise ValueError("Google Drive import job could not be created.")
    _finalize_import_job(refreshed, imported_files)
    completed = None
    with AIDataStore() as store:
        completed = store.get_import_job(refreshed.job_id)
    if completed is None:
        raise ValueError("Google Drive import job did not complete correctly.")
    return completed


def enqueue_kaggle_import(*, workspace_id: str, user_id: str | None, dataset_url: str) -> ImportJobRecord:
    with AIDataStore() as store:
        job = store.create_import_job(
            workspace_id=workspace_id,
            user_id=user_id,
            source_type="kaggle",
            source_ref=dataset_url,
            source_label="Kaggle dataset import",
            status="queued",
        )

    if celery_app is None:
        process_kaggle_import_job(job.job_id)
        with AIDataStore() as store:
            completed = store.get_import_job(job.job_id)
        return completed or job

    from .celery_tasks import run_kaggle_import_job

    try:
        task = run_kaggle_import_job.delay(job.job_id)
    except Exception:
        process_kaggle_import_job(job.job_id)
        with AIDataStore() as store:
            completed = store.get_import_job(job.job_id)
        return completed or job

    with AIDataStore() as store:
        updated = store.update_import_job(job.job_id, celery_task_id=str(task.id or ""), status="queued")
    return updated or job


def process_kaggle_import_job(job_id: str) -> dict[str, Any]:
    with AIDataStore() as store:
        job = store.get_import_job(job_id)
        if job is None:
            raise ValueError(f"Import job '{job_id}' was not found.")
        store.update_import_job(job_id, status="processing")
        refreshed = store.get_import_job(job_id)
    if refreshed is None:
        raise ValueError(f"Import job '{job_id}' disappeared during execution.")

    try:
        imported_files = download_kaggle_dataset(refreshed.source_ref, refreshed.session_id)
        return _finalize_import_job(refreshed, imported_files)
    except Exception as error:
        with AIDataStore() as store:
            store.update_import_job(
                job_id,
                status="failed",
                error_message=str(error),
                completed=True,
            )
        raise


def get_import_job_payload(job_id: str) -> dict[str, Any]:
    with AIDataStore() as store:
        job = store.get_import_job(job_id)
    if job is None:
        raise ValueError(f"Import job '{job_id}' was not found.")
    return {
        "job_id": job.job_id,
        "workspace_id": job.workspace_id,
        "session_id": job.session_id,
        "source_type": job.source_type,
        "source_ref": job.source_ref,
        "source_label": job.source_label,
        "status": job.status,
        "asset_id": job.asset_id,
        "error_message": job.error_message,
        "result": job.result,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
        "completed_at": job.completed_at,
    }
