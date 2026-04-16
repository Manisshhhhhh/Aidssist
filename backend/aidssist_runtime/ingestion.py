from __future__ import annotations

import hashlib
import io
import mimetypes
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

from .chunking import (
    ARCHIVE_FILE_SUFFIXES,
    TABLE_FILE_SUFFIXES,
    build_table_chunks,
    build_text_chunks,
    choose_asset_kind,
    classify_file_kind,
    decode_text_bytes,
    infer_language,
)
from .embedding import embed_texts
from .storage import build_object_key, get_object_store
from backend.aidssist_runtime.analysis_service import create_dataset_from_upload, get_dataset_summary
from backend.data_sources import CSVSourceConfig, ExcelSourceConfig, load_dataframe_from_source
from backend.workflow_store import (
    AssetFileRecord,
    AssetRecord,
    DatasetRecord,
    DerivedDatasetRecord,
    WorkflowStore,
)


SUPPORTED_UPLOAD_SUFFIXES = {
    ".csv",
    ".xlsx",
    ".xlsm",
    ".txt",
    ".md",
    ".json",
    ".py",
    ".js",
    ".ts",
    ".sql",
    ".yaml",
    ".yml",
    ".zip",
}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _guess_media_type(file_name: str, fallback: str = "application/octet-stream") -> str:
    guessed, _ = mimetypes.guess_type(file_name)
    return guessed or fallback


def _is_supported_member(file_name: str) -> bool:
    return Path(file_name or "").suffix.lower() in SUPPORTED_UPLOAD_SUFFIXES - ARCHIVE_FILE_SUFFIXES


def load_dataset_dataframe(dataset_record: DatasetRecord) -> pd.DataFrame:
    file_bytes = get_object_store().get_bytes(dataset_record.object_key)
    if dataset_record.source_kind == "csv":
        loaded = load_dataframe_from_source(
            CSVSourceConfig(file_name=dataset_record.dataset_name, file_bytes=file_bytes)
        )
    else:
        loaded = load_dataframe_from_source(
            ExcelSourceConfig(file_name=dataset_record.dataset_name, file_bytes=file_bytes)
        )
    return loaded.dataframe


def load_derived_dataset_dataframe(derived_dataset: DerivedDatasetRecord) -> pd.DataFrame:
    file_bytes = get_object_store().get_bytes(derived_dataset.object_key)
    suffix = Path(derived_dataset.dataset_name).suffix.lower()
    if suffix == ".csv":
        loaded = load_dataframe_from_source(
            CSVSourceConfig(file_name=derived_dataset.dataset_name, file_bytes=file_bytes)
        )
    else:
        loaded = load_dataframe_from_source(
            ExcelSourceConfig(file_name=derived_dataset.dataset_name, file_bytes=file_bytes)
        )
    return loaded.dataframe


def _persist_chunks(
    *,
    store: WorkflowStore,
    asset_id: str,
    asset_file_id: str | None,
    dataset_id: str | None,
    chunk_payloads: list[dict[str, Any]],
) -> int:
    if not chunk_payloads:
        return 0
    vectors = embed_texts(payload["content_text"] for payload in chunk_payloads)
    count = 0
    for payload, vector in zip(chunk_payloads, vectors):
        chunk = store.record_chunk(
            asset_id=asset_id,
            asset_file_id=asset_file_id,
            dataset_id=dataset_id,
            chunk_index=payload["chunk_index"],
            title=payload["title"],
            content_text=payload["content_text"],
            token_count=payload["token_count"],
            metadata=payload["metadata"],
        )
        store.upsert_embedding(
            chunk_id=chunk.chunk_id,
            model_name="workspace-embedding",
            vector=vector,
        )
        count += 1
    return count


def _ingest_table_file(
    *,
    store: WorkflowStore,
    workspace_id: str,
    asset: AssetRecord,
    file_name: str,
    content: bytes,
    media_type: str,
    user_id: str | None,
) -> tuple[AssetFileRecord, DatasetRecord, DerivedDatasetRecord, int]:
    dataset = create_dataset_from_upload(
        file_name=file_name,
        file_bytes=content,
        content_type=media_type,
        user_id=user_id,
    )
    persisted_dataset = store.get_dataset(dataset.dataset_id)
    if persisted_dataset is None:
        raise ValueError(f"Dataset '{dataset.dataset_id}' was not persisted.")

    asset_file = store.create_asset_file(
        asset_id=asset.asset_id,
        dataset_id=persisted_dataset.dataset_id,
        file_name=file_name,
        media_type=media_type,
        file_kind="table",
        language=infer_language(file_name),
        object_key=persisted_dataset.object_key,
        checksum=_sha256_bytes(content),
        size_bytes=len(content),
    )
    summary = get_dataset_summary(dataset.dataset_id)
    derived = store.create_derived_dataset(
        workspace_id=workspace_id,
        asset_id=asset.asset_id,
        parent_dataset_id=persisted_dataset.dataset_id,
        dataset_name=persisted_dataset.dataset_name,
        dataset_key=persisted_dataset.dataset_key,
        source_fingerprint=persisted_dataset.source_fingerprint,
        object_key=persisted_dataset.object_key,
        content_type=persisted_dataset.content_type,
        transform_prompt=None,
        row_count=int(summary["row_count"]),
        column_count=int(summary["column_count"]),
        preview_columns=list(summary["preview_columns"]),
        preview_rows=list(summary["preview_rows"]),
    )
    chunk_count = _persist_chunks(
        store=store,
        asset_id=asset.asset_id,
        asset_file_id=asset_file.asset_file_id,
        dataset_id=persisted_dataset.dataset_id,
        chunk_payloads=build_table_chunks(
            dataset_name=persisted_dataset.dataset_name,
            dataset_key=persisted_dataset.dataset_key,
            source_fingerprint=persisted_dataset.source_fingerprint,
            df=load_dataset_dataframe(persisted_dataset),
        ),
    )
    return asset_file, persisted_dataset, derived, chunk_count


def _ingest_textual_file(
    *,
    store: WorkflowStore,
    asset: AssetRecord,
    file_name: str,
    content: bytes,
    media_type: str,
) -> tuple[AssetFileRecord, int]:
    object_key = build_object_key("assets", asset.asset_id, file_name)
    get_object_store().put_bytes(object_key, content, content_type=media_type)
    file_kind = classify_file_kind(file_name, media_type)
    asset_file = store.create_asset_file(
        asset_id=asset.asset_id,
        dataset_id=None,
        file_name=file_name,
        media_type=media_type,
        file_kind=file_kind,
        language=infer_language(file_name),
        object_key=object_key,
        checksum=_sha256_bytes(content),
        size_bytes=len(content),
    )
    chunk_count = 0
    if file_kind != "binary":
        chunk_count = _persist_chunks(
            store=store,
            asset_id=asset.asset_id,
            asset_file_id=asset_file.asset_file_id,
            dataset_id=None,
            chunk_payloads=build_text_chunks(
                file_name=file_name,
                text=decode_text_bytes(content),
                file_kind=file_kind,
                language=infer_language(file_name),
            ),
        )
    return asset_file, chunk_count


def _expand_zip_entries(file_name: str, content: bytes) -> list[tuple[str, bytes, str]]:
    entries: list[tuple[str, bytes, str]] = []
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            member_name = member.filename
            if not _is_supported_member(member_name):
                continue
            member_bytes = archive.read(member)
            entries.append(
                (
                    member_name,
                    member_bytes,
                    _guess_media_type(member_name, "text/plain"),
                )
            )
    return entries


def ingest_workspace_files(
    *,
    store: WorkflowStore,
    workspace_id: str,
    uploaded_files: list[dict[str, Any]],
    user_id: str | None = None,
    title: str | None = None,
) -> dict[str, Any]:
    if not uploaded_files:
        raise ValueError("At least one file is required.")

    asset = store.create_asset(
        workspace_id=workspace_id,
        title=title or str(uploaded_files[0].get("file_name") or "Workspace asset"),
        asset_kind="mixed",
    )
    asset_files: list[AssetFileRecord] = []
    datasets: list[DatasetRecord] = []
    derived_datasets: list[DerivedDatasetRecord] = []
    observed_kinds: list[str] = []
    chunk_count = 0

    def process_file(candidate_name: str, candidate_bytes: bytes, candidate_media_type: str) -> None:
        nonlocal chunk_count
        suffix = Path(candidate_name or "").suffix.lower()
        if suffix in TABLE_FILE_SUFFIXES:
            asset_file, dataset, derived, added_chunks = _ingest_table_file(
                store=store,
                workspace_id=workspace_id,
                asset=asset,
                file_name=candidate_name,
                content=candidate_bytes,
                media_type=candidate_media_type,
                user_id=user_id,
            )
            asset_files.append(asset_file)
            datasets.append(dataset)
            derived_datasets.append(derived)
            observed_kinds.append("table")
            chunk_count += added_chunks
            return

        asset_file, added_chunks = _ingest_textual_file(
            store=store,
            asset=asset,
            file_name=candidate_name,
            content=candidate_bytes,
            media_type=candidate_media_type,
        )
        asset_files.append(asset_file)
        observed_kinds.append(asset_file.file_kind)
        chunk_count += added_chunks

    for uploaded_file in uploaded_files:
        file_name = str(uploaded_file.get("file_name") or "")
        file_bytes = bytes(uploaded_file.get("content") or b"")
        media_type = str(uploaded_file.get("content_type") or _guess_media_type(file_name))
        if Path(file_name).suffix.lower() in ARCHIVE_FILE_SUFFIXES:
            archive_file, archive_chunks = _ingest_textual_file(
                store=store,
                asset=asset,
                file_name=file_name,
                content=file_bytes,
                media_type=media_type,
            )
            asset_files.append(archive_file)
            observed_kinds.append("archive")
            chunk_count += archive_chunks
            for member_name, member_bytes, member_media_type in _expand_zip_entries(file_name, file_bytes):
                process_file(member_name, member_bytes, member_media_type)
            continue
        process_file(file_name, file_bytes, media_type)

    finalized_asset = store.update_asset_metadata(
        asset.asset_id,
        asset_kind=choose_asset_kind(observed_kinds),
        primary_dataset_id=datasets[0].dataset_id if datasets else None,
    ) or asset
    return {
        "asset": finalized_asset,
        "asset_files": asset_files,
        "datasets": datasets,
        "derived_datasets": derived_datasets,
        "chunk_count": chunk_count,
    }


def build_asset_detail(store: WorkflowStore, asset_id: str) -> dict[str, Any]:
    asset = store.get_asset(asset_id)
    if asset is None:
        raise ValueError(f"Asset '{asset_id}' was not found.")

    files = store.list_asset_files(asset_id)
    datasets = [get_dataset_summary(dataset.dataset_id) for dataset in store.list_asset_datasets(asset_id)]
    derived_datasets = [
        record
        for record in store.list_workspace_derived_datasets(asset.workspace_id, limit=200)
        if record.asset_id == asset_id
    ]
    chunks = store.list_asset_chunks(asset_id, limit=200)
    return {
        "asset": asset,
        "files": files,
        "datasets": datasets,
        "derived_datasets": derived_datasets,
        "chunk_count": len(chunks),
        "chunk_preview": [
            {
                "chunk_id": chunk.chunk_id,
                "title": chunk.title,
                "metadata": chunk.metadata,
            }
            for chunk in chunks[:12]
        ],
    }
