from __future__ import annotations

from pathlib import Path

from .config import APP_STATE_DIR, PROJECT_ROOT, get_settings


def _resolve_root() -> Path:
    configured = get_settings().dataset_session_dir
    root = configured if configured.is_absolute() else (PROJECT_ROOT / configured)
    root.mkdir(parents=True, exist_ok=True)
    return root


def sanitize_relative_path(value: str, fallback: str = "dataset.csv") -> str:
    raw = str(value or fallback).replace("\\", "/").strip()
    parts = [part for part in raw.split("/") if part and part not in {".", ".."}]
    return "/".join(parts) if parts else fallback


def get_session_root(session_id: str) -> Path:
    root = _resolve_root() / str(session_id or "session")
    root.mkdir(parents=True, exist_ok=True)
    return root


def get_session_sources_dir(session_id: str) -> Path:
    path = get_session_root(session_id) / "sources"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_session_extract_dir(session_id: str) -> Path:
    path = get_session_root(session_id) / "extracted"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_duckdb_path(session_id: str) -> Path:
    return get_session_root(session_id) / "warehouse.duckdb"


def get_catalog_path(session_id: str) -> Path:
    return get_session_root(session_id) / "catalog.json"


def get_schema_path(session_id: str) -> Path:
    return get_session_root(session_id) / "schema.json"


def get_insights_path(session_id: str) -> Path:
    return get_session_root(session_id) / "insights.json"


def write_session_file(session_id: str, relative_path: str, payload: bytes, *, extracted: bool = False) -> Path:
    base_dir = get_session_extract_dir(session_id) if extracted else get_session_sources_dir(session_id)
    safe_relative = sanitize_relative_path(relative_path)
    target_path = base_dir / safe_relative
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_bytes(payload)
    return target_path

