from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Any

import requests

from backend.compat import dataclass

from .config import get_settings
from .dataset_session import get_session_extract_dir, sanitize_relative_path, write_session_file


GOOGLE_FOLDER_MIME = "application/vnd.google-apps.folder"
GOOGLE_SHORTCUT_MIME = "application/vnd.google-apps.shortcut"
SUPPORTED_IMPORT_SUFFIXES = {".csv", ".xlsx", ".xlsm"}


@dataclass(slots=True)
class ImportedBinaryFile:
    file_name: str
    relative_path: str
    content_type: str
    content: bytes


def _google_headers(access_token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {access_token.strip()}"}


def _drive_get(path: str, *, access_token: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    response = requests.get(
        f"{get_settings().google_drive_api_base_url}/{path.lstrip('/')}",
        headers=_google_headers(access_token),
        params=params,
        timeout=get_settings().request_timeout_seconds,
    )
    response.raise_for_status()
    return dict(response.json() or {})


def _drive_download(file_id: str, *, access_token: str) -> tuple[bytes, str]:
    response = requests.get(
        f"{get_settings().google_drive_download_url}/{file_id}",
        headers=_google_headers(access_token),
        params={"alt": "media"},
        timeout=get_settings().request_timeout_seconds,
    )
    response.raise_for_status()
    return response.content, response.headers.get("content-type", "application/octet-stream")


def _supported_drive_file(name: str, mime_type: str | None) -> bool:
    suffix = Path(name or "").suffix.lower()
    if suffix in SUPPORTED_IMPORT_SUFFIXES:
        return True
    return str(mime_type or "").lower() in {
        "text/csv",
        "application/csv",
        "application/vnd.ms-excel",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    }


def _resolve_shortcut(file_id: str, access_token: str) -> dict[str, Any]:
    metadata = _drive_get(
        file_id,
        access_token=access_token,
        params={"fields": "id,name,mimeType,shortcutDetails,targetId"},
    )
    if metadata.get("mimeType") != GOOGLE_SHORTCUT_MIME:
        return metadata
    target_id = str(((metadata.get("shortcutDetails") or {}).get("targetId")) or metadata.get("targetId") or "")
    if not target_id:
        raise ValueError("Google Drive shortcut could not be resolved.")
    return _drive_get(
        target_id,
        access_token=access_token,
        params={"fields": "id,name,mimeType,parents,size"},
    )


def _list_drive_folder_files(folder_id: str, *, access_token: str, parent_path: str = "") -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    page_token: str | None = None

    while True:
        payload = _drive_get(
            "files",
            access_token=access_token,
            params={
                "q": f"'{folder_id}' in parents and trashed = false",
                "fields": "nextPageToken,files(id,name,mimeType,size,shortcutDetails)",
                "pageSize": 200,
                "pageToken": page_token,
                "supportsAllDrives": "true",
                "includeItemsFromAllDrives": "true",
            },
        )
        for entry in payload.get("files", []):
            metadata = dict(entry or {})
            resolved = _resolve_shortcut(str(metadata.get("id") or ""), access_token)
            relative_name = sanitize_relative_path(
                f"{parent_path}/{resolved.get('name')}" if parent_path else str(resolved.get("name") or "dataset.csv")
            )
            if resolved.get("mimeType") == GOOGLE_FOLDER_MIME:
                files.extend(
                    _list_drive_folder_files(
                        str(resolved.get("id") or ""),
                        access_token=access_token,
                        parent_path=relative_name,
                    )
                )
                continue
            if _supported_drive_file(str(resolved.get("name") or ""), str(resolved.get("mimeType") or "")):
                resolved["relative_path"] = relative_name
                files.append(resolved)
        page_token = payload.get("nextPageToken")
        if not page_token:
            break

    return files


def fetch_google_drive_selection(file_id: str, access_token: str) -> list[ImportedBinaryFile]:
    if not str(access_token or "").strip():
        raise ValueError("Google Drive access token is required.")
    if not str(file_id or "").strip():
        raise ValueError("Google Drive file id is required.")

    metadata = _resolve_shortcut(str(file_id), access_token)
    if metadata.get("mimeType") == GOOGLE_FOLDER_MIME:
        selected_items = _list_drive_folder_files(str(metadata.get("id") or ""), access_token=access_token)
    else:
        if not _supported_drive_file(str(metadata.get("name") or ""), str(metadata.get("mimeType") or "")):
            raise ValueError("Aidssist currently supports CSV and Excel files from Google Drive.")
        metadata["relative_path"] = sanitize_relative_path(str(metadata.get("name") or "dataset.csv"))
        selected_items = [metadata]

    imported: list[ImportedBinaryFile] = []
    for item in selected_items:
        content, content_type = _drive_download(str(item.get("id") or ""), access_token=access_token)
        imported.append(
            ImportedBinaryFile(
                file_name=str(item.get("name") or "dataset.csv"),
                relative_path=str(item.get("relative_path") or item.get("name") or "dataset.csv"),
                content_type=content_type,
                content=content,
            )
        )
    if not imported:
        raise ValueError("No supported CSV or Excel files were found in the selected Google Drive item.")
    return imported


def parse_kaggle_dataset_slug(url: str) -> str:
    raw = str(url or "").strip()
    match = re.search(r"kaggle\.com/datasets/([^/?#]+/[^/?#]+)", raw, flags=re.IGNORECASE)
    if not match:
        raise ValueError("Paste a Kaggle dataset URL in the format https://www.kaggle.com/datasets/owner/name")
    return match.group(1)


def _resolve_kaggle_config_dir() -> Path:
    configured = str(get_settings().kaggle_config_dir or "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".kaggle"


def download_kaggle_dataset(dataset_url: str, session_id: str) -> list[ImportedBinaryFile]:
    slug = parse_kaggle_dataset_slug(dataset_url)
    extract_dir = get_session_extract_dir(session_id) / "kaggle"
    extract_dir.mkdir(parents=True, exist_ok=True)

    kaggle_config_dir = _resolve_kaggle_config_dir()
    kaggle_credentials = kaggle_config_dir / "kaggle.json"
    if not kaggle_credentials.exists():
        raise ValueError(
            f"Kaggle credentials were not found at {kaggle_credentials}. Add kaggle.json before importing."
        )

    environment = os.environ.copy()
    environment["KAGGLE_CONFIG_DIR"] = str(kaggle_config_dir)
    command = ["kaggle", "datasets", "download", "-d", slug, "-p", str(extract_dir), "--unzip"]
    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
    except FileNotFoundError as error:
        raise ValueError("The Kaggle CLI is not installed in the Aidssist backend environment.") from error
    except subprocess.CalledProcessError as error:
        stderr = (error.stderr or "").strip()
        raise ValueError(stderr or f"Kaggle download failed for dataset '{slug}'.") from error

    imported: list[ImportedBinaryFile] = []
    for file_path in sorted(extract_dir.rglob("*")):
        if not file_path.is_file() or file_path.suffix.lower() not in SUPPORTED_IMPORT_SUFFIXES:
            continue
        relative_path = sanitize_relative_path(str(file_path.relative_to(extract_dir)))
        payload = file_path.read_bytes()
        write_session_file(session_id, relative_path, payload, extracted=True)
        imported.append(
            ImportedBinaryFile(
                file_name=file_path.name,
                relative_path=relative_path,
                content_type="text/csv"
                if file_path.suffix.lower() == ".csv"
                else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                content=payload,
            )
        )
    if not imported:
        raise ValueError(f"Kaggle dataset '{slug}' did not contain any CSV or Excel files.")
    return imported

