from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from backend.data_sources import build_dataframe_fingerprint, build_dataset_key


SUPPORTED_DATASET_SUFFIXES = {".csv"}
AUTO_CACHE_MAX_FILES = 32
AUTO_CACHE_TOTAL_BYTES = 200 * 1024 * 1024


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sanitize_file_name(file_name: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(file_name or "").strip())
    if not sanitized:
        sanitized = "dataset.csv"
    if not sanitized.lower().endswith(".csv"):
        sanitized = f"{sanitized}.csv"
    return sanitized


def _file_signature(path: Path) -> tuple[int, int]:
    stat_result = path.stat()
    return int(stat_result.st_size), int(stat_result.st_mtime_ns)


def _file_dataset_id(dataset_dir: Path, path: Path) -> str:
    relative_path = str(path.resolve().relative_to(dataset_dir.resolve()))
    digest = hashlib.sha1(relative_path.encode("utf-8")).hexdigest()[:12]
    return f"file_{digest}"


@dataclass(slots=True)
class DatasetRegistryEntry:
    dataset_id: str
    dataset_name: str
    dataset_key: str
    source_type: str
    source_label: str
    path: str | None = None
    size_bytes: int = 0
    modified_at: str | None = None
    parents: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    row_count: int | None = None
    column_count: int | None = None
    cached: bool = False
    virtual: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "dataset_name": self.dataset_name,
            "dataset_key": self.dataset_key,
            "source_type": self.source_type,
            "source_label": self.source_label,
            "path": self.path,
            "size_bytes": self.size_bytes,
            "modified_at": self.modified_at,
            "parents": list(self.parents),
            "metadata": dict(self.metadata),
            "row_count": self.row_count,
            "column_count": self.column_count,
            "cached": self.cached,
            "virtual": self.virtual,
        }


class DatasetManager:
    """Scans a folder of CSV datasets, caches loaded frames, and tracks virtual datasets."""

    def __init__(self, dataset_dir: str | Path, *, recursive: bool = True) -> None:
        self.dataset_dir = Path(dataset_dir).expanduser().resolve()
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.recursive = bool(recursive)
        self._file_registry: dict[str, DatasetRegistryEntry] = {}
        self._virtual_registry: dict[str, DatasetRegistryEntry] = {}
        self._dataframe_cache: dict[str, pd.DataFrame] = {}
        self._file_signatures: dict[str, tuple[int, int]] = {}

    def _iter_dataset_paths(self) -> list[Path]:
        pattern = "**/*" if self.recursive else "*"
        dataset_paths = [
            path
            for path in self.dataset_dir.glob(pattern)
            if path.is_file() and path.suffix.lower() in SUPPORTED_DATASET_SUFFIXES
        ]
        return sorted(dataset_paths, key=lambda item: str(item.relative_to(self.dataset_dir)).lower())

    def scan_datasets(self, *, force: bool = False) -> list[DatasetRegistryEntry]:
        discovered_registry: dict[str, DatasetRegistryEntry] = {}
        discovered_signatures: dict[str, tuple[int, int]] = {}

        for path in self._iter_dataset_paths():
            dataset_id = _file_dataset_id(self.dataset_dir, path)
            signature = _file_signature(path)
            discovered_signatures[dataset_id] = signature

            existing_entry = self._file_registry.get(dataset_id)
            if force or existing_entry is None or self._file_signatures.get(dataset_id) != signature:
                self._dataframe_cache.pop(dataset_id, None)
                relative_label = str(path.relative_to(self.dataset_dir))
                entry = DatasetRegistryEntry(
                    dataset_id=dataset_id,
                    dataset_name=path.name,
                    dataset_key=f"{relative_label}:{signature[0]}:{signature[1]}",
                    source_type="file",
                    source_label=relative_label,
                    path=str(path),
                    size_bytes=int(signature[0]),
                    modified_at=datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
                    cached=False,
                    virtual=False,
                )
            else:
                entry = existing_entry

            cached_frame = self._dataframe_cache.get(dataset_id)
            if cached_frame is not None:
                entry.row_count = int(len(cached_frame))
                entry.column_count = int(len(cached_frame.columns))
                entry.cached = True
            discovered_registry[dataset_id] = entry

        removed_ids = set(self._file_registry) - set(discovered_registry)
        for dataset_id in removed_ids:
            self._dataframe_cache.pop(dataset_id, None)

        self._file_registry = discovered_registry
        self._file_signatures = discovered_signatures
        return self.list_datasets()

    def list_datasets(self, *, include_virtual: bool = True) -> list[DatasetRegistryEntry]:
        entries = list(self._file_registry.values())
        if include_virtual:
            entries.extend(self._virtual_registry.values())
        for entry in entries:
            entry.cached = entry.dataset_id in self._dataframe_cache
        return sorted(entries, key=lambda item: (item.virtual, item.dataset_name.lower(), item.dataset_id))

    def get_entry(self, dataset_id: str) -> DatasetRegistryEntry | None:
        return self._file_registry.get(dataset_id) or self._virtual_registry.get(dataset_id)

    def has_dataset(self, dataset_id: str) -> bool:
        return self.get_entry(dataset_id) is not None

    def load_dataset(self, dataset_id: str, *, force: bool = False) -> pd.DataFrame:
        entry = self.get_entry(dataset_id)
        if entry is None:
            raise ValueError(f"Dataset '{dataset_id}' is not registered.")

        if entry.virtual:
            cached_virtual = self._dataframe_cache.get(dataset_id)
            if cached_virtual is None:
                raise ValueError(f"Virtual dataset '{dataset_id}' is no longer available.")
            return cached_virtual.copy(deep=True)

        path = Path(str(entry.path or "")).expanduser().resolve()
        if not path.exists():
            raise ValueError(f"Dataset file is missing: {path}")

        signature = _file_signature(path)
        cached_frame = self._dataframe_cache.get(dataset_id)
        if not force and cached_frame is not None and self._file_signatures.get(dataset_id) == signature:
            entry.cached = True
            entry.row_count = int(len(cached_frame))
            entry.column_count = int(len(cached_frame.columns))
            return cached_frame.copy(deep=True)

        dataframe = pd.read_csv(path, low_memory=False)
        dataframe.columns = [str(column) for column in dataframe.columns]
        self._dataframe_cache[dataset_id] = dataframe.copy(deep=True)
        self._file_signatures[dataset_id] = signature
        entry.row_count = int(len(dataframe))
        entry.column_count = int(len(dataframe.columns))
        entry.cached = True
        return dataframe.copy(deep=True)

    def warm_cache(
        self,
        *,
        max_files: int = AUTO_CACHE_MAX_FILES,
        max_total_bytes: int = AUTO_CACHE_TOTAL_BYTES,
    ) -> int:
        cached_count = 0
        total_bytes = 0

        for entry in self.list_datasets(include_virtual=False):
            if cached_count >= max_files:
                break

            projected_size = total_bytes + int(entry.size_bytes or 0)
            if cached_count > 0 and projected_size > max_total_bytes:
                break

            self.load_dataset(entry.dataset_id)
            total_bytes = projected_size
            cached_count += 1

        return cached_count

    def register_uploaded_file(
        self,
        file_name: str,
        file_bytes: bytes,
        *,
        overwrite: bool = False,
    ) -> DatasetRegistryEntry:
        sanitized_name = _sanitize_file_name(file_name)
        target_path = self.dataset_dir / sanitized_name

        if target_path.exists() and not overwrite:
            stem = target_path.stem
            suffix = target_path.suffix
            index = 1
            while target_path.exists():
                target_path = self.dataset_dir / f"{stem}_{index}{suffix}"
                index += 1

        target_path.write_bytes(file_bytes)
        self.scan_datasets(force=True)
        dataset_id = _file_dataset_id(self.dataset_dir, target_path)
        entry = self.get_entry(dataset_id)
        if entry is None:
            raise ValueError("Uploaded dataset could not be registered.")
        return entry

    def create_virtual_dataset(
        self,
        dataset_name: str,
        dataframe: pd.DataFrame,
        *,
        parents: list[str] | None = None,
        source_label: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> DatasetRegistryEntry:
        virtual_name = str(dataset_name or "").strip() or "derived_dataset.csv"
        if not virtual_name.lower().endswith(".csv"):
            virtual_name = f"{virtual_name}.csv"

        dataset_id = f"virtual_{uuid.uuid4().hex[:12]}"
        normalized_frame = dataframe.copy(deep=True)
        normalized_frame.columns = [str(column) for column in normalized_frame.columns]
        fingerprint = build_dataframe_fingerprint(normalized_frame)
        entry = DatasetRegistryEntry(
            dataset_id=dataset_id,
            dataset_name=virtual_name,
            dataset_key=build_dataset_key(virtual_name, fingerprint),
            source_type="virtual",
            source_label=source_label or "Derived in-memory dataset",
            path=None,
            size_bytes=0,
            modified_at=_utc_now(),
            parents=list(parents or []),
            metadata=dict(metadata or {}),
            row_count=int(len(normalized_frame)),
            column_count=int(len(normalized_frame.columns)),
            cached=True,
            virtual=True,
        )
        self._virtual_registry[dataset_id] = entry
        self._dataframe_cache[dataset_id] = normalized_frame
        return entry

    def remove_virtual_dataset(self, dataset_id: str) -> None:
        if dataset_id in self._virtual_registry:
            self._virtual_registry.pop(dataset_id, None)
            self._dataframe_cache.pop(dataset_id, None)

    def common_columns(self, dataset_ids: list[str]) -> list[str]:
        column_sets: list[set[str]] = []
        for dataset_id in dataset_ids:
            dataframe = self.load_dataset(dataset_id)
            column_sets.append({str(column) for column in dataframe.columns})
        if not column_sets:
            return []
        common = set.intersection(*column_sets)
        return sorted(common)


__all__ = [
    "AUTO_CACHE_MAX_FILES",
    "AUTO_CACHE_TOTAL_BYTES",
    "DatasetManager",
    "DatasetRegistryEntry",
    "SUPPORTED_DATASET_SUFFIXES",
]
