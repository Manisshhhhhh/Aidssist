from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd

from .config import get_settings


TEXT_FILE_SUFFIXES = {".txt", ".md", ".json", ".yaml", ".yml", ".py", ".js", ".ts", ".sql"}
TABLE_FILE_SUFFIXES = {".csv", ".xlsx", ".xlsm"}
CODE_FILE_SUFFIXES = {".py", ".js", ".ts", ".sql"}
CONFIG_FILE_SUFFIXES = {".json", ".yaml", ".yml"}
ARCHIVE_FILE_SUFFIXES = {".zip"}


def infer_language(file_name: str) -> str | None:
    suffix = Path(file_name or "").suffix.lower()
    return {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".sql": "sql",
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".md": "markdown",
        ".txt": "text",
        ".csv": "csv",
        ".xlsx": "excel",
        ".xlsm": "excel",
        ".zip": "zip",
    }.get(suffix)


def classify_file_kind(file_name: str, media_type: str | None = None) -> str:
    suffix = Path(file_name or "").suffix.lower()
    normalized_media_type = str(media_type or "").lower()

    if suffix in TABLE_FILE_SUFFIXES:
        return "table"
    if suffix in CODE_FILE_SUFFIXES:
        return "code"
    if suffix in CONFIG_FILE_SUFFIXES:
        return "config"
    if suffix in ARCHIVE_FILE_SUFFIXES:
        return "archive"
    if suffix in TEXT_FILE_SUFFIXES:
        return "text"
    if normalized_media_type.startswith("text/"):
        return "text"
    return "binary"


def estimate_token_count(text: str) -> int:
    words = [token for token in str(text or "").split() if token]
    return max(1, len(words))


def decode_text_bytes(data: bytes) -> str:
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            return data.decode(encoding)
        except Exception:
            continue
    return data.decode("utf-8", errors="replace")


def build_text_chunks(
    *,
    file_name: str,
    text: str,
    file_kind: str,
    language: str | None,
) -> list[dict[str, Any]]:
    normalized_text = str(text or "").strip()
    if not normalized_text:
        return []

    settings = get_settings()
    chunk_size = max(int(settings.chunk_size_chars), 400)
    overlap = max(0, min(int(settings.chunk_overlap_chars), chunk_size // 2))
    step = max(chunk_size - overlap, 1)
    total_length = len(normalized_text)

    chunks: list[dict[str, Any]] = []
    for index, start in enumerate(range(0, total_length, step)):
        end = min(start + chunk_size, total_length)
        excerpt = normalized_text[start:end].strip()
        if not excerpt:
            continue
        chunks.append(
            {
                "chunk_index": index,
                "title": f"{Path(file_name).name} chunk {index + 1}",
                "content_text": excerpt,
                "token_count": estimate_token_count(excerpt),
                "metadata": {
                    "file_name": file_name,
                    "file_kind": file_kind,
                    "language": language,
                    "char_start": start,
                    "char_end": end,
                    "chunk_span_ratio": round(end / max(total_length, 1), 4),
                },
            }
        )
        if end >= total_length:
            break
    return chunks


def build_table_chunks(
    *,
    dataset_name: str,
    dataset_key: str,
    source_fingerprint: str,
    df: pd.DataFrame,
) -> list[dict[str, Any]]:
    preview = df.head(8).copy()
    preview = preview.where(pd.notna(preview), None)
    schema_lines = [
        f"- {column}: {dtype}"
        for column, dtype in df.dtypes.astype(str).items()
    ]
    summary_sections = [
        f"Dataset: {dataset_name}",
        f"Dataset key: {dataset_key}",
        f"Source fingerprint: {source_fingerprint}",
        f"Rows: {len(df):,}",
        f"Columns: {len(df.columns):,}",
        "Schema:",
        "\n".join(schema_lines) or "- no columns",
    ]
    preview_sections = [
        f"Dataset preview for {dataset_name}:",
        preview.to_string(index=False),
    ]
    missing_summary = (
        df.isna().sum().sort_values(ascending=False).head(12).rename("missing_count").to_string()
        if len(df.columns)
        else "No columns"
    )
    quality_sections = [
        f"Duplicate rows: {int(df.duplicated().sum()):,}",
        f"Missing cells: {int(df.isna().sum().sum()):,}",
        "Top missing columns:",
        missing_summary,
    ]

    rendered_sections = [
        ("schema", "\n".join(summary_sections)),
        ("preview", "\n".join(preview_sections)),
        ("quality", "\n".join(quality_sections)),
    ]

    chunks: list[dict[str, Any]] = []
    for index, (section_name, content_text) in enumerate(rendered_sections):
        chunks.append(
            {
                "chunk_index": index,
                "title": f"{Path(dataset_name).name} {section_name}",
                "content_text": content_text,
                "token_count": estimate_token_count(content_text),
                "metadata": {
                    "file_name": dataset_name,
                    "file_kind": "table",
                    "language": "table",
                    "section": section_name,
                    "row_count": int(len(df)),
                    "column_count": int(len(df.columns)),
                },
            }
        )
    return chunks


def choose_asset_kind(file_kinds: list[str]) -> str:
    normalized_kinds = {str(kind or "") for kind in file_kinds if kind}
    if not normalized_kinds:
        return "mixed"
    if normalized_kinds == {"table"}:
        return "table"
    if normalized_kinds <= {"code", "config", "text"} and "code" in normalized_kinds:
        return "code"
    if "table" in normalized_kinds and any(kind in normalized_kinds for kind in {"code", "config", "text"}):
        return "hybrid"
    if "archive" in normalized_kinds:
        return "project"
    if len(normalized_kinds) == 1:
        return normalized_kinds.pop()
    return "mixed"


def build_similarity_excerpt(text: str, *, limit: int = 220) -> str:
    normalized = " ".join(str(text or "").split())
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[: max(limit - 1, 1)].rstrip()}..."


def score_to_confidence(score: float) -> str:
    if score >= 0.82:
        return "high"
    if score >= 0.62:
        return "medium"
    return "low"
