from __future__ import annotations

import mimetypes
import re
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.api.types import is_numeric_dtype

from .config import APP_STATE_DIR
from .ingestion import load_dataset_dataframe
from backend.aidssist_runtime.analysis_service import get_dataset_summary
from backend.services.assistant_engine import detect_dataset_domain
from backend.services.join_engine import detect_common_columns
from backend.services.target_detector import coerce_datetime_series
from backend.workflow_store import WorkflowStore


ALLOWED_FOLDER_SUFFIXES = {".csv", ".xlsx"}
MAX_FOLDER_FILE_BYTES = 100 * 1024 * 1024
MAX_RELATIONSHIPS = 8
MAX_RELATIONSHIP_SAMPLE_ROWS = 800
MARKETING_KEYWORDS = {
    "campaign",
    "channel",
    "click",
    "clicks",
    "lead",
    "leads",
    "impression",
    "impressions",
    "traffic",
    "spend",
    "ad",
    "ads",
    "conversion",
    "conversions",
}
FINANCE_KEYWORDS = {
    "budget",
    "cash",
    "cashflow",
    "expense",
    "expenses",
    "finance",
    "financial",
    "invoice",
    "margin",
    "pnl",
    "portfolio",
    "profit",
    "revenue",
    "return",
}
SALES_KEYWORDS = {
    "customer",
    "customers",
    "discount",
    "order",
    "orders",
    "price",
    "product",
    "products",
    "quantity",
    "region",
    "revenue",
    "sales",
}


def _normalize_identifier(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").strip().lower())


def _singularize(value: str) -> str:
    normalized = _normalize_identifier(value)
    if normalized.endswith("ies") and len(normalized) > 3:
        return f"{normalized[:-3]}y"
    if normalized.endswith("ses") and len(normalized) > 3:
        return normalized[:-2]
    if normalized.endswith("s") and len(normalized) > 1:
        return normalized[:-1]
    return normalized


def normalize_relative_path(relative_path: str, fallback_name: str) -> str:
    raw_path = str(relative_path or fallback_name or "dataset.csv").replace("\\", "/").strip().lstrip("/")
    parts = [part for part in raw_path.split("/") if part and part != "."]
    if any(part == ".." for part in parts):
        raise ValueError("Folder upload paths cannot contain parent-directory segments.")
    if not parts:
        parts = [Path(fallback_name or "dataset.csv").name]
    return "/".join(parts)


def validate_folder_file(relative_path: str, content: bytes) -> None:
    suffix = Path(relative_path).suffix.lower()
    if suffix not in ALLOWED_FOLDER_SUFFIXES:
        raise ValueError("Only CSV and XLSX files are supported for folder uploads.")
    if len(content) > MAX_FOLDER_FILE_BYTES:
        raise ValueError("Each file must be 100 MB or smaller.")


def _session_root(session_id: str) -> Path:
    safe_session_id = re.sub(r"[^a-zA-Z0-9_-]+", "", str(session_id or "").strip()) or "folder-upload"
    return APP_STATE_DIR / "uploads" / safe_session_id


def cleanup_staged_folder_upload(session_id: str) -> None:
    root = _session_root(session_id)
    shutil.rmtree(root, ignore_errors=True)


def stage_folder_upload_files(
    *,
    session_id: str,
    uploaded_files: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    root = _session_root(session_id)
    root.mkdir(parents=True, exist_ok=True)
    staged_files: list[dict[str, Any]] = []
    failed_files: list[dict[str, Any]] = []

    for uploaded_file in uploaded_files:
        fallback_name = str(uploaded_file.get("file_name") or "dataset.csv")
        raw_relative_path = str(uploaded_file.get("relative_path") or fallback_name)
        content = bytes(uploaded_file.get("content") or b"")
        try:
            relative_path = normalize_relative_path(raw_relative_path, fallback_name)
            validate_folder_file(relative_path, content)
            destination = (root / relative_path).resolve()
            if root.resolve() not in destination.parents and destination != root.resolve():
                raise ValueError("Folder upload path escapes the upload root.")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
            staged_files.append(
                {
                    "file_name": Path(relative_path).name,
                    "relative_path": relative_path,
                    "size_bytes": len(content),
                    "status": "staged",
                }
            )
        except ValueError as error:
            failed_files.append(
                {
                    "file_name": Path(raw_relative_path).name or fallback_name,
                    "relative_path": raw_relative_path or fallback_name,
                    "size_bytes": len(content),
                    "status": "failed",
                    "error": str(error),
                }
            )
    return staged_files, failed_files


def list_staged_folder_files(session_id: str) -> list[dict[str, Any]]:
    root = _session_root(session_id)
    if not root.exists():
        return []

    staged_files: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative_path = path.relative_to(root).as_posix()
        staged_files.append(
            {
                "file_name": relative_path,
                "relative_path": relative_path,
                "content": path.read_bytes(),
                "content_type": mimetypes.guess_type(relative_path)[0]
                or "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            }
        )
    return staged_files


def infer_folder_name(folder_name: str | None, paths: list[str]) -> str:
    if folder_name and str(folder_name).strip():
        return str(folder_name).strip()
    normalized_paths = [normalize_relative_path(path, Path(path).name or "dataset.csv") for path in paths if path]
    if not normalized_paths:
        return "Dataset Folder"
    first_segments = {path.split("/", 1)[0] for path in normalized_paths}
    if len(first_segments) == 1 and any("/" in path for path in normalized_paths):
        return next(iter(first_segments))
    if len(normalized_paths) == 1:
        return Path(normalized_paths[0]).stem or "Dataset Folder"
    return "Dataset Folder"


def _dataset_tokens(dataset_name: str, dataframe: pd.DataFrame) -> set[str]:
    tokens: set[str] = set()
    parts = re.split(r"[^a-z0-9]+", Path(dataset_name).stem.lower())
    tokens.update(part for part in parts if part)
    for column in dataframe.columns:
        tokens.update(part for part in re.split(r"[^a-z0-9]+", str(column).lower()) if part)
    preview = dataframe.head(20)
    for column in preview.columns:
        series = preview[column]
        if is_numeric_dtype(series):
            continue
        for value in series.dropna().astype("string").head(12):
            tokens.update(part for part in re.split(r"[^a-z0-9]+", str(value).lower()) if part)
    return tokens


def classify_dataset_tag(dataset_name: str, dataframe: pd.DataFrame) -> str:
    tokens = _dataset_tokens(dataset_name, dataframe)
    marketing_score = len(tokens.intersection(MARKETING_KEYWORDS))
    finance_score = len(tokens.intersection(FINANCE_KEYWORDS))
    sales_score = len(tokens.intersection(SALES_KEYWORDS))
    if marketing_score > 0 and marketing_score >= max(finance_score, sales_score):
        return "marketing"
    if finance_score > 0 and finance_score >= sales_score:
        return "finance"
    detected_domain = detect_dataset_domain(dataframe)
    if detected_domain == "finance":
        return "finance"
    return "sales"


def _normalize_series_for_relationship(series: pd.Series) -> pd.Series:
    parsed_datetime = coerce_datetime_series(series, column_name=str(series.name or ""))
    if parsed_datetime is not None:
        return parsed_datetime.dt.strftime("%Y-%m-%dT%H:%M:%S").fillna("")
    if is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").round(10).astype("string").fillna("")
    return series.astype("string").str.strip().str.lower().fillna("")


def _column_unique_ratio(series: pd.Series) -> float:
    non_null = series.dropna()
    if non_null.empty:
        return 0.0
    return float(non_null.nunique(dropna=True) / max(len(non_null), 1))


def _key_candidates(table_name: str, dataframe: pd.DataFrame) -> list[str]:
    singular_name = _singularize(table_name)
    candidates: list[tuple[float, str]] = []
    for column in dataframe.columns:
        column_name = str(column)
        normalized = _normalize_identifier(column_name)
        unique_ratio = _column_unique_ratio(dataframe[column])
        if not normalized:
            continue
        score = unique_ratio
        if normalized == "id":
            score += 1.0
        if normalized.endswith("id"):
            score += 0.7
        if singular_name and normalized in {f"{singular_name}id", f"{singular_name}key"}:
            score += 0.8
        if score < 0.3:
            continue
        candidates.append((score, column_name))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return [column_name for _, column_name in candidates[:8]]


def _relationship_name_score(
    *,
    left_table: str,
    left_column: str,
    right_table: str,
    right_column: str,
) -> float:
    left_norm = _normalize_identifier(left_column)
    right_norm = _normalize_identifier(right_column)
    left_table_norm = _singularize(left_table)
    right_table_norm = _singularize(right_table)
    score = 0.0

    if left_norm == right_norm:
        score = max(score, 0.45)
    if left_norm == "id" and right_norm in {f"{left_table_norm}id", f"{left_table_norm}key"}:
        score = max(score, 0.72)
    if right_norm == "id" and left_norm in {f"{right_table_norm}id", f"{right_table_norm}key"}:
        score = max(score, 0.72)
    if left_norm.endswith("id") and right_norm.endswith("id"):
        if left_table_norm and right_norm.startswith(left_table_norm):
            score = max(score, 0.62)
        if right_table_norm and left_norm.startswith(right_table_norm):
            score = max(score, 0.62)
        shared_suffix = left_norm.split("id")[0] and left_norm.split("id")[0] == right_norm.split("id")[0]
        if shared_suffix:
            score = max(score, 0.55)
    return score


def _best_relationship(
    *,
    left_name: str,
    left_df: pd.DataFrame,
    right_name: str,
    right_df: pd.DataFrame,
) -> dict[str, Any] | None:
    left_candidates = _key_candidates(left_name, left_df)
    right_candidates = _key_candidates(right_name, right_df)
    if not left_candidates or not right_candidates:
        return None

    best: dict[str, Any] | None = None
    for left_column in left_candidates:
        left_series = _normalize_series_for_relationship(left_df[left_column].head(MAX_RELATIONSHIP_SAMPLE_ROWS))
        left_values = {value for value in left_series.tolist() if value}
        if len(left_values) < 3:
            continue
        left_unique_ratio = _column_unique_ratio(left_df[left_column])

        for right_column in right_candidates:
            name_score = _relationship_name_score(
                left_table=left_name,
                left_column=left_column,
                right_table=right_name,
                right_column=right_column,
            )
            if name_score <= 0.0:
                continue

            right_series = _normalize_series_for_relationship(right_df[right_column].head(MAX_RELATIONSHIP_SAMPLE_ROWS))
            right_values = {value for value in right_series.tolist() if value}
            if len(right_values) < 3:
                continue

            overlap_count = len(left_values.intersection(right_values))
            if overlap_count < 3:
                continue

            match_rate = overlap_count / max(1, min(len(left_values), len(right_values)))
            if match_rate < 0.2:
                continue

            right_unique_ratio = _column_unique_ratio(right_df[right_column])
            common_column_bonus = 0.0
            common_matches = detect_common_columns(
                left_df[[left_column]].head(MAX_RELATIONSHIP_SAMPLE_ROWS),
                right_df[[right_column]].head(MAX_RELATIONSHIP_SAMPLE_ROWS),
            )
            if common_matches:
                common_column_bonus = float(common_matches[0].get("compatibility_score") or 0.0) * 0.08

            primary_is_left = left_unique_ratio >= right_unique_ratio
            confidence = min(
                0.99,
                round(
                    (name_score * 0.55)
                    + (match_rate * 0.3)
                    + (max(left_unique_ratio, right_unique_ratio) * 0.15)
                    + common_column_bonus,
                    2,
                ),
            )
            relationship = {
                "left_table": left_name if primary_is_left else right_name,
                "left_column": left_column if primary_is_left else right_column,
                "right_table": right_name if primary_is_left else left_name,
                "right_column": right_column if primary_is_left else left_column,
                "confidence": confidence,
                "match_rate": round(match_rate, 2),
            }
            if best is None or relationship["confidence"] > best["confidence"]:
                best = relationship
    return best


def detect_dataset_relationships(dataset_payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    relationships: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str, str, str]] = set()
    for left_index, left_payload in enumerate(dataset_payloads):
        for right_payload in dataset_payloads[left_index + 1 :]:
            relationship = _best_relationship(
                left_name=str(left_payload["table_name"]),
                left_df=left_payload["dataframe"],
                right_name=str(right_payload["table_name"]),
                right_df=right_payload["dataframe"],
            )
            if relationship is None:
                continue
            key = (
                relationship["left_table"],
                relationship["left_column"],
                relationship["right_table"],
                relationship["right_column"],
            )
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            relationships.append(relationship)
    relationships.sort(key=lambda item: (-float(item["confidence"]), -float(item["match_rate"])))
    return relationships[:MAX_RELATIONSHIPS]


def build_folder_dataset_summary(
    *,
    store: WorkflowStore,
    asset_id: str,
    folder_name: str,
    session_id: str,
) -> dict[str, Any]:
    datasets = store.list_asset_datasets(asset_id)
    asset_files = store.list_asset_files(asset_id)
    file_sizes = {str(file_record.file_name): int(file_record.size_bytes) for file_record in asset_files}

    dataset_payloads: list[dict[str, Any]] = []
    tags: list[str] = []
    processed_files: list[dict[str, Any]] = []
    for dataset in datasets:
        summary = get_dataset_summary(dataset.dataset_id)
        dataframe = load_dataset_dataframe(dataset)
        tag = classify_dataset_tag(dataset.dataset_name, dataframe)
        if tag not in tags:
            tags.append(tag)
        table_name = Path(summary["dataset_name"]).stem
        dataset_payloads.append(
            {
                "dataset_id": dataset.dataset_id,
                "table_name": table_name,
                "dataset_name": summary["dataset_name"],
                "dataframe": dataframe,
                "summary": summary,
                "tag": tag,
            }
        )
        processed_files.append(
            {
                "file_name": Path(summary["dataset_name"]).name,
                "relative_path": summary["dataset_name"],
                "size_bytes": file_sizes.get(summary["dataset_name"], dataset.size_bytes),
                "status": "success",
                "dataset_id": dataset.dataset_id,
                "file_tag": tag,
            }
        )

    relationships = detect_dataset_relationships(dataset_payloads)
    previews = [
        {
            "dataset_id": payload["dataset_id"],
            "file_name": payload["summary"]["dataset_name"],
            "table_name": payload["table_name"],
            "file_tag": payload["tag"],
            "row_count": int(payload["summary"]["row_count"]),
            "column_count": int(payload["summary"]["column_count"]),
            "preview_columns": list(payload["summary"]["preview_columns"]),
            "preview_rows": list(payload["summary"]["preview_rows"][:10]),
        }
        for payload in dataset_payloads
    ]
    tables = [str(payload["table_name"]) for payload in dataset_payloads]
    suggested_tables = ", ".join(tables[:3]) if tables else "the uploaded tables"
    return {
        "session_id": session_id,
        "folder_name": folder_name,
        "processed_files": processed_files,
        "dataset_summary": {
            "tables": tables,
            "tags": tags,
            "relationships": relationships,
            "previews": previews,
            "suggested_analysis_prompt": (
                f"Analyze the uploaded dataset folder using {suggested_tables}. "
                "Detect joins, summarize data quality, and recommend the highest-value insights."
            ),
            "ready_message": "Dataset Ready -> Generate Insights",
        },
    }
