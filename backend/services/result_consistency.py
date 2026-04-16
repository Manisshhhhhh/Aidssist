from __future__ import annotations

import hashlib
import json
import re
from typing import Any

import pandas as pd


def _normalize_for_hash(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        normalized = value.copy()
        normalized.columns = [str(column) for column in normalized.columns]
        normalized = normalized.where(pd.notna(normalized), None)
        return normalized.to_dict(orient="records")
    if isinstance(value, pd.Series):
        normalized = value.where(pd.notna(value), None)
        return normalized.to_list()
    if isinstance(value, dict):
        return {str(key): _normalize_for_hash(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_normalize_for_hash(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def hash_result(result):
    normalized = _normalize_for_hash(result)
    payload = json.dumps(normalized, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def normalize_query(query: str | None) -> str:
    return re.sub(r"\s+", " ", str(query or "").strip().lower())


def _build_consistency_payload(result: Any, prior_hashes: list[str] | None) -> dict[str, Any]:
    result_hash = hash_result(result)
    filtered_hashes = [str(item) for item in list(prior_hashes or []) if str(item).strip()]
    inconsistency_detected = any(existing_hash != result_hash for existing_hash in filtered_hashes)
    return {
        "result_hash": result_hash,
        "inconsistency_detected": bool(inconsistency_detected),
        "prior_hash_count": len(filtered_hashes),
        "consistency_validated": bool(filtered_hashes),
    }


def build_reproducibility_metadata(
    *,
    source_fingerprint: str | None,
    pipeline_trace: list[dict[str, Any]] | None = None,
    result_hash: str | None = None,
    consistency_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_trace = list(pipeline_trace or [])
    pipeline_trace_hash = hashlib.sha256(
        json.dumps(_normalize_for_hash(normalized_trace), sort_keys=True, default=str, separators=(",", ":")).encode()
    ).hexdigest()
    consistency_payload = dict(consistency_payload or {})
    resolved_result_hash = str(result_hash or consistency_payload.get("result_hash") or "")
    return {
        "dataset_fingerprint": str(source_fingerprint or ""),
        "pipeline_trace_hash": pipeline_trace_hash,
        "result_hash": resolved_result_hash,
        "consistent_with_prior_runs": not bool(consistency_payload.get("inconsistency_detected")),
        "prior_hash_count": int(consistency_payload.get("prior_hash_count") or 0),
        "consistency_validated": bool(consistency_payload.get("consistency_validated")),
    }


def build_analysis_consistency(
    *,
    store,
    result: Any,
    source_fingerprint: str | None,
    query: str | None,
    analysis_intent: str | None,
) -> dict[str, Any]:
    prior_hashes = []
    if store is not None and source_fingerprint:
        prior_hashes = store.list_matching_run_result_hashes(
            source_fingerprint=source_fingerprint,
            normalized_query=normalize_query(query),
            analysis_intent=analysis_intent,
            limit=100,
        )
    return _build_consistency_payload(result, prior_hashes)


def build_forecast_consistency(
    *,
    store,
    result: Any,
    source_fingerprint: str | None,
    target_column: str | None,
    horizon: str | None,
) -> dict[str, Any]:
    prior_hashes = []
    if store is not None and source_fingerprint and target_column and horizon:
        prior_hashes = store.list_matching_forecast_result_hashes(
            source_fingerprint=source_fingerprint,
            target_column=target_column,
            horizon=horizon,
            limit=100,
        )
    return _build_consistency_payload(result, prior_hashes)


def build_solve_consistency(
    *,
    store,
    result: Any,
    source_fingerprint: str | None,
    query: str | None,
    route: str = "data",
) -> dict[str, Any]:
    prior_hashes = []
    if store is not None and source_fingerprint:
        prior_hashes = store.list_matching_solve_result_hashes(
            source_fingerprint=source_fingerprint,
            normalized_query=normalize_query(query),
            route=route,
            limit=100,
        )
    return _build_consistency_payload(result, prior_hashes)
