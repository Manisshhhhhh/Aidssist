from __future__ import annotations

from collections import Counter
from typing import Any


failure_log: list[dict[str, Any]] = []
failure_patterns: Counter[str] = Counter(
    {
        "missing_date_column": 0,
        "invalid_target": 0,
    }
)


def _infer_failure_pattern(error: object, stage: object, metadata: dict[str, Any] | None) -> str | None:
    metadata = dict(metadata or {})
    explicit_pattern = str(metadata.get("failure_pattern") or "").strip()
    if explicit_pattern:
        return explicit_pattern

    lowered_error = str(error or "").lower()
    lowered_stage = str(stage or "").lower()

    if any(token in lowered_error for token in ("missing date", "date column", "datetime column", "no clear date")):
        return "missing_date_column"
    if any(token in lowered_error for token in ("invalid target", "target column", "usable numeric values", "prediction requires a clear numeric target")):
        return "invalid_target"
    if "forecast" in lowered_stage and "date" in lowered_stage:
        return "missing_date_column"
    return None


def get_failure_patterns() -> dict[str, int]:
    return {key: int(value) for key, value in failure_patterns.items()}


def log_failure(
    query,
    error,
    stage,
    *,
    store=None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    detected_pattern = _infer_failure_pattern(error, stage, metadata)
    if detected_pattern:
        failure_patterns[detected_pattern] += 1

    record = {
        "query": str(query or ""),
        "error": str(error),
        "stage": str(stage or "unknown"),
        "metadata": dict(metadata or {}),
        "pattern": detected_pattern,
        "failure_patterns": get_failure_patterns(),
    }
    failure_log.append(record)

    if store is not None:
        try:
            store.record_failure_log(
                query=record["query"],
                error=record["error"],
                stage=record["stage"],
                metadata=record["metadata"],
            )
        except Exception:
            pass

    return record
