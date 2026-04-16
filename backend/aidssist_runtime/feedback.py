from __future__ import annotations

from typing import Any

from backend.workflow_store import WorkflowStore


def record_retrieval_feedback(
    *,
    store: WorkflowStore,
    run_id: str,
    retrieval_trace: dict[str, Any],
    succeeded: bool,
) -> None:
    delta = 1 if succeeded else -1
    for item in retrieval_trace.get("items", []) or []:
        chunk_id = item.get("chunk_id")
        if not chunk_id:
            continue
        store.record_feedback_event(
            run_id=run_id,
            chunk_id=str(chunk_id),
            event_type="retrieval_rank",
            score=delta,
            metadata={
                "score": item.get("score"),
                "confidence": item.get("confidence"),
            },
        )
