from __future__ import annotations

from typing import Any

from .chunking import build_similarity_excerpt, score_to_confidence
from .config import get_settings
from .embedding import cosine_similarity, embed_texts
from backend.workflow_store import WorkflowStore


def retrieve_workspace_context(
    *,
    store: WorkflowStore,
    workspace_id: str,
    query: str,
    top_k: int | None = None,
) -> dict[str, Any]:
    chunks = store.list_workspace_chunks(workspace_id, limit=500)
    chunk_ids = [chunk.chunk_id for chunk in chunks]
    embeddings = {item.chunk_id: item for item in store.list_embeddings_for_chunk_ids(chunk_ids)}
    feedback_scores = store.get_chunk_feedback_scores(chunk_ids)
    query_vector = embed_texts([query])[0] if query else []
    retrieved: list[dict[str, Any]] = []

    for chunk in chunks:
        embedding = embeddings.get(chunk.chunk_id)
        if embedding is None:
            continue
        similarity = cosine_similarity(query_vector, embedding.vector)
        feedback_boost = float(feedback_scores.get(chunk.chunk_id, 0.0)) * 0.02
        final_score = similarity + feedback_boost
        retrieved.append(
            {
                "chunk_id": chunk.chunk_id,
                "asset_id": chunk.asset_id,
                "asset_file_id": chunk.asset_file_id,
                "dataset_id": chunk.dataset_id,
                "title": chunk.title,
                "score": round(final_score, 5),
                "confidence": score_to_confidence(final_score),
                "excerpt": build_similarity_excerpt(chunk.content_text),
                "metadata": chunk.metadata,
            }
        )

    effective_top_k = max(1, int(top_k or get_settings().retrieval_top_k))
    retrieved.sort(key=lambda item: item["score"], reverse=True)
    return {
        "query": str(query or ""),
        "items": retrieved[:effective_top_k],
        "scanned_chunk_count": len(chunks),
    }
