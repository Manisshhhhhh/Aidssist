from __future__ import annotations

import hashlib
import math
import re
from typing import Iterable

try:
    from google import genai
except ImportError:  # pragma: no cover - optional in some local envs
    genai = None

from .config import get_settings
from backend import prompt_pipeline


_TOKEN_PATTERN = re.compile(r"[a-z0-9_]+", re.IGNORECASE)


def embedding_provider_ready() -> bool:
    is_ready, _ = prompt_pipeline.get_gemini_configuration_status()
    return bool(is_ready and genai is not None)


def _normalize_vector(values: list[float]) -> list[float]:
    length = math.sqrt(sum(value * value for value in values))
    if length <= 1e-12:
        return values
    return [value / length for value in values]


def deterministic_embedding(text: str, *, dimensions: int | None = None) -> list[float]:
    dims = max(int(dimensions or get_settings().embedding_dimensions), 8)
    vector = [0.0] * dims
    tokens = _TOKEN_PATTERN.findall(str(text or "").lower())
    if not tokens:
        return vector

    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        slot = int.from_bytes(digest[:4], "big") % dims
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        magnitude = 1.0 + (digest[5] / 255.0)
        vector[slot] += sign * magnitude
    return _normalize_vector(vector)


def _extract_embedding_values(response) -> list[list[float]]:
    embeddings = getattr(response, "embeddings", None) or []
    values: list[list[float]] = []
    for item in embeddings:
        raw_values = getattr(item, "values", None) or []
        values.append([float(value) for value in raw_values])
    return values


def embed_texts(texts: Iterable[str], *, model_name: str | None = None) -> list[list[float]]:
    rendered_texts = [str(text or "").strip() for text in texts]
    if not rendered_texts:
        return []

    settings = get_settings()
    if embedding_provider_ready():
        try:
            client = genai.Client(api_key=prompt_pipeline._get_gemini_api_key())  # type: ignore[attr-defined]
            response = client.models.embed_content(
                model=model_name or settings.embedding_model,
                contents=rendered_texts,
                config={"output_dimensionality": settings.embedding_dimensions},
            )
            embedded_values = _extract_embedding_values(response)
            if len(embedded_values) == len(rendered_texts):
                return [
                    _normalize_vector([float(value) for value in values])
                    for values in embedded_values
                ]
        except Exception:
            pass

    return [
        deterministic_embedding(text, dimensions=settings.embedding_dimensions)
        for text in rendered_texts
    ]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    length = min(len(left), len(right))
    if length == 0:
        return 0.0
    return float(sum(float(left[index]) * float(right[index]) for index in range(length)))
