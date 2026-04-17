from __future__ import annotations

import json
from typing import Any

import requests

from .config import get_settings


class LLMUnavailableError(RuntimeError):
    pass


def llm_is_configured() -> bool:
    settings = get_settings()
    return bool(settings.llm_base_url and settings.llm_model)


def request_json_completion(
    *,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.2,
    max_tokens: int = 1800,
) -> dict[str, Any]:
    settings = get_settings()
    if not llm_is_configured():
        raise LLMUnavailableError("LLM gateway is not configured.")

    headers = {"Content-Type": "application/json"}
    if settings.llm_api_key:
        headers["Authorization"] = f"Bearer {settings.llm_api_key}"

    response = requests.post(
        f"{settings.llm_base_url}/chat/completions",
        headers=headers,
        timeout=settings.request_timeout_seconds,
        json={
            "model": settings.llm_model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
        },
    )
    response.raise_for_status()
    payload = dict(response.json() or {})
    choices = list(payload.get("choices") or [])
    if not choices:
        raise LLMUnavailableError("LLM gateway returned no choices.")
    message = dict((choices[0] or {}).get("message") or {})
    content = message.get("content")
    if isinstance(content, list):
        content = "".join(str(item.get("text") or "") for item in content if isinstance(item, dict))
    if not str(content or "").strip():
        raise LLMUnavailableError("LLM gateway returned an empty response.")
    return dict(json.loads(str(content)))

