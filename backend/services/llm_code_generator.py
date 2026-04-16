from __future__ import annotations

import os
import warnings
from typing import Any

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        import google.generativeai as genai
except ImportError:  # pragma: no cover - depends on local environment setup
    genai = None


if genai is not None:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel("gemini-1.5-flash")
else:  # pragma: no cover - depends on local environment setup
    model = None


def _strip_code_fences(code: str) -> str:
    stripped = str(code or "").strip()
    if stripped.startswith("```"):
        stripped = stripped.removeprefix("```python").removeprefix("```").strip()
        if stripped.endswith("```"):
            stripped = stripped[:-3].strip()
    return stripped


def _format_memory_context(memory_context: Any) -> str:
    if not memory_context:
        return "No previous context."
    return str(memory_context)


def generate_code_with_llm(df_head: Any, user_query: str, memory_context: Any | None = None) -> str:
    if model is None:
        raise RuntimeError("Gemini SDK is not available.")

    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY is not set.")

    prompt = f"""
You are an expert Python data analyst.

Your task:
Generate ONLY executable Python code using pandas.

STRICT RULES:
- Use ONLY dataframe named df
- DO NOT create sample data
- DO NOT import anything
- DO NOT explain anything
- ALWAYS store output in variable 'result'
- Handle missing values if needed

DATAFRAME PREVIEW:
{df_head}

PREVIOUS CONTEXT:
{_format_memory_context(memory_context)}

USER QUERY:
{user_query}

OUTPUT:
Return ONLY Python code.
"""

    response = model.generate_content(prompt)
    code = _strip_code_fences(getattr(response, "text", ""))

    if not code:
        raise RuntimeError("Gemini returned empty code.")

    return code


def validate_generated_code(code: str) -> tuple[bool, str | None]:
    blocked = [
        "import os",
        "import sys",
        "open(",
        "exec(",
        "eval(",
        "__",
        "subprocess",
    ]

    lowered_code = str(code).lower()
    for blocked_pattern in blocked:
        if blocked_pattern in lowered_code:
            return False, "Unsafe code detected"

    try:
        compile(code, "<llm-generated-code>", "exec")
    except SyntaxError as exc:
        return False, f"Invalid generated code: {exc}"

    if "result" not in code:
        return False, "Code must assign to 'result'"

    return True, None


__all__ = ["generate_code_with_llm", "validate_generated_code"]
