from __future__ import annotations

from typing import Any

import pandas as pd


def _context_text(context: Any) -> str:
    if isinstance(context, list):
        return " ".join(str((item or {}).get("query", "")) for item in context if isinstance(item, dict)).lower()
    return str(context).lower()


def generate_suggestions(context, df: pd.DataFrame):
    del df  # Reserved for future dataset-aware suggestion rules.

    context_text = _context_text(context)
    suggestions = []

    if "group" not in context_text:
        suggestions.append("Group data by category")

    if "trend" not in context_text:
        suggestions.append("Analyze trend over time")

    if "top" not in context_text:
        suggestions.append("Find top 5 values")

    return suggestions[:3]


__all__ = ["generate_suggestions"]
