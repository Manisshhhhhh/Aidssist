from __future__ import annotations

from typing import Any, Callable

import pandas as pd


INSIGHT_TEMPLATE_NAME = "insight_prompt.txt"


def build_decision_grade_insight_values(
    *,
    user_query: str,
    result: Any,
    df: pd.DataFrame,
    format_result: Callable[[Any], str],
    format_dataframe_context: Callable[[pd.DataFrame], dict[str, str]],
) -> dict[str, str]:
    return {
        **format_dataframe_context(df),
        "user_query": str(user_query),
        "result": format_result(result),
    }


def generate_decision_grade_insights(
    *,
    user_query: str,
    result: Any,
    df: pd.DataFrame,
    model: str,
    prompt_runner: Callable[..., str],
    format_result: Callable[[Any], str],
    format_dataframe_context: Callable[[pd.DataFrame], dict[str, str]],
) -> str:
    prompt_values = build_decision_grade_insight_values(
        user_query=user_query,
        result=result,
        df=df,
        format_result=format_result,
        format_dataframe_context=format_dataframe_context,
    )
    return prompt_runner(
        INSIGHT_TEMPLATE_NAME,
        prompt_values,
        model=model,
        use_search=False,
    )
