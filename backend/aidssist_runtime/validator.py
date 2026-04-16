from __future__ import annotations

from typing import Any

import pandas as pd


def validate_solver_output(
    *,
    route: str,
    query: str,
    candidate_output: dict[str, Any],
) -> tuple[bool, list[dict[str, Any]], str | None]:
    checks: list[dict[str, Any]] = []
    error_message: str | None = None

    summary = str(candidate_output.get("summary") or "").strip()
    checks.append(
        {
            "name": "summary_present",
            "status": "passed" if bool(summary) else "failed",
            "detail": "A user-facing summary must be present.",
        }
    )

    if route == "data":
        pipeline_output = candidate_output.get("pipeline_output") or {}
        result = pipeline_output.get("result")
        passed = result is not None
        checks.append(
            {
                "name": "data_result_present",
                "status": "passed" if passed else "failed",
                "detail": "Data solve runs must produce a result payload.",
            }
        )
        if isinstance(result, pd.DataFrame):
            checks.append(
                {
                    "name": "dataframe_not_empty",
                    "status": "passed" if not result.empty else "failed",
                    "detail": "Transformed dataframe results cannot be empty.",
                }
            )
    else:
        solution_markdown = str(candidate_output.get("solution_markdown") or "").strip()
        redesign = candidate_output.get("redesign_recommendations") or []
        checks.append(
            {
                "name": "solution_markdown_present",
                "status": "passed" if bool(solution_markdown) else "failed",
                "detail": "Code and hybrid routes must include a written solution body.",
            }
        )
        checks.append(
            {
                "name": "redesign_present",
                "status": "passed" if bool(redesign) else "failed",
                "detail": "Code-oriented runs must include redesign recommendations.",
            }
        )

    failed_checks = [check for check in checks if check["status"] != "passed"]
    if failed_checks:
        error_message = (
            f"Solve output did not pass validation for query '{query}'. "
            f"Failed checks: {', '.join(check['name'] for check in failed_checks)}."
        )
    return not failed_checks, checks, error_message
