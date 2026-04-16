from __future__ import annotations

from collections.abc import Callable
from typing import Any


def bounded_refinement_loop(
    *,
    build_candidate: Callable[[int, str | None], dict[str, Any]],
    validate_candidate: Callable[[dict[str, Any]], tuple[bool, list[dict[str, Any]], str | None]],
    max_retries: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    steps: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    latest_error: str | None = None
    latest_candidate: dict[str, Any] = {}

    for attempt_index in range(max(1, int(max_retries) + 1)):
        latest_candidate = build_candidate(attempt_index, latest_error)
        steps.append(
            {
                "attempt_index": attempt_index,
                "stage": "reason",
                "status": "completed",
                "title": f"Candidate attempt {attempt_index + 1}",
                "detail": {"feedback": latest_error},
            }
        )
        passed, checks, error_message = validate_candidate(latest_candidate)
        reports.append(
            {
                "attempt_index": attempt_index,
                "status": "passed" if passed else "failed",
                "checks": checks,
                "error_message": error_message,
            }
        )
        if passed:
            return latest_candidate, steps, reports
        latest_error = error_message

    return latest_candidate, steps, reports
