from __future__ import annotations

from dataclasses import dataclass


WORKFLOW_STEPS = ("Upload", "Clean", "Explore", "Forecast", "Analyze", "Export")


@dataclass(frozen=True)
class StepState:
    name: str
    accessible: bool
    locked_reason: str | None = None


def build_step_states(
    *,
    has_loaded_source: bool,
    has_cleaned_dataset: bool,
    can_forecast: bool,
    has_successful_forecast: bool,
    forecast_skipped: bool,
    can_analyze: bool,
    has_exportable_output: bool,
) -> dict[str, StepState]:
    return {
        "Upload": StepState("Upload", True, None),
        "Clean": StepState(
            "Clean",
            has_loaded_source,
            None if has_loaded_source else "Load a dataset first.",
        ),
        "Explore": StepState(
            "Explore",
            has_cleaned_dataset,
            None if has_cleaned_dataset else "Apply cleaning to continue.",
        ),
        "Forecast": StepState(
            "Forecast",
            can_forecast,
            None if can_forecast else "Resolve or acknowledge validation issues first.",
        ),
        "Analyze": StepState(
            "Analyze",
            can_analyze and (has_successful_forecast or forecast_skipped),
            None
            if can_analyze and (has_successful_forecast or forecast_skipped)
            else (
                "Resolve or acknowledge validation issues first."
                if not can_analyze
                else "Run or skip the forecast step first."
            ),
        ),
        "Export": StepState(
            "Export",
            has_exportable_output,
            None if has_exportable_output else "Run a forecast or successful analysis first.",
        ),
    }


def resolve_active_step(requested_step: str, step_states: dict[str, StepState]) -> tuple[str, str | None]:
    if requested_step not in step_states:
        return "Upload", None

    requested = step_states[requested_step]
    if requested.accessible:
        return requested_step, None

    fallback = "Upload"
    for step_name in WORKFLOW_STEPS:
        if step_states[step_name].accessible:
            fallback = step_name
        else:
            break

    return fallback, requested.locked_reason


def build_step_label(step_state: StepState) -> str:
    if step_state.accessible:
        return step_state.name
    return f"{step_state.name} (Locked)"
