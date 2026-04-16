from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from typing import Any, Mapping


def build_cleaning_options_signature(cleaning_options: Any) -> str:
    if cleaning_options is None:
        payload: dict[str, Any] = {}
    elif is_dataclass(cleaning_options):
        payload = asdict(cleaning_options)
    elif isinstance(cleaning_options, Mapping):
        payload = dict(cleaning_options)
    else:
        raise TypeError("Cleaning options must be a dataclass or mapping.")

    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def should_reuse_cleaning_preview(
    preview_state: Mapping[str, Any] | None,
    source_fingerprint: str | None,
    options_signature: str,
) -> bool:
    if preview_state is None or not source_fingerprint:
        return False

    return (
        preview_state.get("_source_fingerprint") == source_fingerprint
        and preview_state.get("_options_signature") == options_signature
    )


def is_cleaned_dataset_current(
    *,
    active_dataset: Mapping[str, Any] | None,
    loaded_source_state: Mapping[str, Any] | None,
    current_options_signature: str,
    applied_source_fingerprint: str | None,
    applied_options_signature: str | None,
) -> bool:
    if active_dataset is None:
        return False

    if loaded_source_state is None:
        return True

    source_fingerprint = loaded_source_state.get("source_fingerprint")
    if not source_fingerprint:
        return False

    return (
        active_dataset.get("source_fingerprint") == source_fingerprint
        and applied_source_fingerprint == source_fingerprint
        and active_dataset.get("cleaning_options_signature") == current_options_signature
        and applied_options_signature == current_options_signature
    )


def is_forecast_result_current(
    *,
    forecast_output: Mapping[str, Any] | None,
    active_dataset: Mapping[str, Any] | None,
    current_config_signature: str,
) -> bool:
    if forecast_output is None or active_dataset is None:
        return False

    return (
        forecast_output.get("dataset_key") == active_dataset.get("dataset_key")
        and forecast_output.get("source_fingerprint") == active_dataset.get("source_fingerprint")
        and forecast_output.get("forecast_config_signature") == current_config_signature
        and not forecast_output.get("error")
    )
