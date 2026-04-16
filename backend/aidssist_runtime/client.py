from __future__ import annotations

import time

import requests

from .config import get_settings
from .serialization import deserialize_analysis_output
from backend.forecasting import deserialize_forecast_output


def analysis_service_enabled() -> bool:
    return bool(get_settings().api_base_url)


def get_runtime_configuration_status(local_status_resolver):
    settings = get_settings()
    if not settings.api_base_url:
        return local_status_resolver()

    try:
        response = requests.get(
            f"{settings.api_base_url}/readyz",
            timeout=settings.request_timeout_seconds,
        )
        response.raise_for_status()
        return True, f"Aidssist analysis service is ready at {settings.api_base_url}."
    except Exception as error:
        return False, f"Aidssist analysis service is unavailable: {error}"


def ensure_remote_dataset(dataset_state: dict) -> str:
    if dataset_state.get("remote_dataset_id"):
        return str(dataset_state["remote_dataset_id"])

    settings = get_settings()
    file_name = str(dataset_state.get("dataset_name") or "dataset.csv")
    if not file_name.lower().endswith(".csv"):
        file_name = f"{file_name}.csv"

    file_bytes = dataset_state["df"].to_csv(index=False).encode("utf-8")
    response = requests.post(
        f"{settings.api_base_url}/v1/uploads",
        files={"file": (file_name, file_bytes, "text/csv")},
        timeout=settings.request_timeout_seconds,
    )
    response.raise_for_status()
    payload = response.json()
    dataset_state["remote_dataset_id"] = payload["dataset_id"]
    return payload["dataset_id"]


def run_remote_analysis(dataset_state: dict, query: str, workflow_context: dict | None = None) -> dict:
    settings = get_settings()
    dataset_id = ensure_remote_dataset(dataset_state)
    response = requests.post(
        f"{settings.api_base_url}/v1/jobs/analyze",
        json={
            "dataset_id": dataset_id,
            "query": str(query or ""),
            "workflow_context": workflow_context or {},
        },
        timeout=settings.request_timeout_seconds,
    )
    response.raise_for_status()
    job_payload = response.json()
    job_id = job_payload["job_id"]

    deadline = time.time() + settings.job_timeout_seconds
    while time.time() < deadline:
        status_response = requests.get(
            f"{settings.api_base_url}/v1/jobs/{job_id}",
            timeout=settings.request_timeout_seconds,
        )
        status_response.raise_for_status()
        status_payload = status_response.json()
        if status_payload["status"] in {"completed", "failed"}:
            analysis_output = deserialize_analysis_output(status_payload.get("analysis_output") or {})
            analysis_output["job_id"] = job_id
            analysis_output["cache_hit"] = bool(status_payload.get("cache_hit"))
            if status_payload["status"] == "failed" and not analysis_output.get("error"):
                analysis_output["error"] = status_payload.get("error_message")
            return analysis_output
        time.sleep(settings.job_poll_interval_seconds)

    raise RuntimeError("Analysis job timed out while waiting for the remote worker.")


def run_remote_forecast(dataset_state: dict, forecast_config: dict, workflow_context: dict | None = None) -> dict:
    settings = get_settings()
    dataset_id = ensure_remote_dataset(dataset_state)
    response = requests.post(
        f"{settings.api_base_url}/v1/jobs/forecast",
        json={
            "dataset_id": dataset_id,
            "forecast_config": forecast_config or {},
            "workflow_context": workflow_context or {},
        },
        timeout=settings.request_timeout_seconds,
    )
    response.raise_for_status()
    job_payload = response.json()
    job_id = job_payload["job_id"]

    deadline = time.time() + settings.job_timeout_seconds
    while time.time() < deadline:
        status_response = requests.get(
            f"{settings.api_base_url}/v1/jobs/{job_id}",
            timeout=settings.request_timeout_seconds,
        )
        status_response.raise_for_status()
        status_payload = status_response.json()
        if status_payload["status"] in {"completed", "failed"}:
            forecast_output = deserialize_forecast_output(status_payload.get("forecast_output") or {})
            forecast_output["job_id"] = job_id
            forecast_output["cache_hit"] = bool(status_payload.get("cache_hit"))
            if status_payload["status"] == "failed" and not forecast_output.get("error"):
                forecast_output["error"] = status_payload.get("error_message")
            return forecast_output
        time.sleep(settings.job_poll_interval_seconds)

    raise RuntimeError("Forecast job timed out while waiting for the remote worker.")
