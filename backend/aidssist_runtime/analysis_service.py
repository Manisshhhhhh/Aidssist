from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import pandas as pd

from .cache import get_cache_store
from .config import get_settings
from .logging_utils import get_logger
from .metrics import observe_job_completion, observe_queue_wait
from .serialization import deserialize_analysis_output, serialize_analysis_output
from .storage import build_object_key, get_object_store
from backend.forecasting import (
    auto_detect_time_column,
    build_auto_forecast_config,
    build_forecast_eligibility,
    deserialize_forecast_output,
    forecast_config_from_dict,
    forecast_config_to_dict,
    infer_frequency_from_dates,
    persist_forecast_artifact,
    run_forecast_pipeline,
    serialize_forecast_output,
)
from backend.question_engine import build_question_payload
from backend.suggestion_engine import record_user_interaction_memory
from backend.data_sources import (
    CSVSourceConfig,
    ExcelSourceConfig,
    build_dataframe_fingerprint,
    build_dataset_key,
    load_dataframe_from_source,
)
from backend.dashboard_helpers import profile_dataset
from backend.analysis_contract import ensure_analysis_contract_defaults
from backend.prompt_pipeline import PIPELINE_CACHE_VERSION, detect_intent, run_builder_pipeline
from backend.services.auto_analysis_engine import build_auto_analysis_payload
from backend.services.dashboard_engine import build_dashboard_output
from backend.services.decision_engine import (
    build_business_decisions_text,
    derive_recommendations_from_decision_layer,
    ensure_decision_layer_defaults,
)
from backend.services.failure_logging import log_failure
from backend.services.ml_postprocessor import postprocess_ml_output
from backend.services.ml_schema_validator import validate_ml_output
from backend.services.result_consistency import hash_result
from backend.services.target_detector import detect_target_column as detect_target_details
from backend.workflow_store import WorkflowStore


LOGGER = get_logger(__name__)
FORECAST_CACHE_VERSION = "2026-04-03-v1"
EMPTY_AUTO_ANALYSIS = {"auto_analysis": {"tasks": [], "results": [], "summary": []}}


def _infer_upload_kind(file_name: str) -> str:
    suffix = Path(file_name or "").suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix in {".xlsx", ".xlsm"}:
        return "excel"
    raise ValueError("Only CSV and Excel uploads are supported by the analysis API.")


def _load_dataframe_for_upload(file_name: str, file_bytes: bytes):
    upload_kind = _infer_upload_kind(file_name)
    if upload_kind == "csv":
        config = CSVSourceConfig(file_name=file_name, file_bytes=file_bytes)
    else:
        config = ExcelSourceConfig(file_name=file_name, file_bytes=file_bytes)
    loaded = load_dataframe_from_source(config)
    return upload_kind, loaded


def extract_schema(df: pd.DataFrame) -> dict[str, object]:
    return {
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
    }


def detect_date_column(df: pd.DataFrame) -> str | None:
    detected = auto_detect_time_column(df)
    return detected or None


def detect_target_column(df: pd.DataFrame) -> str | None:
    detected = detect_target_details(df)
    return str(detected.get("target") or "") or None


def _build_history_stats(df: pd.DataFrame, date_column: str | None) -> dict[str, object]:
    history_points = int(len(df))
    date_range: tuple[str | None, str | None] = (None, None)

    if date_column and date_column in df.columns:
        parsed_dates = pd.to_datetime(df[date_column], errors="coerce", format="mixed").dropna()
        if not parsed_dates.empty:
            date_range = (
                str(parsed_dates.min().date()),
                str(parsed_dates.max().date()),
            )

    return {
        "history_points": history_points,
        "date_range": date_range,
    }


def _build_cache_key(source_fingerprint: str, query: str, intent: str, workflow_context: dict | None = None) -> str:
    custom_time_range = dict((workflow_context or {}).get("custom_time_range") or {})
    payload = json.dumps(
        {
            "fingerprint": source_fingerprint,
            "query": str(query or "").strip(),
            "intent": intent,
            "time_filter": str((workflow_context or {}).get("time_filter") or ""),
            "custom_time_range": {
                "start_date": str(custom_time_range.get("start_date") or custom_time_range.get("start") or ""),
                "end_date": str(custom_time_range.get("end_date") or custom_time_range.get("end") or ""),
            },
            "pipeline_version": PIPELINE_CACHE_VERSION or get_settings().pipeline_cache_version,
        },
        sort_keys=True,
    ).encode("utf-8")
    return f"analysis:{hashlib.sha256(payload).hexdigest()}"


def _build_forecast_cache_key(source_fingerprint: str, forecast_config: dict, workflow_context: dict | None = None) -> str:
    custom_time_range = dict((workflow_context or {}).get("custom_time_range") or {})
    payload = json.dumps(
        {
            "fingerprint": source_fingerprint,
            "forecast_config": forecast_config,
            "time_filter": str((workflow_context or {}).get("time_filter") or ""),
            "custom_time_range": {
                "start_date": str(custom_time_range.get("start_date") or custom_time_range.get("start") or ""),
                "end_date": str(custom_time_range.get("end_date") or custom_time_range.get("end") or ""),
            },
            "forecast_version": FORECAST_CACHE_VERSION,
        },
        sort_keys=True,
    ).encode("utf-8")
    return f"forecast:{hashlib.sha256(payload).hexdigest()}"


def _build_auto_analysis_cache_key(source_fingerprint: str) -> str:
    return f"auto-analysis:{source_fingerprint}"


def _build_forecast_query_label(forecast_config: dict) -> str:
    target_column = str(forecast_config.get("target_column") or forecast_config.get("target") or "auto-selected KPI")
    horizon = str(forecast_config.get("horizon") or "auto horizon")
    return f"Forecast {target_column} for {horizon.replace('_', ' ')}"


def create_dataset_from_upload(
    file_name: str,
    file_bytes: bytes,
    content_type: str | None = None,
    *,
    user_id: str | None = None,
):
    settings = get_settings()
    max_size_bytes = settings.max_upload_mb * 1024 * 1024
    if len(file_bytes) > max_size_bytes:
        raise ValueError(f"Upload is too large. Maximum allowed size is {settings.max_upload_mb} MB.")

    upload_kind, loaded = _load_dataframe_for_upload(file_name, file_bytes)
    object_store = get_object_store()
    object_key = build_object_key("datasets", loaded.source_fingerprint, file_name)
    object_store.put_bytes(
        object_key,
        file_bytes,
        content_type=content_type or ("text/csv" if upload_kind == "csv" else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
    )

    with WorkflowStore() as store:
        dataset_record = store.create_dataset(
            dataset_name=loaded.dataset_name,
            dataset_key=loaded.dataset_key,
            source_fingerprint=loaded.source_fingerprint,
            source_kind=upload_kind,
            source_label=loaded.source_label,
            object_key=object_key,
            content_type=content_type or ("text/csv" if upload_kind == "csv" else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
            size_bytes=len(file_bytes),
            user_id=user_id,
        )

    profile = profile_dataset(loaded.dataframe, dataset_name=loaded.dataset_name, dataset_key=loaded.dataset_key)
    get_cache_store().set_json(
        f"dataset-profile:{loaded.source_fingerprint}",
        {
            "dataset_name": profile.dataset_name,
            "dataset_key": profile.dataset_key,
            "row_count": profile.row_count,
            "column_count": profile.column_count,
            "numeric_column_count": profile.numeric_column_count,
            "categorical_column_count": profile.categorical_column_count,
            "datetime_column_count": profile.datetime_column_count,
        },
        ttl_seconds=settings.profile_cache_ttl_seconds,
    )

    try:
        auto_analysis_payload = build_auto_analysis_payload(loaded.dataframe)
        get_cache_store().set_json(
            _build_auto_analysis_cache_key(loaded.source_fingerprint),
            auto_analysis_payload,
            ttl_seconds=settings.profile_cache_ttl_seconds,
        )
    except Exception:
        LOGGER.warning(
            "Auto analysis generation failed during upload",
            extra={"source_fingerprint": loaded.source_fingerprint},
            exc_info=True,
        )

    return dataset_record


def _get_cached_or_compute_auto_analysis(dataset_record, dataframe: pd.DataFrame | None = None) -> dict:
    cache_key = _build_auto_analysis_cache_key(dataset_record.source_fingerprint)
    cached_payload = get_cache_store().get_json(cache_key)
    if cached_payload is not None:
        return cached_payload

    try:
        resolved_dataframe = dataframe if dataframe is not None else _load_dataframe_for_dataset(dataset_record)
        payload = build_auto_analysis_payload(resolved_dataframe)
        get_cache_store().set_json(
            cache_key,
            payload,
            ttl_seconds=get_settings().profile_cache_ttl_seconds,
        )
        return payload
    except Exception:
        LOGGER.warning(
            "Auto analysis generation failed",
            extra={"dataset_id": dataset_record.dataset_id, "source_fingerprint": dataset_record.source_fingerprint},
            exc_info=True,
        )
        return dict(EMPTY_AUTO_ANALYSIS)


def get_dataset_auto_analysis(dataset_id: str) -> dict:
    with WorkflowStore() as store:
        dataset_record = store.get_dataset(dataset_id)
        if dataset_record is None:
            raise ValueError(f"Dataset '{dataset_id}' was not found.")

    return _get_cached_or_compute_auto_analysis(dataset_record)


def get_dataset_summary(dataset_id: str) -> dict:
    with WorkflowStore() as store:
        dataset_record = store.get_dataset(dataset_id)
        if dataset_record is None:
            raise ValueError(f"Dataset '{dataset_id}' was not found.")

    dataframe = _load_dataframe_for_dataset(dataset_record)
    profile = profile_dataset(
        dataframe,
        dataset_name=dataset_record.dataset_name,
        dataset_key=dataset_record.dataset_key,
    )
    schema = extract_schema(dataframe)
    columns = [str(column) for column in list(schema.get("columns", []))]
    dtypes = {
        str(column): str(dtype)
        for column, dtype in dict(schema.get("dtypes") or {}).items()
    }
    auto_config = build_auto_forecast_config(dataframe)
    forecast_eligibility = build_forecast_eligibility(dataframe)
    detected_date_column = str(auto_config.get("date_column") or "") or None
    detected_target = detect_target_details(dataframe)
    detected_target_column = str(detected_target.get("target") or auto_config.get("target") or "") or None
    history_stats = _build_history_stats(dataframe, detected_date_column)
    preview_frame = dataframe.head(12).copy()
    preview_frame = preview_frame.where(pd.notna(preview_frame), None)
    auto_analysis_payload = _get_cached_or_compute_auto_analysis(dataset_record, dataframe)
    question_payload = build_question_payload(
        dataframe,
        source_fingerprint=dataset_record.source_fingerprint,
    )

    return {
        "dataset_id": dataset_record.dataset_id,
        "dataset_name": dataset_record.dataset_name,
        "dataset_key": dataset_record.dataset_key,
        "source_fingerprint": dataset_record.source_fingerprint,
        "source_kind": dataset_record.source_kind,
        "source_label": dataset_record.source_label,
        "created_at": dataset_record.created_at,
        "row_count": profile.row_count,
        "column_count": profile.column_count,
        "missing_cell_count": profile.missing_cell_count,
        "duplicate_row_count": profile.duplicate_row_count,
        "numeric_column_count": profile.numeric_column_count,
        "categorical_column_count": profile.categorical_column_count,
        "datetime_column_count": profile.datetime_column_count,
        "columns": columns,
        "dtypes": dtypes,
        "date_column": detected_date_column,
        "target_column": detected_target_column,
        "auto_config": auto_config,
        "forecast_eligibility": forecast_eligibility,
        "stats": history_stats,
        "preview_columns": [str(column) for column in preview_frame.columns.tolist()],
        "preview_rows": preview_frame.to_dict(orient="records"),
        "auto_analysis": dict(auto_analysis_payload.get("auto_analysis") or {}),
        "suggested_questions": list(question_payload.get("suggested_questions") or []),
    }


def _load_dataframe_for_dataset(dataset_record):
    file_bytes = get_object_store().get_bytes(dataset_record.object_key)
    if dataset_record.source_kind == "csv":
        loaded = load_dataframe_from_source(
            CSVSourceConfig(file_name=dataset_record.dataset_name, file_bytes=file_bytes)
        )
    else:
        loaded = load_dataframe_from_source(
            ExcelSourceConfig(file_name=dataset_record.dataset_name, file_bytes=file_bytes)
        )
    return loaded.dataframe


def _record_run(store: WorkflowStore, dataset_record, workflow_context: dict, analysis_output: dict):
    export_settings = workflow_context.get("export_settings", {})
    export_artifacts: list[str] = []
    if export_settings.get("include_csv", True):
        export_artifacts.append("csv")
    if export_settings.get("include_json", True):
        export_artifacts.append("json")
    if analysis_output.get("generated_code"):
        export_artifacts.append("python")

    run_record = store.build_run_record(
        workflow_id=workflow_context.get("workflow_id"),
        workflow_version=workflow_context.get("workflow_version"),
        workflow_name=workflow_context.get("workflow_name"),
        source_fingerprint=dataset_record.source_fingerprint,
        source_label=dataset_record.source_label,
        validation_findings=list(workflow_context.get("validation_findings", [])),
        cleaning_actions=list(workflow_context.get("cleaning_actions", [])),
        generated_code=analysis_output.get("generated_code"),
        final_status=analysis_output.get("test_status") or "UNKNOWN",
        error_message=analysis_output.get("error"),
        export_artifacts=export_artifacts,
        analysis_query=analysis_output.get("query") or "",
        result_summary=analysis_output.get("summary"),
        result_hash=analysis_output.get("result_hash"),
    )
    store.record_run(run_record)
    return run_record


def _build_analysis_forecast_metadata(df: pd.DataFrame, analysis_output: dict) -> dict | None:
    analysis_plan = dict(analysis_output.get("analysis_plan") or {})
    datetime_column = str(analysis_plan.get("datetime_column") or "")
    if not datetime_column or datetime_column not in df.columns:
        return None

    parsed_dates = pd.to_datetime(df[datetime_column], errors="coerce", format="mixed").dropna().sort_values()
    if parsed_dates.empty:
        return None

    resolved_frequency = infer_frequency_from_dates(parsed_dates)
    frequency_label = {
        "D": "daily",
        "W": "weekly",
        "M": "monthly",
        "Q": "quarterly",
    }.get(str(resolved_frequency or "").strip(), "irregular")
    return {
        "time_column": datetime_column,
        "data_points": int(parsed_dates.shape[0]),
        "frequency": frequency_label,
        "filled_missing_timestamps": 0,
    }


def _enforce_ml_intelligence_contract(df: pd.DataFrame, analysis_output: dict) -> dict:
    updated_output = dict(analysis_output or {})
    raw_ml_output = updated_output.get("ml_intelligence")
    if not raw_ml_output:
        raw_ml_output = dict(updated_output.get("analysis_contract") or {}).get("ml_intelligence")
    if not raw_ml_output:
        return updated_output

    try:
        ml_output = postprocess_ml_output(raw_ml_output, df)
        validate_ml_output(ml_output)
    except Exception as error:
        ml_output = {
            "error": str(error),
            "fallback": "analysis_mode",
        }
        warning_messages = list(updated_output.get("warnings") or [])
        warning_messages.append(f"ML intelligence fallback activated: {error}")
        updated_output["warnings"] = list(dict.fromkeys(warning_messages))

    updated_output["ml_intelligence"] = ml_output
    contract = dict(updated_output.get("analysis_contract") or {})
    contract["ml_intelligence"] = ml_output
    updated_output["analysis_contract"] = contract
    return updated_output


def _augment_analysis_output(df: pd.DataFrame, analysis_output: dict) -> dict:
    updated_output = _enforce_ml_intelligence_contract(df, analysis_output)
    workflow_context = dict(updated_output.get("workflow_context") or {})
    question_payload = build_question_payload(
        df,
        source_fingerprint=str(workflow_context.get("source_fingerprint") or ""),
        recent_queries=[str(updated_output.get("query") or "")] if str(updated_output.get("query") or "").strip() else None,
    )

    dashboard = dict(updated_output.get("dashboard") or {})
    if not dashboard:
        try:
            dashboard_payload = build_dashboard_output(
                str(updated_output.get("query") or ""),
                df,
                result=updated_output.get("result"),
                plan=dict(updated_output.get("analysis_plan") or {}) | {
                    "time_filter": workflow_context.get("time_filter"),
                    "custom_time_range": workflow_context.get("custom_time_range"),
                    "source_fingerprint": workflow_context.get("source_fingerprint"),
                },
                preflight={
                    "warnings": list(updated_output.get("warnings") or []),
                    "blocking_errors": [],
                    "time_filter": workflow_context.get("time_filter"),
                    "custom_time_range": workflow_context.get("custom_time_range"),
                },
            )
            dashboard = dict(dashboard_payload.get("dashboard") or {})
            updated_output.setdefault("warnings", [])
            updated_output["warnings"] = list(
                dict.fromkeys(list(updated_output.get("warnings") or []) + list(dashboard_payload.get("warnings") or []))
            )
            updated_output["domain"] = dashboard_payload.get("domain")
            if dashboard_payload.get("suggested_questions"):
                updated_output["suggested_questions"] = list(dashboard_payload.get("suggested_questions") or [])
        except Exception:
            dashboard = {}
    updated_output["dashboard"] = dashboard or None
    updated_output["active_filter"] = str(
        updated_output.get("active_filter")
        or ((updated_output.get("dashboard") or {}).get("active_filter"))
        or workflow_context.get("time_filter")
        or ""
    ) or None
    updated_output["visualization_type"] = str(
        updated_output.get("visualization_type")
        or ((updated_output.get("dashboard") or {}).get("visualization_type"))
        or ""
    ) or None

    if not updated_output.get("suggested_questions"):
        updated_output["suggested_questions"] = list(question_payload.get("suggested_questions") or [])
    if not updated_output.get("context"):
        updated_output["context"] = dict(question_payload.get("context") or {}) or None
    if not updated_output.get("suggestions"):
        updated_output["suggestions"] = list(question_payload.get("suggestions") or [])
    if updated_output.get("recommended_next_step") is None:
        updated_output["recommended_next_step"] = question_payload.get("recommended_next_step")
    updated_output["forecast_metadata"] = dict(
        updated_output.get("forecast_metadata")
        or _build_analysis_forecast_metadata(df, updated_output)
        or {}
    ) or None

    contract_input = dict(updated_output.get("analysis_contract") or {})
    if updated_output.get("ml_intelligence") and not contract_input.get("ml_intelligence"):
        contract_input["ml_intelligence"] = dict(updated_output.get("ml_intelligence") or {})
    contract = ensure_analysis_contract_defaults(contract_input)
    if updated_output.get("dashboard"):
        contract["dashboard"] = dict(updated_output.get("dashboard") or {})
    if updated_output.get("forecast_metadata"):
        contract["forecast_metadata"] = dict(updated_output.get("forecast_metadata") or {})
    if updated_output.get("system_decision"):
        contract["system_decision"] = dict(updated_output.get("system_decision") or {})
    if updated_output.get("context"):
        contract["context"] = dict(updated_output.get("context") or {})
    if updated_output.get("suggestions"):
        contract["suggestions"] = list(updated_output.get("suggestions") or [])
    if updated_output.get("recommended_next_step") is not None:
        contract["recommended_next_step"] = updated_output.get("recommended_next_step")
    if updated_output.get("suggested_questions"):
        contract["suggested_questions"] = list(updated_output.get("suggested_questions") or [])
    if updated_output.get("active_filter") is not None:
        contract["active_filter"] = updated_output.get("active_filter")
    if updated_output.get("visualization_type") is not None:
        contract["visualization_type"] = updated_output.get("visualization_type")
    if updated_output.get("cleaning_report") or workflow_context.get("cleaning_report"):
        contract["cleaning_report"] = dict(
            updated_output.get("cleaning_report")
            or workflow_context.get("cleaning_report")
            or {}
        )
    if updated_output.get("ml_intelligence"):
        contract["ml_intelligence"] = dict(updated_output.get("ml_intelligence") or {})
    updated_output["analysis_contract"] = ensure_analysis_contract_defaults(contract)
    return updated_output


def _normalize_analysis_output_contract(analysis_output: dict) -> dict:
    updated_output = dict(analysis_output or {})
    contract_input = dict(updated_output.get("analysis_contract") or {})
    if updated_output.get("ml_intelligence") and not contract_input.get("ml_intelligence"):
        contract_input["ml_intelligence"] = dict(updated_output.get("ml_intelligence") or {})
    contract = ensure_analysis_contract_defaults(contract_input)
    if not updated_output.get("result_hash"):
        updated_output["result_hash"] = hash_result(updated_output.get("result"))
        contract["result_hash"] = str(contract.get("result_hash") or updated_output["result_hash"])
    updated_output["analysis_contract"] = contract
    updated_output["system_decision"] = dict(updated_output.get("system_decision") or contract.get("system_decision") or {})
    updated_output.setdefault("summary", contract.get("result_summary"))
    updated_output.setdefault("recommendations", list(contract.get("recommendations") or []))
    updated_output.setdefault("warnings", list(contract.get("warnings") or []))
    updated_output.setdefault("confidence", contract.get("confidence"))
    updated_output["decision_layer"] = ensure_decision_layer_defaults(
        updated_output.get("decision_layer") or contract.get("decision_layer"),
        risk=updated_output.get("risk") or contract.get("risk"),
        recommendations=list(updated_output.get("recommendations") or contract.get("recommendations") or []),
    )
    if not updated_output.get("recommendations"):
        updated_output["recommendations"] = derive_recommendations_from_decision_layer(updated_output["decision_layer"])
    if not updated_output.get("business_decisions"):
        updated_output["business_decisions"] = build_business_decisions_text(updated_output["decision_layer"])
    updated_output["data_quality"] = dict(updated_output.get("data_quality") or contract.get("data_quality") or {"score": 0.0, "issues": [], "profile": {}})
    updated_output["cleaning_report"] = dict(updated_output.get("cleaning_report") or contract.get("cleaning_report") or {})
    updated_output["model_metrics"] = dict(updated_output.get("model_metrics") or contract.get("model_metrics") or {"mae": None, "r2": None})
    updated_output["explanation"] = dict(updated_output.get("explanation") or contract.get("explanation") or {"top_features": [], "impact": []})
    updated_output["ml_intelligence"] = dict(contract.get("ml_intelligence") or updated_output.get("ml_intelligence") or {})
    updated_output["model_quality"] = str(updated_output.get("model_quality") or contract.get("model_quality") or "weak")
    updated_output["risk"] = str(updated_output.get("risk") or contract.get("risk") or "")
    updated_output["result_hash"] = str(updated_output.get("result_hash") or contract.get("result_hash") or "")
    updated_output["dataset_fingerprint"] = str(updated_output.get("dataset_fingerprint") or contract.get("dataset_fingerprint") or "")
    updated_output["tool_used"] = str(updated_output.get("tool_used") or contract.get("tool_used") or "PYTHON")
    updated_output["analysis_mode"] = str(updated_output.get("analysis_mode") or contract.get("analysis_mode") or "ad-hoc")
    updated_output["execution_plan"] = list(contract.get("execution_plan") or updated_output.get("execution_plan") or [])
    updated_output["execution_trace"] = list(contract.get("execution_trace") or updated_output.get("execution_trace") or [])
    updated_output["optimization"] = dict(updated_output.get("optimization") or contract.get("optimization") or {})
    updated_output["excel_analysis"] = dict(updated_output.get("excel_analysis") or contract.get("excel_analysis") or {}) or None
    updated_output["dashboard"] = dict(updated_output.get("dashboard") or contract.get("dashboard") or {}) or None
    updated_output["forecast_metadata"] = dict(updated_output.get("forecast_metadata") or contract.get("forecast_metadata") or {}) or None
    updated_output["context"] = dict(updated_output.get("context") or contract.get("context") or {}) or None
    updated_output["suggestions"] = list(updated_output.get("suggestions") or contract.get("suggestions") or [])
    updated_output["recommended_next_step"] = str(
        updated_output.get("recommended_next_step")
        or contract.get("recommended_next_step")
        or ""
    ) or None
    updated_output["suggested_questions"] = list(updated_output.get("suggested_questions") or contract.get("suggested_questions") or [])
    updated_output["active_filter"] = str(updated_output.get("active_filter") or contract.get("active_filter") or "") or None
    updated_output["visualization_type"] = str(updated_output.get("visualization_type") or contract.get("visualization_type") or "") or None
    updated_output["reproducibility"] = dict(
        updated_output.get("reproducibility")
        or contract.get("reproducibility")
        or {
            "dataset_fingerprint": "",
            "pipeline_trace_hash": "",
            "result_hash": "",
            "consistent_with_prior_runs": True,
            "prior_hash_count": 0,
            "consistency_validated": False,
        }
    )
    updated_output["inconsistency_detected"] = bool(
        updated_output.get("inconsistency_detected")
        if updated_output.get("inconsistency_detected") is not None
        else contract.get("inconsistency_detected")
    )
    updated_output["limitations"] = list(updated_output.get("limitations") or contract.get("limitations") or [])
    updated_output["failure_patterns"] = dict(updated_output.get("failure_patterns") or {})
    trace = []
    for stage in list(updated_output.get("pipeline_trace") or []):
        stage_payload = dict(stage)
        if stage_payload.get("stage") == "contract_execution":
            detail = dict(stage_payload.get("detail") or {})
            detail.setdefault("tool_used", updated_output["tool_used"])
            detail.setdefault("analysis_mode", updated_output["analysis_mode"])
            detail.setdefault("execution_plan", list(updated_output.get("execution_plan") or []))
            detail.setdefault("execution_trace", list(updated_output.get("execution_trace") or []))
            detail.setdefault("optimization", dict(updated_output.get("optimization") or {}))
            stage_payload["detail"] = detail
        trace.append(stage_payload)
    if trace:
        updated_output["pipeline_trace"] = trace
    return updated_output


def _record_decision_history_entries(
    store: WorkflowStore,
    *,
    output: dict,
    query: str,
    source_fingerprint: str | None,
    job_id: str | None = None,
    forecast_artifact_id: str | None = None,
) -> None:
    decision_layer = ensure_decision_layer_defaults(
        output.get("decision_layer")
        or (output.get("analysis_contract") or {}).get("decision_layer"),
        risk=output.get("risk") or ((output.get("analysis_contract") or {}).get("risk")),
        recommendations=list(output.get("recommendations") or []),
    )
    decisions = list(decision_layer.get("decisions") or [])
    if not decisions:
        return

    for decision in decisions:
        record = store.build_decision_history_record(
            job_id=job_id,
            forecast_artifact_id=forecast_artifact_id,
            source_fingerprint=str(source_fingerprint or output.get("dataset_fingerprint") or ""),
            query=query,
            decision=decision,
            decision_confidence=str(decision_layer.get("decision_confidence") or decision.get("confidence") or "low"),
            result_hash=output.get("result_hash"),
        )
        store.record_decision_history(record)


def _apply_runtime_pipeline_state(
    analysis_output: dict,
    *,
    cache_key: str,
    cache_status: str,
    cache_hit: bool,
    memory_status: str,
    run_record=None,
    memory_metadata: dict | None = None,
) -> dict:
    updated_output = _normalize_analysis_output_contract(analysis_output)
    created_at_value = getattr(run_record, "created_at", None)
    if hasattr(created_at_value, "isoformat"):
        created_at_value = created_at_value.isoformat()
    updated_output["cache_hit"] = bool(cache_hit)
    updated_output["cache_status"] = {
        "status": cache_status,
        "cache_key": cache_key,
    }
    updated_output["memory_update"] = {
        "status": memory_status,
        "run_id": getattr(run_record, "run_id", None),
        "workflow_id": getattr(run_record, "workflow_id", None),
        "created_at": created_at_value,
    }
    if memory_metadata:
        updated_output["memory_update"].update(memory_metadata)

    if not updated_output.get("data_score"):
        warning_count = len(updated_output.get("warnings") or [])
        score = max(0, 100 - (warning_count * 8) - (10 if updated_output.get("error") else 0))
        updated_output["data_score"] = {
            "score": int(score),
            "band": "usable" if score >= 70 else "watch",
            "warning_count": warning_count,
            "blocking_issue_count": 1 if updated_output.get("error") else 0,
        }

    trace = []
    existing_trace = list(updated_output.get("pipeline_trace") or [])
    if not existing_trace:
        uses_integrated_ml = (
            any(
                str(step.get("tool") or "").strip().upper() == "PYTHON"
                for step in list(updated_output.get("execution_plan") or [])
            ) or updated_output.get("tool_used") == "PYTHON"
        ) and (
            updated_output.get("intent") == "forecast"
            or (updated_output.get("analysis_contract") or {}).get("intent") == "prediction"
        )
        existing_trace = [
            {
                "stage": "user_query",
                "title": "User Query",
                "status": "completed",
                "detail": {"query": updated_output.get("query")},
            },
            {
                "stage": "intent_detection",
                "title": "Intent Detection",
                "status": "completed",
                "detail": {"legacy_intent": updated_output.get("intent")},
            },
            {
                "stage": "contract_execution",
                "title": "Contract Execution",
                "status": "completed",
                "detail": {
                    "tool_used": updated_output.get("tool_used"),
                    "analysis_mode": updated_output.get("analysis_mode"),
                    "execution_plan": list(updated_output.get("execution_plan") or []),
                    "execution_trace": list(updated_output.get("execution_trace") or []),
                    "optimization": dict(updated_output.get("optimization") or {}),
                },
            },
            {
                "stage": "forecast_ml",
                "title": "Forecast / ML (integrated)",
                "status": "completed" if uses_integrated_ml else "skipped",
                "detail": {"tool_used": updated_output.get("tool_used")},
            },
            {
                "stage": "validation_data_score",
                "title": "Validation + Data Score",
                "status": "failed" if updated_output.get("error") else "completed",
                "detail": {"data_score": updated_output["data_score"]},
            },
            {
                "stage": "execution",
                "title": "Execution",
                "status": "failed" if updated_output.get("error") else "completed",
                "detail": {"error": updated_output.get("error")},
            },
            {
                "stage": "model_evaluation",
                "title": "Model Evaluation",
                "status": "completed" if any(value is not None for value in updated_output.get("model_metrics", {}).values()) else "skipped",
                "detail": {"metrics": dict(updated_output.get("model_metrics") or {"mae": None, "r2": None})},
            },
            {
                "stage": "explainability",
                "title": "Explainability",
                "status": "completed" if list((updated_output.get("explanation") or {}).get("top_features") or []) else "skipped",
                "detail": {"top_features": list((updated_output.get("explanation") or {}).get("top_features") or [])},
            },
            {
                "stage": "ml_intelligence",
                "title": "ML Intelligence",
                "status": "completed" if (updated_output.get("ml_intelligence") or {}).get("target") else "skipped",
                "detail": {
                    "target": (updated_output.get("ml_intelligence") or {}).get("target"),
                    "top_features": list((updated_output.get("ml_intelligence") or {}).get("top_features") or []),
                    "recommendation_count": len((updated_output.get("ml_intelligence") or {}).get("recommendations") or []),
                },
            },
            {
                "stage": "decision_engine",
                "title": "Decision Engine",
                "status": "completed" if list((updated_output.get("decision_layer") or {}).get("decisions") or []) else "skipped",
                "detail": {
                    "decision_count": len((updated_output.get("decision_layer") or {}).get("decisions") or []),
                    "decision_confidence": str((updated_output.get("decision_layer") or {}).get("decision_confidence") or "low"),
                },
            },
            {
                "stage": "learning_engine",
                "title": "Learning Engine",
                "status": "completed" if dict((updated_output.get("decision_layer") or {}).get("learning_insights") or {}) else "skipped",
                "detail": {
                    "pattern_count": len((((updated_output.get("decision_layer") or {}).get("learning_insights") or {}).get("patterns") or [])),
                    "confidence_adjustment": (((updated_output.get("decision_layer") or {}).get("learning_insights") or {}).get("confidence_adjustment")),
                },
            },
            {
                "stage": "insight_decisions",
                "title": "Insight + Decisions",
                "status": "completed" if updated_output.get("summary") else "pending",
                "detail": {},
            },
            {
                "stage": "failure_logging",
                "title": "Failure Logging",
                "status": "completed" if updated_output.get("error") else "skipped",
                "detail": {"logged_failures": 1 if updated_output.get("error") else 0},
            },
            {
                "stage": "consistency_check",
                "title": "Consistency Check",
                "status": "completed" if updated_output.get("result_hash") else "skipped",
                "detail": {
                    "result_hash": updated_output.get("result_hash"),
                    "inconsistency_detected": bool(updated_output.get("inconsistency_detected")),
                    "limitations": list(updated_output.get("limitations") or []),
                },
            },
            {"stage": "caching", "title": "Caching", "status": "pending", "detail": {"status": "pending"}},
            {"stage": "memory_update", "title": "Memory Update", "status": "pending", "detail": {"status": "pending"}},
        ]

    for stage in existing_trace:
        stage_payload = dict(stage)
        if stage_payload.get("stage") == "caching":
            stage_payload["status"] = cache_status
            stage_payload["detail"] = dict(updated_output["cache_status"])
        elif stage_payload.get("stage") == "memory_update":
            stage_payload["status"] = memory_status
            stage_payload["detail"] = dict(updated_output["memory_update"])
        trace.append(stage_payload)
    if trace:
        updated_output["pipeline_trace"] = trace

    return updated_output


def _record_forecast_artifact(store: WorkflowStore, dataset_record, workflow_context: dict, forecast_output: dict) -> dict:
    artifact_metadata = dict(forecast_output.get("artifact_metadata") or {})
    if not artifact_metadata.get("artifact_key"):
        artifact_key, artifact_metadata = persist_forecast_artifact(
            forecast_output,
            workflow_id=workflow_context.get("workflow_id"),
            workflow_version=workflow_context.get("workflow_version"),
            source_fingerprint=dataset_record.source_fingerprint,
            dataset_name=dataset_record.dataset_name,
        )
        artifact_metadata["artifact_key"] = artifact_key

    record = store.build_forecast_artifact_record(
        workflow_id=workflow_context.get("workflow_id"),
        workflow_version=workflow_context.get("workflow_version"),
        workflow_name=workflow_context.get("workflow_name"),
        source_fingerprint=dataset_record.source_fingerprint,
        source_label=dataset_record.source_label,
        target_column=artifact_metadata.get("target_column") or forecast_output.get("config", {}).get("target_column"),
        horizon=artifact_metadata.get("horizon") or forecast_output.get("horizon"),
        model_name=artifact_metadata.get("model_name") or forecast_output.get("chosen_model"),
        training_mode=forecast_output.get("config", {}).get("training_mode") or "auto",
        status=artifact_metadata.get("status") or forecast_output.get("status") or "UNKNOWN",
        artifact_key=artifact_metadata.get("artifact_key") or "",
        forecast_config=forecast_output.get("config", {}),
        evaluation_metrics=artifact_metadata.get("evaluation_metrics") or forecast_output.get("evaluation_metrics", {}),
        recommendation_payload=artifact_metadata.get("recommendations") or forecast_output.get("recommendations", []),
        summary=artifact_metadata.get("summary") or forecast_output.get("summary"),
        result_hash=forecast_output.get("result_hash"),
    )
    store.record_forecast_artifact(record)
    artifact_metadata["artifact_id"] = record.artifact_id
    return artifact_metadata


def _should_record_forecast_artifact(forecast_output: dict) -> bool:
    return (
        str(forecast_output.get("status") or "").upper() == "PASSED"
        and bool(forecast_output.get("artifact_payload"))
    )


def submit_analysis_job(
    dataset_id: str,
    query: str,
    workflow_context: dict | None = None,
    *,
    user_id: str | None = None,
):
    with WorkflowStore() as store:
        dataset_record = store.get_dataset(dataset_id)
        if dataset_record is None:
            raise ValueError(f"Dataset '{dataset_id}' was not found.")

        intent = detect_intent(query)
        cache_key = _build_cache_key(dataset_record.source_fingerprint, query, intent, workflow_context)
        cached_output = get_cache_store().get_json(cache_key)
        if cached_output is not None:
            cached_analysis_output = _normalize_analysis_output_contract(deserialize_analysis_output(cached_output))
            run_record = _record_run(store, dataset_record, workflow_context or {}, cached_analysis_output)
            finalized_cached_output = _apply_runtime_pipeline_state(
                cached_analysis_output,
                cache_key=cache_key,
                cache_status="hit",
                cache_hit=True,
                memory_status="replayed",
                run_record=run_record,
            )
            record_user_interaction_memory(
                source_fingerprint=dataset_record.source_fingerprint,
                dataset_type=((finalized_cached_output.get("context") or {}).get("dataset_type") or (finalized_cached_output.get("context") or {}).get("domain")),
                query=query,
                successful_action=query if not finalized_cached_output.get("error") else None,
            )
            serialized_cached_output = serialize_analysis_output(finalized_cached_output)
            job = store.create_job(
                dataset_id=dataset_id,
                query=query,
                intent=intent,
                workflow_context=workflow_context or {},
                cache_key=cache_key,
                status="completed",
                analysis_output=serialized_cached_output,
                result_summary=str(serialized_cached_output.get("summary") or ""),
                cache_hit=True,
                user_id=user_id,
            )
            _record_decision_history_entries(
                store,
                output=serialized_cached_output,
                query=query,
                source_fingerprint=dataset_record.source_fingerprint,
                job_id=job.job_id,
            )
            return job

        job = store.create_job(
            dataset_id=dataset_id,
            query=query,
            intent=intent,
            workflow_context=workflow_context or {},
            cache_key=cache_key,
            user_id=user_id,
        )
        from .queueing import enqueue_analysis_job

        enqueue_analysis_job(job.job_id)
        return store.get_job(job.job_id)


def submit_forecast_job(
    dataset_id: str,
    forecast_config: dict,
    workflow_context: dict | None = None,
    *,
    user_id: str | None = None,
):
    normalized_config = forecast_config_from_dict(forecast_config)
    serialized_config = forecast_config_to_dict(normalized_config)
    query_label = _build_forecast_query_label(serialized_config)

    with WorkflowStore() as store:
        dataset_record = store.get_dataset(dataset_id)
        if dataset_record is None:
            raise ValueError(f"Dataset '{dataset_id}' was not found.")

        forecast_workflow_context = dict(workflow_context or {})
        forecast_workflow_context["forecast_config"] = serialized_config
        cache_key = _build_forecast_cache_key(
            dataset_record.source_fingerprint,
            serialized_config,
            forecast_workflow_context,
        )
        cached_output = get_cache_store().get_json(cache_key)

        if cached_output is not None:
            deserialized_output = deserialize_forecast_output(cached_output)
            artifact_metadata = (
                _record_forecast_artifact(store, dataset_record, forecast_workflow_context, deserialized_output)
                if _should_record_forecast_artifact(deserialized_output)
                else {}
            )
            deserialized_output["artifact_metadata"] = artifact_metadata
            finalized_cached_output = _apply_runtime_pipeline_state(
                deserialized_output,
                cache_key=cache_key,
                cache_status="hit",
                cache_hit=True,
                memory_status="replayed",
                run_record=None,
                memory_metadata={"artifact_id": artifact_metadata.get("artifact_id")},
            )
            record_user_interaction_memory(
                source_fingerprint=dataset_record.source_fingerprint,
                dataset_type=((finalized_cached_output.get("context") or {}).get("dataset_type") or (finalized_cached_output.get("context") or {}).get("domain")),
                query=query_label,
                successful_action=query_label if not finalized_cached_output.get("error") else None,
            )
            job = store.create_job(
                dataset_id=dataset_id,
                query=query_label,
                intent="forecast",
                workflow_context=forecast_workflow_context,
                cache_key=cache_key,
                status="completed",
                analysis_output=serialize_forecast_output(finalized_cached_output),
                result_summary=str(finalized_cached_output.get("summary") or ""),
                cache_hit=True,
                user_id=user_id,
            )
            updated_job = store.complete_job(
                job.job_id,
                analysis_output=serialize_forecast_output(finalized_cached_output),
                result_summary=str(finalized_cached_output.get("summary") or ""),
                cache_hit=True,
            )
            _record_decision_history_entries(
                store,
                output=finalized_cached_output,
                query=query_label,
                source_fingerprint=dataset_record.source_fingerprint,
                job_id=(updated_job or job).job_id,
                forecast_artifact_id=artifact_metadata.get("artifact_id"),
            )
            return updated_job or job

        job = store.create_job(
            dataset_id=dataset_id,
            query=query_label,
            intent="forecast",
            workflow_context=forecast_workflow_context,
            cache_key=cache_key,
            user_id=user_id,
        )
        from .queueing import enqueue_forecast_job

        enqueue_forecast_job(job.job_id)
        return store.get_job(job.job_id)


def process_analysis_job(job_id: str):
    with WorkflowStore() as store:
        job_record = store.get_job(job_id)
        if job_record is None:
            raise ValueError(f"Job '{job_id}' was not found.")

        dataset_record = store.get_dataset(job_record.dataset_id)
        if dataset_record is None:
            raise ValueError(f"Dataset '{job_record.dataset_id}' was not found.")

        running_job = store.mark_job_running(job_id) or job_record
        if running_job.queue_wait_ms is not None:
            observe_queue_wait(running_job.intent, running_job.queue_wait_ms / 1000)

        started = time.perf_counter()
        try:
            dataframe = _load_dataframe_for_dataset(dataset_record)
            runtime_workflow_context = dict(running_job.workflow_context or {})
            runtime_workflow_context.setdefault("dataset_id", dataset_record.dataset_id)
            runtime_workflow_context.setdefault("source_fingerprint", dataset_record.source_fingerprint)
            runtime_workflow_context.setdefault("source_label", dataset_record.source_label)
            analysis_output = run_builder_pipeline(
                running_job.query,
                dataframe,
                workflow_context=runtime_workflow_context,
            )
            analysis_output = _augment_analysis_output(dataframe, analysis_output)
            analysis_output = _normalize_analysis_output_contract(analysis_output)
            cache_key = running_job.cache_key or _build_cache_key(
                dataset_record.source_fingerprint,
                running_job.query,
                running_job.intent,
                runtime_workflow_context,
            )
            cached_analysis_output = _apply_runtime_pipeline_state(
                analysis_output,
                cache_key=cache_key,
                cache_status="stored",
                cache_hit=False,
                memory_status="pending",
                run_record=None,
            )
            serialized_output = serialize_analysis_output(cached_analysis_output)
            get_cache_store().set_json(cache_key, serialized_output)
            run_record = _record_run(store, dataset_record, runtime_workflow_context, cached_analysis_output)
            finalized_analysis_output = _apply_runtime_pipeline_state(
                cached_analysis_output,
                cache_key=cache_key,
                cache_status="stored",
                cache_hit=False,
                memory_status="recorded",
                run_record=run_record,
            )
            serialized_output = serialize_analysis_output(finalized_analysis_output)
            get_cache_store().set_json(cache_key, serialized_output)
            record_user_interaction_memory(
                source_fingerprint=dataset_record.source_fingerprint,
                dataset_type=((finalized_analysis_output.get("context") or {}).get("dataset_type") or (finalized_analysis_output.get("context") or {}).get("domain")),
                query=running_job.query,
                successful_action=running_job.query if not finalized_analysis_output.get("error") else None,
            )
            store.complete_job(
                job_id,
                analysis_output=serialized_output,
                result_summary=finalized_analysis_output.get("summary"),
            )
            _record_decision_history_entries(
                store,
                output=finalized_analysis_output,
                query=running_job.query,
                source_fingerprint=dataset_record.source_fingerprint,
                job_id=job_id,
            )
            observe_job_completion(running_job.intent, time.perf_counter() - started, success=not bool(finalized_analysis_output.get("error")))
            return serialized_output
        except Exception as error:
            LOGGER.exception("analysis job failed", extra={"component": "worker", "job_id": job_id})
            log_failure(
                running_job.query,
                error,
                "analysis_service",
                store=store,
                metadata={"job_id": job_id, "dataset_id": dataset_record.dataset_id},
            )
            store.fail_job(job_id, error_message=str(error))
            observe_job_completion(running_job.intent, time.perf_counter() - started, success=False)
            raise


def process_forecast_job(job_id: str):
    with WorkflowStore() as store:
        job_record = store.get_job(job_id)
        if job_record is None:
            raise ValueError(f"Job '{job_id}' was not found.")

        dataset_record = store.get_dataset(job_record.dataset_id)
        if dataset_record is None:
            raise ValueError(f"Dataset '{job_record.dataset_id}' was not found.")

        running_job = store.mark_job_running(job_id) or job_record
        if running_job.queue_wait_ms is not None:
            observe_queue_wait("forecast", running_job.queue_wait_ms / 1000)

        started = time.perf_counter()
        try:
            dataframe = _load_dataframe_for_dataset(dataset_record)
            forecast_config = running_job.workflow_context.get("forecast_config") or {}
            runtime_workflow_context = dict(running_job.workflow_context or {})
            runtime_workflow_context.setdefault("dataset_id", dataset_record.dataset_id)
            runtime_workflow_context.setdefault("source_fingerprint", dataset_record.source_fingerprint)
            runtime_workflow_context.setdefault("source_label", dataset_record.source_label)
            forecast_output = run_forecast_pipeline(
                dataframe,
                forecast_config,
                workflow_context=runtime_workflow_context,
            )
            artifact_metadata = (
                _record_forecast_artifact(
                    store,
                    dataset_record,
                    runtime_workflow_context,
                    forecast_output,
                )
                if _should_record_forecast_artifact(forecast_output)
                else {}
            )
            forecast_output["artifact_metadata"] = artifact_metadata
            cache_key = running_job.cache_key or _build_forecast_cache_key(
                dataset_record.source_fingerprint,
                forecast_output.get("config", {}),
                runtime_workflow_context,
            )
            cached_forecast_output = _apply_runtime_pipeline_state(
                forecast_output,
                cache_key=cache_key,
                cache_status="stored",
                cache_hit=False,
                memory_status="recorded",
                run_record=None,
                memory_metadata={"artifact_id": artifact_metadata.get("artifact_id")},
            )
            serialized_output = serialize_forecast_output(cached_forecast_output)
            get_cache_store().set_json(cache_key, serialized_output)
            record_user_interaction_memory(
                source_fingerprint=dataset_record.source_fingerprint,
                dataset_type=((cached_forecast_output.get("context") or {}).get("dataset_type") or (cached_forecast_output.get("context") or {}).get("domain")),
                query=running_job.query,
                successful_action=running_job.query if not cached_forecast_output.get("error") else None,
            )
            store.complete_job(
                job_id,
                analysis_output=serialized_output,
                result_summary=cached_forecast_output.get("summary"),
            )
            _record_decision_history_entries(
                store,
                output=cached_forecast_output,
                query=running_job.query,
                source_fingerprint=dataset_record.source_fingerprint,
                job_id=job_id,
                forecast_artifact_id=artifact_metadata.get("artifact_id"),
            )
            observe_job_completion("forecast", time.perf_counter() - started, success=not bool(cached_forecast_output.get("error")))
            return serialized_output
        except Exception as error:
            LOGGER.exception("forecast job failed", extra={"component": "worker", "job_id": job_id})
            log_failure(
                running_job.query,
                error,
                "forecast_service",
                store=store,
                metadata={"job_id": job_id, "dataset_id": dataset_record.dataset_id},
            )
            store.fail_job(job_id, error_message=str(error))
            observe_job_completion("forecast", time.perf_counter() - started, success=False)
            raise


def build_result_artifact(job_record) -> tuple[bytes, str, str]:
    analysis_output = job_record.analysis_output or {}
    result_payload = analysis_output.get("result")
    if not result_payload:
        return b"", "text/plain", "analysis_result.txt"

    kind = result_payload.get("kind")
    if kind == "dataframe":
        frame = pd.DataFrame(result_payload.get("records", []), columns=result_payload.get("columns"))
        return frame.to_csv(index=False).encode("utf-8"), "text/csv", "analysis_result.csv"
    if kind == "series":
        series = pd.Series(result_payload.get("values", []), index=result_payload.get("index", []), name=result_payload.get("name"))
        return series.to_frame().to_csv(index=True).encode("utf-8"), "text/csv", "analysis_result.csv"
    if kind == "json":
        return json.dumps(result_payload.get("value"), indent=2, default=str).encode("utf-8"), "application/json", "analysis_result.json"

    return str(result_payload.get("value")).encode("utf-8"), "text/plain", "analysis_result.txt"
