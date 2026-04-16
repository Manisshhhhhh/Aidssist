from __future__ import annotations

import json
from collections import Counter
from typing import Any

import pandas as pd

from backend.analysis_contract import (
    build_analysis_contract,
    build_analysis_plan,
    classify_analysis_intent,
    ensure_analysis_contract_defaults,
    validate_analysis_request,
)
from backend.services.failure_logging import log_failure
from backend.services.limitations import build_limitations
from backend.services.result_consistency import build_solve_consistency
from .config import get_settings
from .feedback import record_retrieval_feedback
from .ingestion import load_dataset_dataframe, load_derived_dataset_dataframe
from .logging_utils import get_logger
from .refinement import bounded_refinement_loop
from .retrieval import retrieve_workspace_context
from .serialization import serialize_analysis_output, serialize_result
from .validator import validate_solver_output
from backend import prompt_pipeline
from backend.dashboard_helpers import profile_dataset
from backend.workflow_store import WorkflowStore


LOGGER = get_logger(__name__)
VALID_ROUTES = {"data", "code", "hybrid"}


def _keyword_score(text: str, keywords: tuple[str, ...]) -> float:
    lowered = str(text or "").lower()
    return float(sum(1 for keyword in keywords if keyword in lowered))


def _route_history_scores(store: WorkflowStore, workspace_id: str) -> dict[str, float]:
    scores = {route: 0.0 for route in VALID_ROUTES}
    for run in store.list_workspace_solve_runs(workspace_id, limit=100):
        if run.route not in scores:
            continue
        if run.status == "completed":
            scores[run.route] += 0.25
        elif run.status == "failed":
            scores[run.route] -= 0.05
    return scores


def classify_solver_route(
    *,
    store: WorkflowStore,
    workspace_id: str,
    query: str,
    route_hint: str | None = None,
) -> tuple[str, dict[str, float]]:
    normalized_hint = str(route_hint or "").strip().lower()
    history_scores = _route_history_scores(store, workspace_id)
    if normalized_hint in VALID_ROUTES:
        history_scores[normalized_hint] += 10.0
        return normalized_hint, history_scores

    assets = store.list_workspace_assets(workspace_id, limit=200)
    asset_kinds = Counter(asset.asset_kind for asset in assets)
    scores = {route: history_scores.get(route, 0.0) for route in VALID_ROUTES}

    data_keywords = (
        "dataset",
        "csv",
        "excel",
        "column",
        "row",
        "chart",
        "forecast",
        "trend",
        "anomaly",
        "segment",
        "clean",
        "analyze",
        "analysis",
        "table",
        "metric",
    )
    code_keywords = (
        "code",
        "project",
        "bug",
        "error",
        "fix",
        "refactor",
        "schema",
        "api",
        "backend",
        "frontend",
        "python",
        "typescript",
        "javascript",
        "sql",
        "yaml",
        "redesign",
        "architecture",
    )

    scores["data"] += _keyword_score(query, data_keywords)
    scores["code"] += _keyword_score(query, code_keywords)

    if asset_kinds.get("table"):
        scores["data"] += 2.0
    if asset_kinds.get("code") or asset_kinds.get("document"):
        scores["code"] += 1.5
    if asset_kinds.get("mixed") or (scores["data"] > 0 and scores["code"] > 0):
        scores["hybrid"] += 2.0
    if asset_kinds.get("table") and (asset_kinds.get("code") or asset_kinds.get("document")):
        scores["hybrid"] += 1.5

    if not any(score > 0 for score in scores.values()):
        scores["data"] = 1.0 if asset_kinds.get("table") else 0.5
        scores["code"] = 1.0 if asset_kinds.get("code") else 0.5
        scores["hybrid"] = 0.75 if asset_kinds.get("mixed") else 0.0

    chosen_route = max(scores.items(), key=lambda item: item[1])[0]
    return chosen_route, scores


def _resolve_dataset_bundle(
    *,
    store: WorkflowStore,
    workspace_id: str,
    asset_id: str | None,
    requested_dataset_id: str | None,
) -> dict[str, Any] | None:
    derived_candidates = store.list_workspace_derived_datasets(workspace_id, limit=200)
    if asset_id:
        derived_candidates = [item for item in derived_candidates if item.asset_id == asset_id]
    if requested_dataset_id:
        requested_matches = [
            item for item in derived_candidates if item.parent_dataset_id == requested_dataset_id
        ]
        if requested_matches:
            selected = requested_matches[0]
            dataframe = load_derived_dataset_dataframe(selected)
            return {
                "kind": "derived",
                "dataframe": dataframe,
                "dataset_id": requested_dataset_id,
                "derived_dataset_id": selected.derived_dataset_id,
                "dataset_name": selected.dataset_name,
                "source_fingerprint": selected.source_fingerprint,
                "row_count": selected.row_count,
                "column_count": selected.column_count,
                "preview_columns": selected.preview_columns,
            }

    if derived_candidates:
        selected = derived_candidates[0]
        dataframe = load_derived_dataset_dataframe(selected)
        return {
            "kind": "derived",
            "dataframe": dataframe,
            "dataset_id": selected.parent_dataset_id,
            "derived_dataset_id": selected.derived_dataset_id,
            "dataset_name": selected.dataset_name,
            "source_fingerprint": selected.source_fingerprint,
            "row_count": selected.row_count,
            "column_count": selected.column_count,
            "preview_columns": selected.preview_columns,
        }

    dataset = None
    if requested_dataset_id:
        dataset = store.get_dataset(requested_dataset_id)
    elif asset_id:
        asset = store.get_asset(asset_id)
        if asset and asset.primary_dataset_id:
            dataset = store.get_dataset(asset.primary_dataset_id)
    else:
        for asset in store.list_workspace_assets(workspace_id, limit=200):
            if asset.primary_dataset_id:
                dataset = store.get_dataset(asset.primary_dataset_id)
                if dataset is not None:
                    break

    if dataset is None:
        return None

    dataframe = load_dataset_dataframe(dataset)
    return {
        "kind": "source",
        "dataframe": dataframe,
        "dataset_id": dataset.dataset_id,
        "derived_dataset_id": None,
        "dataset_name": dataset.dataset_name,
        "source_fingerprint": dataset.source_fingerprint,
        "row_count": int(len(dataframe)),
        "column_count": int(len(dataframe.columns)),
        "preview_columns": [str(column) for column in dataframe.columns.tolist()[:12]],
    }


def _build_ml_modules(df: pd.DataFrame) -> list[dict[str, str]]:
    profile = profile_dataset(df, dataset_name="workspace", dataset_key="workspace")
    modules: list[dict[str, str]] = []
    if profile.datetime_column_count and profile.numeric_column_count:
        modules.append(
            {
                "name": "Forecast Engine",
                "mode": "time-series",
                "description": "A time column and numeric metrics are present, so forecasting can be enabled.",
            }
        )
    if profile.numeric_column_count >= 3:
        modules.append(
            {
                "name": "Anomaly Radar",
                "mode": "outlier-detection",
                "description": "The dataset has enough numeric depth for anomaly and risk scoring.",
            }
        )
    if profile.categorical_column_count and profile.numeric_column_count:
        modules.append(
            {
                "name": "Recommendation Layer",
                "mode": "segmentation",
                "description": "Categorical and numeric fields can be combined for segment-aware guidance.",
            }
        )
    if not modules:
        modules.append(
            {
                "name": "Schema Profiler",
                "mode": "profiling",
                "description": "The workspace remains eligible for schema checks and deterministic validation.",
            }
        )
    return modules


def _build_deterministic_data_candidate(
    *,
    store: WorkflowStore,
    query: str,
    dataset_bundle: dict[str, Any],
    retrieval_trace: dict[str, Any],
    failure_reason: str | None,
) -> dict[str, Any]:
    df = dataset_bundle["dataframe"]
    profile = profile_dataset(
        df,
        dataset_name=dataset_bundle["dataset_name"],
        dataset_key=str(dataset_bundle.get("source_fingerprint") or "workspace"),
    )
    analysis_intent = classify_analysis_intent(query)
    analysis_plan = build_analysis_plan(query, df, intent=analysis_intent)
    preflight = validate_analysis_request(query, df, analysis_plan)
    preview = df.head(25).copy()
    summary_parts = [
        f"{dataset_bundle['dataset_name']} contains {profile.row_count:,} rows and {profile.column_count:,} columns.",
        f"The workspace query was: {query}.",
    ]
    if profile.missing_cell_count:
        summary_parts.append(f"{profile.missing_cell_count:,} missing cells were detected.")
    if profile.duplicate_row_count:
        summary_parts.append(f"{profile.duplicate_row_count:,} duplicate rows remain in scope.")
    if failure_reason:
        summary_parts.append("The deterministic path was used because the LLM builder was unavailable.")

    consistency = build_solve_consistency(
        store=store,
        result=preview.to_dict(orient="records"),
        source_fingerprint=dataset_bundle.get("source_fingerprint"),
        query=query,
        route="data",
    )
    limitations = build_limitations(
        query=query,
        result=preview,
        df=df,
        warnings=list(preflight.get("warnings") or []),
        data_score=prompt_pipeline._build_data_score(preflight),  # type: ignore[attr-defined]
        model_metrics={"mae": None, "r2": None},
        explanation={"top_features": [], "impact": []},
        inconsistency_detected=bool(consistency.get("inconsistency_detected")),
        use_llm=False,
        store=store,
        metadata={"route": "data", "dataset_id": dataset_bundle.get("dataset_id")},
    )

    analysis_contract = build_analysis_contract(
        query=query,
        df=df,
        result=preview,
        executed_code="result = df.head(25).copy()",
        plan=analysis_plan,
        preflight=preflight,
        method="deterministic_fallback",
        result_hash=consistency.get("result_hash"),
        dataset_fingerprint=dataset_bundle.get("source_fingerprint"),
        inconsistency_detected=bool(consistency.get("inconsistency_detected")),
        limitations=limitations,
    )
    analysis_contract = ensure_analysis_contract_defaults(analysis_contract)
    if failure_reason:
        warning_messages = list(analysis_contract.get("warnings") or [])
        warning_messages.append(str(failure_reason))
        analysis_contract["warnings"] = list(dict.fromkeys(warning_messages))
        analysis_contract["result_summary"] = " ".join(summary_parts)

    analysis_output = {
        "query": query,
        "intent": "general",
        "analysis_contract": analysis_contract,
        "system_decision": dict(analysis_contract.get("system_decision") or {}),
        "analysis_plan": analysis_plan,
        "summary": analysis_contract["result_summary"],
        "insights": "\n".join(analysis_contract["insights"]),
        "recommendations": list(analysis_contract["recommendations"]),
        "warnings": list(analysis_contract["warnings"]),
        "confidence": analysis_contract["confidence"],
        "decision_layer": dict(
            analysis_contract.get("decision_layer")
            or {
                "decisions": [],
                "top_decision": None,
                "decision_confidence": "low",
                "risk_summary": str(analysis_contract.get("risk") or ""),
            }
        ),
        "data_quality": dict(analysis_contract.get("data_quality") or {"score": 0.0, "issues": [], "profile": {}}),
        "data_score": prompt_pipeline._build_data_score(preflight),  # type: ignore[attr-defined]
        "pipeline_trace": prompt_pipeline._build_pipeline_trace(  # type: ignore[attr-defined]
            query=query,
            detected_intent="general",
            analysis_plan=analysis_plan,
            analysis_contract=analysis_contract,
            preflight=preflight,
            method="deterministic_fallback",
            error=None,
            data_quality=analysis_contract.get("data_quality"),
            model_quality=analysis_contract.get("model_quality"),
            risk=analysis_contract.get("risk"),
        ),
        "cache_status": {"status": "pending"},
        "memory_update": {"status": "pending"},
        "model_metrics": dict(analysis_contract.get("model_metrics") or {"mae": None, "r2": None}),
        "explanation": dict(analysis_contract.get("explanation") or {"top_features": [], "impact": []}),
        "ml_intelligence": dict(analysis_contract.get("ml_intelligence") or {}),
        "model_quality": str(analysis_contract.get("model_quality") or "weak"),
        "risk": str(analysis_contract.get("risk") or ""),
        "dataset_fingerprint": str(analysis_contract.get("dataset_fingerprint") or ""),
        "reproducibility": dict(analysis_contract.get("reproducibility") or {}),
        "result_hash": analysis_contract.get("result_hash"),
        "inconsistency_detected": bool(analysis_contract.get("inconsistency_detected")),
        "limitations": list(analysis_contract.get("limitations") or []),
        "business_decisions": "\n".join(analysis_contract["recommendations"]),
        "result": preview,
        "error": None,
        "workflow_context": {"solver_fallback": True},
    }
    return {
        "summary": analysis_output["summary"],
        "pipeline_output": analysis_output,
        "solution_markdown": "",
        "redesign_recommendations": [
            {
                "title": "Profile the active schema",
                "detail": "Review missingness, duplicate rows, and column types before applying heavy transforms.",
                "priority": "medium",
            },
            {
                "title": "Use derived datasets",
                "detail": "Version transformations instead of overwriting the uploaded source dataset.",
                "priority": "high",
            },
        ],
        "ml_modules": _build_ml_modules(df),
    }


def _build_data_candidate(
    *,
    store: WorkflowStore,
    query: str,
    dataset_bundle: dict[str, Any],
    retrieval_trace: dict[str, Any],
    feedback: str | None,
) -> dict[str, Any]:
    df = dataset_bundle["dataframe"]
    settings = get_settings()
    try:
        pipeline_output = prompt_pipeline.run_builder_pipeline(
            query if not feedback else f"{query}\n\nValidator feedback to address:\n{feedback}",
            df,
            max_retries=settings.max_solver_retries,
            model=prompt_pipeline.DEFAULT_GEMINI_MODEL,
            workflow_context={
                "solver_route": "data",
                "retrieved_chunk_ids": [item["chunk_id"] for item in retrieval_trace.get("items", [])],
                "dataset_kind": dataset_bundle["kind"],
                "dataset_id": dataset_bundle.get("dataset_id"),
                "source_fingerprint": dataset_bundle.get("source_fingerprint"),
            },
        )
        pipeline_output["analysis_contract"] = ensure_analysis_contract_defaults(pipeline_output.get("analysis_contract"))
        if pipeline_output.get("error") and pipeline_output.get("result") is None:
            return _build_deterministic_data_candidate(
                store=store,
                query=query,
                dataset_bundle=dataset_bundle,
                retrieval_trace=retrieval_trace,
                failure_reason=str(pipeline_output.get("error")),
            )
        summary = (
            str((pipeline_output.get("analysis_contract") or {}).get("result_summary") or "").strip()
            or str(pipeline_output.get("summary") or "").strip()
            or str(pipeline_output.get("insights") or "").strip()
            or f"Analysis completed for {dataset_bundle['dataset_name']}."
        )
        return {
            "summary": summary,
            "pipeline_output": pipeline_output,
            "solution_markdown": "",
            "redesign_recommendations": [
                {
                    "title": "Persist analysis context",
                    "detail": "Keep this solve run linked to the workspace so future retrieval can reuse it.",
                    "priority": "medium",
                }
            ],
            "ml_modules": _build_ml_modules(df),
        }
    except Exception as error:
        LOGGER.warning(
            "solver data route fell back to deterministic mode",
            extra={"component": "solver", "error": str(error)},
        )
        log_failure(
            query,
            error,
            "solver_data_candidate",
            store=store,
            metadata={"dataset_id": dataset_bundle.get("dataset_id")},
        )
        return _build_deterministic_data_candidate(
            store=store,
            query=query,
            dataset_bundle=dataset_bundle,
            retrieval_trace=retrieval_trace,
            failure_reason=str(error),
        )


def _extract_json_payload(raw_text: str) -> dict[str, Any] | None:
    text = str(raw_text or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    fenced = text.strip("`")
    if fenced.startswith("json"):
        fenced = fenced[4:].strip()
    try:
        return json.loads(fenced)
    except json.JSONDecodeError:
        return None


def _build_deterministic_solution_candidate(
    *,
    route: str,
    query: str,
    retrieval_trace: dict[str, Any],
    dataset_bundle: dict[str, Any] | None,
    failure_reason: str | None,
) -> dict[str, Any]:
    top_items = retrieval_trace.get("items", [])[:4]
    redesign_recommendations = []
    for index, item in enumerate(top_items, start=1):
        redesign_recommendations.append(
            {
                "title": f"Redesign track {index}",
                "detail": f"Use {item.get('title', 'retrieved context')} as the anchor for the next implementation slice.",
                "priority": "high" if index == 1 else "medium",
            }
        )
    if not redesign_recommendations:
        redesign_recommendations.append(
            {
                "title": "Seed the workspace with more project context",
                "detail": "Upload the relevant source files or archives so retrieval can ground the redesign loop.",
                "priority": "high",
            }
        )

    dataset_note = ""
    if dataset_bundle is not None:
        dataset_note = (
            f"\n\nData companion: `{dataset_bundle['dataset_name']}` is available with "
            f"{dataset_bundle['row_count']:,} rows and {dataset_bundle['column_count']:,} columns."
        )

    failure_note = ""
    if failure_reason:
        failure_note = f"\n\nFallback note: {failure_reason}"

    return {
        "summary": f"The {route} solver prepared a grounded redesign plan for the request: {query}.",
        "solution_markdown": (
            "### Proposed solution\n"
            "1. Stabilize ingestion and chunking boundaries so each file type has a deterministic path.\n"
            "2. Persist retrieval traces, validator reports, and derived outputs for replay and ranking.\n"
            "3. Apply the redesign in small, testable slices and use validator feedback to tighten each retry."
            f"{dataset_note}{failure_note}"
        ),
        "redesign_recommendations": redesign_recommendations,
        "validator_guidance": [
            "Run deterministic logic checks before trusting a generated patch.",
            "Keep data and code changes versioned so the refinement loop can compare outcomes.",
        ],
        "implementation_outline": [
            "ingestion -> chunking -> embeddings -> retrieval",
            "reasoner -> validator -> bounded refinement",
            "packaged output -> timeline -> replay memory",
        ],
        "ml_modules": _build_ml_modules(dataset_bundle["dataframe"]) if dataset_bundle is not None else [],
    }


def _build_solution_candidate(
    *,
    store: WorkflowStore,
    route: str,
    query: str,
    retrieval_trace: dict[str, Any],
    dataset_bundle: dict[str, Any] | None,
    feedback: str | None,
) -> dict[str, Any]:
    context_lines = []
    for item in retrieval_trace.get("items", [])[:6]:
        context_lines.append(
            f"- {item.get('title', 'Context')} ({item.get('confidence', 'medium')} confidence): {item.get('excerpt', '')}"
        )

    dataset_context = "No tabular dataset is attached to this solve run."
    if dataset_bundle is not None:
        dataset_context = (
            f"Dataset companion: {dataset_bundle['dataset_name']} with "
            f"{dataset_bundle['row_count']:,} rows and {dataset_bundle['column_count']:,} columns."
        )

    prompt = "\n".join(
        (
            "You are the solution planner for Aidssist Solver Platform v1.",
            "Return valid JSON only.",
            "Required keys: summary, solution_markdown, redesign_recommendations, validator_guidance, implementation_outline.",
            "redesign_recommendations must be an array of objects with title, detail, priority.",
            f"Route: {route}",
            f"User request: {query}",
            f"{dataset_context}",
            "Retrieved workspace context:",
            "\n".join(context_lines) or "- No retrieved chunks were available.",
            f"Validator feedback to address: {feedback or 'None'}",
        )
    )

    try:
        raw_text = prompt_pipeline._generate_groq_content(  # type: ignore[attr-defined]
            model=prompt_pipeline.DEFAULT_GEMINI_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        payload = _extract_json_payload(raw_text)
        if payload is None:
            raise ValueError("Model returned a non-JSON payload.")
        recommendations = payload.get("redesign_recommendations") or []
        if not isinstance(recommendations, list):
            recommendations = []
        payload["redesign_recommendations"] = recommendations
        payload["ml_modules"] = (
            _build_ml_modules(dataset_bundle["dataframe"]) if dataset_bundle is not None else []
        )
        return payload
    except Exception as error:
        LOGGER.warning(
            "solver solution route fell back to deterministic mode",
            extra={"component": "solver", "error": str(error)},
        )
        log_failure(
            query,
            error,
            "solver_solution_candidate",
            store=store,
            metadata={"route": route},
        )
        return _build_deterministic_solution_candidate(
            route=route,
            query=query,
            retrieval_trace=retrieval_trace,
            dataset_bundle=dataset_bundle,
            failure_reason=str(error),
        )


def _build_plan_text(
    *,
    route: str,
    query: str,
    retrieval_trace: dict[str, Any],
    dataset_bundle: dict[str, Any] | None,
    route_scores: dict[str, float],
) -> str:
    lines = [
        f"Route: {route}",
        f"Query: {query}",
        "Pipeline: ingestion -> chunking -> embeddings -> retrieval -> reasoning -> validator -> refinement -> packaged output",
        f"Route scores: {json.dumps(route_scores, sort_keys=True)}",
        f"Retrieved chunks: {len(retrieval_trace.get('items', []))} of {retrieval_trace.get('scanned_chunk_count', 0)} scanned",
    ]
    if dataset_bundle is not None:
        lines.append(
            f"Dataset companion: {dataset_bundle['dataset_name']} ({dataset_bundle['row_count']:,} rows x {dataset_bundle['column_count']:,} columns)"
        )
    return "\n".join(lines)


def _serialize_candidate_for_storage(route: str, candidate: dict[str, Any]) -> dict[str, Any]:
    payload = dict(candidate)
    if route == "data":
        pipeline_output = dict(candidate.get("pipeline_output") or {})
        payload["pipeline_output"] = serialize_analysis_output(pipeline_output) if pipeline_output else {}
    return payload


def _package_output(
    *,
    route: str,
    candidate: dict[str, Any],
    dataset_bundle: dict[str, Any] | None,
) -> dict[str, Any]:
    packaged = {
        "route": route,
        "summary": candidate.get("summary"),
        "artifact_kind": "analysis_bundle" if route == "data" else "solution_bundle",
        "next_actions": candidate.get("redesign_recommendations") or [],
    }
    if dataset_bundle is not None:
        packaged["dataset"] = {
            "dataset_name": dataset_bundle["dataset_name"],
            "row_count": dataset_bundle["row_count"],
            "column_count": dataset_bundle["column_count"],
        }
    if route == "data":
        pipeline_output = candidate.get("pipeline_output") or {}
        packaged["result"] = serialize_result(pipeline_output.get("result"))
    return packaged


def submit_solve_run(
    *,
    workspace_id: str,
    query: str,
    user_id: str | None,
    asset_id: str | None = None,
    dataset_id: str | None = None,
    route_hint: str | None = None,
):
    with WorkflowStore() as store:
        workspace = store.get_workspace(workspace_id)
        if workspace is None:
            raise ValueError(f"Workspace '{workspace_id}' was not found.")
        route, route_scores = classify_solver_route(
            store=store,
            workspace_id=workspace_id,
            query=query,
            route_hint=route_hint,
        )
        run = store.create_solve_run(
            workspace_id=workspace_id,
            user_id=user_id,
            asset_id=asset_id,
            dataset_id=dataset_id,
            query=query,
            route=route,
            plan_text=json.dumps({"route_scores": route_scores}, sort_keys=True),
        )
        store.record_solve_step(
            run_id=run.run_id,
            step_index=0,
            stage="queue",
            status="queued",
            title="Solve run queued",
            detail={"route": route, "route_scores": route_scores},
        )
        from .queueing import enqueue_solve_run

        enqueue_solve_run(run.run_id)
        return store.get_solve_run(run.run_id)


def process_solve_run(run_id: str):
    with WorkflowStore() as store:
        initial_run = store.get_solve_run(run_id)
        if initial_run is None:
            raise ValueError(f"Solve run '{run_id}' was not found.")
        run = store.mark_solve_run_running(run_id) or initial_run
        route_scores = classify_solver_route(
            store=store,
            workspace_id=run.workspace_id,
            query=run.query,
            route_hint=run.route,
        )[1]
        retrieval_trace = retrieve_workspace_context(
            store=store,
            workspace_id=run.workspace_id,
            query=run.query,
        )
        dataset_bundle = _resolve_dataset_bundle(
            store=store,
            workspace_id=run.workspace_id,
            asset_id=run.asset_id,
            requested_dataset_id=run.dataset_id,
        )
        plan_text = _build_plan_text(
            route=run.route,
            query=run.query,
            retrieval_trace=retrieval_trace,
            dataset_bundle=dataset_bundle,
            route_scores=route_scores,
        )
        store.record_solve_step(
            run_id=run.run_id,
            step_index=1,
            stage="retrieve",
            status="completed",
            title="Workspace context retrieved",
            detail=retrieval_trace,
        )
        store.record_solve_step(
            run_id=run.run_id,
            step_index=2,
            stage="plan",
            status="completed",
            title="Execution plan drafted",
            detail={"plan_text": plan_text},
        )

        def build_candidate(attempt_index: int, feedback: str | None) -> dict[str, Any]:
            del attempt_index
            if run.route == "data":
                if dataset_bundle is None:
                    return {
                        "summary": "",
                        "pipeline_output": {"result": None},
                        "solution_markdown": "",
                        "redesign_recommendations": [],
                    }
                return _build_data_candidate(
                    store=store,
                    query=run.query,
                    dataset_bundle=dataset_bundle,
                    retrieval_trace=retrieval_trace,
                    feedback=feedback,
                )
            return _build_solution_candidate(
                store=store,
                route=run.route,
                query=run.query,
                retrieval_trace=retrieval_trace,
                dataset_bundle=dataset_bundle,
                feedback=feedback,
            )

        candidate, refinement_steps, validator_reports = bounded_refinement_loop(
            build_candidate=build_candidate,
            validate_candidate=lambda payload: validate_solver_output(
                route=run.route,
                query=run.query,
                candidate_output=payload,
            ),
            max_retries=get_settings().max_solver_retries,
        )

        step_index = 3
        for step in refinement_steps:
            store.record_solve_step(
                run_id=run.run_id,
                step_index=step_index,
                stage=step["stage"],
                status=step["status"],
                title=step["title"],
                detail=step["detail"],
            )
            step_index += 1

        for report in validator_reports:
            store.record_validator_report(
                run_id=run.run_id,
                attempt_index=report["attempt_index"],
                status=report["status"],
                checks=report["checks"],
                error_message=report["error_message"],
            )

        last_report = validator_reports[-1] if validator_reports else None
        succeeded = bool(last_report and last_report["status"] == "passed")
        if succeeded:
            final_output = _serialize_candidate_for_storage(run.route, candidate)
            packaged_output = _package_output(
                route=run.route,
                candidate=candidate,
                dataset_bundle=dataset_bundle,
            )
            store.complete_solve_run(
                run.run_id,
                plan_text=plan_text,
                retrieval_trace=retrieval_trace,
                retrieved_chunk_ids=[item["chunk_id"] for item in retrieval_trace.get("items", [])],
                final_output=final_output,
                final_summary=str(candidate.get("summary") or "").strip() or None,
                packaged_output=packaged_output,
                result_hash=(
                    ((candidate.get("pipeline_output") or {}).get("result_hash"))
                    if run.route == "data"
                    else None
                ),
            )
        else:
            log_failure(
                run.query,
                str(last_report.get("error_message") or "Solve validation failed.") if last_report else "Solve validation failed.",
                "solver_validation",
                store=store,
                metadata={"run_id": run.run_id, "route": run.route},
            )
            store.fail_solve_run(
                run.run_id,
                error_message=str(last_report.get("error_message") or "Solve validation failed.") if last_report else "Solve validation failed.",
                plan_text=plan_text,
                retrieval_trace=retrieval_trace,
                retrieved_chunk_ids=[item["chunk_id"] for item in retrieval_trace.get("items", [])],
            )

        record_retrieval_feedback(
            store=store,
            run_id=run.run_id,
            retrieval_trace=retrieval_trace,
            succeeded=succeeded,
        )
        return store.get_solve_run(run.run_id)
