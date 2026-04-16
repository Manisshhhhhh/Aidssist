from __future__ import annotations

from backend.runtime_preflight import ensure_runtime_or_raise

ensure_runtime_or_raise("api")

import time
import uuid

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .analysis_service import (
    build_result_artifact,
    create_dataset_from_upload,
    get_dataset_auto_analysis,
    get_dataset_summary,
    submit_analysis_job,
    submit_forecast_job,
)
from .auth_service import authenticate_user, get_user_from_token, register_user, revoke_token
from .cache import get_cache_store
from .config import get_settings
from .dataset_transform import transform_dataset
from .ingestion import build_asset_detail, ingest_workspace_files
from .logging_utils import configure_logging, get_logger
from .metrics import render_metrics_payload
from .queueing import get_redis_connection
from .schemas import (
    AnalysisHistoryItemResponse,
    AnalyzeJobRequest,
    AnalyzeJobResponse,
    AutoAnalysisEnvelopeResponse,
    AssetDetail,
    AssetFileSummary,
    AssetSummary,
    AuthRequest,
    AuthResponse,
    DatasetSummaryResponse,
    DatasetTransformRequest,
    DatasetTransformResponse,
    DecisionHistoryItemResponse,
    DecisionOutcomeUpdateRequest,
    DemoOutputResponse,
    DemoFlowStepResponse,
    DemoResponse,
    DemoStatResponse,
    DerivedDatasetSummary,
    ForecastJobRequest,
    ForecastJobResponse,
    JobStatusResponse,
    RetrievalTrace,
    RetrievalTraceItem,
    SolveRequest,
    SolveRunStatus,
    SolveStepResponse,
    TimelineItemResponse,
    UploadResponse,
    UserResponse,
    ValidationReport,
    WorkspaceCreateRequest,
    WorkspaceDetail,
    WorkspaceSummary,
)
from .solver_orchestrator import submit_solve_run
from .storage import get_object_store
from backend.services.demo_service import get_demo_payload
from backend.services.learning_engine import refresh_learning_patterns
from backend.workflow_store import WorkflowStore


configure_logging("api")
settings = get_settings()
LOGGER = get_logger(__name__)
app = FastAPI(title="Aidssist Analysis API", version="1.0.0")
bearer_scheme = HTTPBearer(auto_error=False)

if settings.cors_origins:
    allow_all_origins = "*" in settings.cors_origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[] if allow_all_origins else list(settings.cors_origins),
        allow_origin_regex=".*" if allow_all_origins else None,
        allow_credentials=not allow_all_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def _user_response(user) -> UserResponse:
    return UserResponse(
        user_id=user.user_id,
        email=user.email,
        display_name=user.display_name,
        created_at=user.created_at,
        updated_at=user.updated_at,
    )


def _derived_dataset_response(record) -> DerivedDatasetSummary:
    return DerivedDatasetSummary(
        derived_dataset_id=record.derived_dataset_id,
        workspace_id=record.workspace_id,
        asset_id=record.asset_id,
        parent_dataset_id=record.parent_dataset_id,
        dataset_name=record.dataset_name,
        dataset_key=record.dataset_key,
        source_fingerprint=record.source_fingerprint,
        content_type=record.content_type,
        transform_prompt=record.transform_prompt,
        row_count=record.row_count,
        column_count=record.column_count,
        preview_columns=list(record.preview_columns),
        preview_rows=list(record.preview_rows),
        created_at=record.created_at,
    )


def _asset_file_response(record) -> AssetFileSummary:
    return AssetFileSummary(
        asset_file_id=record.asset_file_id,
        asset_id=record.asset_id,
        dataset_id=record.dataset_id,
        file_name=record.file_name,
        media_type=record.media_type,
        file_kind=record.file_kind,
        language=record.language,
        object_key=record.object_key,
        checksum=record.checksum,
        size_bytes=record.size_bytes,
        created_at=record.created_at,
    )


def _decision_history_response(record) -> DecisionHistoryItemResponse:
    return DecisionHistoryItemResponse(
        decision_history_id=record.decision_history_id,
        job_id=record.job_id,
        forecast_artifact_id=record.forecast_artifact_id,
        source_fingerprint=record.source_fingerprint,
        query=record.query,
        decision_id=record.decision_id,
        decision_json=dict(record.decision_json or {}),
        priority=record.priority,
        decision_confidence=record.decision_confidence,
        risk_level=record.risk_level,
        result_hash=record.result_hash,
        outcome=record.outcome,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def _retrieval_trace_response(payload: dict | None) -> RetrievalTrace:
    payload = payload or {}
    return RetrievalTrace(
        query=str(payload.get("query") or ""),
        scanned_chunk_count=int(payload.get("scanned_chunk_count") or 0),
        items=[
            RetrievalTraceItem(
                chunk_id=str(item.get("chunk_id") or ""),
                asset_id=str(item.get("asset_id") or ""),
                asset_file_id=item.get("asset_file_id"),
                dataset_id=item.get("dataset_id"),
                title=str(item.get("title") or ""),
                score=float(item.get("score") or 0.0),
                confidence=str(item.get("confidence") or "low"),
                excerpt=str(item.get("excerpt") or ""),
                metadata=dict(item.get("metadata") or {}),
            )
            for item in payload.get("items", [])
        ],
    )


def _solve_step_response(record) -> SolveStepResponse:
    return SolveStepResponse(
        step_id=record.step_id,
        run_id=record.run_id,
        step_index=record.step_index,
        stage=record.stage,
        status=record.status,
        title=record.title,
        detail=dict(record.detail),
        created_at=record.created_at,
    )


def _validator_report_response(record) -> ValidationReport:
    return ValidationReport(
        report_id=record.report_id,
        run_id=record.run_id,
        attempt_index=record.attempt_index,
        status=record.status,
        checks=list(record.checks),
        error_message=record.error_message,
        created_at=record.created_at,
    )


def _solve_run_response(store: WorkflowStore, run) -> SolveRunStatus:
    return SolveRunStatus(
        run_id=run.run_id,
        workspace_id=run.workspace_id,
        user_id=run.user_id,
        asset_id=run.asset_id,
        dataset_id=run.dataset_id,
        query=run.query,
        route=run.route,
        status=run.status,
        plan_text=run.plan_text,
        retrieval_trace=_retrieval_trace_response(run.retrieval_trace),
        retrieved_chunk_ids=list(run.retrieved_chunk_ids),
        final_output=run.final_output,
        final_summary=run.final_summary,
        packaged_output=run.packaged_output,
        error_message=run.error_message,
        queued_at=run.queued_at,
        started_at=run.started_at,
        finished_at=run.finished_at,
        queue_wait_ms=run.queue_wait_ms,
        elapsed_ms=run.elapsed_ms,
        steps=[_solve_step_response(item) for item in store.list_solve_steps(run.run_id)],
        validator_reports=[_validator_report_response(item) for item in store.list_validator_reports(run.run_id)],
    )


def _asset_summary_response(store: WorkflowStore, asset) -> AssetSummary:
    files = store.list_asset_files(asset.asset_id)
    derived_datasets = [
        item for item in store.list_workspace_derived_datasets(asset.workspace_id, limit=200) if item.asset_id == asset.asset_id
    ]
    return AssetSummary(
        asset_id=asset.asset_id,
        workspace_id=asset.workspace_id,
        title=asset.title,
        asset_kind=asset.asset_kind,
        primary_dataset_id=asset.primary_dataset_id,
        created_at=asset.created_at,
        updated_at=asset.updated_at,
        file_count=len(files),
        dataset_count=len([item for item in files if item.dataset_id]),
        derived_dataset_count=len(derived_datasets),
        chunk_count=len(store.list_asset_chunks(asset.asset_id, limit=500)),
    )


def _workspace_summary_response(store: WorkflowStore, workspace) -> WorkspaceSummary:
    assets = store.list_workspace_assets(workspace.workspace_id, limit=500)
    derived_datasets = store.list_workspace_derived_datasets(workspace.workspace_id, limit=500)
    runs = store.list_workspace_solve_runs(workspace.workspace_id, limit=500)
    dataset_count = sum(1 for asset in assets if asset.primary_dataset_id)
    return WorkspaceSummary(
        workspace_id=workspace.workspace_id,
        user_id=workspace.user_id,
        name=workspace.name,
        description=workspace.description,
        created_at=workspace.created_at,
        updated_at=workspace.updated_at,
        asset_count=len(assets),
        dataset_count=dataset_count,
        solve_run_count=len(runs),
        derived_dataset_count=len(derived_datasets),
    )


def _workspace_detail_response(store: WorkflowStore, workspace) -> WorkspaceDetail:
    summary = _workspace_summary_response(store, workspace)
    assets = store.list_workspace_assets(workspace.workspace_id, limit=20)
    derived_datasets = store.list_workspace_derived_datasets(workspace.workspace_id, limit=20)
    recent_runs = store.list_workspace_solve_runs(workspace.workspace_id, limit=10)
    return WorkspaceDetail(
        **summary.model_dump(),
        assets=[_asset_summary_response(store, asset) for asset in assets],
        derived_datasets=[_derived_dataset_response(item) for item in derived_datasets],
        recent_runs=[_solve_run_response(store, run) for run in recent_runs],
    )


def _asset_detail_response(store: WorkflowStore, asset_id: str) -> AssetDetail:
    detail = build_asset_detail(store, asset_id)
    summary = _asset_summary_response(store, detail["asset"])
    return AssetDetail(
        **summary.model_dump(),
        files=[_asset_file_response(item) for item in detail["files"]],
        datasets=[DatasetSummaryResponse(**item) for item in detail["datasets"]],
        derived_datasets=[_derived_dataset_response(item) for item in detail["derived_datasets"]],
        chunk_preview=list(detail["chunk_preview"]),
    )


def _workspace_timeline(store: WorkflowStore, workspace_id: str) -> list[TimelineItemResponse]:
    items: list[TimelineItemResponse] = []
    for asset in store.list_workspace_assets(workspace_id, limit=50):
        items.append(
            TimelineItemResponse(
                event_id=f"asset-{asset.asset_id}",
                event_type="asset_uploaded",
                title=asset.title,
                summary=f"{asset.asset_kind.title()} asset uploaded into the workspace.",
                created_at=asset.created_at,
                metadata={"asset_id": asset.asset_id, "asset_kind": asset.asset_kind},
            )
        )
    for dataset in store.list_workspace_derived_datasets(workspace_id, limit=50):
        items.append(
            TimelineItemResponse(
                event_id=f"derived-{dataset.derived_dataset_id}",
                event_type="derived_dataset",
                title=dataset.dataset_name,
                summary=f"Derived dataset created with {dataset.row_count:,} rows and {dataset.column_count:,} columns.",
                created_at=dataset.created_at,
                metadata={
                    "derived_dataset_id": dataset.derived_dataset_id,
                    "parent_dataset_id": dataset.parent_dataset_id,
                },
            )
        )
    for run in store.list_workspace_solve_runs(workspace_id, limit=50):
        items.append(
            TimelineItemResponse(
                event_id=f"solve-{run.run_id}",
                event_type="solve_run",
                title=run.query,
                summary=f"{run.route.title()} solve run finished with status {run.status}.",
                created_at=run.finished_at or run.queued_at,
                metadata={"run_id": run.run_id, "status": run.status, "route": run.route},
            )
        )
    return sorted(items, key=lambda item: item.created_at, reverse=True)


def get_optional_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
):
    if credentials is None or credentials.scheme.lower() != "bearer":
        return None

    user = get_user_from_token(credentials.credentials)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired session.")
    return user


def get_current_user(user=Depends(get_optional_current_user)):
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required.")
    return user


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    started = time.perf_counter()
    status_code = 500
    request_id = request.headers.get("X-Request-Id") or uuid.uuid4().hex
    try:
        response = await call_next(request)
        status_code = response.status_code
        response.headers["X-Request-Id"] = request_id
        return response
    finally:
        from .metrics import REQUEST_COUNT, REQUEST_LATENCY

        endpoint = request.url.path
        duration = time.perf_counter() - started
        REQUEST_LATENCY.labels(method=request.method, endpoint=endpoint).observe(duration)
        REQUEST_COUNT.labels(method=request.method, endpoint=endpoint, status_code=str(status_code)).inc()
        LOGGER.info(
            "request completed",
            extra={
                "component": "api",
                "request_id": request_id,
                "method": request.method,
                "endpoint": endpoint,
                "status_code": status_code,
                "duration_ms": round(duration * 1000, 2),
            },
        )


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/readyz")
def readyz():
    try:
        with WorkflowStore() as store:
            store.list_workflows()
        get_cache_store()
        get_object_store()
        if settings.redis_url and get_redis_connection() is None:
            raise RuntimeError("Redis is unavailable.")
    except Exception as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return {"status": "ready"}


def _demo_response() -> DemoResponse:
    payload = get_demo_payload()
    return DemoResponse(
        dataset=dict(payload.get("dataset") or {}),
        datasets=[dict(item) for item in payload.get("datasets", []) if isinstance(item, dict)],
        queries=[str(item) for item in payload.get("queries", [])],
        outputs=[
            DemoOutputResponse(
                query=str(item.get("query") or ""),
                intent=str(item.get("intent") or "analysis"),
                output=dict(item.get("output") or {}),
            )
            for item in payload.get("outputs", [])
        ],
        dashboard=dict(payload.get("dashboard") or {}),
        stats=[
            DemoStatResponse(
                label=str(item.get("label") or ""),
                value=item.get("value", ""),
                detail=str(item.get("detail")) if item.get("detail") is not None else None,
            )
            for item in payload.get("stats", [])
            if isinstance(item, dict)
        ],
        flow=[
            DemoFlowStepResponse(
                title=str(item.get("title") or ""),
                description=str(item.get("description") or ""),
            )
            for item in payload.get("flow", [])
            if isinstance(item, dict)
        ],
        suggestions=[
            dict(item)
            for item in payload.get("suggestions", [])
            if isinstance(item, dict)
        ],
    )


@app.get("/demo", response_model=DemoResponse)
def demo():
    return _demo_response()


@app.get("/demo-data", response_model=DemoResponse)
def demo_data():
    return _demo_response()


@app.get(settings.metrics_path)
def metrics():
    payload, content_type = render_metrics_payload()
    return Response(content=payload, media_type=content_type)


@app.post("/v1/auth/register", response_model=AuthResponse)
def register(payload: AuthRequest):
    try:
        user, token = register_user(payload.email, payload.password, payload.display_name)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    return AuthResponse(token=token, user=_user_response(user))


@app.post("/v1/auth/login", response_model=AuthResponse)
def login(payload: AuthRequest):
    try:
        user, token = authenticate_user(payload.email, payload.password)
    except ValueError as error:
        raise HTTPException(status_code=401, detail=str(error)) from error

    return AuthResponse(token=token, user=_user_response(user))


@app.get("/v1/auth/me", response_model=UserResponse)
def me(current_user=Depends(get_current_user)):
    return _user_response(current_user)


@app.post("/v1/auth/logout")
def logout(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    current_user=Depends(get_current_user),
):
    del current_user
    if credentials is None:
        raise HTTPException(status_code=401, detail="Authentication required.")
    revoke_token(credentials.credentials)
    return {"status": "signed_out"}


@app.post("/v1/uploads", response_model=UploadResponse)
async def upload_dataset(
    file: UploadFile = File(...),
    current_user=Depends(get_optional_current_user),
):
    file_bytes = await file.read()
    try:
        dataset_record = create_dataset_from_upload(
            file.filename or "dataset.csv",
            file_bytes,
            file.content_type,
            user_id=current_user.user_id if current_user else None,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error

    auto_analysis_payload = get_dataset_auto_analysis(dataset_record.dataset_id)
    dataset_summary_payload = get_dataset_summary(dataset_record.dataset_id)
    return UploadResponse(
        dataset_id=dataset_record.dataset_id,
        dataset_name=dataset_record.dataset_name,
        dataset_key=dataset_record.dataset_key,
        source_fingerprint=dataset_record.source_fingerprint,
        source_kind=dataset_record.source_kind,
        source_label=dataset_record.source_label,
        auto_analysis=auto_analysis_payload.get("auto_analysis", {}),
        suggested_questions=list(dataset_summary_payload.get("suggested_questions") or []),
    )


@app.get("/v1/datasets", response_model=list[DatasetSummaryResponse])
def list_datasets(current_user=Depends(get_current_user)):
    with WorkflowStore() as store:
        datasets = store.list_user_datasets(current_user.user_id, limit=25)

    summaries: list[DatasetSummaryResponse] = []
    for dataset in datasets:
        summary = get_dataset_summary(dataset.dataset_id)
        summaries.append(DatasetSummaryResponse(**summary))
    return summaries


@app.get("/v1/datasets/{dataset_id}", response_model=DatasetSummaryResponse)
def dataset_summary(dataset_id: str, current_user=Depends(get_optional_current_user)):
    if current_user is not None:
        with WorkflowStore() as store:
            if not store.user_has_dataset_access(current_user.user_id, dataset_id):
                raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' was not found.")
    try:
        return DatasetSummaryResponse(**get_dataset_summary(dataset_id))
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.get("/v1/datasets/{dataset_id}/auto-analysis", response_model=AutoAnalysisEnvelopeResponse)
def dataset_auto_analysis(dataset_id: str, current_user=Depends(get_optional_current_user)):
    if current_user is not None:
        with WorkflowStore() as store:
            if not store.user_has_dataset_access(current_user.user_id, dataset_id):
                raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' was not found.")
    try:
        return AutoAnalysisEnvelopeResponse(**get_dataset_auto_analysis(dataset_id))
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.get("/v1/workspaces", response_model=list[WorkspaceSummary])
def list_workspaces(current_user=Depends(get_current_user)):
    with WorkflowStore() as store:
        workspaces = store.list_user_workspaces(current_user.user_id, limit=100)
        return [_workspace_summary_response(store, workspace) for workspace in workspaces]


@app.post("/v1/workspaces", response_model=WorkspaceDetail)
def create_workspace(payload: WorkspaceCreateRequest, current_user=Depends(get_current_user)):
    with WorkflowStore() as store:
        workspace = store.create_workspace(
            user_id=current_user.user_id,
            name=payload.name,
            description=payload.description,
        )
        return _workspace_detail_response(store, workspace)


@app.get("/v1/workspaces/{workspace_id}", response_model=WorkspaceDetail)
def get_workspace_detail(workspace_id: str, current_user=Depends(get_current_user)):
    with WorkflowStore() as store:
        if not store.user_has_workspace_access(current_user.user_id, workspace_id):
            raise HTTPException(status_code=404, detail=f"Workspace '{workspace_id}' was not found.")
        workspace = store.get_workspace(workspace_id)
        if workspace is None:
            raise HTTPException(status_code=404, detail=f"Workspace '{workspace_id}' was not found.")
        return _workspace_detail_response(store, workspace)


@app.get("/v1/workspaces/{workspace_id}/timeline", response_model=list[TimelineItemResponse])
def get_workspace_timeline(workspace_id: str, current_user=Depends(get_current_user)):
    with WorkflowStore() as store:
        if not store.user_has_workspace_access(current_user.user_id, workspace_id):
            raise HTTPException(status_code=404, detail=f"Workspace '{workspace_id}' was not found.")
        return _workspace_timeline(store, workspace_id)


@app.post("/v1/assets", response_model=AssetDetail)
async def upload_workspace_assets(
    workspace_id: str = Form(...),
    title: str | None = Form(None),
    files: list[UploadFile] = File(...),
    current_user=Depends(get_current_user),
):
    with WorkflowStore() as store:
        if not store.user_has_workspace_access(current_user.user_id, workspace_id):
            raise HTTPException(status_code=404, detail=f"Workspace '{workspace_id}' was not found.")
        uploaded_files = []
        for item in files:
            uploaded_files.append(
                {
                    "file_name": item.filename or "workspace_asset.bin",
                    "content_type": item.content_type or "application/octet-stream",
                    "content": await item.read(),
                }
            )
        try:
            result = ingest_workspace_files(
                store=store,
                workspace_id=workspace_id,
                uploaded_files=uploaded_files,
                user_id=current_user.user_id,
                title=title,
            )
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return _asset_detail_response(store, result["asset"].asset_id)


@app.get("/v1/assets/{asset_id}", response_model=AssetDetail)
def get_asset(asset_id: str, current_user=Depends(get_current_user)):
    with WorkflowStore() as store:
        asset = store.get_asset(asset_id)
        if asset is None:
            raise HTTPException(status_code=404, detail=f"Asset '{asset_id}' was not found.")
        if not store.user_has_workspace_access(current_user.user_id, asset.workspace_id):
            raise HTTPException(status_code=404, detail=f"Asset '{asset_id}' was not found.")
        return _asset_detail_response(store, asset_id)


@app.post("/v1/solve", response_model=SolveRunStatus)
def create_solve_run(payload: SolveRequest, current_user=Depends(get_current_user)):
    with WorkflowStore() as store:
        if not store.user_has_workspace_access(current_user.user_id, payload.workspace_id):
            raise HTTPException(status_code=404, detail=f"Workspace '{payload.workspace_id}' was not found.")
    try:
        run = submit_solve_run(
            workspace_id=payload.workspace_id,
            query=payload.query,
            user_id=current_user.user_id,
            asset_id=payload.asset_id,
            dataset_id=payload.dataset_id,
            route_hint=payload.route_hint,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    if run is None:
        raise HTTPException(status_code=500, detail="Solve run could not be created.")
    with WorkflowStore() as store:
        persisted = store.get_solve_run(run.run_id)
        if persisted is None:
            raise HTTPException(status_code=500, detail="Solve run could not be loaded.")
        return _solve_run_response(store, persisted)


@app.get("/v1/solve/{run_id}", response_model=SolveRunStatus)
def get_solve_run_status(run_id: str, current_user=Depends(get_current_user)):
    with WorkflowStore() as store:
        run = store.get_solve_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Solve run '{run_id}' was not found.")
        if not store.user_has_workspace_access(current_user.user_id, run.workspace_id):
            raise HTTPException(status_code=404, detail=f"Solve run '{run_id}' was not found.")
        return _solve_run_response(store, run)


@app.post("/v1/datasets/{dataset_id}/transform", response_model=DatasetTransformResponse)
def transform_workspace_dataset(
    dataset_id: str,
    payload: DatasetTransformRequest,
    current_user=Depends(get_current_user),
):
    with WorkflowStore() as store:
        workspace = store.get_workspace_by_dataset_id(dataset_id)
        if workspace is None or not store.user_has_workspace_access(current_user.user_id, workspace.workspace_id):
            raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' was not found.")
        try:
            result = transform_dataset(
                store=store,
                dataset_id=dataset_id,
                instruction=payload.instruction,
            )
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return DatasetTransformResponse(
            derived_dataset=_derived_dataset_response(result["derived_dataset"]),
            transform_code=result["transform_code"],
        )


@app.post("/v1/jobs/analyze", response_model=AnalyzeJobResponse)
def enqueue_analysis_job(
    payload: AnalyzeJobRequest,
    current_user=Depends(get_optional_current_user),
):
    if current_user is not None:
        with WorkflowStore() as store:
            if not store.user_has_dataset_access(current_user.user_id, payload.dataset_id):
                raise HTTPException(status_code=404, detail=f"Dataset '{payload.dataset_id}' was not found.")
    try:
        job = submit_analysis_job(
            payload.dataset_id,
            payload.query,
            payload.workflow_context,
            user_id=current_user.user_id if current_user else None,
        )
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error

    return AnalyzeJobResponse(
        job_id=job.job_id,
        dataset_id=job.dataset_id,
        status=job.status,
        intent=job.intent,
        queued_at=job.queued_at,
        cache_hit=job.cache_hit,
    )


@app.post("/v1/jobs/forecast", response_model=ForecastJobResponse)
def enqueue_forecast_job(
    payload: ForecastJobRequest,
    current_user=Depends(get_optional_current_user),
):
    if current_user is not None:
        with WorkflowStore() as store:
            if not store.user_has_dataset_access(current_user.user_id, payload.dataset_id):
                raise HTTPException(status_code=404, detail=f"Dataset '{payload.dataset_id}' was not found.")
    try:
        job = submit_forecast_job(
            payload.dataset_id,
            payload.forecast_config,
            payload.workflow_context,
            user_id=current_user.user_id if current_user else None,
        )
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error

    return ForecastJobResponse(
        job_id=job.job_id,
        dataset_id=job.dataset_id,
        status=job.status,
        intent=job.intent,
        queued_at=job.queued_at,
        cache_hit=job.cache_hit,
    )


@app.get("/v1/jobs/{job_id}", response_model=JobStatusResponse)
def get_job(job_id: str, current_user=Depends(get_optional_current_user)):
    with WorkflowStore() as store:
        if current_user is not None and not store.user_has_job_access(current_user.user_id, job_id):
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' was not found.")
        job = store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' was not found.")

    return JobStatusResponse(
        job_id=job.job_id,
        dataset_id=job.dataset_id,
        status=job.status,
        intent=job.intent,
        query=job.query,
        queued_at=job.queued_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        queue_wait_ms=job.queue_wait_ms,
        elapsed_ms=job.elapsed_ms,
        error_message=job.error_message,
        analysis_output=job.analysis_output if job.intent != "forecast" else None,
        forecast_output=job.analysis_output if job.intent == "forecast" else None,
        cache_hit=job.cache_hit,
    )


@app.get("/v1/jobs/{job_id}/artifacts/result")
def download_job_result(job_id: str, current_user=Depends(get_optional_current_user)):
    with WorkflowStore() as store:
        if current_user is not None and not store.user_has_job_access(current_user.user_id, job_id):
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' was not found.")
        job = store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' was not found.")
    if job.analysis_output is None:
        raise HTTPException(status_code=409, detail="Job has not completed yet.")

    payload, media_type, file_name = build_result_artifact(job)
    headers = {"Content-Disposition": f'attachment; filename="{file_name}"'}
    return Response(content=payload, media_type=media_type, headers=headers)


@app.get("/v1/history", response_model=list[AnalysisHistoryItemResponse])
def history(current_user=Depends(get_current_user)):
    with WorkflowStore() as store:
        items = store.list_user_history(current_user.user_id, limit=50)

    return [
        AnalysisHistoryItemResponse(
            job_id=item.job_id,
            dataset_id=item.dataset_id,
            dataset_name=item.dataset_name,
            query=item.query,
            status=item.status,
            intent=item.intent,
            queued_at=item.queued_at,
            finished_at=item.finished_at,
            result_summary=item.result_summary,
            error_message=item.error_message,
            cache_hit=item.cache_hit,
        )
        for item in items
    ]


@app.get("/v1/decisions", response_model=list[DecisionHistoryItemResponse])
def list_decisions(current_user=Depends(get_current_user)):
    with WorkflowStore() as store:
        items = store.list_decision_history(user_id=current_user.user_id, limit=100)
    return [_decision_history_response(item) for item in items]


@app.get("/v1/decisions/{decision_history_id}", response_model=DecisionHistoryItemResponse)
def get_decision(decision_history_id: str, current_user=Depends(get_current_user)):
    with WorkflowStore() as store:
        if not store.user_has_decision_access(current_user.user_id, decision_history_id):
            raise HTTPException(status_code=404, detail=f"Decision '{decision_history_id}' was not found.")
        item = store.get_decision_history(decision_history_id, user_id=current_user.user_id)
    if item is None:
        raise HTTPException(status_code=404, detail=f"Decision '{decision_history_id}' was not found.")
    return _decision_history_response(item)


@app.patch("/v1/decisions/{decision_history_id}/outcome", response_model=DecisionHistoryItemResponse)
def update_decision_outcome(
    decision_history_id: str,
    payload: DecisionOutcomeUpdateRequest,
    current_user=Depends(get_current_user),
):
    with WorkflowStore() as store:
        if not store.user_has_decision_access(current_user.user_id, decision_history_id):
            raise HTTPException(status_code=404, detail=f"Decision '{decision_history_id}' was not found.")
        updated = store.update_decision_outcome(decision_history_id, payload.outcome)
        if updated is not None:
            refresh_learning_patterns(store, updated.source_fingerprint)
    if updated is None:
        raise HTTPException(status_code=404, detail=f"Decision '{decision_history_id}' was not found.")
    return _decision_history_response(updated)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    del request
    return JSONResponse(status_code=500, content={"detail": str(exc)})
