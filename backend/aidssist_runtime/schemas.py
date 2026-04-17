from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class WorkspaceCreateRequest(BaseModel):
    name: str
    description: str | None = None


class AssetFileSummary(BaseModel):
    asset_file_id: str
    asset_id: str
    dataset_id: str | None = None
    file_name: str
    media_type: str
    file_kind: str
    language: str | None = None
    object_key: str
    checksum: str
    size_bytes: int
    created_at: str


class DerivedDatasetSummary(BaseModel):
    derived_dataset_id: str
    workspace_id: str
    asset_id: str | None = None
    parent_dataset_id: str | None = None
    dataset_name: str
    dataset_key: str
    source_fingerprint: str
    content_type: str
    transform_prompt: str | None = None
    row_count: int
    column_count: int
    preview_columns: list[str] = Field(default_factory=list)
    preview_rows: list[dict[str, Any]] = Field(default_factory=list)
    created_at: str


class DatasetForecastStatsResponse(BaseModel):
    history_points: int = 0
    date_range: tuple[str | None, str | None] = (None, None)


class AutoForecastConfigResponse(BaseModel):
    date_column: str | None = None
    target: str | None = None
    frequency: str | None = None
    frequency_code: str | None = None
    data_points: int = 0
    horizon: str | None = None
    horizon_label: str | None = None
    model_strategy: str | None = None
    training_mode: str | None = None
    confidence: str | None = None
    confidence_score: float = 0.0
    date_confidence: dict[str, Any] = Field(default_factory=dict)
    target_confidence: dict[str, Any] = Field(default_factory=dict)


class ForecastEligibilityResponse(BaseModel):
    allowed: bool = True
    reason: str | None = None
    detected_time_column: str | None = None
    suggestions: list[str] = Field(default_factory=list)


class AssetSummary(BaseModel):
    asset_id: str
    workspace_id: str
    title: str
    asset_kind: str
    primary_dataset_id: str | None = None
    created_at: str
    updated_at: str
    file_count: int = 0
    dataset_count: int = 0
    derived_dataset_count: int = 0
    chunk_count: int = 0


class WorkspaceSummary(BaseModel):
    workspace_id: str
    user_id: str
    name: str
    description: str | None = None
    created_at: str
    updated_at: str
    asset_count: int = 0
    dataset_count: int = 0
    solve_run_count: int = 0
    derived_dataset_count: int = 0


class RetrievalTraceItem(BaseModel):
    chunk_id: str
    asset_id: str
    asset_file_id: str | None = None
    dataset_id: str | None = None
    title: str
    score: float
    confidence: str
    excerpt: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalTrace(BaseModel):
    query: str
    scanned_chunk_count: int
    items: list[RetrievalTraceItem] = Field(default_factory=list)


class SolveStepResponse(BaseModel):
    step_id: str
    run_id: str
    step_index: int
    stage: str
    status: str
    title: str
    detail: dict[str, Any] = Field(default_factory=dict)
    created_at: str


class ValidationReport(BaseModel):
    report_id: str
    run_id: str
    attempt_index: int
    status: str
    checks: list[dict[str, Any]] = Field(default_factory=list)
    error_message: str | None = None
    created_at: str


class SolveRunStatus(BaseModel):
    run_id: str
    workspace_id: str
    user_id: str | None = None
    asset_id: str | None = None
    dataset_id: str | None = None
    query: str
    route: str
    status: str
    plan_text: str | None = None
    retrieval_trace: RetrievalTrace = Field(default_factory=lambda: RetrievalTrace(query="", scanned_chunk_count=0))
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    final_output: dict[str, Any] | None = None
    final_summary: str | None = None
    packaged_output: dict[str, Any] | None = None
    error_message: str | None = None
    queued_at: str
    started_at: str | None = None
    finished_at: str | None = None
    queue_wait_ms: int | None = None
    elapsed_ms: int | None = None
    steps: list[SolveStepResponse] = Field(default_factory=list)
    validator_reports: list[ValidationReport] = Field(default_factory=list)


class AssetDetail(AssetSummary):
    files: list[AssetFileSummary] = Field(default_factory=list)
    datasets: list["DatasetSummaryResponse"] = Field(default_factory=list)
    derived_datasets: list[DerivedDatasetSummary] = Field(default_factory=list)
    chunk_preview: list[dict[str, Any]] = Field(default_factory=list)


class WorkspaceDetail(WorkspaceSummary):
    assets: list[AssetSummary] = Field(default_factory=list)
    derived_datasets: list[DerivedDatasetSummary] = Field(default_factory=list)
    recent_runs: list[SolveRunStatus] = Field(default_factory=list)


class SolveRequest(BaseModel):
    workspace_id: str
    query: str
    asset_id: str | None = None
    dataset_id: str | None = None
    route_hint: str | None = None


class DatasetTransformRequest(BaseModel):
    instruction: str


class DatasetTransformResponse(BaseModel):
    derived_dataset: DerivedDatasetSummary
    transform_code: str | None = None


class TimelineItemResponse(BaseModel):
    event_id: str
    event_type: str
    title: str
    summary: str
    created_at: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class DemoOutputResponse(BaseModel):
    query: str
    intent: str
    output: dict[str, Any] = Field(default_factory=dict)


class DemoStatResponse(BaseModel):
    label: str
    value: str | int | float
    detail: str | None = None


class DemoFlowStepResponse(BaseModel):
    title: str
    description: str


class DemoResponse(BaseModel):
    dataset: dict[str, Any] = Field(default_factory=dict)
    datasets: list[dict[str, Any]] = Field(default_factory=list)
    queries: list[str] = Field(default_factory=list)
    outputs: list[DemoOutputResponse] = Field(default_factory=list)
    dashboard: dict[str, Any] = Field(default_factory=dict)
    stats: list[DemoStatResponse] = Field(default_factory=list)
    flow: list[DemoFlowStepResponse] = Field(default_factory=list)
    suggestions: list[dict[str, Any]] = Field(default_factory=list)


class UploadResponse(BaseModel):
    dataset_id: str
    dataset_name: str
    dataset_key: str
    source_fingerprint: str
    source_kind: str
    source_label: str
    auto_analysis: "AutoAnalysisResponse" = Field(default_factory=lambda: AutoAnalysisResponse())
    suggested_questions: list[str] = Field(default_factory=list)


class FolderUploadFileResultResponse(BaseModel):
    file_name: str
    relative_path: str
    size_bytes: int = 0
    status: str
    dataset_id: str | None = None
    file_tag: str | None = None
    error: str | None = None


class FolderUploadRelationshipResponse(BaseModel):
    left_table: str
    left_column: str
    right_table: str
    right_column: str
    confidence: float = 0.0
    match_rate: float = 0.0


class FolderUploadPreviewResponse(BaseModel):
    dataset_id: str
    file_name: str
    table_name: str
    file_tag: str | None = None
    row_count: int = 0
    column_count: int = 0
    preview_columns: list[str] = Field(default_factory=list)
    preview_rows: list[dict[str, Any]] = Field(default_factory=list)


class FolderUploadDatasetSummaryResponse(BaseModel):
    tables: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    relationships: list[FolderUploadRelationshipResponse] = Field(default_factory=list)
    previews: list[FolderUploadPreviewResponse] = Field(default_factory=list)
    suggested_analysis_prompt: str | None = None
    ready_message: str = "Dataset Ready -> Generate Insights"


class FolderUploadResponse(BaseModel):
    status: str
    session_id: str
    folder_name: str
    files_processed: int = 0
    file_count: int = 0
    total_size_bytes: int = 0
    asset: AssetDetail | None = None
    processed_files: list[FolderUploadFileResultResponse] = Field(default_factory=list)
    failed_files: list[FolderUploadFileResultResponse] = Field(default_factory=list)
    dataset_summary: FolderUploadDatasetSummaryResponse = Field(default_factory=FolderUploadDatasetSummaryResponse)


class UserResponse(BaseModel):
    user_id: str
    email: str
    display_name: str
    created_at: str
    updated_at: str


class AuthRequest(BaseModel):
    email: str
    password: str
    display_name: str | None = None


class AuthResponse(BaseModel):
    token: str
    user: UserResponse


class AnalyzeJobRequest(BaseModel):
    dataset_id: str
    query: str
    workflow_context: dict[str, Any] = Field(default_factory=dict)


class AnalyzeJobResponse(BaseModel):
    job_id: str
    dataset_id: str
    status: str
    intent: str
    queued_at: str
    cache_hit: bool = False


class ForecastJobRequest(BaseModel):
    dataset_id: str
    forecast_config: dict[str, Any] = Field(default_factory=dict)
    workflow_context: dict[str, Any] = Field(default_factory=dict)


class ForecastJobResponse(BaseModel):
    job_id: str
    dataset_id: str
    status: str
    intent: str
    queued_at: str
    cache_hit: bool = False


class JobStatusResponse(BaseModel):
    job_id: str
    dataset_id: str
    status: str
    intent: str
    query: str
    queued_at: str
    started_at: str | None = None
    finished_at: str | None = None
    queue_wait_ms: int | None = None
    elapsed_ms: int | None = None
    error_message: str | None = None
    analysis_output: dict[str, Any] | None = None
    forecast_output: dict[str, Any] | None = None
    cache_hit: bool = False


class AutoAnalysisResultResponse(BaseModel):
    task: str
    result: Any = None
    insight: str = ""


class AutoAnalysisResponse(BaseModel):
    tasks: list[str] = Field(default_factory=list)
    results: list[AutoAnalysisResultResponse] = Field(default_factory=list)
    summary: list[str] = Field(default_factory=list)


class AutoAnalysisEnvelopeResponse(BaseModel):
    auto_analysis: AutoAnalysisResponse = Field(default_factory=AutoAnalysisResponse)


class DatasetSummaryResponse(BaseModel):
    dataset_id: str
    dataset_name: str
    dataset_key: str
    source_fingerprint: str
    source_kind: str
    source_label: str
    created_at: str
    row_count: int
    column_count: int
    missing_cell_count: int
    duplicate_row_count: int
    numeric_column_count: int
    categorical_column_count: int
    datetime_column_count: int
    columns: list[str] = Field(default_factory=list)
    dtypes: dict[str, str] = Field(default_factory=dict)
    date_column: str | None = None
    target_column: str | None = None
    auto_config: AutoForecastConfigResponse = Field(default_factory=AutoForecastConfigResponse)
    forecast_eligibility: ForecastEligibilityResponse = Field(default_factory=ForecastEligibilityResponse)
    stats: DatasetForecastStatsResponse = Field(default_factory=DatasetForecastStatsResponse)
    preview_columns: list[str] = Field(default_factory=list)
    preview_rows: list[dict[str, Any]] = Field(default_factory=list)
    auto_analysis: AutoAnalysisResponse = Field(default_factory=AutoAnalysisResponse)
    suggested_questions: list[str] = Field(default_factory=list)


class AnalysisHistoryItemResponse(BaseModel):
    job_id: str
    dataset_id: str
    dataset_name: str
    query: str
    status: str
    intent: str
    queued_at: str
    finished_at: str | None = None
    result_summary: str | None = None
    error_message: str | None = None
    cache_hit: bool = False


class DecisionHistoryItemResponse(BaseModel):
    decision_history_id: str
    job_id: str | None = None
    forecast_artifact_id: str | None = None
    source_fingerprint: str
    query: str
    decision_id: str
    decision_json: dict[str, Any] = Field(default_factory=dict)
    priority: str
    decision_confidence: str
    risk_level: str
    result_hash: str | None = None
    outcome: str | None = None
    created_at: str
    updated_at: str


class DecisionOutcomeUpdateRequest(BaseModel):
    outcome: str | None = None


class GoogleDriveImportRequest(BaseModel):
    workspace_id: str
    file_id: str
    access_token: str


class KaggleImportRequest(BaseModel):
    workspace_id: str
    dataset_url: str


class ImportJobResponse(BaseModel):
    job_id: str
    workspace_id: str
    session_id: str
    source_type: str
    source_ref: str
    source_label: str
    status: str
    asset_id: str | None = None
    error_message: str | None = None
    result: dict[str, Any] = Field(default_factory=dict)
    created_at: str
    updated_at: str
    completed_at: str | None = None


class SchemaColumnResponse(BaseModel):
    name: str
    sql_type: str
    semantic_type: str
    nullable: bool = True
    non_null_count: int = 0
    missing_count: int = 0
    null_ratio: float = 0.0
    unique_count: int = 0
    unique_ratio: float = 0.0
    sample_values: list[Any] = Field(default_factory=list)


class SchemaTableResponse(BaseModel):
    name: str
    source_name: str | None = None
    dataset_id: str | None = None
    source_kind: str | None = None
    row_count: int = 0
    column_count: int = 0
    primary_keys: list[str] = Field(default_factory=list)
    columns: list[SchemaColumnResponse] = Field(default_factory=list)
    preview_rows: list[dict[str, Any]] = Field(default_factory=list)


class SchemaRelationshipResponse(BaseModel):
    left_table: str
    left_column: str
    right_table: str
    right_column: str
    relationship_type: str = "one_to_many"
    match_rate: float = 0.0
    confidence: float = 0.0


class GraphNodeResponse(BaseModel):
    id: str
    label: str
    dataset_id: str | None = None
    row_count: int = 0
    column_count: int = 0
    dataset_type: str | None = None


class GraphEdgeResponse(BaseModel):
    id: str
    source: str
    target: str
    label: str
    confidence: float = 0.0


class DatasetGraphResponse(BaseModel):
    nodes: list[GraphNodeResponse] = Field(default_factory=list)
    edges: list[GraphEdgeResponse] = Field(default_factory=list)


class InsightChartResponse(BaseModel):
    type: str
    title: str | None = None
    x: str
    y: str | None = None
    rows: list[dict[str, Any]] = Field(default_factory=list)


class InsightCardResponse(BaseModel):
    kind: str
    title: str
    narrative: str
    confidence: str | None = None
    chart: InsightChartResponse | None = None


class RecommendationCardResponse(BaseModel):
    title: str
    body: str
    priority: str | None = None


class AssetIntelligenceResponse(BaseModel):
    asset_id: str
    session_id: str
    dataset_type: str | None = None
    status: str
    catalog: dict[str, Any] = Field(default_factory=dict)
    schema: dict[str, Any] = Field(default_factory=dict)
    insights: dict[str, Any] = Field(default_factory=dict)
    chat_context: dict[str, Any] = Field(default_factory=dict)


class AIInsightsRequest(BaseModel):
    asset_id: str
    force_refresh: bool = False


class AIChatRequest(BaseModel):
    asset_id: str
    question: str


class AIChatResponse(BaseModel):
    question: str
    sql: str
    answer: str
    columns: list[str] = Field(default_factory=list)
    rows: list[dict[str, Any]] = Field(default_factory=list)
    chart: dict[str, Any] | None = None


UploadResponse.model_rebuild()
AssetDetail.model_rebuild()
WorkspaceDetail.model_rebuild()
