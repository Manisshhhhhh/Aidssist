export type User = {
  user_id: string;
  email: string;
  display_name: string;
  created_at: string;
  updated_at: string;
};

export type AuthResponse = {
  token: string;
  user: User;
};

export type DatasetSummary = {
  dataset_id: string;
  dataset_name: string;
  dataset_key: string;
  source_fingerprint: string;
  source_kind: string;
  source_label: string;
  created_at: string;
  row_count: number;
  column_count: number;
  missing_cell_count: number;
  duplicate_row_count: number;
  numeric_column_count: number;
  categorical_column_count: number;
  datetime_column_count: number;
  columns: string[];
  dtypes: Record<string, string>;
  date_column?: string | null;
  target_column?: string | null;
  auto_config?: ForecastAutoConfig | null;
  forecast_eligibility?: ForecastEligibility | null;
  stats: {
    history_points: number;
    date_range: [string | null, string | null];
  };
  preview_columns: string[];
  preview_rows: Record<string, unknown>[];
};

export type AssetFileSummary = {
  asset_file_id: string;
  asset_id: string;
  dataset_id?: string | null;
  file_name: string;
  media_type: string;
  file_kind: string;
  language?: string | null;
  object_key: string;
  checksum: string;
  size_bytes: number;
  created_at: string;
};

export type DerivedDatasetSummary = {
  derived_dataset_id: string;
  workspace_id: string;
  asset_id?: string | null;
  parent_dataset_id?: string | null;
  dataset_name: string;
  dataset_key: string;
  source_fingerprint: string;
  content_type: string;
  transform_prompt?: string | null;
  row_count: number;
  column_count: number;
  preview_columns: string[];
  preview_rows: Record<string, unknown>[];
  created_at: string;
};

export type AssetSummary = {
  asset_id: string;
  workspace_id: string;
  title: string;
  asset_kind: string;
  primary_dataset_id?: string | null;
  created_at: string;
  updated_at: string;
  file_count: number;
  dataset_count: number;
  derived_dataset_count: number;
  chunk_count: number;
};

export type RetrievalTraceItem = {
  chunk_id: string;
  asset_id: string;
  asset_file_id?: string | null;
  dataset_id?: string | null;
  title: string;
  score: number;
  confidence: string;
  excerpt: string;
  metadata: Record<string, unknown>;
};

export type RetrievalTrace = {
  query: string;
  scanned_chunk_count: number;
  items: RetrievalTraceItem[];
};

export type SolveStep = {
  step_id: string;
  run_id: string;
  step_index: number;
  stage: string;
  status: string;
  title: string;
  detail: Record<string, unknown>;
  created_at: string;
};

export type ValidationReport = {
  report_id: string;
  run_id: string;
  attempt_index: number;
  status: string;
  checks: Array<Record<string, unknown>>;
  error_message?: string | null;
  created_at: string;
};

export type ExecutionPlanStep = {
  step: number;
  tool: "SQL" | "PYTHON" | "EXCEL" | "BI";
  task: string;
  query?: string;
  depends_on?: number[];
  uses_context?: boolean;
  sql_plan?: string | null;
  python_steps?: string[];
  excel_logic?: Record<string, unknown>;
  fallback_reason?: string | null;
  cost_estimate?: "low" | "medium" | "high";
};

export type ExecutionPlan = ExecutionPlanStep[];

export type ExecutionTraceStep = {
  step: number;
  tool?: "SQL" | "PYTHON" | "EXCEL" | "BI" | null;
  task?: string | null;
  status: string;
  execution_time_ms?: number;
  cost_estimate?: "low" | "medium" | "high";
  warnings?: string[];
  error?: string | null;
  fallback_tool?: "SQL" | "PYTHON" | "EXCEL" | "BI" | null;
};

export type OptimizationPayload = {
  execution_time_total: number;
  cost_estimate: "low" | "medium" | "high";
  optimized: boolean;
  parallel_execution: boolean;
  plans_considered: number;
  selected_plan_score: number;
  constraints_applied: Record<string, unknown>;
};

export type ExcelAnalysis = {
  pivot_table?: Record<string, unknown>;
  aggregations?: Record<string, unknown>;
  summary?: Record<string, unknown> | string | null;
};

export type DashboardChart = {
  type: string;
  purpose?: string | null;
  x: string;
  y?: string | null;
  title?: string | null;
  series_key?: string | null;
  time_column?: string | null;
  rows?: Record<string, unknown>[];
  layout?: Record<string, unknown>;
  drilldown?: Record<string, unknown>;
};

export type DashboardKpi = {
  metric: string;
  value: number | string;
};

export type DashboardPayload = {
  charts: DashboardChart[];
  filters?: string[];
  kpis: DashboardKpi[];
  layout?: Record<string, unknown>;
  drilldown_ready?: boolean;
  time_column?: string | null;
  applied_time_filter?: string | null;
  active_filter?: string | null;
  visualization_type?: string | null;
};

export type ForecastMetadata = {
  time_column?: string | null;
  data_points: number;
  frequency: string;
  filled_missing_timestamps?: number;
};

export type ForecastAutoConfig = {
  date_column?: string | null;
  target?: string | null;
  frequency?: string | null;
  frequency_code?: string | null;
  data_points: number;
  horizon?: string | null;
  horizon_label?: string | null;
  model_strategy?: string | null;
  training_mode?: string | null;
  confidence?: string | null;
  confidence_score?: number;
  date_confidence?: { label?: string; score?: number } | null;
  target_confidence?: { label?: string; score?: number } | null;
};

export type ForecastRecommendation = {
  category?: string;
  priority?: number;
  priority_label?: string;
  title?: string;
  recommended_action?: string;
  rationale?: string;
  impact_direction?: string;
  expected_impact?: string;
  confidence?: string;
  risk_level?: string;
  decision_id?: string;
};

export type ForecastEligibility = {
  allowed: boolean;
  reason?: string | null;
  detected_time_column?: string | null;
  suggestions?: string[];
};

export type InteractionMemory = {
  dataset_type: string;
  queries: string[];
  successful_actions: string[];
};

export type DatasetContext = {
  row_count: number;
  column_count: number;
  column_summary: Array<{
    name: string;
    dtype: string;
    non_null_count: number;
    missing_count: number;
    unique_count: number;
    role: string;
  }>;
  data_types: Record<string, string>;
  missing_values: Record<string, number>;
  domain: string;
  dataset_type: string;
  is_time_series: boolean;
  time_columns: string[];
  primary_metrics: string[];
  categorical_features: string[];
  interaction_memory?: InteractionMemory;
};

export type SuggestedAction = {
  title: string;
  prompt: string;
  action_type: "analysis" | "forecast" | string;
  category?: string;
  goal?: string;
  rationale?: string;
  score: number;
  rank: number;
};

export type ForecastPoint = {
  date: string;
  value: number;
  lower_bound?: number | null;
  upper_bound?: number | null;
};

export type ForecastOutput = {
  status: string;
  error?: { message?: string; suggestion?: string } | null;
  error_message?: string | null;
  summary?: string | null;
  dashboard?: DashboardPayload | null;
  context?: DatasetContext | null;
  suggestions?: SuggestedAction[];
  recommended_next_step?: string | null;
  suggested_questions?: string[];
  auto_config?: ForecastAutoConfig | null;
  forecast_eligibility?: ForecastEligibility | null;
  forecast_metadata?: ForecastMetadata | null;
  forecast?: Record<string, ForecastPoint[]>;
  time_series?: Record<string, unknown>[];
  confidence?: { score?: number; label?: string; warnings?: string[] } | null;
  recommendations?: ForecastRecommendation[];
  chosen_model?: string | null;
  trend_status?: string | null;
  history_points?: number;
  resolved_frequency?: string | null;
  active_filter?: string | null;
  visualization_type?: string | null;
};

export type CleaningReport = {
  quality_score: number;
  missing_handled: number;
  duplicates_removed: number;
  outliers_detected: number;
  outlier_columns?: Record<string, number>;
  issues?: string[];
  actions?: string[];
  columns_dropped?: string[];
  type_conversions?: Record<string, { from: string; to: string }>;
  before?: {
    row_count: number;
    column_count: number;
    missing_cells: number;
    duplicate_rows: number;
  };
  after?: {
    row_count: number;
    column_count: number;
    missing_cells: number;
    duplicate_rows: number;
  };
};

export type AnalysisOutput = {
  summary?: string | null;
  confidence?: string | null;
  warnings?: string[];
  dashboard?: DashboardPayload | null;
  context?: DatasetContext | null;
  suggestions?: SuggestedAction[];
  recommended_next_step?: string | null;
  suggested_questions?: string[];
  forecast_metadata?: ForecastMetadata | null;
  active_filter?: string | null;
  visualization_type?: string | null;
  cleaning_report?: CleaningReport | null;
  analysis_contract?: AnalysisContract | null;
  result?: Record<string, unknown> | null;
};

export type AnalysisContract = {
  intent: string;
  code: string;
  result_summary: string;
  insights: string[];
  recommendations: string[];
  confidence: string;
  warnings: string[];
  cleaning_report?: CleaningReport | null;
  tool_used?: "SQL" | "PYTHON" | "EXCEL" | "BI";
  analysis_mode?: "ad-hoc" | "dashboard" | "prediction";
  execution_plan?: ExecutionPlan;
  execution_trace?: ExecutionTraceStep[];
  optimization?: OptimizationPayload;
  excel_analysis?: ExcelAnalysis | null;
  dashboard?: DashboardPayload | null;
  forecast_metadata?: ForecastMetadata | null;
  context?: DatasetContext | null;
  suggestions?: SuggestedAction[];
  recommended_next_step?: string | null;
  suggested_questions?: string[];
  active_filter?: string | null;
  visualization_type?: string | null;
  decision_layer?: DecisionLayer;
};

export type DecisionObject = {
  decision_id: string;
  action: string;
  expected_impact: string;
  confidence: "high" | "medium" | "low";
  risk_level: "low" | "medium" | "high";
  priority: "HIGH" | "MEDIUM" | "LOW";
  reasoning: string;
  decision_performance?: {
    historical_success_rate: number;
    avg_impact: number;
    sample_size: number;
  };
};

export type DecisionLayer = {
  decisions: DecisionObject[];
  top_decision?: DecisionObject | null;
  decision_confidence: "high" | "medium" | "low";
  risk_summary: string;
  learning_insights?: {
    patterns: Array<{
      decision_type: string;
      success_rate: number;
      avg_impact: number;
      sample_size: number;
      uncertainty: string;
    }>;
    confidence_adjustment: string;
    risk_adjustment: string;
  };
};

export type SolveRunStatus = {
  run_id: string;
  workspace_id: string;
  user_id?: string | null;
  asset_id?: string | null;
  dataset_id?: string | null;
  query: string;
  route: string;
  status: string;
  plan_text?: string | null;
  retrieval_trace: RetrievalTrace;
  retrieved_chunk_ids: string[];
  final_output?: Record<string, unknown> | null;
  final_summary?: string | null;
  packaged_output?: Record<string, unknown> | null;
  error_message?: string | null;
  queued_at: string;
  started_at?: string | null;
  finished_at?: string | null;
  queue_wait_ms?: number | null;
  elapsed_ms?: number | null;
  steps: SolveStep[];
  validator_reports: ValidationReport[];
};

export type AssetDetail = AssetSummary & {
  files: AssetFileSummary[];
  datasets: DatasetSummary[];
  derived_datasets: DerivedDatasetSummary[];
  chunk_preview: Array<Record<string, unknown>>;
};

export type WorkspaceSummary = {
  workspace_id: string;
  user_id: string;
  name: string;
  description?: string | null;
  created_at: string;
  updated_at: string;
  asset_count: number;
  dataset_count: number;
  solve_run_count: number;
  derived_dataset_count: number;
};

export type WorkspaceDetail = WorkspaceSummary & {
  assets: AssetSummary[];
  derived_datasets: DerivedDatasetSummary[];
  recent_runs: SolveRunStatus[];
};

export type TimelineItem = {
  event_id: string;
  event_type: string;
  title: string;
  summary: string;
  created_at: string;
  metadata: Record<string, unknown>;
};

export type DatasetTransformResponse = {
  derived_dataset: DerivedDatasetSummary;
  transform_code?: string | null;
};

export type AnalyzeJobResponse = {
  job_id: string;
  dataset_id: string;
  status: string;
  intent: string;
  queued_at: string;
  cache_hit: boolean;
};

export type ForecastJobResponse = {
  job_id: string;
  dataset_id: string;
  status: string;
  intent: string;
  queued_at: string;
  cache_hit: boolean;
};

export type JobStatusResponse = {
  job_id: string;
  dataset_id: string;
  status: string;
  intent: string;
  query: string;
  queued_at: string;
  started_at?: string | null;
  finished_at?: string | null;
  queue_wait_ms?: number | null;
  elapsed_ms?: number | null;
  error_message?: string | null;
  analysis_output?: AnalysisOutput | null;
  forecast_output?: ForecastOutput | null;
  cache_hit: boolean;
};

export type DemoDataset = {
  metadata: {
    file_name: string;
    row_count: number;
    column_count: number;
    columns: string[];
    time_column?: string;
    target_column?: string;
    sample_domain?: string;
  };
  rows: Record<string, unknown>[];
};

export type DemoOutput = {
  query: string;
  intent: string;
  output: Record<string, unknown>;
};

export type DemoStat = {
  label: string;
  value: string | number;
  detail?: string | null;
};

export type DemoFlowStep = {
  title: string;
  description: string;
};

export type DemoResponse = {
  dataset: DemoDataset;
  datasets: DemoDataset[];
  queries: string[];
  outputs: DemoOutput[];
  dashboard?: DashboardPayload | null;
  stats: DemoStat[];
  flow: DemoFlowStep[];
  suggestions: SuggestedAction[];
};
