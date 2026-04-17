import { startTransition, useDeferredValue, useEffect, useState } from "react";

import {
  createWorkspace,
  getDemoData,
  getAsset,
  getCurrentUser,
  getJobStatus,
  getSolveRun,
  getWorkspace,
  getWorkspaceTimeline,
  listWorkspaces,
  login,
  logout,
  register,
  startAnalysisJob,
  startForecastJob,
  startSolve,
  transformDataset,
  uploadAsset
} from "./lib/api";
import { clearStoredToken, getStoredToken, setStoredToken } from "./lib/auth";
import demoDashboardScreenshot from "./assets/demo-dashboard.png";
import AskDataChatPanel from "./components/AskDataChatPanel";
import AssetIntelligencePanel from "./components/AssetIntelligencePanel";
import FolderUploadPanel from "./components/FolderUploadPanel";
import ImportHubPanel from "./components/ImportHubPanel";
import type {
  AnalysisOutput,
  AssetDetail,
  DashboardChart,
  DashboardPayload,
  DatasetSummary,
  DemoResponse,
  ForecastOutput,
  FolderUploadResponse,
  JobStatusResponse,
  SuggestedAction,
  SolveRunStatus,
  TimelineItem,
  User,
  WorkspaceDetail,
  WorkspaceSummary
} from "./types/api";

type AuthMode = "login" | "register";
type PanelName = "mission" | "assets" | "studio" | "solve" | "history";
type PublicView = "landing" | "demo";
type RouteHint = "auto" | "data" | "code" | "hybrid";
type WorkbenchMode = "analysis" | "forecast" | "orchestrator";
type FlowStepState = "pending" | "active" | "completed" | "error";
type ActivityTone = "info" | "success" | "error";
type ActivityFlow = {
  title: string;
  message?: string;
  tone: ActivityTone;
  steps: Array<{ label: string; state: FlowStepState }>;
};

const PANEL_ITEMS: Array<{ id: PanelName; label: string; description: string }> = [
  { id: "mission", label: "Mission", description: "Workspace control room" },
  { id: "assets", label: "Assets", description: "Upload CSV, code, and archives" },
  { id: "studio", label: "Dataset Studio", description: "Version transforms and previews" },
  { id: "solve", label: "Solve Trace", description: "Reasoning, validation, and output" },
  { id: "history", label: "Benchmarks", description: "Timeline, success rate, replay memory" }
];
const DEMO_TIME_FILTER = "last_quarter";

const DEFAULT_DEMO_FLOW = [
  {
    title: "Load sample dataset",
    description: "Open a retail sales dataset instantly without signup, uploads, or API keys."
  },
  {
    title: "Show the dashboard",
    description: "Land on KPI cards, charts, and talking points that explain the product in seconds."
  },
  {
    title: "Pivot into planning",
    description: "Switch between analysis, forecasting, and root-cause stories during the same demo."
  }
];

const DEFAULT_DEMO_STATS = [
  { label: "Demo ready", value: "3 flows", detail: "Analysis, forecast, and root cause preloaded." },
  { label: "Setup time", value: "< 10 sec", detail: "Cached public payload for repeat demos." },
  { label: "Target user", value: "Recruiters", detail: "Fast, visual, and easy to explain." },
  { label: "Mode", value: "Public", detail: "No login required for first-run exploration." }
];

const LANDING_FEATURES = [
  {
    title: "Public demo path",
    body: "Anyone can open the product, load sample data, and understand the value without friction."
  },
  {
    title: "AI analysis + forecast",
    body: "The product moves from dashboards to forward-looking recommendations in one coherent workflow."
  },
  {
    title: "Recruiter-friendly narrative",
    body: "Every screen is optimized to explain the system fast, with strong copy and visible outputs."
  },
  {
    title: "Deployment-ready stack",
    body: "FastAPI, React, Render/Railway backend config, and Vercel frontend config are wired in."
  }
];

const RECRUITER_HIGHLIGHTS = [
  "See the core value in under a minute with a guided sample workflow.",
  "Review precomputed outputs so the demo stays stable even without external AI credentials.",
  "Switch from dashboarding to forecasting to root-cause analysis without resetting context."
];

function formatCount(value: number): string {
  return new Intl.NumberFormat().format(value);
}

function formatMs(value?: number | null): string {
  if (!value) {
    return "n/a";
  }
  if (value < 1000) {
    return `${value} ms`;
  }
  return `${(value / 1000).toFixed(1)} s`;
}

function formatPercent(value: number): string {
  return `${value.toFixed(0)}%`;
}

function formatStatValue(value: string | number): string {
  return typeof value === "number" ? formatCount(value) : value;
}

function formatDateRange(dateRange?: [string | null, string | null] | null): string {
  if (!dateRange || !dateRange[0] || !dateRange[1]) {
    return "n/a";
  }
  if (dateRange[0] === dateRange[1]) {
    return dateRange[0];
  }
  return `${dateRange[0]} to ${dateRange[1]}`;
}

function formatColumnOptionLabel(column: string, dtypes: Record<string, string>): string {
  const dtype = dtypes[column];
  return dtype ? `${column} (${dtype})` : column;
}

function safeObject(value: unknown): Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
}

function safeArray(value: unknown): Array<Record<string, unknown>> {
  return Array.isArray(value)
    ? value.filter((item) => typeof item === "object" && item !== null) as Array<Record<string, unknown>>
    : [];
}

function resultPreview(run: SolveRunStatus | null): Array<Record<string, unknown>> {
  const finalOutput = safeObject(run?.final_output);
  const pipelineOutput = safeObject(finalOutput.pipeline_output);
  const result = safeObject(pipelineOutput.result);
  const records = result.records;
  return safeArray(records).slice(0, 12);
}

function runSuccessMetrics(runs: SolveRunStatus[]): {
  successRate: number;
  averageLatency: string;
  completedCount: number;
} {
  const completed = runs.filter((run) => run.status === "completed" || run.status === "failed");
  const successful = completed.filter((run) => run.status === "completed");
  const durations = completed
    .map((run) => run.elapsed_ms || 0)
    .filter((value) => value > 0);
  const averageLatency =
    durations.length > 0
      ? formatMs(Math.round(durations.reduce((sum, value) => sum + value, 0) / durations.length))
      : "n/a";

  return {
    successRate: completed.length > 0 ? (successful.length / completed.length) * 100 : 0,
    averageLatency,
    completedCount: completed.length
  };
}

function firstDatasetId(asset: AssetDetail | null): string | null {
  return asset?.datasets[0]?.dataset_id || null;
}

const UNIVERSAL_TIME_FILTERS: Array<{ value: string; label: string }> = [
  { value: "current_month", label: "Current month" },
  { value: "last_month", label: "Last month" },
  { value: "last_quarter", label: "Last quarter" },
  { value: "last_year", label: "Last year" },
  { value: "custom_range", label: "Custom range" }
];

const FORECAST_HORIZON_OPTIONS: Array<{ value: string; label: string }> = [
  { value: "next_week", label: "Next week" },
  { value: "next_month", label: "Next month" },
  { value: "next_quarter", label: "Next quarter" },
  { value: "next_year", label: "Next year" }
];

const FORECAST_FREQUENCY_OPTIONS: Array<{ value: string; label: string }> = [
  { value: "auto", label: "Auto detect" },
  { value: "D", label: "Daily" },
  { value: "W", label: "Weekly" },
  { value: "M", label: "Monthly" },
  { value: "Q", label: "Quarterly" }
];

const FORECAST_MODEL_STRATEGY_OPTIONS: Array<{ value: string; label: string }> = [
  { value: "hybrid", label: "Hybrid AI" },
  { value: "explainable", label: "Explainable" },
  { value: "accuracy", label: "Accuracy first" }
];

const FORECAST_TRAINING_MODE_OPTIONS: Array<{ value: string; label: string }> = [
  { value: "auto", label: "Auto" },
  { value: "local", label: "Local" },
  { value: "background", label: "Background" }
];

const FORECAST_ALTERNATIVE_ACTIONS: Array<{ label: string; prompt: string }> = [
  {
    label: "Correlation analysis",
    prompt: "Run a correlation analysis across the key variables and explain the strongest relationships."
  },
  {
    label: "Clustering",
    prompt: "Segment this dataset with clustering and describe the most meaningful groups."
  },
  {
    label: "Feature importance",
    prompt: "Estimate feature importance for the main business outcome and explain the strongest drivers."
  },
  {
    label: "Classification",
    prompt: "Check whether this dataset supports a useful classification analysis and outline the best target options."
  }
];

function safeStringArray(value: unknown): string[] {
  return Array.isArray(value)
    ? value.filter((item): item is string => typeof item === "string" && item.length > 0)
    : [];
}

function safeSuggestedActions(value: unknown): SuggestedAction[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value
    .filter((item): item is Record<string, unknown> => typeof item === "object" && item !== null)
    .map((item, index) => ({
      title: typeof item.title === "string" && item.title.trim() ? item.title : `Suggestion ${index + 1}`,
      prompt: typeof item.prompt === "string" && item.prompt.trim() ? item.prompt : typeof item.title === "string" ? item.title : "",
      action_type: typeof item.action_type === "string" ? item.action_type : "analysis",
      category: typeof item.category === "string" ? item.category : undefined,
      goal: typeof item.goal === "string" ? item.goal : undefined,
      rationale: typeof item.rationale === "string" ? item.rationale : undefined,
      score: typeof item.score === "number" ? item.score : Number(item.score || 0),
      rank: typeof item.rank === "number" ? item.rank : Number(item.rank || index + 1)
    }))
    .filter((item) => item.prompt.trim().length > 0 || item.title.trim().length > 0);
}

function safeForecastRecommendations(value: unknown): Array<Record<string, unknown>> {
  return safeArray(value);
}

function labelForOption(options: Array<{ value: string; label: string }>, value: string | null | undefined): string {
  return options.find((option) => option.value === value)?.label || (value ? value.replace(/_/g, " ") : "Auto");
}

function titleCaseSlug(value: string | null | undefined): string {
  if (!value) {
    return "n/a";
  }
  return value
    .replace(/_/g, " ")
    .replace(/\b\w/g, (match) => match.toUpperCase());
}

function parseDateValue(value: unknown): Date | null {
  if (typeof value !== "string" && !(value instanceof Date)) {
    return null;
  }
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? null : parsed;
}

function getFilterBounds(
  anchor: Date,
  filterType: string,
  customRange?: { startDate?: string; endDate?: string }
): { start: Date; end: Date } {
  const end = new Date(anchor);
  if (filterType === "custom_range") {
    const start = customRange?.startDate ? new Date(customRange.startDate) : null;
    const finish = customRange?.endDate ? new Date(customRange.endDate) : null;
    if (!start || Number.isNaN(start.getTime()) || !finish || Number.isNaN(finish.getTime())) {
      return { start: new Date(-8640000000000000), end: new Date(8640000000000000) };
    }
    finish.setHours(23, 59, 59, 999);
    return { start, end: finish };
  }
  if (filterType === "current_month") {
    return {
      start: new Date(anchor.getFullYear(), anchor.getMonth(), 1),
      end
    };
  }
  if (filterType === "last_month") {
    return {
      start: new Date(anchor.getFullYear(), anchor.getMonth() - 1, 1),
      end: new Date(anchor.getFullYear(), anchor.getMonth(), 0, 23, 59, 59, 999)
    };
  }
  if (filterType === "last_quarter") {
    const currentQuarterStartMonth = Math.floor(anchor.getMonth() / 3) * 3;
    return {
      start: new Date(anchor.getFullYear(), currentQuarterStartMonth - 3, 1),
      end: new Date(anchor.getFullYear(), currentQuarterStartMonth, 0, 23, 59, 59, 999)
    };
  }
  return {
    start: new Date(anchor.getFullYear() - 1, anchor.getMonth(), anchor.getDate()),
    end
  };
}

function filterRowsByTime(
  rows: Array<Record<string, unknown>>,
  timeColumn: string | null | undefined,
  filterType: string,
  customRange?: { startDate?: string; endDate?: string }
) {
  if (!timeColumn || !filterType) {
    return rows;
  }
  const parsedDates = rows
    .map((row) => parseDateValue(row[timeColumn]))
    .filter((value): value is Date => value instanceof Date);
  if (!parsedDates.length) {
    return rows;
  }
  const anchor = parsedDates.reduce((latest, current) => (current > latest ? current : latest), parsedDates[0]);
  const { start, end } = getFilterBounds(anchor, filterType, customRange);
  return rows.filter((row) => {
    const dateValue = parseDateValue(row[timeColumn]);
    return dateValue ? dateValue >= start && dateValue <= end : true;
  });
}

function chartValueKey(chart: DashboardChart): string {
  return chart.y || "value";
}

function toNumeric(value: unknown): number {
  return typeof value === "number" ? value : Number(value || 0);
}

function normalizeErrorMessage(error: unknown, fallback: string): string {
  return error instanceof Error ? error.message : fallback;
}

function buildActivityFlow(
  title: string,
  stepLabels: string[],
  activeIndex: number,
  tone: ActivityTone = "info",
  message?: string
): ActivityFlow {
  const boundedIndex = Math.max(0, Math.min(activeIndex, Math.max(stepLabels.length - 1, 0)));
  return {
    title,
    message,
    tone,
    steps: stepLabels.map((label, index) => ({
      label,
      state: index < boundedIndex ? "completed" : index === boundedIndex ? "active" : "pending"
    }))
  };
}

function completeActivityFlow(title: string, stepLabels: string[], message?: string): ActivityFlow {
  return {
    title,
    message,
    tone: "success",
    steps: stepLabels.map((label) => ({ label, state: "completed" }))
  };
}

function failActivityFlow(
  title: string,
  stepLabels: string[],
  failedIndex: number,
  message: string
): ActivityFlow {
  const boundedIndex = Math.max(0, Math.min(failedIndex, Math.max(stepLabels.length - 1, 0)));
  return {
    title,
    message,
    tone: "error",
    steps: stepLabels.map((label, index) => ({
      label,
      state: index < boundedIndex ? "completed" : index === boundedIndex ? "error" : "pending"
    }))
  };
}

function buildAsyncJobFlow(
  title: string,
  stepLabels: string[],
  status: string | null | undefined,
  errorMessage?: string | null
): ActivityFlow {
  const normalized = String(status || "").toLowerCase();
  if (normalized === "completed") {
    return completeActivityFlow(title, stepLabels, "Results are ready.");
  }
  if (normalized === "failed") {
    return failActivityFlow(title, stepLabels, 1, errorMessage || "The run could not be completed.");
  }
  if (normalized === "running" || normalized === "processing" || normalized === "started") {
    return buildActivityFlow(title, stepLabels, 1, "info", "Aidssist is processing the request.");
  }
  return buildActivityFlow(title, stepLabels, 0, "info", "Aidssist has queued the request.");
}

function ActivityTracker({ flow }: { flow: ActivityFlow | null }) {
  if (!flow) {
    return null;
  }

  return (
    <section className={`activity-panel activity-panel--${flow.tone}`}>
      <div className="activity-panel__header">
        <div>
          <span className="section-kicker">Progress</span>
          <h3>{flow.title}</h3>
          {flow.message ? <p>{flow.message}</p> : null}
        </div>
        {flow.steps.some((step) => step.state === "active") ? <span className="loading-dot" aria-hidden="true" /> : null}
      </div>
      <div className="activity-steps">
        {flow.steps.map((step, index) => (
          <div key={`${step.label}-${index}`} className={`activity-step activity-step--${step.state}`}>
            <span className="activity-step__index">{index + 1}</span>
            <strong>{step.label}</strong>
          </div>
        ))}
      </div>
    </section>
  );
}

function InsightChartCard({
  chart,
  filterType,
  customRange
}: {
  chart: DashboardChart;
  filterType: string;
  customRange?: { startDate?: string; endDate?: string };
}) {
  const timeColumn = chart.time_column || (chart.type === "line" ? chart.x : null);
  const rows = filterRowsByTime(chart.rows || [], timeColumn, filterType, customRange);
  if (!rows.length) {
    return (
      <article className="viz-card">
        <div className="section-kicker">{chart.purpose || chart.type}</div>
        <h4>{chart.title || "Chart"}</h4>
        <p>No rows match the selected time filter. Try a broader range or switch to the public demo dataset.</p>
      </article>
    );
  }

  if (chart.type === "line") {
    const valueKey = chartValueKey(chart);
    const values = rows.map((row) => toNumeric(row[valueKey]));
    const min = Math.min(...values);
    const max = Math.max(...values);
    const points = values
      .map((value, index) => {
        const x = (index / Math.max(values.length - 1, 1)) * 100;
        const y = max === min ? 50 : 100 - ((value - min) / (max - min)) * 100;
        return `${x},${y}`;
      })
      .join(" ");
    return (
      <article className="viz-card viz-card--line">
        <div className="section-kicker">{chart.purpose || chart.type}</div>
        <h4>{chart.title || "Line chart"}</h4>
        <svg viewBox="0 0 100 100" preserveAspectRatio="none" className="line-chart">
          <polyline points={points} />
        </svg>
        <div className="viz-axis-row">
          <span>{String(rows[0][chart.x] ?? "")}</span>
          <strong>{values[values.length - 1]?.toLocaleString()}</strong>
        </div>
      </article>
    );
  }

  if (chart.type === "pie") {
    const valueKey = chartValueKey(chart);
    const total = rows.reduce((sum, row) => sum + Math.max(0, toNumeric(row[valueKey])), 0) || 1;
    let cursor = 0;
    const palette = ["#ff8c42", "#65d8ff", "#7ce38b", "#ffd166", "#f78fb3", "#8ea0b9"];
    const segments = rows
      .map((row, index) => {
        const value = Math.max(0, toNumeric(row[valueKey]));
        const angle = (value / total) * 360;
        const segment = `${palette[index % palette.length]} ${cursor}deg ${cursor + angle}deg`;
        cursor += angle;
        return segment;
      })
      .join(", ");
    return (
      <article className="viz-card viz-card--pie">
        <div className="section-kicker">{chart.purpose || chart.type}</div>
        <h4>{chart.title || "Pie chart"}</h4>
        <div className="pie-chart" style={{ background: `conic-gradient(${segments})` }} />
        <div className="viz-legend">
          {rows.map((row, index) => (
            <div key={`${chart.title || "pie"}-${index}`}>
              <span className="legend-dot" style={{ backgroundColor: palette[index % palette.length] }} />
              <strong>{String(row[chart.x] ?? "Segment")}</strong>
              <small>{toNumeric(row[valueKey]).toLocaleString()}</small>
            </div>
          ))}
        </div>
      </article>
    );
  }

  const valueKey = chartValueKey(chart);
  const max = Math.max(...rows.map((row) => Math.max(0, toNumeric(row[valueKey]))), 1);
  return (
    <article className="viz-card">
      <div className="section-kicker">{chart.purpose || chart.type}</div>
      <h4>{chart.title || "Bar chart"}</h4>
      <div className="bar-chart-list">
        {rows.map((row, index) => {
          const value = Math.max(0, toNumeric(row[valueKey]));
          return (
            <div key={`${chart.title || "bar"}-${index}`} className="bar-chart-row">
              <div>
                <strong>{String(row[chart.x] ?? "")}</strong>
                <small>{value.toLocaleString()}</small>
              </div>
              <div className="bar-track">
                <div className="bar-fill" style={{ width: `${(value / max) * 100}%` }} />
              </div>
            </div>
          );
        })}
      </div>
    </article>
  );
}

function DashboardGrid({
  dashboard,
  filterType,
  customRange
}: {
  dashboard: DashboardPayload | null | undefined;
  filterType: string;
  customRange?: { startDate?: string; endDate?: string };
}) {
  if (!dashboard) {
    return (
      <div className="dashboard-empty">
        <strong>Dashboard will appear here.</strong>
        <p>Run analysis, load the public demo, or select a dataset to populate KPI cards and charts.</p>
      </div>
    );
  }

  return (
    <div className="dashboard-stack">
      <div className="kpi-grid">
        {(dashboard.kpis || []).map((kpi) => (
          <div key={kpi.metric} className="kpi-card">
            <span>{kpi.metric.replace(/_/g, " ")}</span>
            <strong>
              {typeof kpi.value === "number" ? kpi.value.toLocaleString() : String(kpi.value)}
            </strong>
          </div>
        ))}
      </div>
      <div className="viz-grid">
        {(dashboard.charts || []).map((chart, index) => (
          <InsightChartCard
            key={`${chart.title || chart.type}-${index}`}
            chart={chart}
            filterType={filterType}
            customRange={customRange}
          />
        ))}
      </div>
    </div>
  );
}

export default function App() {
  const [token, setToken] = useState<string | null>(() => getStoredToken());
  const [user, setUser] = useState<User | null>(null);
  const [authMode, setAuthMode] = useState<AuthMode>("register");
  const [publicView, setPublicView] = useState<PublicView>("landing");
  const [authError, setAuthError] = useState<string | null>(null);
  const [demoError, setDemoError] = useState<string | null>(null);
  const [statusMessage, setStatusMessage] = useState("Solver platform ready.");
  const [working, setWorking] = useState(false);
  const [demoLoading, setDemoLoading] = useState(false);
  const [activityFlow, setActivityFlow] = useState<ActivityFlow | null>(null);
  const [demoData, setDemoData] = useState<DemoResponse | null>(null);
  const [demoOutputIndex, setDemoOutputIndex] = useState(0);

  const [panel, setPanel] = useState<PanelName>("mission");
  const [workspaces, setWorkspaces] = useState<WorkspaceSummary[]>([]);
  const [activeWorkspaceId, setActiveWorkspaceId] = useState<string | null>(null);
  const [workspaceDetail, setWorkspaceDetail] = useState<WorkspaceDetail | null>(null);
  const [timeline, setTimeline] = useState<TimelineItem[]>([]);
  const [activeAssetId, setActiveAssetId] = useState<string | null>(null);
  const [assetDetail, setAssetDetail] = useState<AssetDetail | null>(null);
  const [latestFolderUpload, setLatestFolderUpload] = useState<FolderUploadResponse | null>(null);
  const [uploadTitle, setUploadTitle] = useState("");
  const [queuedFiles, setQueuedFiles] = useState<File[]>([]);
  const [fileInputResetKey, setFileInputResetKey] = useState(0);
  const [selectedDatasetId, setSelectedDatasetId] = useState<string | null>(null);
  const [activeRunId, setActiveRunId] = useState<string | null>(null);
  const [activeRun, setActiveRun] = useState<SolveRunStatus | null>(null);

  const [transformPrompt, setTransformPrompt] = useState("Fill numeric missing values with mean");
  const [solvePrompt, setSolvePrompt] = useState(
    "Redesign this workspace and propose the highest-value next steps."
  );
  const [workbenchMode, setWorkbenchMode] = useState<WorkbenchMode>("analysis");
  const [analysisPrompt, setAnalysisPrompt] = useState("Build a dashboard and explain the biggest shifts.");
  const [forecastHorizon, setForecastHorizon] = useState("next_month");
  const [forecastDateColumn, setForecastDateColumn] = useState("");
  const [forecastTargetColumn, setForecastTargetColumn] = useState("");
  const [forecastFrequency, setForecastFrequency] = useState("auto");
  const [forecastModelStrategy, setForecastModelStrategy] = useState("hybrid");
  const [forecastTrainingMode, setForecastTrainingMode] = useState("auto");
  const [showForecastAdvanced, setShowForecastAdvanced] = useState(false);
  const [timeFilter, setTimeFilter] = useState("last_quarter");
  const [customRangeStart, setCustomRangeStart] = useState("");
  const [customRangeEnd, setCustomRangeEnd] = useState("");
  const [analysisJobId, setAnalysisJobId] = useState<string | null>(null);
  const [analysisJob, setAnalysisJob] = useState<JobStatusResponse | null>(null);
  const [forecastJobId, setForecastJobId] = useState<string | null>(null);
  const [forecastJob, setForecastJob] = useState<JobStatusResponse | null>(null);
  const [chatSeedQuestion, setChatSeedQuestion] = useState<string | null>(null);
  const [routeHint, setRouteHint] = useState<RouteHint>("auto");
  const [timelineFilter, setTimelineFilter] = useState("");
  const deferredTimelineFilter = useDeferredValue(timelineFilter);
  const customRangeComplete = timeFilter !== "custom_range" || Boolean(customRangeStart && customRangeEnd);
  const selectedDataset: DatasetSummary | undefined = assetDetail?.datasets.find(
    (dataset) => dataset.dataset_id === selectedDatasetId
  );
  const forecastColumnOptions: string[] = Array.from(
    new Set([...(selectedDataset?.columns || []), ...(selectedDataset?.preview_columns || [])])
  );
  const forecastColumnTypes: Record<string, string> = selectedDataset?.dtypes ?? {};
  const forecastAutoConfig = selectedDataset?.auto_config ?? null;
  const forecastEligibility = selectedDataset?.forecast_eligibility ?? null;
  const forecastEligibilityAllowed = forecastEligibility?.allowed ?? true;
  const forecastEligibilityReason =
    typeof forecastEligibility?.reason === "string" && forecastEligibility.reason.trim()
      ? forecastEligibility.reason.trim()
      : null;
  const forecastEligibilitySuggestions = safeStringArray(forecastEligibility?.suggestions);
  const forecastDateAutoDetected = forecastAutoConfig?.date_column || selectedDataset?.date_column || "";
  const forecastTargetAutoDetected = forecastAutoConfig?.target || selectedDataset?.target_column || "";
  const forecastAutoHorizon = forecastAutoConfig?.horizon || "next_month";
  const forecastAutoFrequencyLabel = forecastAutoConfig?.frequency || "Auto detect";
  const forecastAutoConfidence = forecastAutoConfig?.confidence || null;
  const forecastAutoConfidenceScore = forecastAutoConfig?.confidence_score;
  const forecastHistoryPoints = selectedDataset?.stats?.history_points ?? selectedDataset?.row_count ?? 0;
  const forecastDateRange: [string | null, string | null] = selectedDataset?.stats?.date_range ?? [null, null];
  const forecastCanRun = Boolean(selectedDatasetId && customRangeComplete && forecastEligibilityAllowed);
  const forecastEffectiveDateColumn = forecastDateColumn || forecastDateAutoDetected;
  const forecastEffectiveTargetColumn = forecastTargetColumn || forecastTargetAutoDetected;
  const forecastDateNeedsAttention = Boolean(
    selectedDatasetId && forecastColumnOptions.length > 0 && !forecastEffectiveDateColumn
  );
  const forecastTargetNeedsAttention = Boolean(
    selectedDatasetId && forecastColumnOptions.length > 0 && !forecastEffectiveTargetColumn
  );
  const forecastUsingManualDateOverride = Boolean(
    forecastDateColumn && forecastDateColumn !== forecastDateAutoDetected
  );
  const forecastUsingManualTargetOverride = Boolean(
    forecastTargetColumn && forecastTargetColumn !== forecastTargetAutoDetected
  );
  const forecastResolvedFrequencyLabel =
    forecastFrequency === "auto"
      ? forecastAutoFrequencyLabel
      : labelForOption(FORECAST_FREQUENCY_OPTIONS, forecastFrequency);
  const forecastResolvedHorizonLabel = labelForOption(FORECAST_HORIZON_OPTIONS, forecastHorizon);
  const forecastBlockReason =
    !forecastEligibilityAllowed
      ? forecastEligibilityReason === "No valid time column detected"
        ? "Aidssist couldn’t validate a reliable time column in this dataset, so forecasting is disabled."
        : forecastEligibilityReason || "Aidssist couldn’t validate a reliable time column in this dataset, so forecasting is disabled."
      : null;

  function buildTimeFilterContext(): Record<string, unknown> {
    if (timeFilter !== "custom_range") {
      return { time_filter: timeFilter };
    }
    return {
      time_filter: timeFilter,
      custom_time_range: {
        start_date: customRangeStart,
        end_date: customRangeEnd
      }
    };
  }

  function loadAnalysisAlternative(prompt: string, status: string) {
    startTransition(() => {
      setWorkbenchMode("analysis");
      setAnalysisPrompt(prompt);
    });
    setStatusMessage(status);
  }

  async function handleLoadDemo(forceFresh = false) {
    const steps = ["Load sample dataset", "Apply precomputed outputs", "Open recruiter-ready dashboard"];
    setDemoLoading(true);
    setDemoError(null);
    setActivityFlow(
      buildActivityFlow(
        "Preparing public demo",
        steps,
        0,
        "info",
        "Loading the bundled dataset and precomputed analysis outputs."
      )
    );

    try {
      const payload = await getDemoData(forceFresh);
      setActivityFlow(
        buildActivityFlow(
          "Preparing public demo",
          steps,
          1,
          "info",
          "Sample outputs are ready. Opening the dashboard view."
        )
      );
      const analysisIndex = payload.outputs.findIndex((item) => item.intent === "analysis");
      startTransition(() => {
        setDemoData(payload);
        setDemoOutputIndex(analysisIndex >= 0 ? analysisIndex : 0);
        setPublicView("demo");
      });
      setActivityFlow(
        completeActivityFlow(
          "Public demo ready",
          steps,
          "The analysis dashboard, forecast, and dataset preview are available."
        )
      );
    } catch (error) {
      const message = normalizeErrorMessage(error, "The public demo could not be loaded.");
      setDemoError(message);
      setPublicView("landing");
      setActivityFlow(failActivityFlow("Public demo failed", steps, 1, message));
    } finally {
      setDemoLoading(false);
    }
  }

  function handleExitDemo(targetAuthMode: AuthMode = "register") {
    startTransition(() => {
      setPublicView("landing");
      setAuthMode(targetAuthMode);
    });
    setDemoError(null);
  }

  async function refreshWorkspace(workspaceId: string, preferredAssetId?: string | null) {
    if (!token) {
      return;
    }
    const [detail, timelineItems] = await Promise.all([
      getWorkspace(workspaceId, token),
      getWorkspaceTimeline(workspaceId, token)
    ]);
    startTransition(() => {
      setWorkspaceDetail(detail);
      setTimeline(timelineItems);
      setWorkspaces((existing) => {
        const summary: WorkspaceSummary = {
          workspace_id: detail.workspace_id,
          user_id: detail.user_id,
          name: detail.name,
          description: detail.description,
          created_at: detail.created_at,
          updated_at: detail.updated_at,
          asset_count: detail.asset_count,
          dataset_count: detail.dataset_count,
          solve_run_count: detail.solve_run_count,
          derived_dataset_count: detail.derived_dataset_count
        };
        const next = [summary, ...existing.filter((item) => item.workspace_id !== detail.workspace_id)];
        return next;
      });
      const nextAssetId = preferredAssetId ?? activeAssetId ?? detail.assets[0]?.asset_id ?? null;
      setActiveAssetId(nextAssetId);
      if (!selectedDatasetId && detail.assets[0]?.primary_dataset_id) {
        setSelectedDatasetId(detail.assets[0].primary_dataset_id);
      }
    });
  }

  useEffect(() => {
    if (!token) {
      setUser(null);
      setWorkspaces([]);
      setWorkspaceDetail(null);
      setTimeline([]);
      setAssetDetail(null);
      setActiveWorkspaceId(null);
      setActiveAssetId(null);
      setActiveRun(null);
      setActiveRunId(null);
      setAnalysisJob(null);
      setAnalysisJobId(null);
      setForecastJob(null);
      setForecastJobId(null);
      setLatestFolderUpload(null);
      setActivityFlow(null);
      return;
    }

    void (async () => {
      try {
        const currentUser = await getCurrentUser(token);
        let workspaceItems = await listWorkspaces(token);
        if (workspaceItems.length === 0) {
          const created = await createWorkspace(
            "Primary Mission Control",
            "Default solver workspace for mixed datasets and project files.",
            token
          );
          workspaceItems = [created];
        }
        const initialWorkspaceId =
          workspaceItems.find((item) => item.workspace_id === activeWorkspaceId)?.workspace_id ||
          workspaceItems[0]?.workspace_id ||
          null;
        startTransition(() => {
          setUser(currentUser);
          setWorkspaces(workspaceItems);
          setActiveWorkspaceId(initialWorkspaceId);
        });
        setStatusMessage("Mission control synchronized.");
      } catch (error) {
        clearStoredToken();
        setToken(null);
        setAuthError(error instanceof Error ? error.message : "Failed to restore session.");
      }
    })();
  }, [token]);

  useEffect(() => {
    if (!token || !activeWorkspaceId) {
      return;
    }
    setLatestFolderUpload(null);
    void refreshWorkspace(activeWorkspaceId);
  }, [token, activeWorkspaceId]);

  useEffect(() => {
    if (!token || !activeAssetId) {
      setAssetDetail(null);
      if (!activeAssetId) {
        setSelectedDatasetId(null);
      }
      return;
    }
    void (async () => {
      const detail = await getAsset(activeAssetId, token);
      startTransition(() => {
        setAssetDetail(detail);
        setSelectedDatasetId((existing) =>
          detail.datasets.some((dataset) => dataset.dataset_id === existing)
            ? existing
            : detail.datasets[0]?.dataset_id || existing || null
        );
      });
    })();
  }, [token, activeAssetId]);

  useEffect(() => {
    setAnalysisJob(null);
    setAnalysisJobId(null);
    setForecastJob(null);
    setForecastJobId(null);
    setForecastDateColumn(forecastDateAutoDetected);
    setForecastTargetColumn(forecastTargetAutoDetected);
    setForecastHorizon(forecastAutoHorizon);
    setForecastFrequency("auto");
    setForecastModelStrategy(forecastAutoConfig?.model_strategy || "hybrid");
    setForecastTrainingMode(forecastAutoConfig?.training_mode || "auto");
    setShowForecastAdvanced(false);
  }, [
    selectedDatasetId,
    forecastDateAutoDetected,
    forecastTargetAutoDetected,
    forecastAutoHorizon,
    forecastAutoConfig?.model_strategy,
    forecastAutoConfig?.training_mode
  ]);

  useEffect(() => {
    if (!token || !activeRunId) {
      return;
    }

    let timerId = 0;
    let cancelled = false;

    const poll = async () => {
      try {
        const run = await getSolveRun(activeRunId, token);
        if (cancelled) {
          return;
        }
        setActiveRun(run);
        setActivityFlow(
          buildAsyncJobFlow(
            "Solver run in progress",
            ["Queue orchestration", "Execute solver plan", "Validate final output"],
            run.status,
            run.error_message
          )
        );
        if (run.status === "completed" || run.status === "failed") {
          setStatusMessage(
            run.status === "completed"
              ? `Solve run ${run.run_id.slice(0, 8)} completed.`
              : run.error_message || "Solve run failed."
          );
          if (activeWorkspaceId) {
            await refreshWorkspace(activeWorkspaceId, run.asset_id || activeAssetId);
          }
          return;
        }
        timerId = window.setTimeout(poll, 1500);
      } catch (error) {
        if (!cancelled) {
          setStatusMessage(error instanceof Error ? error.message : "Polling failed.");
        }
      }
    };

    void poll();

    return () => {
      cancelled = true;
      window.clearTimeout(timerId);
    };
  }, [token, activeRunId]);

  useEffect(() => {
    if (!token || !analysisJobId) {
      return;
    }

    let timerId = 0;
    let cancelled = false;

    const poll = async () => {
      try {
        const job = await getJobStatus(analysisJobId, token);
        if (cancelled) {
          return;
        }
        setAnalysisJob(job);
        setActivityFlow(
          buildAsyncJobFlow(
            "Analysis in progress",
            ["Queue analysis", "Generate insights", "Render dashboard"],
            job.status,
            job.error_message
          )
        );
        if (job.status === "completed" || job.status === "failed") {
          setStatusMessage(
            job.status === "completed"
              ? `Analysis job ${job.job_id.slice(0, 8)} completed.`
              : job.error_message || "Analysis job failed."
          );
          return;
        }
        timerId = window.setTimeout(poll, 1200);
      } catch (error) {
        if (!cancelled) {
          setStatusMessage(error instanceof Error ? error.message : "Analysis polling failed.");
        }
      }
    };

    void poll();

    return () => {
      cancelled = true;
      window.clearTimeout(timerId);
    };
  }, [token, analysisJobId]);

  useEffect(() => {
    if (!token || !forecastJobId) {
      return;
    }

    let timerId = 0;
    let cancelled = false;

    const poll = async () => {
      try {
        const job = await getJobStatus(forecastJobId, token);
        if (cancelled) {
          return;
        }
        setForecastJob(job);
        setActivityFlow(
          buildAsyncJobFlow(
            "Forecast in progress",
            ["Queue forecast", "Model trend", "Render outlook"],
            job.status,
            job.error_message
          )
        );
        if (job.status === "completed" || job.status === "failed") {
          const forecastOutput = job.forecast_output as ForecastOutput | undefined;
          setStatusMessage(
            job.status === "completed"
              ? `Forecast job ${job.job_id.slice(0, 8)} ${forecastOutput?.status === "FAILED" ? "returned a structured error." : "completed"}.`
              : job.error_message || "Forecast job failed."
          );
          return;
        }
        timerId = window.setTimeout(poll, 1200);
      } catch (error) {
        if (!cancelled) {
          setStatusMessage(error instanceof Error ? error.message : "Forecast polling failed.");
        }
      }
    };

    void poll();

    return () => {
      cancelled = true;
      window.clearTimeout(timerId);
    };
  }, [token, forecastJobId]);

  async function handleAuthSubmit(formData: FormData) {
    const email = String(formData.get("email") || "");
    const password = String(formData.get("password") || "");
    const displayName = String(formData.get("displayName") || "");
    const steps = ["Validate credentials", "Open workspace"];
    setWorking(true);
    setAuthError(null);
    setActivityFlow(
      buildActivityFlow(
        authMode === "register" ? "Creating workspace account" : "Signing you in",
        steps,
        0,
        "info",
        "We’re validating your credentials and loading the workspace."
      )
    );
    try {
      const payload =
        authMode === "register"
          ? await register(email, password, displayName)
          : await login(email, password);
      setStoredToken(payload.token);
      setToken(payload.token);
      setUser(payload.user);
      setStatusMessage("Authentication successful.");
      setActivityFlow(
        completeActivityFlow(
          authMode === "register" ? "Workspace account ready" : "Signed in",
          steps,
          "Mission control will sync your workspace in the background."
        )
      );
    } catch (error) {
      const message = normalizeErrorMessage(error, "Authentication failed.");
      setAuthError(message);
      setActivityFlow(
        failActivityFlow(
          authMode === "register" ? "Workspace account failed" : "Sign-in failed",
          steps,
          0,
          message
        )
      );
    } finally {
      setWorking(false);
    }
  }

  async function handleCreateWorkspace() {
    if (!token) {
      return;
    }
    const steps = ["Provision workspace", "Sync control room"];
    setWorking(true);
    setActivityFlow(
      buildActivityFlow(
        "Creating workspace",
        steps,
        0,
        "info",
        "Provisioning a fresh workspace shell."
      )
    );
    try {
      const created = await createWorkspace(
        `Workspace ${workspaces.length + 1}`,
        "AI + ML solver workspace for uploaded datasets and project files.",
        token
      );
      startTransition(() => {
        setActiveWorkspaceId(created.workspace_id);
        setPanel("mission");
      });
      setStatusMessage(`Created ${created.name}.`);
      setActivityFlow(completeActivityFlow("Workspace created", steps, `${created.name} is ready.`));
    } catch (error) {
      const message = normalizeErrorMessage(error, "Workspace creation failed.");
      setStatusMessage(message);
      setActivityFlow(failActivityFlow("Workspace creation failed", steps, 0, message));
    } finally {
      setWorking(false);
    }
  }

  async function handleAssetUpload() {
    if (!token || !activeWorkspaceId || queuedFiles.length === 0) {
      return;
    }
    const steps = ["Upload files", "Index workspace content", "Refresh workspace view"];
    setWorking(true);
    setStatusMessage("Ingesting files into the workspace...");
    setActivityFlow(
      buildActivityFlow(
        "Uploading asset",
        steps,
        0,
        "info",
        "Aidssist is ingesting files and preparing searchable context."
      )
    );
    try {
      const asset = await uploadAsset(activeWorkspaceId, uploadTitle, queuedFiles, token);
      setActivityFlow(
        buildActivityFlow(
          "Uploading asset",
          steps,
          1,
          "info",
          "Files uploaded. Building the indexed workspace view now."
        )
      );
      setUploadTitle("");
      setQueuedFiles([]);
      setLatestFolderUpload(null);
      setFileInputResetKey((value: number) => value + 1);
      startTransition(() => {
        setActiveAssetId(asset.asset_id);
        setPanel("assets");
      });
      await refreshWorkspace(activeWorkspaceId, asset.asset_id);
      setStatusMessage(`Asset ${asset.title} uploaded and indexed.`);
      setActivityFlow(
        completeActivityFlow("Asset uploaded", steps, `${asset.title} is now ready for analysis.`)
      );
    } catch (error) {
      const message = normalizeErrorMessage(error, "Asset upload failed.");
      setStatusMessage(message);
      setActivityFlow(failActivityFlow("Asset upload failed", steps, 1, message));
    } finally {
      setWorking(false);
    }
  }

  async function handleFolderUploadCompleted(payload: FolderUploadResponse) {
    const steps = ["Stage files", "Register datasets", "Detect relationships", "Sync workspace"];
    setLatestFolderUpload(payload);
    setStatusMessage(payload.dataset_summary.ready_message || "Dataset folder uploaded.");
    setActivityFlow(
      completeActivityFlow(
        "Dataset folder ready",
        steps,
        `${payload.files_processed} files processed and ${payload.dataset_summary.relationships.length} join candidates detected.`
      )
    );

    if (!payload.asset) {
      return;
    }

    startTransition(() => {
      setActiveAssetId(payload.asset?.asset_id || null);
      setAssetDetail(payload.asset || null);
      setSelectedDatasetId(payload.asset?.datasets[0]?.dataset_id || null);
      setPanel("assets");
    });

    if (activeWorkspaceId) {
      await refreshWorkspace(activeWorkspaceId, payload.asset.asset_id);
    }
  }

  async function handleExternalImportCompleted(job: { asset_id?: string | null; source_type: string }) {
    if (!token || !activeWorkspaceId || !job.asset_id) {
      return;
    }
    setLatestFolderUpload(null);
    await refreshWorkspace(activeWorkspaceId, job.asset_id);
    const detail = await getAsset(job.asset_id, token);
    startTransition(() => {
      setActiveAssetId(job.asset_id || null);
      setAssetDetail(detail);
      setPanel("assets");
    });
    setStatusMessage(
      `${job.source_type === "google_drive" ? "Google Drive" : "Kaggle"} import is ready for AI analysis.`
    );
  }

  async function handleFolderAutoAnalyze(prompt: string, assetId?: string | null) {
    if (!token) {
      return;
    }

    const resolvedAssetId = assetId || latestFolderUpload?.asset?.asset_id || activeAssetId;
    let resolvedAssetDetail = assetDetail;
    if (resolvedAssetId && assetDetail?.asset_id !== resolvedAssetId) {
      resolvedAssetDetail = await getAsset(resolvedAssetId, token);
    }

    const resolvedDatasetId =
      resolvedAssetDetail?.datasets[0]?.dataset_id || latestFolderUpload?.asset?.datasets[0]?.dataset_id || null;
    if (!resolvedDatasetId) {
      setStatusMessage("Upload a dataset folder with CSV or XLSX files before running auto analysis.");
      return;
    }

    startTransition(() => {
      setPanel("solve");
      setWorkbenchMode("analysis");
      setAnalysisPrompt(prompt);
      setActiveAssetId(resolvedAssetId || null);
      setAssetDetail(resolvedAssetDetail || null);
      setSelectedDatasetId(resolvedDatasetId);
    });
    await handleStartAnalysis(prompt, resolvedDatasetId);
  }

  function handleAskDataShortcut(question: string) {
    startTransition(() => {
      setPanel("solve");
      setChatSeedQuestion(question);
    });
    setStatusMessage("Loaded the question into Ask Your Data.");
  }

  async function handleTransformDataset() {
    if (!token || !selectedDatasetId || !transformPrompt.trim() || !activeWorkspaceId) {
      return;
    }
    const steps = ["Prepare transform", "Create derived dataset", "Refresh lineage"];
    setWorking(true);
    setStatusMessage("Creating a derived dataset...");
    setActivityFlow(
      buildActivityFlow(
        "Building derived dataset",
        steps,
        0,
        "info",
        "Aidssist is turning your instruction into a reusable dataset version."
      )
    );
    try {
      await transformDataset(selectedDatasetId, transformPrompt.trim(), token);
      setActivityFlow(
        buildActivityFlow(
          "Building derived dataset",
          steps,
          1,
          "info",
          "Derived dataset created. Syncing lineage and previews."
        )
      );
      await refreshWorkspace(activeWorkspaceId, activeAssetId);
      if (activeAssetId) {
        const detail = await getAsset(activeAssetId, token);
        setAssetDetail(detail);
      }
      setStatusMessage("Derived dataset created successfully.");
      startTransition(() => setPanel("studio"));
      setActivityFlow(
        completeActivityFlow("Derived dataset ready", steps, "Workspace lineage has been updated.")
      );
    } catch (error) {
      const message = normalizeErrorMessage(error, "Dataset transform failed.");
      setStatusMessage(message);
      setActivityFlow(failActivityFlow("Dataset transform failed", steps, 1, message));
    } finally {
      setWorking(false);
    }
  }

  async function handleStartSolve() {
    if (!token || !activeWorkspaceId || !solvePrompt.trim()) {
      return;
    }
    const steps = ["Queue orchestration", "Execute solver plan", "Validate final output"];
    setWorking(true);
    setStatusMessage("Launching the solver orchestrator...");
    setActivityFlow(
      buildActivityFlow(
        "Starting solver run",
        steps,
        0,
        "info",
        "Aidssist is queuing the orchestration request."
      )
    );
    try {
      const run = await startSolve(
        {
          workspace_id: activeWorkspaceId,
          query: solvePrompt.trim(),
          asset_id: activeAssetId,
          dataset_id: selectedDatasetId,
          route_hint: routeHint === "auto" ? null : routeHint
        },
        token
      );
      setActiveRunId(run.run_id);
      setActiveRun(run);
      startTransition(() => setPanel("solve"));
      setStatusMessage(`Solve run ${run.run_id.slice(0, 8)} started.`);
      setActivityFlow(buildAsyncJobFlow("Solver run in progress", steps, run.status, run.error_message));
    } catch (error) {
      const message = normalizeErrorMessage(error, "Solver run failed to start.");
      setStatusMessage(message);
      setActivityFlow(failActivityFlow("Solver run failed", steps, 0, message));
    } finally {
      setWorking(false);
    }
  }

  async function handleStartAnalysis(promptOverride?: string, datasetIdOverride?: string | null) {
    const resolvedPrompt = (promptOverride ?? analysisPrompt).trim();
    const targetDatasetId = datasetIdOverride ?? selectedDatasetId;
    if (!token || !targetDatasetId || !resolvedPrompt || !customRangeComplete) {
      return;
    }
    const steps = ["Queue analysis", "Generate insights", "Render dashboard"];
    setWorking(true);
    setStatusMessage("Launching dataset analysis...");
    setActivityFlow(
      buildActivityFlow(
        "Starting analysis",
        steps,
        0,
        "info",
        "Aidssist is preparing the dataset analysis flow."
      )
    );
    try {
      const job = await startAnalysisJob(
        {
          dataset_id: targetDatasetId,
          query: resolvedPrompt,
          workflow_context: buildTimeFilterContext()
        },
        token
      );
      setAnalysisJobId(job.job_id);
      setAnalysisJob(null);
      setAnalysisPrompt(resolvedPrompt);
      if (targetDatasetId !== selectedDatasetId) {
        setSelectedDatasetId(targetDatasetId);
      }
      setStatusMessage(`Analysis job ${job.job_id.slice(0, 8)} started.`);
      setActivityFlow(buildAsyncJobFlow("Analysis in progress", steps, job.status));
    } catch (error) {
      const message = normalizeErrorMessage(error, "Analysis failed to start.");
      setStatusMessage(message);
      setActivityFlow(failActivityFlow("Analysis failed", steps, 0, message));
    } finally {
      setWorking(false);
    }
  }

  async function handleStartForecast() {
    if (!token || !selectedDatasetId || !customRangeComplete) {
      return;
    }
    if (!forecastEligibilityAllowed) {
      setStatusMessage("Forecasting is blocked for this dataset because no reliable time column was detected.");
      return;
    }
    const steps = ["Queue forecast", "Model trend", "Render outlook"];
    setWorking(true);
    setStatusMessage("Generating AI forecast...");
    setActivityFlow(
      buildActivityFlow(
        "Starting forecast",
        steps,
        0,
        "info",
        "Aidssist is preparing a forward-looking forecast view."
      )
    );
    try {
      const job = await startForecastJob(
        {
          dataset_id: selectedDatasetId,
          forecast_config: {
            date_column: forecastDateColumn,
            target_column: forecastTargetColumn,
            horizon: forecastHorizon,
            aggregation_frequency: forecastFrequency,
            model_strategy: forecastModelStrategy,
            training_mode: forecastTrainingMode
          },
          workflow_context: buildTimeFilterContext()
        },
        token
      );
      setForecastJobId(job.job_id);
      setForecastJob(null);
      setStatusMessage(`AI forecast job ${job.job_id.slice(0, 8)} started.`);
      setActivityFlow(buildAsyncJobFlow("Forecast in progress", steps, job.status));
    } catch (error) {
      const message = normalizeErrorMessage(error, "Forecast failed to start.");
      setStatusMessage(message);
      setActivityFlow(failActivityFlow("Forecast failed", steps, 0, message));
    } finally {
      setWorking(false);
    }
  }

  async function handleSuggestedAction(action: SuggestedAction) {
    if (action.action_type === "forecast") {
      if (!forecastEligibilityAllowed) {
        const fallbackAction = FORECAST_ALTERNATIVE_ACTIONS[0];
        loadAnalysisAlternative(
          fallbackAction.prompt,
          "Forecasting is unavailable for this dataset, so we loaded an analysis alternative instead."
        );
        return;
      }
      startTransition(() => setWorkbenchMode("forecast"));
      if (!forecastCanRun) {
        setStatusMessage("Select a dataset and complete the time filter to generate the AI forecast.");
        return;
      }
      await handleStartForecast();
      return;
    }

    startTransition(() => setWorkbenchMode("analysis"));
    await handleStartAnalysis(action.prompt || action.title);
  }

  async function handleSignOut() {
    if (token) {
      try {
        await logout(token);
      } catch {
        // best effort
      }
    }
    clearStoredToken();
    setToken(null);
    setPublicView("landing");
    setActivityFlow(null);
    setStatusMessage("Signed out.");
  }

  const filteredTimeline = timeline.filter((item) => {
    const haystack = `${item.title} ${item.summary} ${item.event_type}`.toLowerCase();
    return haystack.includes(deferredTimelineFilter.trim().toLowerCase());
  });
  const metrics = runSuccessMetrics(workspaceDetail?.recent_runs || []);
  const previewRows = resultPreview(activeRun);
  const analysisOutput = (analysisJob?.analysis_output || null) as AnalysisOutput | null;
  const forecastOutput = (forecastJob?.forecast_output || null) as ForecastOutput | null;
  const analysisDashboard = analysisOutput?.dashboard || undefined;
  const forecastDashboard = forecastOutput?.dashboard || undefined;
  const analysisSuggestions = safeSuggestedActions(analysisOutput?.suggestions);
  const forecastSuggestions = safeSuggestedActions(forecastOutput?.suggestions);
  const analysisQuestions = safeStringArray(analysisOutput?.suggested_questions);
  const forecastQuestions = safeStringArray(forecastOutput?.suggested_questions);
  const forecastRecommendations = safeForecastRecommendations(forecastOutput?.recommendations);
  const primaryForecastRecommendation = forecastRecommendations[0] || null;
  const forecastSeries = safeArray(forecastOutput?.time_series);
  const customRange = { startDate: customRangeStart, endDate: customRangeEnd };
  const analysisActiveFilter =
    typeof analysisOutput?.active_filter === "string"
      ? String(analysisOutput.active_filter)
      : analysisDashboard?.active_filter || timeFilter;
  const forecastActiveFilter = forecastOutput?.active_filter || forecastDashboard?.active_filter || timeFilter;
  const analysisVisualizationType =
    typeof analysisOutput?.visualization_type === "string"
      ? String(analysisOutput.visualization_type)
      : analysisDashboard?.visualization_type || null;
  const forecastVisualizationType = forecastOutput?.visualization_type || forecastDashboard?.visualization_type || null;
  const forecastTrendSummary =
    forecastOutput?.summary ||
    (forecastOutput?.trend_status ? `${titleCaseSlug(forecastOutput.trend_status)} trend detected.` : "Forecast insights will appear here after the run.");
  const forecastConfidenceSummary = titleCaseSlug(
    forecastOutput?.confidence?.label || forecastOutput?.auto_config?.confidence || forecastAutoConfidence
  );
  const forecastFallbackActions: SuggestedAction[] = FORECAST_ALTERNATIVE_ACTIONS.map((action, index) => ({
    title: action.label,
    prompt: action.prompt,
    action_type: "analysis",
    rationale: "Recommended because this dataset is not forecast-ready.",
    score: Math.max(1, 10 - index),
    rank: index + 1
  }));
  const forecastRecommendedAction =
    (typeof primaryForecastRecommendation?.recommended_action === "string" && primaryForecastRecommendation.recommended_action) ||
    (typeof forecastOutput?.recommended_next_step === "string" && forecastOutput.recommended_next_step) ||
    (!forecastEligibilityAllowed
      ? "Switch to analysis mode and explore relationships before attempting prediction."
      : "Review the AI forecast before making operational changes.");
  const forecastSummaryText =
    forecastOutput?.summary ||
    forecastOutput?.error?.message ||
    (!forecastEligibilityAllowed && selectedDataset
      ? "This dataset needs a reliable time column before forecasting can run. Use one of the suggested analysis paths instead."
      : "Run a forecast to populate the time-series outlook.");
  const forecastStatusLabel =
    forecastOutput?.status ||
    forecastJob?.status ||
    (!forecastEligibilityAllowed && selectedDataset ? "blocked" : "idle");
  const forecastModelLabel =
    !forecastEligibilityAllowed && !forecastOutput
      ? "blocked"
      : forecastOutput?.chosen_model || "auto";
  const forecastGuidanceSuggestions =
    forecastEligibilitySuggestions.length > 0
      ? forecastEligibilitySuggestions
      : [
          "Use analysis mode instead",
          "Upload a time-based dataset",
          "Try correlation or segmentation analysis"
        ];
  const forecastHasRenderableResults = Boolean(forecastOutput && forecastOutput.status !== "FAILED");
  const activeSuggestions: SuggestedAction[] =
    workbenchMode === "analysis"
      ? analysisSuggestions.length
        ? analysisSuggestions
        : analysisQuestions.map((question, index) => ({
            title: question,
            prompt: question,
            action_type: "analysis",
            rationale: undefined,
            score: Math.max(1, 10 - index),
            rank: index + 1
          }))
      : !forecastEligibilityAllowed && selectedDataset && !forecastSuggestions.length && !forecastQuestions.length
        ? forecastFallbackActions
        : forecastSuggestions.length
        ? forecastSuggestions
        : forecastQuestions.map((question, index) => ({
            title: question,
            prompt: question,
            action_type: "analysis",
            rationale: undefined,
            score: Math.max(1, 10 - index),
            rank: index + 1
          }));
  const activeRecommendedNextStep =
    (workbenchMode === "analysis"
      ? typeof analysisOutput?.recommended_next_step === "string"
        ? String(analysisOutput.recommended_next_step)
        : null
      : forecastOutput?.recommended_next_step || null) ||
    activeSuggestions[0]?.prompt ||
    null;
  const activeContext =
    workbenchMode === "analysis"
      ? ((analysisOutput?.context as unknown as Record<string, unknown> | undefined) || undefined)
      : ((forecastOutput?.context as unknown as Record<string, unknown> | undefined) || undefined);
  const publicDemoFlow = demoData?.flow.length ? demoData.flow : DEFAULT_DEMO_FLOW;
  const publicDemoStats = demoData?.stats.length ? demoData.stats : DEFAULT_DEMO_STATS;
  const demoDataset = demoData?.datasets[0] || demoData?.dataset || null;
  const demoActiveEntry = demoData?.outputs[demoOutputIndex] || null;
  const demoActivePayload = safeObject(demoActiveEntry?.output);
  const demoActiveDashboard =
    (demoActivePayload.dashboard as DashboardPayload | undefined) || demoData?.dashboard || undefined;
  const demoActiveSummary =
    typeof demoActivePayload.summary === "string"
      ? demoActivePayload.summary
      : typeof demoActivePayload.error_message === "string"
        ? demoActivePayload.error_message
        : "Switch between analysis, forecast, and root-cause views to walk through the product story.";
  const demoActiveSuggestions = safeSuggestedActions(demoActivePayload.suggestions || demoData?.suggestions || []);
  const demoSuggestedQuestions = safeStringArray(demoActivePayload.suggested_questions);
  const demoActiveFilter =
    typeof demoActivePayload.active_filter === "string"
      ? demoActivePayload.active_filter
      : demoActiveDashboard?.active_filter || DEMO_TIME_FILTER;
  const demoVisualizationType =
    typeof demoActivePayload.visualization_type === "string"
      ? demoActivePayload.visualization_type
      : demoActiveDashboard?.visualization_type || null;
  const demoTimeSeries = safeArray(demoActivePayload.time_series);
  const demoOutputStatus =
    typeof demoActivePayload.status === "string"
      ? demoActivePayload.status
      : demoActiveEntry
        ? "READY"
        : "IDLE";
  const demoOutputConfidence =
    typeof safeObject(demoActivePayload.confidence).label === "string"
      ? String(safeObject(demoActivePayload.confidence).label)
      : typeof demoActivePayload.confidence === "string"
        ? String(demoActivePayload.confidence)
        : "high";
  const demoRecommendedNextStep =
    typeof demoActivePayload.recommended_next_step === "string"
      ? demoActivePayload.recommended_next_step
      : demoActiveSuggestions[0]?.prompt || null;
  const demoCustomRange = { startDate: "", endDate: "" };

  if (!token || !user) {
    if (publicView === "demo" && demoData) {
      return (
        <main className="public-shell public-shell--demo">
          <header className="public-nav">
            <div className="public-brand">
              <span className="brand-kicker">Public Demo</span>
              <h1>Aidssist</h1>
            </div>
            <div className="public-nav__actions">
              <button type="button" className="ghost-button" onClick={() => void handleLoadDemo(true)} disabled={demoLoading}>
                {demoLoading ? "Refreshing..." : "Refresh demo"}
              </button>
              <button type="button" className="ghost-button" onClick={() => handleExitDemo("register")}>
                Back to landing
              </button>
            </div>
          </header>

          <section className="demo-hero">
            <div className="demo-hero__copy">
              <span className="section-kicker">Demo Mode</span>
              <h2>Preloaded dashboard, forecast, and AI suggestions</h2>
              <p>
                This public workspace uses sample retail data so you can show the full product flow without
                waiting for uploads or external model credentials.
              </p>
              <div className="landing-actions">
                <button type="button" className="primary-button-new" onClick={() => handleExitDemo("register")}>
                  Create workspace account
                </button>
                <button type="button" className="ghost-button" onClick={() => handleExitDemo("login")}>
                  Sign in
                </button>
              </div>
            </div>
            <div className="landing-stat-grid">
              {publicDemoStats.map((stat) => (
                <article key={stat.label} className="landing-stat-card">
                  <span>{stat.label}</span>
                  <strong>{formatStatValue(stat.value)}</strong>
                  {stat.detail ? <small>{stat.detail}</small> : null}
                </article>
              ))}
            </div>
          </section>

          <ActivityTracker flow={activityFlow} />
          {demoError ? <div className="notice-card notice-card--error">{demoError}</div> : null}

          <section className="surface-card demo-tabs-panel">
            <span className="section-kicker">Demo Views</span>
            <h3>Switch the story in one click</h3>
            <div className="demo-output-tabs">
              {demoData.outputs.map((item, index) => (
                <button
                  key={`${item.intent}-${item.query}`}
                  type="button"
                  className={`demo-output-tab ${demoOutputIndex === index ? "demo-output-tab--active" : ""}`}
                  onClick={() => setDemoOutputIndex(index)}
                >
                  <span>{item.intent.replace(/_/g, " ")}</span>
                  <strong>{item.query}</strong>
                </button>
              ))}
            </div>
          </section>

          <section className="panel-grid panel-grid--public">
            <article className="surface-card surface-card--wide">
              <span className="section-kicker">Active Output</span>
              <h3>{demoActiveEntry?.query || "Demo output"}</h3>
              <div className="mini-stat-grid">
                <div>
                  <span>Status</span>
                  <strong>{demoOutputStatus}</strong>
                </div>
                <div>
                  <span>Confidence</span>
                  <strong>{demoOutputConfidence}</strong>
                </div>
                <div>
                  <span>Filter</span>
                  <strong>{demoActiveFilter.replace(/_/g, " ")}</strong>
                </div>
                <div>
                  <span>Primary view</span>
                  <strong>{demoVisualizationType || "dashboard"}</strong>
                </div>
              </div>
              <p>{demoActiveSummary}</p>
              <DashboardGrid dashboard={demoActiveDashboard} filterType={demoActiveFilter} customRange={demoCustomRange} />
            </article>

            <article className="surface-card">
              <span className="section-kicker">Next Step</span>
              <h3>Helpful guidance, never a blank screen</h3>
              {demoRecommendedNextStep ? (
                <div className="best-next-step">
                  <span>Best next step</span>
                  <strong>{demoRecommendedNextStep}</strong>
                </div>
              ) : null}
              <div className="recommendation-list">
                {demoActiveSuggestions.length ? (
                  demoActiveSuggestions.map((suggestion, index) => (
                    <div key={`${suggestion.title}-${index}`} className={`recommendation-card ${index === 0 ? "recommendation-card--best" : ""}`}>
                      <strong>{suggestion.title}</strong>
                      <p>{suggestion.prompt}</p>
                      <small>{suggestion.rationale || "Ranked to keep the demo moving with a clear story."}</small>
                      <div className="suggestion-actions">
                        <button
                          type="button"
                          className="primary-button-new"
                          onClick={() => {
                            const targetIndex = demoData.outputs.findIndex((item) =>
                              suggestion.action_type === "forecast" ? item.intent === "forecast" : item.intent !== "forecast"
                            );
                            setDemoOutputIndex(targetIndex >= 0 ? targetIndex : 0);
                          }}
                        >
                          {suggestion.action_type === "forecast" ? "Open forecast" : "Open analysis"}
                        </button>
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="notice-card">
                    Load the public demo to see suggested questions and recruiter-friendly talking points.
                  </div>
                )}
              </div>
              {demoSuggestedQuestions.length ? (
                <div className="demo-question-list">
                  {demoSuggestedQuestions.map((question) => (
                    <div key={question} className="forecast-block-suggestion">
                      {question}
                    </div>
                  ))}
                </div>
              ) : null}
            </article>

            <article className="surface-card surface-card--wide">
              <span className="section-kicker">Sample Dataset</span>
              <h3>{demoDataset?.metadata.file_name || "Bundled demo data"}</h3>
              <div className="mini-stat-grid">
                <div>
                  <span>Rows</span>
                  <strong>{formatCount(demoDataset?.metadata.row_count || 0)}</strong>
                </div>
                <div>
                  <span>Columns</span>
                  <strong>{formatCount(demoDataset?.metadata.column_count || 0)}</strong>
                </div>
                <div>
                  <span>Time column</span>
                  <strong>{demoDataset?.metadata.time_column || "n/a"}</strong>
                </div>
                <div>
                  <span>Target</span>
                  <strong>{demoDataset?.metadata.target_column || "n/a"}</strong>
                </div>
              </div>
              {demoDataset ? (
                <div className="table-shell">
                  <table>
                    <thead>
                      <tr>
                        {demoDataset.metadata.columns.map((column) => (
                          <th key={column}>{column}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {demoDataset.rows.slice(0, 8).map((row, rowIndex) => (
                        <tr key={`demo-row-${rowIndex}`}>
                          {demoDataset.metadata.columns.map((column) => (
                            <td key={`${column}-${rowIndex}`}>{String(row[column] ?? "")}</td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : null}
              {demoTimeSeries.length ? (
                <div className="table-shell demo-table-shell">
                  <table>
                    <thead>
                      <tr>
                        <th>Date</th>
                        <th>Value</th>
                        <th>Series</th>
                      </tr>
                    </thead>
                    <tbody>
                      {demoTimeSeries.slice(0, 8).map((row, rowIndex) => (
                        <tr key={`demo-series-${rowIndex}`}>
                          <td>{String(row.date ?? "")}</td>
                          <td>{toNumeric(row.value).toLocaleString()}</td>
                          <td>{String(row.series ?? "")}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : null}
            </article>

            <article className="surface-card">
              <span className="section-kicker">Demo Script</span>
              <h3>Fast to explain</h3>
              <div className="public-flow-list">
                {publicDemoFlow.map((step, index) => (
                  <div key={`${step.title}-${index}`} className="public-flow-item">
                    <span>{index + 1}</span>
                    <div>
                      <strong>{step.title}</strong>
                      <p>{step.description}</p>
                    </div>
                  </div>
                ))}
              </div>
              <div className="notice-card">
                Use this view for demos, then create an account to upload your own datasets and project files.
              </div>
            </article>
          </section>
        </main>
      );
    }

    return (
      <main className="public-shell">
        <header className="public-nav">
          <div className="public-brand">
            <span className="brand-kicker">AI Data Intelligence System</span>
            <h1>Aidssist</h1>
          </div>
          <div className="public-nav__actions">
            <button type="button" className="ghost-button" onClick={() => setAuthMode("login")}>
              Sign in
            </button>
            <button type="button" className="primary-button-new" onClick={() => void handleLoadDemo()} disabled={demoLoading}>
              {demoLoading ? "Loading demo..." : "Try Demo"}
            </button>
          </div>
        </header>

        <section className="landing-hero">
          <div className="landing-copy">
            <span className="section-kicker">Demo-ready product</span>
            <h2>Public AI analytics workspace with dashboards, forecasts, and AI suggestions</h2>
            <p>
              Explore a polished sample workflow instantly, then sign in to upload your own datasets, run
              forecasting, and inspect the full reasoning trace behind every result.
            </p>
            <div className="landing-actions">
              <button type="button" className="primary-button-new" onClick={() => void handleLoadDemo()} disabled={demoLoading}>
                {demoLoading ? "Loading demo..." : "Try Demo"}
              </button>
              <button type="button" className="ghost-button" onClick={() => setAuthMode("login")}>
                Open sign in
              </button>
            </div>
            <div className="landing-stat-grid">
              {publicDemoStats.map((stat) => (
                <article key={stat.label} className="landing-stat-card">
                  <span>{stat.label}</span>
                  <strong>{formatStatValue(stat.value)}</strong>
                  {stat.detail ? <small>{stat.detail}</small> : null}
                </article>
              ))}
            </div>
          </div>

          <div className="landing-shot">
            <img src={demoDashboardScreenshot} alt="Aidssist dashboard screenshot" />
            <div className="landing-shot__badge">
              <span>Live preview</span>
              <strong>Dashboard + forecasting + AI suggestions</strong>
            </div>
          </div>
        </section>

        <ActivityTracker flow={activityFlow} />
        {demoError ? <div className="notice-card notice-card--error">{demoError}</div> : null}

        <section className="landing-grid">
          <article className="surface-card landing-story-card">
            <span className="section-kicker">Why It Works</span>
            <h3>Fast to understand, stable to demo, flexible to deploy</h3>
            <div className="feature-grid">
              {LANDING_FEATURES.map((feature) => (
                <article key={feature.title} className="feature-card">
                  <strong>{feature.title}</strong>
                  <p>{feature.body}</p>
                </article>
              ))}
            </div>

            <div className="public-flow-list">
              {publicDemoFlow.map((step, index) => (
                <div key={`${step.title}-${index}`} className="public-flow-item">
                  <span>{index + 1}</span>
                  <div>
                    <strong>{step.title}</strong>
                    <p>{step.description}</p>
                  </div>
                </div>
              ))}
            </div>
          </article>

          <article className="surface-card landing-auth-card">
            <span className="section-kicker">Get Started</span>
            <h3>{authMode === "register" ? "Create a workspace account" : "Sign in to your workspace"}</h3>
            <p>
              Start with the public demo, or sign in now to upload real data, run analyses, and save your
              workspace history.
            </p>
            <form
              className="auth-form-new"
              onSubmit={(event) => {
                event.preventDefault();
                void handleAuthSubmit(new FormData(event.currentTarget));
              }}
            >
              {authMode === "register" ? (
                <input name="displayName" placeholder="Display name" required />
              ) : null}
              <input name="email" type="email" placeholder="Work email" required />
              <input name="password" type="password" placeholder="Password" required />
              <button type="submit" disabled={working}>
                {working ? "Working..." : authMode === "register" ? "Create workspace account" : "Sign in"}
              </button>
            </form>
            {authError ? <div className="notice-card notice-card--error">{authError}</div> : null}
            <button
              className="auth-switch"
              type="button"
              onClick={() => setAuthMode(authMode === "register" ? "login" : "register")}
            >
              {authMode === "register"
                ? "Already have an account? Sign in."
                : "Need an account? Create one."}
            </button>
            <div className="recruiter-list">
              {RECRUITER_HIGHLIGHTS.map((item) => (
                <div key={item} className="recruiter-list__item">
                  {item}
                </div>
              ))}
            </div>
          </article>
        </section>
      </main>
    );
  }

  return (
    <div className="solver-shell">
      <aside className="control-rail">
        <div className="brand-block">
          <span className="brand-kicker">React First</span>
          <h1>Aidssist</h1>
          <p>Solver workspace for mixed files, retrieval memory, validator loops, and packaged outputs.</p>
        </div>

        <div className="operator-card">
          <span>Operator</span>
          <strong>{user.display_name}</strong>
          <small>{user.email}</small>
        </div>

        <nav className="panel-nav">
          {PANEL_ITEMS.map((item) => (
            <button
              key={item.id}
              type="button"
              className={`panel-nav__item ${panel === item.id ? "panel-nav__item--active" : ""}`}
              onClick={() => startTransition(() => setPanel(item.id))}
            >
              <span>{item.label}</span>
              <small>{item.description}</small>
            </button>
          ))}
        </nav>

        <div className="rail-stats">
          <div>
            <span>Success rate</span>
            <strong>{formatPercent(metrics.successRate)}</strong>
          </div>
          <div>
            <span>Avg latency</span>
            <strong>{metrics.averageLatency}</strong>
          </div>
          <div>
            <span>Completed runs</span>
            <strong>{formatCount(metrics.completedCount)}</strong>
          </div>
        </div>
      </aside>

      <main className="solver-main">
        <header className="mission-bar">
          <div>
            <span className="section-kicker">Workspace Shell</span>
            <h2>{workspaceDetail?.name || "Loading workspace..."}</h2>
            <p>{workspaceDetail?.description || "Canonical workspace flow for ingestion, transforms, and solve traces."}</p>
          </div>
          <div className="mission-bar__actions">
            <select
              value={activeWorkspaceId || ""}
              onChange={(event) => startTransition(() => setActiveWorkspaceId(event.target.value))}
            >
              {workspaces.map((workspace) => (
                <option key={workspace.workspace_id} value={workspace.workspace_id}>
                  {workspace.name}
                </option>
              ))}
            </select>
            <button type="button" className="ghost-button" onClick={() => void handleCreateWorkspace()}>
              New workspace
            </button>
            <button type="button" className="ghost-button" onClick={() => void handleSignOut()}>
              Sign out
            </button>
          </div>
        </header>

        <div className="status-ribbon">{statusMessage}</div>
        <ActivityTracker flow={activityFlow} />

        <section className="hero-grid-new">
          <article className="hero-card hero-card--primary">
            <span className="section-kicker">Pipeline</span>
            <h3>{"Ingestion -> Retrieval -> Validation -> Output"}</h3>
            <p>
              The workspace stores uploaded files, chunked context, embedding memory, validator reports, and
              final packaged outputs so every solve run gets better grounded over time.
            </p>
          </article>
          <article className="hero-card">
            <span className="metric-label">Assets</span>
            <strong>{formatCount(workspaceDetail?.asset_count || 0)}</strong>
            <p>Uploaded datasets, code files, archives, and mixed project bundles.</p>
          </article>
          <article className="hero-card">
            <span className="metric-label">Derived datasets</span>
            <strong>{formatCount(workspaceDetail?.derived_dataset_count || 0)}</strong>
            <p>Versioned transforms stay linked to the original workspace history.</p>
          </article>
          <article className="hero-card">
            <span className="metric-label">Solve runs</span>
            <strong>{formatCount(workspaceDetail?.solve_run_count || 0)}</strong>
            <p>Reasoning traces, retries, and validator loops are all preserved.</p>
          </article>
        </section>

        {panel === "mission" ? (
          <section className="panel-grid">
            <article className="surface-card">
              <span className="section-kicker">Mission Control</span>
              <h3>Workspace overview</h3>
              <div className="mini-stat-grid">
                <div>
                  <span>Assets</span>
                  <strong>{formatCount(workspaceDetail?.asset_count || 0)}</strong>
                </div>
                <div>
                  <span>Datasets</span>
                  <strong>{formatCount(workspaceDetail?.dataset_count || 0)}</strong>
                </div>
                <div>
                  <span>Recent runs</span>
                  <strong>{formatCount(workspaceDetail?.recent_runs.length || 0)}</strong>
                </div>
              </div>
              <div className="insight-list">
                <div>
                  <h4>Upload mixed inputs</h4>
                  <p>CSV, XLSX, ZIP archives, code files, and config files share one canonical workspace model.</p>
                </div>
                <div>
                  <h4>Ground every solve</h4>
                  <p>Retrieval uses stored chunks, embeddings, and prior feedback scores before reasoning begins.</p>
                </div>
                <div>
                  <h4>Keep the full trace</h4>
                  <p>Each run stores the plan, refinement attempts, validator reports, and packaged output.</p>
                </div>
              </div>
            </article>

            <article className="surface-card">
              <span className="section-kicker">Recent Assets</span>
              <h3>Latest uploads</h3>
              <div className="asset-stack">
                {(workspaceDetail?.assets || []).slice(0, 5).map((asset) => (
                  <button
                    key={asset.asset_id}
                    type="button"
                    className="asset-chip"
                    onClick={() => {
                      startTransition(() => {
                        setActiveAssetId(asset.asset_id);
                        setPanel("assets");
                      });
                    }}
                  >
                    <strong>{asset.title}</strong>
                    <span>
                      {asset.asset_kind} · {asset.file_count} files · {asset.chunk_count} chunks
                    </span>
                  </button>
                ))}
              </div>
            </article>

            <article className="surface-card surface-card--wide">
              <span className="section-kicker">Timeline</span>
              <h3>Workspace pulse</h3>
              <div className="timeline-list">
                {filteredTimeline.slice(0, 8).map((item) => (
                  <div key={item.event_id} className="timeline-item">
                    <span>{item.event_type.replace(/_/g, " ")}</span>
                    <strong>{item.title}</strong>
                    <p>{item.summary}</p>
                    <small>{item.created_at}</small>
                  </div>
                ))}
              </div>
            </article>
          </section>
        ) : null}

        {panel === "assets" ? (
          <section className="panel-grid">
            <FolderUploadPanel
              token={token}
              workspaceId={activeWorkspaceId}
              latestUpload={latestFolderUpload}
              onUploaded={(payload) => void handleFolderUploadCompleted(payload)}
              onAutoAnalyze={(prompt, assetId) => void handleFolderAutoAnalyze(prompt, assetId)}
            />

            <ImportHubPanel
              workspaceId={activeWorkspaceId}
              token={token}
              onImported={(job) => void handleExternalImportCompleted(job)}
              onStatus={setStatusMessage}
            />

            <article className="surface-card">
              <span className="section-kicker">Mixed Asset Intake</span>
              <h3>Upload scripts, notes, archives, or one-off files</h3>
              <p>
                Keep the premium folder flow for datasets, and use this lane for code, markdown, ZIP bundles,
                and supporting project files that should still be indexed into the workspace.
              </p>
              <div className="upload-controls">
                <input
                  value={uploadTitle}
                  onChange={(event) => setUploadTitle(event.target.value)}
                  placeholder="Optional asset title"
                />
                <input
                  key={fileInputResetKey}
                  type="file"
                  multiple
                  accept=".csv,.xlsx,.xlsm,.txt,.md,.json,.py,.js,.ts,.sql,.yaml,.yml,.zip"
                  onChange={(event) => setQueuedFiles(Array.from(event.target.files || []))}
                />
                <button type="button" className="primary-button-new" disabled={working} onClick={() => void handleAssetUpload()}>
                  {working ? "Uploading..." : "Ingest files"}
                </button>
              </div>
              <div className="file-pill-row">
                {queuedFiles.map((file) => (
                  <span key={`${file.name}-${file.size}`} className="file-pill">
                    {file.name}
                  </span>
                ))}
              </div>
            </article>

            <article className="surface-card">
              <span className="section-kicker">Asset Library</span>
              <h3>Workspace assets</h3>
              <div className="asset-stack">
                {(workspaceDetail?.assets || []).map((asset) => (
                  <button
                    key={asset.asset_id}
                    type="button"
                    className={`asset-chip ${activeAssetId === asset.asset_id ? "asset-chip--active" : ""}`}
                    onClick={() => setActiveAssetId(asset.asset_id)}
                  >
                    <strong>{asset.title}</strong>
                    <span>
                      {asset.asset_kind} · {asset.file_count} files · {asset.dataset_count} datasets
                    </span>
                  </button>
                ))}
              </div>
            </article>

            <article className="surface-card surface-card--wide">
              <span className="section-kicker">Selected Asset</span>
              <h3>{assetDetail?.title || "Choose an asset"}</h3>
              {assetDetail ? (
                <div className="detail-grid">
                  <div>
                    <h4>Files</h4>
                    <div className="table-shell">
                      <table>
                        <thead>
                          <tr>
                            <th>Name</th>
                            <th>Kind</th>
                            <th>Language</th>
                            <th>Size</th>
                          </tr>
                        </thead>
                        <tbody>
                          {assetDetail.files.map((file) => (
                            <tr key={file.asset_file_id}>
                              <td>{file.file_name}</td>
                              <td>{file.file_kind}</td>
                              <td>{file.language || "n/a"}</td>
                              <td>{formatCount(file.size_bytes)} B</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                  <div>
                    <h4>Datasets</h4>
                    <div className="asset-stack">
                      {assetDetail.datasets.map((dataset) => (
                        <button
                          key={dataset.dataset_id}
                          type="button"
                          className={`dataset-pill ${selectedDatasetId === dataset.dataset_id ? "dataset-pill--active" : ""}`}
                          onClick={() => setSelectedDatasetId(dataset.dataset_id)}
                        >
                          <strong>{dataset.dataset_name}</strong>
                          <span>
                            {formatCount(dataset.row_count)} rows · {formatCount(dataset.column_count)} columns
                          </span>
                        </button>
                      ))}
                    </div>
                  </div>
                </div>
              ) : (
                <p>No asset is selected yet.</p>
              )}
            </article>

            <AssetIntelligencePanel
              assetId={activeAssetId}
              token={token}
              onAskQuestion={handleAskDataShortcut}
            />
          </section>
        ) : null}

        {panel === "studio" ? (
          <section className="panel-grid">
            <article className="surface-card surface-card--wide">
              <span className="section-kicker">Dataset Studio</span>
              <h3>Version transforms instead of overwriting the source</h3>
              <div className="studio-controls">
                <select
                  value={selectedDatasetId || ""}
                  onChange={(event) => setSelectedDatasetId(event.target.value)}
                >
                  {(assetDetail?.datasets || []).map((dataset) => (
                    <option key={dataset.dataset_id} value={dataset.dataset_id}>
                      {dataset.dataset_name}
                    </option>
                  ))}
                </select>
                <textarea
                  value={transformPrompt}
                  onChange={(event) => setTransformPrompt(event.target.value)}
                  placeholder="Example: Rename burnout score to burnout_index and keep columns student_id, burnout_score"
                />
                <button
                  type="button"
                  className="primary-button-new"
                  disabled={working || !selectedDatasetId}
                  onClick={() => void handleTransformDataset()}
                >
                  {working ? "Transforming..." : "Create derived dataset"}
                </button>
              </div>
            </article>

            <article className="surface-card">
              <span className="section-kicker">Source Preview</span>
              <h3>{selectedDataset?.dataset_name || "Select a dataset"}</h3>
              {selectedDataset ? (
                <>
                  <div className="mini-stat-grid">
                    <div>
                      <span>Rows</span>
                      <strong>{formatCount(selectedDataset.row_count)}</strong>
                    </div>
                    <div>
                      <span>Columns</span>
                      <strong>{formatCount(selectedDataset.column_count)}</strong>
                    </div>
                    <div>
                      <span>Missing cells</span>
                      <strong>{formatCount(selectedDataset.missing_cell_count)}</strong>
                    </div>
                  </div>
                  <div className="table-shell">
                    <table>
                      <thead>
                        <tr>
                          {selectedDataset.preview_columns.map((column) => (
                            <th key={column}>{column}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {selectedDataset.preview_rows.slice(0, 6).map((row, index) => (
                          <tr key={`preview-${index}`}>
                            {selectedDataset.preview_columns.map((column) => (
                              <td key={`${column}-${index}`}>{String(row[column] ?? "")}</td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </>
              ) : (
                <p>Choose an uploaded dataset to unlock transform workflows.</p>
              )}
            </article>

            <article className="surface-card">
              <span className="section-kicker">Derived Versions</span>
              <h3>Workspace lineage</h3>
              <div className="asset-stack">
                {(workspaceDetail?.derived_datasets || []).map((dataset) => (
                  <div key={dataset.derived_dataset_id} className="asset-chip asset-chip--static">
                    <strong>{dataset.dataset_name}</strong>
                    <span>
                      {formatCount(dataset.row_count)} rows · {formatCount(dataset.column_count)} columns
                    </span>
                  </div>
                ))}
              </div>
            </article>
          </section>
        ) : null}

        {panel === "solve" ? (
          <section className="panel-grid">
            <AskDataChatPanel
              assetId={activeAssetId}
              token={token}
              initialQuestion={chatSeedQuestion}
              onQuestionConsumed={() => setChatSeedQuestion(null)}
            />

            <article className="surface-card surface-card--wide">
              <span className="section-kicker">AI Workbench</span>
              <h3>Analysis, forecasting, and orchestration in one workspace</h3>
              <div className="workbench-tabs">
                {(["analysis", "forecast", "orchestrator"] as WorkbenchMode[]).map((mode) => (
                  <button
                    key={mode}
                    type="button"
                    className={`workbench-tab ${workbenchMode === mode ? "workbench-tab--active" : ""}`}
                    onClick={() => setWorkbenchMode(mode)}
                  >
                    {mode === "analysis" ? "Analysis" : mode === "forecast" ? "Forecast" : "AI Solve"}
                  </button>
                ))}
              </div>
              <div className="solve-controls solve-controls--filters">
                <div>
                  <span className="section-kicker">Global Time Filter</span>
                  <p>Applied across analysis, forecasting, dashboards, and chart rendering.</p>
                </div>
                <div className="solve-actions">
                  <select value={timeFilter} onChange={(event) => setTimeFilter(event.target.value)}>
                    {UNIVERSAL_TIME_FILTERS.map((option) => (
                      <option key={option.value} value={option.value}>
                        {option.label}
                      </option>
                    ))}
                  </select>
                  {timeFilter === "custom_range" ? (
                    <>
                      <input
                        type="date"
                        value={customRangeStart}
                        onChange={(event) => setCustomRangeStart(event.target.value)}
                        aria-label="Custom range start"
                      />
                      <input
                        type="date"
                        value={customRangeEnd}
                        onChange={(event) => setCustomRangeEnd(event.target.value)}
                        aria-label="Custom range end"
                      />
                    </>
                  ) : null}
                </div>
                {timeFilter === "custom_range" && !customRangeComplete ? (
                  <p>Select both a start date and an end date to apply the custom range.</p>
                ) : null}
              </div>

              {workbenchMode === "analysis" ? (
                <div className="solve-controls">
                  <textarea
                    value={analysisPrompt}
                    onChange={(event) => setAnalysisPrompt(event.target.value)}
                    placeholder="Ask a business question, request a dashboard, or explore the dataset."
                  />
                  <div className="solve-actions">
                    <button
                      type="button"
                      className="primary-button-new"
                      disabled={working || !selectedDatasetId || !customRangeComplete}
                      onClick={() => void handleStartAnalysis()}
                    >
                      {working ? "Submitting..." : "Run analysis"}
                    </button>
                  </div>
                </div>
              ) : null}

              {workbenchMode === "forecast" ? (
                <div className="solve-controls solve-controls--forecast">
                  <div className="forecast-control-header">
                    <div>
                      <span className="section-kicker">AI Settings (Auto)</span>
                      <p>
                        {forecastEligibilityAllowed
                          ? "Forecasts now auto-configure the time column, KPI, frequency, and a sensible horizon before the run starts."
                          : "Aidssist checks whether a dataset is forecast-ready before enabling AI forecasting."}
                      </p>
                    </div>
                    {selectedDataset && forecastEligibilityAllowed ? (
                      <button
                        type="button"
                        className="ghost-button"
                        onClick={() => setShowForecastAdvanced((current) => !current)}
                      >
                        {showForecastAdvanced ? "Hide Advanced Options" : "Advanced Options"}
                      </button>
                    ) : null}
                  </div>

                  {selectedDataset ? (
                    forecastEligibilityAllowed ? (
                      <>
                        <div className="forecast-preview-card">
                          <div className="forecast-preview-card__header">
                            <div>
                              <span className="section-kicker">Detected Configuration</span>
                              <p>Preview the AI-selected setup before you generate the forecast.</p>
                            </div>
                            <div className={`forecast-confidence-pill forecast-confidence-pill--${forecastAutoConfidence || "low"}`}>
                              <strong>{titleCaseSlug(forecastAutoConfidence || "low")} confidence</strong>
                              {typeof forecastAutoConfidenceScore === "number"
                                ? <span>{formatPercent(forecastAutoConfidenceScore * 100)}</span>
                                : null}
                            </div>
                          </div>

                          <div className="forecast-preview-grid">
                            <div className={`forecast-preview-item ${forecastDateNeedsAttention ? "forecast-preview-item--warning" : forecastUsingManualDateOverride ? "forecast-preview-item--manual" : "forecast-preview-item--auto"}`}>
                              <span>Date column</span>
                              <strong>{forecastEffectiveDateColumn || "Not detected"}</strong>
                              <small>
                                {forecastUsingManualDateOverride
                                  ? "Manual override"
                                  : forecastEffectiveDateColumn
                                    ? "Auto-detected"
                                    : "Needs selection"}
                              </small>
                            </div>
                            <div className={`forecast-preview-item ${forecastTargetNeedsAttention ? "forecast-preview-item--warning" : forecastUsingManualTargetOverride ? "forecast-preview-item--manual" : "forecast-preview-item--auto"}`}>
                              <span>Target</span>
                              <strong>{forecastEffectiveTargetColumn || "Not detected"}</strong>
                              <small>
                                {forecastUsingManualTargetOverride
                                  ? "Manual override"
                                  : forecastEffectiveTargetColumn
                                    ? "Auto-detected"
                                    : "Needs selection"}
                              </small>
                            </div>
                            <div className="forecast-preview-item forecast-preview-item--auto">
                              <span>Frequency</span>
                              <strong>{forecastResolvedFrequencyLabel || "Auto detect"}</strong>
                              <small>{forecastFrequency === "auto" ? "AI-managed" : "Manual override"}</small>
                            </div>
                            <div className="forecast-preview-item forecast-preview-item--auto">
                              <span>Data points</span>
                              <strong>{formatCount(forecastAutoConfig?.data_points ?? forecastHistoryPoints)}</strong>
                              <small>{formatDateRange(forecastDateRange)}</small>
                            </div>
                          </div>

                          {forecastDateNeedsAttention ? (
                            <p className="helper-text helper-text--warning">
                              ⚠️ We couldn’t detect a time column. Please select one in Advanced options.
                            </p>
                          ) : null}
                          {forecastTargetNeedsAttention ? (
                            <p className="helper-text helper-text--warning">
                              ⚠️ We couldn’t detect a primary KPI. Please select one numeric column in Advanced options.
                            </p>
                          ) : null}
                        </div>

                        <div className="forecast-horizon-panel">
                          <div className="forecast-horizon-panel__copy">
                            <span className="section-kicker">Forecast Window</span>
                            <p>Default horizon is chosen from the detected frequency and available history.</p>
                          </div>
                          <div className="forecast-chip-row">
                            {FORECAST_HORIZON_OPTIONS.map((option) => (
                              <button
                                key={option.value}
                                type="button"
                                className={`forecast-chip ${forecastHorizon === option.value ? "forecast-chip--active" : ""}`}
                                onClick={() => setForecastHorizon(option.value)}
                              >
                                {option.label}
                              </button>
                            ))}
                          </div>
                          <p className="helper-text">
                            AI default: <strong>{labelForOption(FORECAST_HORIZON_OPTIONS, forecastAutoHorizon)}</strong>
                          </p>
                        </div>

                        <div className="mini-stat-grid">
                          <div>
                            <span>AI model strategy</span>
                            <strong>{labelForOption(FORECAST_MODEL_STRATEGY_OPTIONS, forecastModelStrategy)}</strong>
                          </div>
                          <div>
                            <span>Selected horizon</span>
                            <strong>{forecastResolvedHorizonLabel}</strong>
                          </div>
                          <div>
                            <span>Target dtype</span>
                            <strong>
                              {forecastEffectiveTargetColumn
                                ? forecastColumnTypes[forecastEffectiveTargetColumn] || "n/a"
                                : "n/a"}
                            </strong>
                          </div>
                        </div>

                        {showForecastAdvanced ? (
                          <div className="forecast-advanced-panel">
                            <div className="forecast-mapping-grid">
                              <label className="forecast-field">
                                <span>Date column override</span>
                                <select
                                  value={forecastDateColumn}
                                  onChange={(event) => setForecastDateColumn(event.target.value)}
                                >
                                  <option value="">
                                    {forecastDateNeedsAttention ? "❌ No time column detected" : "Auto detect"}
                                  </option>
                                  {forecastColumnOptions.map((column) => (
                                    <option key={`forecast-date-${column}`} value={column}>
                                      {formatColumnOptionLabel(column, forecastColumnTypes)}
                                    </option>
                                  ))}
                                </select>
                              </label>

                              <label className="forecast-field">
                                <span>Target override</span>
                                <select
                                  value={forecastTargetColumn}
                                  onChange={(event) => setForecastTargetColumn(event.target.value)}
                                >
                                  <option value="">Auto detect</option>
                                  {forecastColumnOptions.map((column) => (
                                    <option key={`forecast-target-${column}`} value={column}>
                                      {formatColumnOptionLabel(column, forecastColumnTypes)}
                                    </option>
                                  ))}
                                </select>
                              </label>
                            </div>

                            <div className="forecast-mapping-grid">
                              <label className="forecast-field">
                                <span>Frequency override</span>
                                <select
                                  value={forecastFrequency}
                                  onChange={(event) => setForecastFrequency(event.target.value)}
                                >
                                  {FORECAST_FREQUENCY_OPTIONS.map((option) => (
                                    <option key={option.value} value={option.value}>
                                      {option.label}
                                    </option>
                                  ))}
                                </select>
                              </label>

                              <label className="forecast-field">
                                <span>Model strategy</span>
                                <select
                                  value={forecastModelStrategy}
                                  onChange={(event) => setForecastModelStrategy(event.target.value)}
                                >
                                  {FORECAST_MODEL_STRATEGY_OPTIONS.map((option) => (
                                    <option key={option.value} value={option.value}>
                                      {option.label}
                                    </option>
                                  ))}
                                </select>
                              </label>
                            </div>

                            <div className="forecast-mapping-grid forecast-mapping-grid--single">
                              <label className="forecast-field">
                                <span>Training mode</span>
                                <select
                                  value={forecastTrainingMode}
                                  onChange={(event) => setForecastTrainingMode(event.target.value)}
                                >
                                  {FORECAST_TRAINING_MODE_OPTIONS.map((option) => (
                                    <option key={option.value} value={option.value}>
                                      {option.label}
                                    </option>
                                  ))}
                                </select>
                              </label>
                            </div>

                            <p className="helper-text">
                              Overrides are optional. Leave values on Auto to keep the AI-driven setup effortless.
                            </p>
                          </div>
                        ) : null}
                      </>
                    ) : (
                      <div className="forecast-block-panel">
                        <div className="forecast-block-panel__header">
                          <div>
                            <span className="section-kicker">Forecast Eligibility</span>
                            <strong className="forecast-block-panel__title">
                              ⚠️ This dataset is not suitable for forecasting.
                            </strong>
                            <p>{forecastBlockReason}</p>
                          </div>
                          <div className="forecast-preview-item forecast-preview-item--warning">
                            <span>Time column</span>
                            <strong>❌ No time column detected</strong>
                            <small>Forecasting is blocked until a reliable time field is available.</small>
                          </div>
                        </div>

                        <div className="forecast-block-suggestion-list">
                          {forecastGuidanceSuggestions.map((suggestion) => (
                            <div key={suggestion} className="forecast-block-suggestion">
                              {suggestion}
                            </div>
                          ))}
                        </div>

                        <div className="forecast-block-actions">
                          {FORECAST_ALTERNATIVE_ACTIONS.map((action, index) => (
                            <button
                              key={action.label}
                              type="button"
                              className={index === 0 ? "primary-button-new" : "ghost-button"}
                              onClick={() =>
                                loadAnalysisAlternative(
                                  action.prompt,
                                  `${action.label} loaded into analysis mode.`
                                )
                              }
                            >
                              {action.label}
                            </button>
                          ))}
                        </div>
                      </div>
                    )
                  ) : (
                    <p>Select a dataset to let Aidssist auto-configure the forecast.</p>
                  )}

                  {selectedDataset && !forecastEligibilityAllowed ? null : (
                    <div className="solve-actions solve-actions--primary">
                      <button
                        type="button"
                        className="primary-button-new"
                        disabled={working || !forecastCanRun}
                        onClick={() => void handleStartForecast()}
                      >
                        {working ? "Generating..." : "Generate AI Forecast"}
                      </button>
                    </div>
                  )}
                </div>
              ) : null}

              {workbenchMode === "orchestrator" ? (
                <div className="solve-controls">
                  <textarea
                    value={solvePrompt}
                    onChange={(event) => setSolvePrompt(event.target.value)}
                    placeholder="Describe the puzzle, redesign request, or dataset problem you want Aidssist to solve."
                  />
                  <div className="solve-actions">
                    <select value={routeHint} onChange={(event) => setRouteHint(event.target.value as RouteHint)}>
                      <option value="auto">Auto route</option>
                      <option value="data">Data route</option>
                      <option value="code">Code route</option>
                      <option value="hybrid">Hybrid route</option>
                    </select>
                    <button type="button" className="primary-button-new" disabled={working} onClick={() => void handleStartSolve()}>
                      {working ? "Submitting..." : "Run solver"}
                    </button>
                  </div>
                </div>
              ) : null}
            </article>

            <article className="surface-card">
              <span className="section-kicker">Run Status</span>
              <h3>
                {workbenchMode === "analysis"
                  ? analysisJob?.status || "No analysis run"
                  : workbenchMode === "forecast"
                    ? forecastJob?.status || "No forecast run"
                    : activeRun?.route.toUpperCase() || "No active run"}
              </h3>
              {workbenchMode === "analysis" ? (
                <>
                  <div className="mini-stat-grid">
                    <div>
                      <span>Status</span>
                      <strong>{analysisJob?.status || "idle"}</strong>
                    </div>
                    <div>
                      <span>Elapsed</span>
                      <strong>{formatMs(analysisJob?.elapsed_ms)}</strong>
                    </div>
                    <div>
                      <span>Cache</span>
                      <strong>{analysisJob?.cache_hit ? "hit" : "fresh"}</strong>
                    </div>
                    <div>
                      <span>Filter</span>
                      <strong>{analysisActiveFilter.replace(/_/g, " ")}</strong>
                    </div>
                  </div>
                  <p>
                    {analysisOutput?.summary
                      ? String(analysisOutput.summary)
                      : "Run an analysis to generate a dashboard, suggested questions, and a structured result."}
                  </p>
                </>
              ) : workbenchMode === "forecast" ? (
                <>
                  <div className="mini-stat-grid">
                    <div>
                      <span>Status</span>
                      <strong>{forecastStatusLabel}</strong>
                    </div>
                    <div>
                      <span>Model</span>
                      <strong>{forecastModelLabel}</strong>
                    </div>
                    <div>
                      <span>History</span>
                      <strong>{formatCount(forecastOutput?.history_points || forecastHistoryPoints)}</strong>
                    </div>
                    <div>
                      <span>Filter</span>
                      <strong>{forecastActiveFilter.replace(/_/g, " ")}</strong>
                    </div>
                  </div>
                  <p>{forecastSummaryText}</p>
                </>
              ) : activeRun ? (
                <>
                  <div className="mini-stat-grid">
                    <div>
                      <span>Status</span>
                      <strong>{activeRun.status}</strong>
                    </div>
                    <div>
                      <span>Elapsed</span>
                      <strong>{formatMs(activeRun.elapsed_ms)}</strong>
                    </div>
                    <div>
                      <span>Retrieved</span>
                      <strong>{formatCount(activeRun.retrieval_trace.items.length)}</strong>
                    </div>
                  </div>
                  <p>{activeRun.final_summary || activeRun.error_message || "Waiting for the solver to finish."}</p>
                </>
              ) : (
                <p>Submit a query to create a solver trace.</p>
              )}
            </article>

            <article className="surface-card">
              <span className="section-kicker">Suggested Actions</span>
              <h3>Context-aware next steps</h3>
              {activeRecommendedNextStep ? (
                <div className="best-next-step">
                  <span>Best next step</span>
                  <strong>{activeRecommendedNextStep}</strong>
                </div>
              ) : null}
              {activeContext ? (
                  <div className="mini-stat-grid">
                    <div>
                      <span>Domain</span>
                      <strong>{String(activeContext.domain || "generic")}</strong>
                    </div>
                    <div>
                      <span>Time series</span>
                      <strong>{Boolean(activeContext.is_time_series) ? "yes" : "no"}</strong>
                    </div>
                  <div>
                    <span>Primary metric</span>
                    <strong>
                      {Array.isArray(activeContext.primary_metrics) && activeContext.primary_metrics.length
                        ? String(activeContext.primary_metrics[0])
                        : "n/a"}
                    </strong>
                  </div>
                </div>
              ) : null}
              <div className="recommendation-list">
                {activeSuggestions.map((suggestion, index) => (
                  <div
                    key={`${suggestion.title}-${suggestion.prompt}-${index}`}
                    className={`recommendation-card ${index === 0 ? "recommendation-card--best" : ""}`}
                  >
                    <strong>{suggestion.title}</strong>
                    <p>{suggestion.prompt}</p>
                    <small>
                      {suggestion.rationale || "Adapted to the active dataset and ranked by likely relevance."}
                    </small>
                    <div className="suggestion-actions">
                      <button
                        type="button"
                        className="primary-button-new"
                        disabled={working || !selectedDatasetId}
                        onClick={() => void handleSuggestedAction(suggestion)}
                      >
                        {suggestion.action_type === "forecast" ? "Generate AI Forecast" : "Run analysis"}
                      </button>
                      <button
                        type="button"
                        className="ghost-button"
                        onClick={() => {
                          if (suggestion.action_type === "forecast") {
                            if (!forecastEligibilityAllowed) {
                              const fallbackAction = FORECAST_ALTERNATIVE_ACTIONS[0];
                              loadAnalysisAlternative(
                                fallbackAction.prompt,
                                "Forecasting is unavailable for this dataset, so we loaded a safer analysis path instead."
                              );
                              return;
                            }
                            startTransition(() => setWorkbenchMode("forecast"));
                            setStatusMessage("Forecast suggestion loaded. The AI settings are ready when you are.");
                            return;
                          }
                          startTransition(() => setWorkbenchMode("analysis"));
                          setAnalysisPrompt(suggestion.prompt);
                          setStatusMessage("Suggested analysis loaded into the editor.");
                        }}
                      >
                        Load only
                      </button>
                    </div>
                  </div>
                ))}
                {!activeSuggestions.length ? (
                  <div className="notice-card">
                    Upload a dataset or run the public demo to unlock suggested next steps and follow-up prompts.
                  </div>
                ) : null}
              </div>
            </article>

            {workbenchMode === "analysis" ? (
              <article className="surface-card surface-card--wide">
                <span className="section-kicker">Dashboard</span>
                <h3>Tableau-style analytical view</h3>
                <p>
                  Active filter: {analysisActiveFilter.replace(/_/g, " ")}
                  {analysisVisualizationType ? ` · Primary view: ${analysisVisualizationType}` : ""}
                </p>
                <DashboardGrid dashboard={analysisDashboard} filterType={analysisActiveFilter} customRange={customRange} />
              </article>
            ) : null}

            {workbenchMode === "forecast" ? (
              <article className="surface-card surface-card--wide">
                <span className="section-kicker">Forecast View</span>
                <h3>Time-series outlook</h3>
                {forecastOutput?.error ? (
                  <div className="validator-card">
                    <strong>{forecastOutput.error.message || "Forecast error"}</strong>
                    <p>{forecastOutput.error.suggestion || forecastOutput.error_message || ""}</p>
                  </div>
                ) : null}

                {forecastOutput?.auto_config ? (
                  <div className="forecast-preview-card forecast-preview-card--result">
                    <div className="forecast-preview-card__header">
                      <div>
                        <span className="section-kicker">Detected Configuration</span>
                        <p>Forecast run used the following AI-ready setup.</p>
                      </div>
                      <div className={`forecast-confidence-pill forecast-confidence-pill--${forecastOutput.auto_config.confidence || "low"}`}>
                        <strong>{titleCaseSlug(forecastOutput.auto_config.confidence || "low")} confidence</strong>
                        {typeof forecastOutput.auto_config.confidence_score === "number"
                          ? <span>{formatPercent(forecastOutput.auto_config.confidence_score * 100)}</span>
                          : null}
                      </div>
                    </div>
                    <div className="forecast-preview-grid">
                      <div className="forecast-preview-item forecast-preview-item--auto">
                        <span>Date column</span>
                        <strong>{forecastOutput.auto_config.date_column || "n/a"}</strong>
                        <small>Applied at run time</small>
                      </div>
                      <div className="forecast-preview-item forecast-preview-item--auto">
                        <span>Target</span>
                        <strong>{forecastOutput.auto_config.target || "n/a"}</strong>
                        <small>Applied at run time</small>
                      </div>
                      <div className="forecast-preview-item forecast-preview-item--auto">
                        <span>Frequency</span>
                        <strong>{forecastOutput.auto_config.frequency || "n/a"}</strong>
                        <small>{forecastOutput.auto_config.horizon_label || "Auto horizon"}</small>
                      </div>
                      <div className="forecast-preview-item forecast-preview-item--auto">
                        <span>Data points</span>
                        <strong>{formatCount(forecastOutput.auto_config.data_points || 0)}</strong>
                        <small>{forecastOutput.auto_config.model_strategy || "hybrid"}</small>
                      </div>
                    </div>
                  </div>
                ) : null}

                {forecastHasRenderableResults && forecastOutput?.forecast_metadata ? (
                  <div className="mini-stat-grid">
                    <div>
                      <span>Time column</span>
                      <strong>{forecastOutput.forecast_metadata.time_column || "auto"}</strong>
                    </div>
                    <div>
                      <span>Frequency</span>
                      <strong>{forecastOutput.forecast_metadata.frequency}</strong>
                    </div>
                    <div>
                      <span>Confidence</span>
                      <strong>{forecastOutput.confidence?.label || "n/a"}</strong>
                    </div>
                    <div>
                      <span>View</span>
                      <strong>{forecastVisualizationType || "line"}</strong>
                    </div>
                  </div>
                ) : null}

                {forecastHasRenderableResults ? (
                  <>
                    <div className="forecast-insight-grid">
                      <div className="forecast-insight-card">
                        <span>Trend summary</span>
                        <strong>{forecastTrendSummary}</strong>
                      </div>
                      <div className="forecast-insight-card">
                        <span>Confidence level</span>
                        <strong>{forecastConfidenceSummary}</strong>
                      </div>
                      <div className="forecast-insight-card">
                        <span>Recommended action</span>
                        <strong>{forecastRecommendedAction}</strong>
                      </div>
                    </div>

                    <p>Active filter: {forecastActiveFilter.replace(/_/g, " ")}</p>
                    <DashboardGrid dashboard={forecastDashboard} filterType={forecastActiveFilter} customRange={customRange} />
                    {forecastSeries.length ? (
                      <div className="table-shell">
                        <table>
                          <thead>
                            <tr>
                              <th>Date</th>
                              <th>Value</th>
                              <th>Series</th>
                            </tr>
                          </thead>
                          <tbody>
                            {filterRowsByTime(forecastSeries, "date", forecastActiveFilter, customRange).slice(0, 12).map((row, rowIndex) => (
                              <tr key={`forecast-row-${rowIndex}`}>
                                <td>{String(row.date ?? "")}</td>
                                <td>{toNumeric(row.value).toLocaleString()}</td>
                                <td>{String(row.series ?? "")}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    ) : null}
                  </>
                ) : !forecastOutput && selectedDataset && !forecastEligibilityAllowed ? (
                  <div className="forecast-block-panel forecast-block-panel--result">
                    <div>
                      <span className="section-kicker">Blocked</span>
                      <strong className="forecast-block-panel__title">⚠️ This dataset is not suitable for forecasting.</strong>
                      <p>{forecastBlockReason}</p>
                    </div>
                    <div className="forecast-block-actions">
                      {FORECAST_ALTERNATIVE_ACTIONS.map((action, index) => (
                        <button
                          key={`forecast-view-${action.label}`}
                          type="button"
                          className={index === 0 ? "primary-button-new" : "ghost-button"}
                          onClick={() =>
                            loadAnalysisAlternative(
                              action.prompt,
                              `${action.label} loaded into analysis mode.`
                            )
                          }
                        >
                          {action.label}
                        </button>
                      ))}
                    </div>
                  </div>
                ) : null}
              </article>
            ) : null}

            {workbenchMode === "orchestrator" ? (
              <>
                <article className="surface-card">
                  <span className="section-kicker">Validation</span>
                  <h3>Checks and refinement loop</h3>
                  <div className="validator-list">
                    {(activeRun?.validator_reports || []).map((report) => (
                      <div key={report.report_id} className="validator-card">
                        <strong>
                          Attempt {report.attempt_index + 1} · {report.status}
                        </strong>
                        <p>{report.error_message || "Candidate passed validation."}</p>
                      </div>
                    ))}
                  </div>
                </article>

                <article className="surface-card surface-card--wide">
                  <span className="section-kicker">Trace</span>
                  <h3>Retrieved context and recorded steps</h3>
                  <div className="trace-grid">
                    <div>
                      <h4>Retrieval memory</h4>
                      <div className="trace-list">
                        {(activeRun?.retrieval_trace.items || []).map((item) => (
                          <div key={item.chunk_id} className="trace-card">
                            <strong>{item.title}</strong>
                            <span>
                              {item.confidence} · score {item.score.toFixed(3)}
                            </span>
                            <p>{item.excerpt}</p>
                          </div>
                        ))}
                      </div>
                    </div>
                    <div>
                      <h4>Execution steps</h4>
                      <div className="trace-list">
                        {(activeRun?.steps || []).map((step) => (
                          <div key={step.step_id} className="trace-card">
                            <strong>
                              {step.step_index}. {step.title}
                            </strong>
                            <span>
                              {step.stage} · {step.status}
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </article>

                <article className="surface-card surface-card--wide">
                  <span className="section-kicker">Final Output</span>
                  <h3>Packaged response</h3>
                  {activeRun ? (
                    <>
                      <p>{activeRun.final_summary || "The final output will appear here when the run completes."}</p>
                      {safeObject(activeRun.final_output).solution_markdown ? (
                        <pre className="code-panel">
                          {String(safeObject(activeRun.final_output).solution_markdown)}
                        </pre>
                      ) : null}
                      {previewRows.length ? (
                        <div className="table-shell">
                          <table>
                            <thead>
                              <tr>
                                {Object.keys(previewRows[0]).map((column) => (
                                  <th key={column}>{column}</th>
                                ))}
                              </tr>
                            </thead>
                            <tbody>
                              {previewRows.map((row, rowIndex) => (
                                <tr key={`result-${rowIndex}`}>
                                  {Object.keys(previewRows[0]).map((column) => (
                                    <td key={`${column}-${rowIndex}`}>{String(row[column] ?? "")}</td>
                                  ))}
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      ) : null}
                      <div className="recommendation-list">
                        {safeArray(safeObject(activeRun.final_output).redesign_recommendations).map((item, index) => (
                          <div key={`recommendation-${index}`} className="recommendation-card">
                            <strong>{String(item.title || "Recommendation")}</strong>
                            <p>{String(item.detail || "")}</p>
                          </div>
                        ))}
                      </div>
                    </>
                  ) : (
                    <p>No run selected yet.</p>
                  )}
                </article>
              </>
            ) : null}
          </section>
        ) : null}

        {panel === "history" ? (
          <section className="panel-grid">
            <article className="surface-card">
              <span className="section-kicker">Benchmarks</span>
              <h3>Workspace quality signals</h3>
              <div className="mini-stat-grid">
                <div>
                  <span>Success rate</span>
                  <strong>{formatPercent(metrics.successRate)}</strong>
                </div>
                <div>
                  <span>Average latency</span>
                  <strong>{metrics.averageLatency}</strong>
                </div>
                <div>
                  <span>Retrieved chunks</span>
                  <strong>{formatCount(activeRun?.retrieval_trace.items.length || 0)}</strong>
                </div>
              </div>
            </article>

            <article className="surface-card">
              <span className="section-kicker">Recent Runs</span>
              <h3>Replay memory</h3>
              <div className="trace-list">
                {(workspaceDetail?.recent_runs || []).map((run) => (
                  <button
                    key={run.run_id}
                    type="button"
                    className="trace-card trace-card--button"
                    onClick={() => {
                      setActiveRunId(run.run_id);
                      setActiveRun(run);
                      startTransition(() => setPanel("solve"));
                    }}
                  >
                    <strong>{run.query}</strong>
                    <span>
                      {run.route} · {run.status} · {formatMs(run.elapsed_ms)}
                    </span>
                    <p>{run.final_summary || run.error_message || "No summary stored yet."}</p>
                  </button>
                ))}
              </div>
            </article>

            <article className="surface-card surface-card--wide">
              <span className="section-kicker">Timeline Search</span>
              <h3>Filter workspace events</h3>
              <input
                value={timelineFilter}
                onChange={(event) => setTimelineFilter(event.target.value)}
                placeholder="Search uploads, derived datasets, and solve events"
              />
              <div className="timeline-list">
                {filteredTimeline.map((item) => (
                  <div key={item.event_id} className="timeline-item">
                    <span>{item.event_type.replace(/_/g, " ")}</span>
                    <strong>{item.title}</strong>
                    <p>{item.summary}</p>
                    <small>{item.created_at}</small>
                  </div>
                ))}
              </div>
            </article>
          </section>
        ) : null}
      </main>
    </div>
  );
}
