import type {
  AnalyzeJobResponse,
  AssetDetail,
  AuthResponse,
  DatasetTransformResponse,
  DemoResponse,
  ForecastJobResponse,
  JobStatusResponse,
  SolveRunStatus,
  TimelineItem,
  User,
  WorkspaceDetail,
  WorkspaceSummary
} from "../types/api";

const API_BASE_URL =
  (import.meta.env.VITE_AIDSSIST_API_URL as string | undefined)?.replace(/\/$/, "") ||
  "/api";
const DEMO_CACHE_KEY = "aidssist-demo-payload-v2";

async function readErrorMessage(response: Response): Promise<string> {
  const contentType = response.headers.get("content-type") || "";
  if (contentType.includes("application/json")) {
    const errorBody = await response.json().catch(() => ({}));
    const detail = (errorBody as { detail?: string }).detail;
    if (detail && detail.trim()) {
      return detail;
    }
  }

  const text = await response.text().catch(() => "");
  if (text.trim()) {
    return text.trim();
  }

  if (response.status === 404) {
    return "The requested Aidssist endpoint could not be found.";
  }
  if (response.status >= 500) {
    return "Aidssist is temporarily unavailable. Please try again in a moment.";
  }
  return `Request failed with status ${response.status}.`;
}

function readCachedDemoData(): DemoResponse | null {
  if (typeof window === "undefined") {
    return null;
  }
  try {
    const raw = window.sessionStorage.getItem(DEMO_CACHE_KEY);
    return raw ? (JSON.parse(raw) as DemoResponse) : null;
  } catch {
    return null;
  }
}

function writeCachedDemoData(payload: DemoResponse): void {
  if (typeof window === "undefined") {
    return;
  }
  try {
    window.sessionStorage.setItem(DEMO_CACHE_KEY, JSON.stringify(payload));
  } catch {
    // best effort cache
  }
}

async function request<T>(path: string, options: RequestInit = {}, token?: string | null): Promise<T> {
  const headers = new Headers(options.headers || {});
  headers.set("Accept", "application/json");
  if (!(options.body instanceof FormData)) {
    headers.set("Content-Type", "application/json");
  }
  if (token) {
    headers.set("Authorization", `Bearer ${token}`);
  }

  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      ...options,
      headers
    });
  } catch (error) {
    const message =
      error instanceof Error ? error.message : "The API request could not reach the server.";
    throw new Error(`Unable to reach the Aidssist API at ${API_BASE_URL}. ${message}`);
  }

  if (!response.ok) {
    const message = await readErrorMessage(response);
    throw new Error(message);
  }

  const contentType = response.headers.get("content-type") || "";
  if (!contentType.includes("application/json")) {
    return {} as T;
  }
  return response.json() as Promise<T>;
}

export async function register(email: string, password: string, displayName: string): Promise<AuthResponse> {
  return request<AuthResponse>("/v1/auth/register", {
    method: "POST",
    body: JSON.stringify({ email, password, display_name: displayName })
  });
}

export async function login(email: string, password: string): Promise<AuthResponse> {
  return request<AuthResponse>("/v1/auth/login", {
    method: "POST",
    body: JSON.stringify({ email, password })
  });
}

export async function getCurrentUser(token: string): Promise<User> {
  return request<User>("/v1/auth/me", { method: "GET" }, token);
}

export async function logout(token: string): Promise<void> {
  await request<{ status: string }>("/v1/auth/logout", { method: "POST" }, token);
}

export async function listWorkspaces(token: string): Promise<WorkspaceSummary[]> {
  return request<WorkspaceSummary[]>("/v1/workspaces", { method: "GET" }, token);
}

export async function createWorkspace(
  name: string,
  description: string,
  token: string
): Promise<WorkspaceDetail> {
  return request<WorkspaceDetail>(
    "/v1/workspaces",
    {
      method: "POST",
      body: JSON.stringify({ name, description })
    },
    token
  );
}

export async function getWorkspace(workspaceId: string, token: string): Promise<WorkspaceDetail> {
  return request<WorkspaceDetail>(`/v1/workspaces/${workspaceId}`, { method: "GET" }, token);
}

export async function getWorkspaceTimeline(workspaceId: string, token: string): Promise<TimelineItem[]> {
  return request<TimelineItem[]>(`/v1/workspaces/${workspaceId}/timeline`, { method: "GET" }, token);
}

export async function uploadAsset(
  workspaceId: string,
  title: string,
  files: File[],
  token: string
): Promise<AssetDetail> {
  const formData = new FormData();
  formData.append("workspace_id", workspaceId);
  if (title.trim()) {
    formData.append("title", title.trim());
  }
  for (const file of files) {
    formData.append("files", file);
  }
  return request<AssetDetail>("/v1/assets", { method: "POST", body: formData }, token);
}

export async function getAsset(assetId: string, token: string): Promise<AssetDetail> {
  return request<AssetDetail>(`/v1/assets/${assetId}`, { method: "GET" }, token);
}

export async function transformDataset(
  datasetId: string,
  instruction: string,
  token: string
): Promise<DatasetTransformResponse> {
  return request<DatasetTransformResponse>(
    `/v1/datasets/${datasetId}/transform`,
    {
      method: "POST",
      body: JSON.stringify({ instruction })
    },
    token
  );
}

export async function startSolve(
  payload: {
    workspace_id: string;
    query: string;
    asset_id?: string | null;
    dataset_id?: string | null;
    route_hint?: string | null;
  },
  token: string
): Promise<SolveRunStatus> {
  return request<SolveRunStatus>(
    "/v1/solve",
    {
      method: "POST",
      body: JSON.stringify(payload)
    },
    token
  );
}

export async function getSolveRun(runId: string, token: string): Promise<SolveRunStatus> {
  return request<SolveRunStatus>(`/v1/solve/${runId}`, { method: "GET" }, token);
}

export async function startAnalysisJob(
  payload: {
    dataset_id: string;
    query: string;
    workflow_context?: Record<string, unknown>;
  },
  token: string
): Promise<AnalyzeJobResponse> {
  return request<AnalyzeJobResponse>(
    "/v1/jobs/analyze",
    {
      method: "POST",
      body: JSON.stringify(payload)
    },
    token
  );
}

export async function startForecastJob(
  payload: {
    dataset_id: string;
    forecast_config?: Record<string, unknown>;
    workflow_context?: Record<string, unknown>;
  },
  token: string
): Promise<ForecastJobResponse> {
  return request<ForecastJobResponse>(
    "/v1/jobs/forecast",
    {
      method: "POST",
      body: JSON.stringify(payload)
    },
    token
  );
}

export async function getJobStatus(jobId: string, token: string): Promise<JobStatusResponse> {
  return request<JobStatusResponse>(`/v1/jobs/${jobId}`, { method: "GET" }, token);
}

export async function getDemoData(forceFresh = false): Promise<DemoResponse> {
  if (!forceFresh) {
    const cached = readCachedDemoData();
    if (cached) {
      return cached;
    }
  }

  const payload = await request<DemoResponse>("/demo-data", { method: "GET" });
  writeCachedDemoData(payload);
  return payload;
}
