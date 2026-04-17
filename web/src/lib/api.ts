import type {
  AssetIntelligence,
  AnalyzeJobResponse,
  AskDataResponse,
  AssetDetail,
  AuthResponse,
  DatasetTransformResponse,
  DemoResponse,
  ForecastJobResponse,
  FolderUploadResponse,
  ImportJob,
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

export type FolderUploadItemInput = {
  id: string;
  file: File;
  relativePath: string;
};

export type FolderUploadProgressSnapshot = {
  overallProgress: number;
  fileProgress: Record<string, number>;
};

const FOLDER_UPLOAD_BATCH_LIMIT = 8;
const FOLDER_UPLOAD_BATCH_BYTES = 24 * 1024 * 1024;
const FOLDER_UPLOAD_CONCURRENCY = 3;

function parseXhrError(xhr: XMLHttpRequest): string {
  if (!xhr.responseText?.trim()) {
    return xhr.status >= 500
      ? "Aidssist is temporarily unavailable. Please try again in a moment."
      : `Request failed with status ${xhr.status || 0}.`;
  }
  try {
    const payload = JSON.parse(xhr.responseText) as { detail?: string };
    if (payload.detail?.trim()) {
      return payload.detail.trim();
    }
  } catch {
    // fall back to raw text
  }
  return xhr.responseText.trim();
}

function createFolderUploadBatches(items: FolderUploadItemInput[]): FolderUploadItemInput[][] {
  const batches: FolderUploadItemInput[][] = [];
  let currentBatch: FolderUploadItemInput[] = [];
  let currentBytes = 0;

  for (const item of items) {
    const nextBytes = currentBytes + item.file.size;
    if (
      currentBatch.length > 0 &&
      (currentBatch.length >= FOLDER_UPLOAD_BATCH_LIMIT || nextBytes > FOLDER_UPLOAD_BATCH_BYTES)
    ) {
      batches.push(currentBatch);
      currentBatch = [];
      currentBytes = 0;
    }
    currentBatch.push(item);
    currentBytes += item.file.size;
  }

  if (currentBatch.length > 0) {
    batches.push(currentBatch);
  }

  return batches;
}

function sendFolderUploadRequest(
  payload: {
    workspaceId: string;
    sessionId: string;
    folderName: string;
    title?: string;
    finalize: boolean;
    items?: FolderUploadItemInput[];
  },
  token: string,
  onProgress?: (loaded: number, total: number) => void
): Promise<FolderUploadResponse> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API_BASE_URL}/v1/upload-folder`);
    xhr.setRequestHeader("Accept", "application/json");
    xhr.setRequestHeader("Authorization", `Bearer ${token}`);

    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          resolve((JSON.parse(xhr.responseText || "{}") || {}) as FolderUploadResponse);
        } catch (error) {
          reject(error instanceof Error ? error : new Error("Folder upload response could not be parsed."));
        }
        return;
      }
      reject(new Error(parseXhrError(xhr)));
    };
    xhr.onerror = () => reject(new Error("Folder upload failed before the server returned a response."));
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable) {
        onProgress?.(event.loaded, event.total);
      }
    };

    const formData = new FormData();
    formData.append("workspace_id", payload.workspaceId);
    formData.append("session_id", payload.sessionId);
    formData.append("folder_name", payload.folderName);
    formData.append("finalize", payload.finalize ? "true" : "false");
    if (payload.title?.trim()) {
      formData.append("title", payload.title.trim());
    }
    for (const item of payload.items || []) {
      formData.append("files", item.file, item.file.name);
      formData.append("relative_paths", item.relativePath);
    }
    xhr.send(formData);
  });
}

async function runBatchedFolderUpload(
  batches: FolderUploadItemInput[][],
  worker: (batch: FolderUploadItemInput[], batchIndex: number) => Promise<void>
): Promise<void> {
  let cursor = 0;
  const workers = Array.from({ length: Math.min(FOLDER_UPLOAD_CONCURRENCY, batches.length) }, async () => {
    while (cursor < batches.length) {
      const batchIndex = cursor;
      cursor += 1;
      await worker(batches[batchIndex], batchIndex);
    }
  });
  await Promise.all(workers);
}

export async function uploadFolderAsset(
  workspaceId: string,
  title: string,
  folderName: string,
  items: FolderUploadItemInput[],
  token: string,
  onProgress?: (snapshot: FolderUploadProgressSnapshot) => void
): Promise<FolderUploadResponse> {
  const sessionId =
    typeof crypto !== "undefined" && "randomUUID" in crypto
      ? crypto.randomUUID()
      : `folder-${Date.now()}`;
  const batches = createFolderUploadBatches(items);
  const totalBytes = items.reduce((sum, item) => sum + item.file.size, 0);
  const batchLoaded = new Map<number, number>();
  const fileProgress: Record<string, number> = Object.fromEntries(items.map((item) => [item.id, 0]));

  const emitProgress = () => {
    const loadedBytes = Array.from(batchLoaded.values()).reduce((sum, value) => sum + value, 0);
    onProgress?.({
      overallProgress: totalBytes > 0 ? Math.min(1, loadedBytes / totalBytes) : 1,
      fileProgress: { ...fileProgress }
    });
  };

  await runBatchedFolderUpload(batches, async (batch, batchIndex) => {
    const batchBytes = batch.reduce((sum, item) => sum + item.file.size, 0);
    await sendFolderUploadRequest(
      {
        workspaceId,
        sessionId,
        folderName,
        title,
        finalize: false,
        items: batch
      },
      token,
      (loaded) => {
        batchLoaded.set(batchIndex, Math.min(loaded, batchBytes));
        let consumed = 0;
        const normalizedLoaded = Math.min(loaded, batchBytes);
        for (const item of batch) {
          const start = consumed;
          const end = consumed + item.file.size;
          const progress =
            end <= start
              ? 1
              : Math.min(1, Math.max(0, (normalizedLoaded - start) / Math.max(item.file.size, 1)));
          fileProgress[item.id] = progress;
          consumed = end;
        }
        emitProgress();
      }
    );
    batchLoaded.set(batchIndex, batchBytes);
    for (const item of batch) {
      fileProgress[item.id] = 1;
    }
    emitProgress();
  });

  const response = await sendFolderUploadRequest(
    {
      workspaceId,
      sessionId,
      folderName,
      title,
      finalize: true
    },
    token
  );
  onProgress?.({
    overallProgress: 1,
    fileProgress: { ...Object.fromEntries(items.map((item) => [item.id, 1])) }
  });
  return response;
}

export async function getAsset(assetId: string, token: string): Promise<AssetDetail> {
  return request<AssetDetail>(`/v1/assets/${assetId}`, { method: "GET" }, token);
}

export async function getAssetIntelligence(
  assetId: string,
  token: string,
  forceRefresh = false
): Promise<AssetIntelligence> {
  const query = forceRefresh ? "?force_refresh=true" : "";
  return request<AssetIntelligence>(`/v1/assets/${assetId}/intelligence${query}`, { method: "GET" }, token);
}

export async function importFromGoogleDrive(
  payload: {
    workspace_id: string;
    file_id: string;
    access_token: string;
  },
  token: string
): Promise<ImportJob> {
  return request<ImportJob>(
    "/v1/import/google-drive",
    {
      method: "POST",
      body: JSON.stringify(payload)
    },
    token
  );
}

export async function importFromKaggle(
  payload: {
    workspace_id: string;
    dataset_url: string;
  },
  token: string
): Promise<ImportJob> {
  return request<ImportJob>(
    "/v1/import/kaggle",
    {
      method: "POST",
      body: JSON.stringify(payload)
    },
    token
  );
}

export async function getImportJob(jobId: string, token: string): Promise<ImportJob> {
  return request<ImportJob>(`/v1/import/jobs/${jobId}`, { method: "GET" }, token);
}

export async function generateAIInsights(
  payload: { asset_id: string; force_refresh?: boolean },
  token: string
): Promise<AssetIntelligence> {
  return request<AssetIntelligence>(
    "/v1/ai/insights",
    {
      method: "POST",
      body: JSON.stringify(payload)
    },
    token
  );
}

export async function askYourData(
  payload: { asset_id: string; question: string },
  token: string
): Promise<AskDataResponse> {
  return request<AskDataResponse>(
    "/v1/ai/chat",
    {
      method: "POST",
      body: JSON.stringify(payload)
    },
    token
  );
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
