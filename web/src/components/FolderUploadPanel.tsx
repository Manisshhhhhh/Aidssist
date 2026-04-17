import { useEffect, useRef, useState, type ChangeEvent, type DragEvent, type InputHTMLAttributes } from "react";
import {
  AlertCircle,
  BarChart3,
  CheckCircle2,
  ChevronRight,
  FileSpreadsheet,
  Files,
  FolderOpen,
  Loader2,
  RefreshCw,
  Sparkles,
  UploadCloud
} from "lucide-react";

import {
  uploadFolderAsset,
  type FolderUploadItemInput,
  type FolderUploadProgressSnapshot
} from "../lib/api";
import type {
  FolderUploadDatasetSummary,
  FolderUploadFileResult,
  FolderUploadPreview,
  FolderUploadResponse
} from "../types/api";

type UploadStatus = "queued" | "uploading" | "success" | "failed";

type UploadItem = {
  id: string;
  file: File;
  relativePath: string;
  fileName: string;
  sizeBytes: number;
  status: UploadStatus;
  progress: number;
  error?: string | null;
  retryable: boolean;
};

type CandidateFile = {
  file: File;
  relativePath: string;
};

type DirectoryInputElement = HTMLInputElement & {
  directory?: boolean;
  webkitdirectory?: boolean;
};

type DirectoryInputAttributes = InputHTMLAttributes<HTMLInputElement> & {
  directory?: string;
  webkitdirectory?: string;
};

type FileSystemEntryLike = {
  isDirectory: boolean;
  isFile: boolean;
  name: string;
  fullPath?: string;
};

type FileSystemFileEntryLike = FileSystemEntryLike & {
  file: (callback: (file: File) => void, error?: (error: DOMException) => void) => void;
};

type FileSystemDirectoryReaderLike = {
  readEntries: (
    callback: (entries: FileSystemEntryLike[]) => void,
    error?: (error: DOMException) => void
  ) => void;
};

type FileSystemDirectoryEntryLike = FileSystemEntryLike & {
  createReader: () => FileSystemDirectoryReaderLike;
};

type DataTransferItemWithEntry = DataTransferItem & {
  webkitGetAsEntry?: () => FileSystemEntryLike | null;
};

type TreeNode = {
  key: string;
  name: string;
  kind: "folder" | "file";
  children: TreeNode[];
  item?: UploadItem;
};

const MAX_FILE_BYTES = 100 * 1024 * 1024;
const SUPPORTED_SUFFIXES = new Set([".csv", ".xlsx"]);
const AUTO_ANALYZE_FALLBACK =
  "Analyze this uploaded dataset folder, detect relationships between the tables, summarize data quality, and recommend the highest-value insights.";

const FOLDER_INPUT_ATTRIBUTES: DirectoryInputAttributes = {
  type: "file",
  accept: ".csv,.xlsx",
  multiple: true,
  hidden: true,
  directory: "",
  webkitdirectory: ""
};

function formatBytes(value: number): string {
  if (value < 1024) {
    return `${value} B`;
  }
  if (value < 1024 * 1024) {
    return `${(value / 1024).toFixed(1)} KB`;
  }
  if (value < 1024 * 1024 * 1024) {
    return `${(value / (1024 * 1024)).toFixed(1)} MB`;
  }
  return `${(value / (1024 * 1024 * 1024)).toFixed(1)} GB`;
}

function sanitizeRelativePath(relativePath: string, fallbackName: string): string {
  const cleaned = (relativePath || fallbackName || "dataset.csv").replace(/\\/g, "/").replace(/^\/+/, "");
  const parts = cleaned.split("/").filter((part) => part && part !== "." && part !== "..");
  return parts.length ? parts.join("/") : fallbackName || "dataset.csv";
}

function extractFolderName(paths: string[]): string {
  if (!paths.length) {
    return "Dataset Folder";
  }
  const normalized = paths.map((path) => sanitizeRelativePath(path, path.split("/").pop() || "dataset.csv"));
  const firstSegments = normalized.map((path) => path.split("/")[0]).filter(Boolean);
  if (firstSegments.length && firstSegments.every((segment) => segment === firstSegments[0]) && normalized.some((path) => path.includes("/"))) {
    return firstSegments[0];
  }
  if (normalized.length === 1) {
    return normalized[0].replace(/\.[^.]+$/, "");
  }
  return "Dataset Folder";
}

function createUploadId(relativePath: string, file: File): string {
  return `${relativePath}:${file.size}:${file.lastModified}`;
}

function classifyClientFile(candidate: CandidateFile): UploadItem {
  const relativePath = sanitizeRelativePath(candidate.relativePath, candidate.file.name);
  const suffix = relativePath.slice(relativePath.lastIndexOf(".")).toLowerCase();
  if (!SUPPORTED_SUFFIXES.has(suffix)) {
    return {
      id: createUploadId(relativePath, candidate.file),
      file: candidate.file,
      relativePath,
      fileName: relativePath.split("/").pop() || candidate.file.name,
      sizeBytes: candidate.file.size,
      status: "failed",
      progress: 0,
      error: "Only CSV and XLSX files can be uploaded in dataset folders.",
      retryable: false
    };
  }
  if (candidate.file.size > MAX_FILE_BYTES) {
    return {
      id: createUploadId(relativePath, candidate.file),
      file: candidate.file,
      relativePath,
      fileName: relativePath.split("/").pop() || candidate.file.name,
      sizeBytes: candidate.file.size,
      status: "failed",
      progress: 0,
      error: "Each file must be 100 MB or smaller.",
      retryable: false
    };
  }
  return {
    id: createUploadId(relativePath, candidate.file),
    file: candidate.file,
    relativePath,
    fileName: relativePath.split("/").pop() || candidate.file.name,
    sizeBytes: candidate.file.size,
    status: "queued",
    progress: 0,
    retryable: true
  };
}

function mergeUploadItems(existing: UploadItem[], incoming: CandidateFile[]): UploadItem[] {
  const next = new Map(existing.map((item) => [item.relativePath, item]));
  for (const candidate of incoming) {
    const classified = classifyClientFile(candidate);
    next.set(classified.relativePath, classified);
  }
  return Array.from(next.values()).sort((left, right) => left.relativePath.localeCompare(right.relativePath));
}

function buildTree(items: UploadItem[]): TreeNode[] {
  type MutableTreeNode = TreeNode & { childMap: Map<string, MutableTreeNode> };

  const root = new Map<string, MutableTreeNode>();
  for (const item of items) {
    const parts = item.relativePath.split("/").filter(Boolean);
    let level = root;
    let currentPath = "";
    parts.forEach((part, index) => {
      currentPath = currentPath ? `${currentPath}/${part}` : part;
      const isLeaf = index === parts.length - 1;
      let node = level.get(part);
      if (!node) {
        node = {
          key: currentPath,
          name: part,
          kind: isLeaf ? "file" : "folder",
          children: [],
          item: isLeaf ? item : undefined,
          childMap: new Map<string, MutableTreeNode>()
        };
        level.set(part, node);
      }
      if (isLeaf) {
        node.kind = "file";
        node.item = item;
      }
      level = node.childMap;
    });
  }

  const sortNodes = (nodes: Iterable<MutableTreeNode>): TreeNode[] =>
    Array.from(nodes)
      .map((node) => ({
        key: node.key,
        name: node.name,
        kind: node.kind,
        item: node.item,
        children: sortNodes(node.childMap.values())
      }))
      .sort((left, right) => {
        if (left.kind !== right.kind) {
          return left.kind === "folder" ? -1 : 1;
        }
        return left.name.localeCompare(right.name);
      });

  return sortNodes(root.values());
}

async function readFileEntry(entry: FileSystemFileEntryLike, relativePath: string): Promise<CandidateFile[]> {
  return new Promise((resolve, reject) => {
    entry.file(
      (file) => resolve([{ file, relativePath }]),
      (error) => reject(error)
    );
  });
}

async function readDirectoryEntries(directory: FileSystemDirectoryEntryLike): Promise<FileSystemEntryLike[]> {
  const reader = directory.createReader();
  const entries: FileSystemEntryLike[] = [];

  async function readChunk(): Promise<void> {
    const chunk = await new Promise<FileSystemEntryLike[]>((resolve, reject) => {
      reader.readEntries(resolve, reject);
    });
    if (!chunk.length) {
      return;
    }
    entries.push(...chunk);
    await readChunk();
  }

  await readChunk();
  return entries;
}

async function walkEntry(entry: FileSystemEntryLike, parentPath = ""): Promise<CandidateFile[]> {
  const relativePath = parentPath ? `${parentPath}/${entry.name}` : entry.name;
  if (entry.isFile) {
    return readFileEntry(entry as FileSystemFileEntryLike, relativePath);
  }
  if (!entry.isDirectory) {
    return [];
  }
  const directoryEntries = await readDirectoryEntries(entry as FileSystemDirectoryEntryLike);
  const nested = await Promise.all(directoryEntries.map((child) => walkEntry(child, relativePath)));
  return nested.flat();
}

async function collectDroppedFiles(dataTransfer: DataTransfer): Promise<CandidateFile[]> {
  const items = Array.from(dataTransfer.items || []) as DataTransferItemWithEntry[];
  if (items.length) {
    const entries = items
      .map((item) => item.webkitGetAsEntry?.())
      .filter((entry): entry is NonNullable<typeof entry> => entry !== null);
    const entryResults = await Promise.all(
      entries.map((entry) => walkEntry(entry as unknown as FileSystemEntryLike))
    );
    const flattened = entryResults.flat();
    if (flattened.length > 0) {
      return flattened;
    }
  }
  return Array.from(dataTransfer.files || []).map((file) => {
    const withRelativePath = file as File & { webkitRelativePath?: string };
    return {
      file,
      relativePath: withRelativePath.webkitRelativePath || file.name
    };
  });
}

function flattenTreeFiles(items: UploadItem[], folderResult: FolderUploadResponse | null): UploadItem[] {
  if (items.length > 0) {
    return items;
  }
  if (!folderResult) {
    return [];
  }
  return [...folderResult.processed_files, ...folderResult.failed_files].map((fileResult) => ({
    id: `${fileResult.relative_path}:${fileResult.size_bytes}`,
    file: new File([], fileResult.file_name),
    relativePath: fileResult.relative_path,
    fileName: fileResult.file_name,
    sizeBytes: fileResult.size_bytes,
    status:
      fileResult.status === "success" || fileResult.status === "staged"
        ? "success"
        : fileResult.status === "uploading"
          ? "uploading"
          : "failed",
    progress: fileResult.status === "success" ? 1 : 0,
    error: fileResult.error,
    retryable: fileResult.status !== "success"
  }));
}

function retryableFromError(error: string | null | undefined): boolean {
  const message = String(error || "").toLowerCase();
  return !(message.includes("csv and xlsx") || message.includes("100 mb"));
}

function fileStatusLabel(status: UploadStatus): string {
  if (status === "uploading") {
    return "uploading";
  }
  if (status === "success") {
    return "success";
  }
  if (status === "failed") {
    return "failed";
  }
  return "queued";
}

function pickPreview(
  datasetSummary: FolderUploadDatasetSummary | null,
  selectedPreviewId: string | null
): FolderUploadPreview | null {
  if (!datasetSummary?.previews.length) {
    return null;
  }
  return datasetSummary.previews.find((preview) => preview.dataset_id === selectedPreviewId) || datasetSummary.previews[0];
}

function FileTree({ nodes, depth = 0 }: { nodes: TreeNode[]; depth?: number }) {
  return (
    <>
      {nodes.map((node) =>
        node.kind === "folder" ? (
          <div key={node.key} className="folder-tree-group">
            <div className="folder-tree-item folder-tree-item--folder" style={{ paddingLeft: `${depth * 0.8 + 0.8}rem` }}>
              <div className="folder-tree-item__icon">
                <FolderOpen size={16} />
              </div>
              <strong>{node.name}</strong>
              <span>{node.children.length}</span>
            </div>
            <FileTree nodes={node.children} depth={depth + 1} />
          </div>
        ) : (
          <div
            key={node.key}
            className={`folder-tree-item folder-tree-item--${node.item?.status || "queued"}`}
            style={{ paddingLeft: `${depth * 0.8 + 0.8}rem` }}
          >
            <div className="folder-tree-item__icon">
              <FileSpreadsheet size={16} />
            </div>
            <div className="folder-tree-item__body">
              <strong>{node.name}</strong>
              <small>
                {formatBytes(node.item?.sizeBytes || 0)} · {fileStatusLabel(node.item?.status || "queued")}
              </small>
              {node.item?.error ? <span>{node.item.error}</span> : null}
            </div>
            <div className="folder-tree-item__trailing">
              {node.item?.status === "uploading" ? (
                <Loader2 size={15} className="folder-tree-spin" />
              ) : node.item?.status === "success" ? (
                <CheckCircle2 size={15} />
              ) : node.item?.status === "failed" ? (
                <AlertCircle size={15} />
              ) : (
                <ChevronRight size={15} />
              )}
            </div>
          </div>
        )
      )}
    </>
  );
}

export default function FolderUploadPanel({
  token,
  workspaceId,
  latestUpload,
  onUploaded,
  onAutoAnalyze
}: {
  token: string | null;
  workspaceId: string | null;
  latestUpload: FolderUploadResponse | null;
  onUploaded: (payload: FolderUploadResponse) => Promise<void> | void;
  onAutoAnalyze: (prompt: string, assetId?: string | null) => void;
}) {
  const [items, setItems] = useState<UploadItem[]>([]);
  const [uploading, setUploading] = useState(false);
  const [dropActive, setDropActive] = useState(false);
  const [overallProgress, setOverallProgress] = useState(0);
  const [selectedPreviewId, setSelectedPreviewId] = useState<string | null>(null);
  const [localMessage, setLocalMessage] = useState<string>("Drop a dataset folder or choose files to start.");
  const filesInputRef = useRef<HTMLInputElement | null>(null);
  const folderInputRef = useRef<DirectoryInputElement | null>(null);

  useEffect(() => {
    if (latestUpload?.dataset_summary.previews?.length) {
      setSelectedPreviewId(latestUpload.dataset_summary.previews[0].dataset_id);
    }
  }, [latestUpload?.session_id]);

  const visibleItems = flattenTreeFiles(items, latestUpload);
  const folderName = extractFolderName(
    visibleItems.map((item) => item.relativePath).filter(Boolean)
  );
  const totalSizeBytes = visibleItems.reduce((sum, item) => sum + item.sizeBytes, 0);
  const queuedCount = visibleItems.filter((item) => item.status === "queued").length;
  const failedCount = visibleItems.filter((item) => item.status === "failed").length;
  const retryableFailedCount = visibleItems.filter((item) => item.status === "failed" && item.retryable).length;
  const tree = buildTree(visibleItems);
  const activePreview = pickPreview(latestUpload?.dataset_summary || null, selectedPreviewId);

  function mergeCandidates(candidates: CandidateFile[]) {
    setItems((current) => mergeUploadItems(current, candidates));
    setLocalMessage(`${candidates.length} file${candidates.length === 1 ? "" : "s"} staged for upload.`);
  }

  function handleFileInputChange(event: ChangeEvent<HTMLInputElement>) {
    const files = Array.from(event.target.files || []).map((file) => {
      const withRelativePath = file as File & { webkitRelativePath?: string };
      return {
        file,
        relativePath: withRelativePath.webkitRelativePath || file.name
      };
    });
    mergeCandidates(files);
    event.target.value = "";
  }

  async function handleDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setDropActive(false);
    try {
      const droppedFiles = await collectDroppedFiles(event.dataTransfer);
      if (droppedFiles.length) {
        mergeCandidates(droppedFiles);
      }
    } catch (error) {
      setLocalMessage(error instanceof Error ? error.message : "The dropped folder could not be parsed.");
    }
  }

  async function startUpload(retryOnly = false) {
    if (!token || !workspaceId) {
      setLocalMessage("Create or select a workspace before uploading.");
      return;
    }

    const uploadCandidates = items.filter((item) =>
      retryOnly ? item.status === "failed" && item.retryable : item.status === "queued"
    );
    if (!uploadCandidates.length) {
      setLocalMessage("Add at least one valid CSV or XLSX file before uploading.");
      return;
    }

    const candidateIds = new Set(uploadCandidates.map((item) => item.id));
    const candidateRelativePaths = new Set(uploadCandidates.map((item) => item.relativePath));
    const resolvedFolderName = extractFolderName(uploadCandidates.map((item) => item.relativePath));
    setUploading(true);
    setOverallProgress(0);
    setLocalMessage("Uploading dataset folder...");
    setItems((current) =>
      current.map((item) =>
        candidateIds.has(item.id)
          ? { ...item, status: "uploading", progress: 0, error: null }
          : item
      )
    );

    try {
      const response = await uploadFolderAsset(
        workspaceId,
        resolvedFolderName,
        resolvedFolderName,
        uploadCandidates.map<FolderUploadItemInput>((item) => ({
          id: item.id,
          file: item.file,
          relativePath: item.relativePath
        })),
        token,
        (snapshot: FolderUploadProgressSnapshot) => {
          setOverallProgress(snapshot.overallProgress);
          setItems((current) =>
            current.map((item) =>
              candidateIds.has(item.id)
                ? {
                    ...item,
                    progress: snapshot.fileProgress[item.id] ?? item.progress,
                    status: "uploading"
                  }
                : item
            )
          );
        }
      );

      const processedByPath = new Map(response.processed_files.map((file) => [file.relative_path, file]));
      const failedByPath = new Map(response.failed_files.map((file) => [file.relative_path, file]));

      setItems((current) =>
        current.map((item) => {
          if (!candidateRelativePaths.has(item.relativePath)) {
            return item;
          }
          const processed = processedByPath.get(item.relativePath);
          const failed = failedByPath.get(item.relativePath);
          if (processed) {
            return {
              ...item,
              status: "success",
              progress: 1,
              error: null,
              retryable: false
            };
          }
          if (failed) {
            return {
              ...item,
              status: "failed",
              progress: 0,
              error: failed.error || "Upload failed.",
              retryable: retryableFromError(failed.error)
            };
          }
          return {
            ...item,
            status: "success",
            progress: 1,
            error: null,
            retryable: false
          };
        })
      );
      setLocalMessage(response.dataset_summary.ready_message || "Dataset Ready -> Generate Insights");
      await onUploaded(response);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Folder upload failed.";
      setLocalMessage(message);
      setItems((current) =>
        current.map((item) =>
          candidateIds.has(item.id)
            ? {
                ...item,
                status: "failed",
                progress: 0,
                error: message,
                retryable: true
              }
            : item
        )
      );
    } finally {
      setUploading(false);
    }
  }

  function removeFailedItems() {
    setItems((current) => current.filter((item) => item.status !== "failed" || item.retryable));
  }

  function renderTagChips(fileResults: FolderUploadFileResult[]) {
    const tags = Array.from(new Set(fileResults.map((result) => result.file_tag).filter(Boolean))) as string[];
    if (!tags.length) {
      return null;
    }
    return (
      <div className="folder-tag-row">
        {tags.map((tag) => (
          <span key={tag} className="folder-tag">
            {tag}
          </span>
        ))}
      </div>
    );
  }

  return (
    <div className="folder-uploader-shell">
      <aside className="folder-sidebar">
        <div className="folder-sidebar__header">
          <div>
            <span className="section-kicker">Dataset Folder</span>
            <h4>{latestUpload?.folder_name || folderName}</h4>
          </div>
          <div className="folder-sidebar__meta">
            <span>{visibleItems.length} files</span>
            <strong>{formatBytes(totalSizeBytes)}</strong>
          </div>
        </div>

        <div className="folder-sidebar__body">
          {tree.length ? <FileTree nodes={tree} /> : <p>No folder staged yet.</p>}
        </div>
      </aside>

      <div className="folder-uploader-main">
        <div
          className={`folder-dropzone ${dropActive ? "folder-dropzone--active" : ""}`}
          onDragEnter={(event) => {
            event.preventDefault();
            setDropActive(true);
          }}
          onDragLeave={(event) => {
            event.preventDefault();
            if (event.currentTarget === event.target) {
              setDropActive(false);
            }
          }}
          onDragOver={(event) => event.preventDefault()}
          onDrop={(event) => void handleDrop(event)}
        >
          <div className="folder-dropzone__icon">
            <UploadCloud size={26} />
          </div>
          <div>
            <h4>Drag and drop a dataset folder</h4>
            <p>CSV and XLSX files are staged recursively, validated, and previewed before analysis.</p>
          </div>
          <div className="folder-dropzone__actions">
            <input
              ref={filesInputRef}
              type="file"
              accept=".csv,.xlsx"
              multiple
              hidden
              onChange={handleFileInputChange}
            />
            <input
              ref={folderInputRef}
              {...FOLDER_INPUT_ATTRIBUTES}
              onChange={handleFileInputChange}
            />
            <button type="button" className="ghost-button" onClick={() => filesInputRef.current?.click()}>
              <Files size={16} />
              Upload Files
            </button>
            <button type="button" className="primary-button-new" onClick={() => folderInputRef.current?.click()}>
              <FolderOpen size={16} />
              Upload Folder
            </button>
          </div>
        </div>

        <div className="folder-upload-toolbar">
          <div className="folder-upload-toolbar__copy">
            <strong>{localMessage}</strong>
            <span>
              {queuedCount} queued · {failedCount} failed
            </span>
          </div>
          <div className="folder-upload-toolbar__actions">
            <button type="button" className="ghost-button" disabled={uploading || !queuedCount} onClick={() => void startUpload(false)}>
              {uploading ? <Loader2 size={16} className="folder-tree-spin" /> : <UploadCloud size={16} />}
              Start Upload
            </button>
            <button
              type="button"
              className="ghost-button"
              disabled={uploading || retryableFailedCount === 0}
              onClick={() => void startUpload(true)}
            >
              <RefreshCw size={16} />
              Retry Failed
            </button>
            <button type="button" className="ghost-button" disabled={uploading || failedCount === 0} onClick={removeFailedItems}>
              <AlertCircle size={16} />
              Clear Invalid
            </button>
          </div>
        </div>

        <div className="folder-progress-panel">
          <div className="folder-progress-panel__header">
            <span>Overall Upload Progress</span>
            <strong>{Math.round(overallProgress * 100)}%</strong>
          </div>
          <div className="folder-progress-track">
            <div
              className="folder-progress-fill"
              style={{ width: `${overallProgress > 0 ? Math.max(4, overallProgress * 100) : 0}%` }}
            />
          </div>
          <div className="folder-progress-grid">
            {visibleItems.slice(0, 6).map((item) => (
              <div key={item.id} className={`folder-progress-card folder-progress-card--${item.status}`}>
                <div className="folder-progress-card__header">
                  <div>
                    <strong>{item.fileName}</strong>
                    <span>{item.relativePath}</span>
                  </div>
                  <span>{Math.round(item.progress * 100)}%</span>
                </div>
                <div className="folder-progress-track folder-progress-track--small">
                  <div
                    className="folder-progress-fill"
                    style={{ width: `${item.progress > 0 ? Math.max(4, item.progress * 100) : 0}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>

        {latestUpload ? (
          <div className="folder-summary-grid">
            <article className="folder-summary-card folder-summary-card--spotlight">
              <div className="folder-summary-card__header">
                <div>
                  <span className="section-kicker">Ready State</span>
                  <h4>{latestUpload.dataset_summary.ready_message}</h4>
                  <p>
                    {latestUpload.files_processed} files processed with{" "}
                    {latestUpload.dataset_summary.relationships.length} detected table relationships.
                  </p>
                </div>
                <Sparkles size={20} />
              </div>
              <div className="folder-summary-actions">
                <button
                  type="button"
                  className="primary-button-new"
                  disabled={!latestUpload.asset}
                  onClick={() =>
                    onAutoAnalyze(
                      latestUpload.dataset_summary.suggested_analysis_prompt || AUTO_ANALYZE_FALLBACK,
                      latestUpload.asset?.asset_id
                    )
                  }
                >
                  <BarChart3 size={16} />
                  Auto Analyze Dataset
                </button>
              </div>
            </article>

            <article className="folder-summary-card">
              <div className="folder-summary-card__header">
                <div>
                  <span className="section-kicker">Detected Tables</span>
                  <h4>{latestUpload.folder_name}</h4>
                </div>
                <Files size={18} />
              </div>
              <div className="folder-table-list">
                {latestUpload.dataset_summary.tables.map((table) => (
                  <div key={table} className="folder-table-list__item">
                    <FileSpreadsheet size={15} />
                    <span>{table}</span>
                  </div>
                ))}
              </div>
              {renderTagChips(latestUpload.processed_files)}
            </article>

            <article className="folder-summary-card">
              <div className="folder-summary-card__header">
                <div>
                  <span className="section-kicker">Auto Relationships</span>
                  <h4>Join candidates</h4>
                </div>
                <Sparkles size={18} />
              </div>
              <div className="folder-relationship-list">
                {latestUpload.dataset_summary.relationships.length ? (
                  latestUpload.dataset_summary.relationships.map((relationship) => (
                    <div
                      key={`${relationship.left_table}-${relationship.left_column}-${relationship.right_table}-${relationship.right_column}`}
                      className="folder-relationship-card"
                    >
                      <strong>
                        {relationship.left_table}.{relationship.left_column}
                      </strong>
                      <span>
                        <ChevronRight size={14} />
                        {relationship.right_table}.{relationship.right_column}
                      </span>
                      <small>
                        Confidence {Math.round(relationship.confidence * 100)}% · Match rate{" "}
                        {Math.round(relationship.match_rate * 100)}%
                      </small>
                    </div>
                  ))
                ) : (
                  <p>No strong table relationships were detected yet.</p>
                )}
              </div>
            </article>
          </div>
        ) : null}

        {latestUpload?.dataset_summary.previews?.length ? (
          <article className="folder-preview-card">
            <div className="folder-preview-card__header">
              <div>
                <span className="section-kicker">Preview Mode</span>
                <h4>{activePreview?.table_name || "Preview"}</h4>
                <p>First 10 rows from each uploaded dataset.</p>
              </div>
              <FileSpreadsheet size={18} />
            </div>

            <div className="folder-preview-tabs">
              {latestUpload.dataset_summary.previews.map((preview) => (
                <button
                  key={preview.dataset_id}
                  type="button"
                  className={`folder-preview-tab ${activePreview?.dataset_id === preview.dataset_id ? "folder-preview-tab--active" : ""}`}
                  onClick={() => setSelectedPreviewId(preview.dataset_id)}
                >
                  <span>{preview.table_name}</span>
                  <small>{preview.file_tag || "sales"}</small>
                </button>
              ))}
            </div>

            {activePreview ? (
              <div className="table-shell">
                <table>
                  <thead>
                    <tr>
                      {activePreview.preview_columns.map((column) => (
                        <th key={column}>{column}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {activePreview.preview_rows.slice(0, 10).map((row, index) => (
                      <tr key={`${activePreview.dataset_id}-${index}`}>
                        {activePreview.preview_columns.map((column) => (
                          <td key={`${column}-${index}`}>{String(row[column] ?? "")}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : null}
          </article>
        ) : null}
      </div>
    </div>
  );
}
