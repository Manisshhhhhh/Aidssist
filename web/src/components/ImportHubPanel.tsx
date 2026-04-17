import { useEffect, useRef, useState } from "react";
import { Cloud, Database, ExternalLink, FolderOpen, Loader2 } from "lucide-react";

import { getImportJob, importFromGoogleDrive, importFromKaggle } from "../lib/api";
import type { ImportJob } from "../types/api";

declare global {
  interface Window {
    gapi?: {
      load: (name: string, callback: () => void) => void;
    };
    google?: any;
  }
}

type ImportHubPanelProps = {
  workspaceId: string | null;
  token: string | null;
  onImported?: (job: ImportJob) => void | Promise<void>;
  onStatus?: (message: string) => void;
};

const GOOGLE_SCOPE = "https://www.googleapis.com/auth/drive.readonly";
const GOOGLE_API_KEY = import.meta.env.VITE_GOOGLE_API_KEY as string | undefined;
const GOOGLE_CLIENT_ID = import.meta.env.VITE_GOOGLE_CLIENT_ID as string | undefined;
const GOOGLE_APP_ID = import.meta.env.VITE_GOOGLE_APP_ID as string | undefined;

const DRIVE_MIME_TYPES = [
  "text/csv",
  "application/vnd.ms-excel",
  "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
  "application/vnd.google-apps.folder"
].join(",");

let googleScriptsPromise: Promise<void> | null = null;

function loadScript(src: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const existing = document.querySelector(`script[src="${src}"]`);
    if (existing) {
      resolve();
      return;
    }
    const script = document.createElement("script");
    script.src = src;
    script.async = true;
    script.defer = true;
    script.onload = () => resolve();
    script.onerror = () => reject(new Error(`Failed to load ${src}`));
    document.head.appendChild(script);
  });
}

function loadGoogleScripts(): Promise<void> {
  if (!googleScriptsPromise) {
    googleScriptsPromise = Promise.all([
      loadScript("https://apis.google.com/js/api.js"),
      loadScript("https://accounts.google.com/gsi/client")
    ]).then(() => undefined);
  }
  return googleScriptsPromise;
}

async function waitForImportCompletion(jobId: string, token: string): Promise<ImportJob> {
  let latest = await getImportJob(jobId, token);
  while (latest.status !== "completed" && latest.status !== "failed") {
    await new Promise((resolve) => window.setTimeout(resolve, 1500));
    latest = await getImportJob(jobId, token);
  }
  return latest;
}

export default function ImportHubPanel({
  workspaceId,
  token,
  onImported,
  onStatus
}: ImportHubPanelProps) {
  const [kaggleUrl, setKaggleUrl] = useState("");
  const [busyAction, setBusyAction] = useState<"google" | "kaggle" | null>(null);
  const [localMessage, setLocalMessage] = useState<string | null>(null);
  const [localError, setLocalError] = useState<string | null>(null);
  const [googleReady, setGoogleReady] = useState(false);
  const tokenClientRef = useRef<any>(null);

  const canUseGooglePicker = Boolean(GOOGLE_API_KEY && GOOGLE_CLIENT_ID);

  useEffect(() => {
    if (!canUseGooglePicker) {
      return;
    }
    let cancelled = false;
    void (async () => {
      try {
        await loadGoogleScripts();
        if (!cancelled) {
          setGoogleReady(true);
        }
      } catch (error) {
        if (!cancelled) {
          setLocalError(error instanceof Error ? error.message : "Google Picker could not be loaded.");
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [canUseGooglePicker]);

  async function finalizeJob(job: ImportJob) {
    if (job.status === "failed") {
      throw new Error(job.error_message || "The import job failed.");
    }
    setLocalMessage(
      job.status === "completed"
        ? `${job.source_type === "google_drive" ? "Google Drive" : "Kaggle"} import completed.`
        : "Import job queued."
    );
    onStatus?.(
      job.status === "completed"
        ? `${job.source_type === "google_drive" ? "Google Drive" : "Kaggle"} import completed.`
        : "Import job queued."
    );
    await onImported?.(job);
  }

  async function handleKaggleImport() {
    if (!workspaceId || !token || !kaggleUrl.trim()) {
      return;
    }
    setBusyAction("kaggle");
    setLocalError(null);
    setLocalMessage("Downloading the Kaggle dataset and registering tables...");
    onStatus?.("Downloading the Kaggle dataset and registering tables...");
    try {
      const job = await importFromKaggle({ workspace_id: workspaceId, dataset_url: kaggleUrl.trim() }, token);
      const completed = job.status === "completed" ? job : await waitForImportCompletion(job.job_id, token);
      await finalizeJob(completed);
      setKaggleUrl("");
    } catch (error) {
      const message = error instanceof Error ? error.message : "Kaggle import failed.";
      setLocalError(message);
      onStatus?.(message);
    } finally {
      setBusyAction(null);
    }
  }

  function openPicker(accessToken: string) {
    if (!window.google?.picker || !workspaceId || !token) {
      setLocalError("Google Picker is not available yet.");
      return;
    }
    const view = new window.google.picker.DocsView(window.google.picker.ViewId.DOCS);
    view.setIncludeFolders(true);
    view.setSelectFolderEnabled(true);
    view.setMimeTypes(DRIVE_MIME_TYPES);
    const picker = new window.google.picker.PickerBuilder()
      .setOAuthToken(accessToken)
      .setDeveloperKey(GOOGLE_API_KEY)
      .setAppId(GOOGLE_APP_ID || "")
      .addView(view)
      .setTitle("Import datasets from Google Drive")
      .setCallback(async (data: any) => {
        if (data.action !== window.google.picker.Action.PICKED) {
          return;
        }
        const doc = data.docs?.[0];
        const fileId = doc?.id;
        if (!fileId) {
          setLocalError("Google Picker did not return a file id.");
          return;
        }
        setBusyAction("google");
        setLocalError(null);
        setLocalMessage("Importing the selected Drive file into Aidssist...");
        onStatus?.("Importing the selected Drive file into Aidssist...");
        try {
          const job = await importFromGoogleDrive(
            {
              workspace_id: workspaceId,
              file_id: fileId,
              access_token: accessToken
            },
            token
          );
          await finalizeJob(job);
        } catch (error) {
          const message = error instanceof Error ? error.message : "Google Drive import failed.";
          setLocalError(message);
          onStatus?.(message);
        } finally {
          setBusyAction(null);
        }
      })
      .build();
    picker.setVisible(true);
  }

  function handleGoogleImport() {
    if (!workspaceId || !token || !canUseGooglePicker || !googleReady) {
      return;
    }
    setLocalError(null);
    window.gapi?.load("picker", () => {
      if (!tokenClientRef.current) {
        tokenClientRef.current = window.google?.accounts?.oauth2?.initTokenClient({
          client_id: GOOGLE_CLIENT_ID,
          scope: GOOGLE_SCOPE,
          callback: (tokenResponse: { access_token?: string }) => {
            if (!tokenResponse?.access_token) {
              setLocalError("Google authentication did not return an access token.");
              return;
            }
            openPicker(tokenResponse.access_token);
          }
        });
      }
      tokenClientRef.current?.requestAccessToken({ prompt: "consent" });
    });
  }

  return (
    <article className="surface-card surface-card--wide import-hub-card">
      <div className="import-hub-card__header">
        <div>
          <span className="section-kicker">Premium Intake</span>
          <h3>Import from Google Drive or Kaggle</h3>
          <p>
            Pull live datasets straight into Aidssist, auto-register every table, and move directly into
            schema mapping, AI insights, and chat.
          </p>
        </div>
        <div className="import-hub-card__pill">
          <Database size={18} />
          <span>DuckDB + AI pipeline</span>
        </div>
      </div>

      <div className="import-hub-grid">
        <section className="import-tile">
          <div className="import-tile__icon">
            <Cloud size={18} />
          </div>
          <span className="section-kicker">Google Drive</span>
          <h4>Use the Google Picker UI</h4>
          <p>Select a CSV, XLSX, or full dataset folder and import it directly into the current workspace.</p>
          <button
            type="button"
            className="primary-button-new"
            disabled={!workspaceId || !token || !googleReady || !canUseGooglePicker || busyAction !== null}
            onClick={handleGoogleImport}
          >
            {busyAction === "google" ? (
              <>
                <Loader2 size={16} className="spin" />
                Importing...
              </>
            ) : (
              <>
                <FolderOpen size={16} />
                Import from Google Drive
              </>
            )}
          </button>
          {!canUseGooglePicker ? (
            <div className="inline-note">
              Add `VITE_GOOGLE_API_KEY` and `VITE_GOOGLE_CLIENT_ID` to enable the Drive picker.
            </div>
          ) : null}
        </section>

        <section className="import-tile">
          <div className="import-tile__icon">
            <ExternalLink size={18} />
          </div>
          <span className="section-kicker">Kaggle</span>
          <h4>Import a full Kaggle dataset</h4>
          <p>Paste a Kaggle dataset URL. Aidssist downloads it on the server, unzips it, and registers every table.</p>
          <div className="import-form">
            <input
              value={kaggleUrl}
              onChange={(event) => setKaggleUrl(event.target.value)}
              placeholder="https://www.kaggle.com/datasets/username/dataset-name"
            />
            <button
              type="button"
              className="primary-button-new"
              disabled={!workspaceId || !token || !kaggleUrl.trim() || busyAction !== null}
              onClick={() => void handleKaggleImport()}
            >
              {busyAction === "kaggle" ? (
                <>
                  <Loader2 size={16} className="spin" />
                  Importing...
                </>
              ) : (
                "Import Kaggle Dataset"
              )}
            </button>
          </div>
        </section>
      </div>

      {localMessage ? <div className="notice-card import-hub-card__notice">{localMessage}</div> : null}
      {localError ? <div className="notice-card notice-card--error import-hub-card__notice">{localError}</div> : null}
    </article>
  );
}
