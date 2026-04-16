import http from "k6/http";
import { check, sleep } from "k6";
import { Counter, Rate, Trend } from "k6/metrics";

const BASE_URL = (__ENV.BASE_URL || "http://localhost:8080").replace(/\/$/, "");
const API_BASE_URL = (__ENV.API_BASE_URL || `${BASE_URL}/api`).replace(/\/$/, "");
const TARGET_VUS = Number(__ENV.TARGET_VUS || "100");
const POLL_TIMEOUT_MS = Number(__ENV.POLL_TIMEOUT_MS || "240000");
const POLL_INTERVAL_SECONDS = Number(__ENV.POLL_INTERVAL_SECONDS || "2");

const salesCsv = open("./fixtures/sales.csv", "b");
const ratingsCsv = open("./fixtures/ratings.csv", "b");

export const completedJobs = new Counter("aidssist_completed_jobs");
export const failedJobs = new Rate("aidssist_job_failures");
export const endToEndDuration = new Trend("aidssist_end_to_end_seconds");

function buildStages(targetVus) {
  const safeTarget = Math.max(1, targetVus);
  return [
    { duration: "1m", target: Math.max(1, Math.floor(safeTarget * 0.25)) },
    { duration: "2m", target: Math.max(1, Math.floor(safeTarget * 0.5)) },
    { duration: "3m", target: safeTarget },
    { duration: "2m", target: safeTarget },
    { duration: "1m", target: 0 },
  ];
}

export const options = {
  scenarios: {
    api_http: {
      executor: "ramping-vus",
      startVUs: 0,
      stages: buildStages(TARGET_VUS),
      gracefulRampDown: "30s",
    },
  },
  thresholds: {
    http_req_failed: ["rate<0.01"],
    "http_req_duration{endpoint:upload}": ["p(95)<2000", "max<5000"],
    "http_req_duration{endpoint:analyze}": ["p(95)<2000", "max<5000"],
    "http_req_duration{endpoint:job_status}": ["p(95)<1500", "max<4000"],
    "http_req_duration{endpoint:result_download}": ["p(95)<2000", "max<8000"],
    aidssist_job_failures: ["rate<0.01"],
    aidssist_end_to_end_seconds: ["p(95)<120"],
  },
};

function randomThinkTime() {
  sleep(Math.random() * 2 + 1);
}

function uploadDataset(buffer, fileName) {
  const response = http.post(
    `${API_BASE_URL}/v1/uploads`,
    { file: http.file(buffer, fileName, "text/csv") },
    { tags: { endpoint: "upload" } },
  );

  check(response, {
    "upload returned 200": (res) => res.status === 200,
    "upload returned dataset_id": (res) => !!res.json("dataset_id"),
  });

  return response.status === 200 ? response.json("dataset_id") : null;
}

function submitJob(datasetId, query) {
  const response = http.post(
    `${API_BASE_URL}/v1/jobs/analyze`,
    JSON.stringify({
      dataset_id: datasetId,
      query,
      workflow_context: {
        source: "k6-api-load",
        requested_by: "performance-suite",
      },
    }),
    {
      headers: { "Content-Type": "application/json" },
      tags: { endpoint: "analyze" },
    },
  );

  check(response, {
    "analyze returned 200": (res) => res.status === 200,
    "analyze returned job_id": (res) => !!res.json("job_id"),
  });

  return response.status === 200 ? response.json("job_id") : null;
}

function pollJob(jobId) {
  const deadline = Date.now() + POLL_TIMEOUT_MS;

  while (Date.now() < deadline) {
    const response = http.get(`${API_BASE_URL}/v1/jobs/${jobId}`, {
      tags: { endpoint: "job_status" },
    });

    check(response, {
      "status endpoint returned 200": (res) => res.status === 200,
      "status endpoint returned status": (res) => !!res.json("status"),
    });

    if (response.status !== 200) {
      return { status: "failed", error_message: `status polling failed with ${response.status}` };
    }

    const payload = response.json();
    if (payload.status === "completed" || payload.status === "failed") {
      return payload;
    }

    sleep(POLL_INTERVAL_SECONDS);
  }

  return { status: "failed", error_message: "job polling timed out" };
}

function downloadArtifact(jobId) {
  const response = http.get(`${API_BASE_URL}/v1/jobs/${jobId}/artifacts/result`, {
    tags: { endpoint: "result_download" },
  });

  check(response, {
    "artifact download returned 200": (res) => res.status === 200,
  });
}

function runJourney(buffer, fileName, query) {
  const started = Date.now();
  const datasetId = uploadDataset(buffer, fileName);
  if (!datasetId) {
    failedJobs.add(1);
    return;
  }

  randomThinkTime();
  const jobId = submitJob(datasetId, query);
  if (!jobId) {
    failedJobs.add(1);
    return;
  }

  const payload = pollJob(jobId);
  const succeeded = payload.status === "completed";
  failedJobs.add(succeeded ? 0 : 1);
  endToEndDuration.add((Date.now() - started) / 1000);

  if (succeeded) {
    completedJobs.add(1);
    if (Math.random() < 0.6) {
      randomThinkTime();
      downloadArtifact(jobId);
    }
  }
}

export default function() {
  const scenarios = [
    {
      fileName: "sales.csv",
      buffer: salesCsv,
      query: "predict sales for next week, next month, next quarter, and next year",
    },
    {
      fileName: "ratings.csv",
      buffer: ratingsCsv,
      query: "analyze product ratings and show rating distribution",
    },
    {
      fileName: "sales.csv",
      buffer: salesCsv,
      query: "top customers by revenue",
    },
  ];
  const picked = scenarios[Math.floor(Math.random() * scenarios.length)];
  runJourney(picked.buffer, picked.fileName, picked.query);
  randomThinkTime();
}
