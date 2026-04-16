import http from "k6/http";
import { check, sleep } from "k6";
import { Counter, Rate, Trend } from "k6/metrics";

const BASE_URL = (__ENV.BASE_URL || "http://localhost:8080").replace(/\/$/, "");
const API_BASE_URL = (__ENV.API_BASE_URL || `${BASE_URL}/api`).replace(/\/$/, "");
const TARGET_VUS = Number(__ENV.TARGET_VUS || "300");
const POLL_TIMEOUT_MS = Number(__ENV.POLL_TIMEOUT_MS || "240000");
const POLL_INTERVAL_SECONDS = Number(__ENV.POLL_INTERVAL_SECONDS || "2");

const salesCsv = open("./fixtures/sales.csv", "b");
const ratingsCsv = open("./fixtures/ratings.csv", "b");

export const completedJobs = new Counter("aidssist_mixed_completed_jobs");
export const failedJobs = new Rate("aidssist_mixed_job_failures");
export const endToEndDuration = new Trend("aidssist_mixed_end_to_end_seconds");

function scenarioVus(share) {
  return Math.max(1, Math.floor(TARGET_VUS * share));
}

export const options = {
  scenarios: {
    ratings_users: {
      executor: "constant-vus",
      vus: scenarioVus(0.25),
      duration: __ENV.DURATION || "8m",
      exec: "ratingsScenario",
    },
    forecast_users: {
      executor: "constant-vus",
      vus: scenarioVus(0.4),
      duration: __ENV.DURATION || "8m",
      exec: "forecastScenario",
    },
    general_users: {
      executor: "constant-vus",
      vus: scenarioVus(0.35),
      duration: __ENV.DURATION || "8m",
      exec: "generalScenario",
    },
  },
  thresholds: {
    http_req_failed: ["rate<0.01"],
    "http_req_duration{endpoint:upload}": ["p(95)<2000"],
    "http_req_duration{endpoint:analyze}": ["p(95)<2000"],
    "http_req_duration{endpoint:job_status}": ["p(95)<1500"],
    "http_req_duration{endpoint:result_download}": ["p(95)<2000"],
    aidssist_mixed_job_failures: ["rate<0.01"],
    aidssist_mixed_end_to_end_seconds: ["p(95)<120"],
  },
};

function uploadDataset(buffer, fileName) {
  const response = http.post(
    `${API_BASE_URL}/v1/uploads`,
    { file: http.file(buffer, fileName, "text/csv") },
    { tags: { endpoint: "upload" } },
  );
  check(response, { "upload OK": (res) => res.status === 200 });
  return response.status === 200 ? response.json("dataset_id") : null;
}

function submitJob(datasetId, query, journey) {
  const response = http.post(
    `${API_BASE_URL}/v1/jobs/analyze`,
    JSON.stringify({
      dataset_id: datasetId,
      query,
      workflow_context: {
        source: "k6-mixed-workload",
        journey,
      },
    }),
    {
      headers: { "Content-Type": "application/json" },
      tags: { endpoint: "analyze", journey },
    },
  );
  check(response, { "job accepted": (res) => res.status === 200 });
  return response.status === 200 ? response.json("job_id") : null;
}

function pollJob(jobId, journey) {
  const deadline = Date.now() + POLL_TIMEOUT_MS;
  while (Date.now() < deadline) {
    const response = http.get(`${API_BASE_URL}/v1/jobs/${jobId}`, {
      tags: { endpoint: "job_status", journey },
    });
    check(response, { "job status OK": (res) => res.status === 200 });
    if (response.status !== 200) {
      return { status: "failed" };
    }
    const payload = response.json();
    if (payload.status === "completed" || payload.status === "failed") {
      return payload;
    }
    sleep(POLL_INTERVAL_SECONDS);
  }
  return { status: "failed", error_message: "poll timeout" };
}

function downloadArtifact(jobId, journey) {
  const response = http.get(`${API_BASE_URL}/v1/jobs/${jobId}/artifacts/result`, {
    tags: { endpoint: "result_download", journey },
  });
  check(response, { "result artifact OK": (res) => res.status === 200 });
}

function runJourney(buffer, fileName, query, journey) {
  const started = Date.now();
  const datasetId = uploadDataset(buffer, fileName);
  if (!datasetId) {
    failedJobs.add(1);
    return;
  }

  sleep(Math.random() + 0.5);
  const jobId = submitJob(datasetId, query, journey);
  if (!jobId) {
    failedJobs.add(1);
    return;
  }

  const payload = pollJob(jobId, journey);
  const succeeded = payload.status === "completed";
  failedJobs.add(succeeded ? 0 : 1);
  endToEndDuration.add((Date.now() - started) / 1000);

  if (succeeded) {
    completedJobs.add(1);
    if (Math.random() < 0.7) {
      sleep(Math.random() + 0.5);
      downloadArtifact(jobId, journey);
    }
  }
}

export function ratingsScenario() {
  runJourney(ratingsCsv, "ratings.csv", "show average rating, top-rated products, worst-rated products, and rating distribution", "ratings");
  sleep(Math.random() * 2 + 1);
}

export function forecastScenario() {
  runJourney(salesCsv, "sales.csv", "predict sales for next week, next month, next quarter, and next year", "forecast");
  sleep(Math.random() * 2 + 1);
}

export function generalScenario() {
  runJourney(salesCsv, "sales.csv", "top customers by revenue", "general");
  sleep(Math.random() * 2 + 1);
}
