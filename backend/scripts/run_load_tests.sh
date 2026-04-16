#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

BASE_URL="${BASE_URL:-http://localhost:8080}"
API_BASE_URL="${API_BASE_URL:-${BASE_URL}/api}"
OUT_DIR="${OUT_DIR:-performance_results}"
K6_BIN="${K6_BIN:-k6}"
ENABLE_BROWSER="${ENABLE_BROWSER:-0}"

mkdir -p "${OUT_DIR}"

run_api_suite() {
  local vus="$1"
  echo "Running API load suite for ${vus} VUs"
  "${K6_BIN}" run \
    --env BASE_URL="${BASE_URL}" \
    --env API_BASE_URL="${API_BASE_URL}" \
    --env TARGET_VUS="${vus}" \
    --summary-export "${OUT_DIR}/api_${vus}.json" \
    "${BACKEND_DIR}/load_tests/api_load.js"
}

run_mixed_suite() {
  local vus="$1"
  echo "Running mixed workload suite for ${vus} VUs"
  "${K6_BIN}" run \
    --env BASE_URL="${BASE_URL}" \
    --env API_BASE_URL="${API_BASE_URL}" \
    --env TARGET_VUS="${vus}" \
    --summary-export "${OUT_DIR}/mixed_${vus}.json" \
    "${BACKEND_DIR}/load_tests/mixed_workload.js"
}

run_browser_suite() {
  echo "Running browser journey suite"
  K6_BROWSER_ENABLED=true "${K6_BIN}" run \
    --env BASE_URL="${BASE_URL}" \
    --env BROWSER_VUS="${BROWSER_VUS:-5}" \
    --env BROWSER_ITERATIONS="${BROWSER_ITERATIONS:-20}" \
    --summary-export "${OUT_DIR}/ui_browser.json" \
    "${BACKEND_DIR}/load_tests/ui_journey.js"
}

run_api_suite 100
run_api_suite 500
run_api_suite 1000
run_mixed_suite "${MIXED_VUS:-300}"

if [[ "${ENABLE_BROWSER}" == "1" ]]; then
  run_browser_suite
fi

echo "Load test summaries written to ${OUT_DIR}"
