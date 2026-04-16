#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path


TAGGED_METRIC_RE = re.compile(r"^(?P<name>[^{]+)\{(?P<tags>.+)\}$")


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze Aidssist load-test and telemetry outputs.")
    parser.add_argument("--k6-summary", action="append", default=[], help="Path to a k6 --summary-export JSON file.")
    parser.add_argument("--docker-stats-csv", help="Optional CSV export with service,cpu_pct,memory_pct columns.")
    parser.add_argument("--pg-statements-csv", help="Optional CSV export of pg_stat_statements.")
    parser.add_argument("--app-log", help="Optional JSON-lines application log file.")
    parser.add_argument("--latency-threshold-ms", type=float, default=2000.0)
    parser.add_argument("--error-threshold-rate", type=float, default=0.01)
    parser.add_argument("--cpu-threshold", type=float, default=80.0)
    parser.add_argument("--memory-threshold", type=float, default=85.0)
    parser.add_argument("--output-json", help="Optional path to write JSON analysis.")
    return parser.parse_args()


def parse_metric_tags(metric_name: str):
    match = TAGGED_METRIC_RE.match(metric_name)
    if not match:
        return metric_name, {}

    tags = {}
    for raw_tag in match.group("tags").split(","):
        if ":" not in raw_tag:
            continue
        key, value = raw_tag.split(":", 1)
        tags[key.strip()] = value.strip()
    return match.group("name"), tags


def analyze_k6_summaries(paths: list[str], latency_threshold_ms: float, error_threshold_rate: float):
    overall = []
    endpoint_metrics = defaultdict(dict)
    issues = []

    for path in paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        metrics = payload.get("metrics", {})
        overall_entry = {
            "file": path,
            "requests_per_second": metrics.get("http_reqs", {}).get("values", {}).get("rate"),
            "error_rate": metrics.get("http_req_failed", {}).get("values", {}).get("rate"),
            "avg_latency_ms": metrics.get("http_req_duration", {}).get("values", {}).get("avg"),
            "p95_latency_ms": metrics.get("http_req_duration", {}).get("values", {}).get("p(95)"),
            "max_latency_ms": metrics.get("http_req_duration", {}).get("values", {}).get("max"),
        }
        overall.append(overall_entry)

        for metric_name, metric_payload in metrics.items():
            base_name, tags = parse_metric_tags(metric_name)
            endpoint = tags.get("endpoint")
            if not endpoint:
                continue
            endpoint_metrics[(path, endpoint)][base_name] = metric_payload.get("values", {})

    for (path, endpoint), metrics in sorted(endpoint_metrics.items()):
        latency = metrics.get("http_req_duration", {})
        failures = metrics.get("http_req_failed", {})
        p95 = float(latency.get("p(95)", 0.0) or 0.0)
        max_latency = float(latency.get("max", 0.0) or 0.0)
        error_rate = float(failures.get("rate", 0.0) or 0.0)

        if p95 > latency_threshold_ms:
            issues.append(
                {
                    "type": "slow_endpoint",
                    "file": path,
                    "endpoint": endpoint,
                    "p95_latency_ms": round(p95, 2),
                    "max_latency_ms": round(max_latency, 2),
                    "recommendation": "Reduce synchronous work, add caching, or move CPU-heavy processing behind the async worker queue.",
                }
            )
        if error_rate > error_threshold_rate:
            issues.append(
                {
                    "type": "failing_endpoint",
                    "file": path,
                    "endpoint": endpoint,
                    "error_rate": round(error_rate, 4),
                    "recommendation": "Inspect application logs for non-2xx responses and harden retry, validation, or timeout handling.",
                }
            )

    return {"overall": overall, "issues": issues}


def analyze_docker_stats(path: str | None, cpu_threshold: float, memory_threshold: float):
    if not path:
        return {"issues": []}

    issues = []
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            service = row.get("service") or row.get("name") or row.get("container") or "unknown"
            cpu = float(row.get("cpu_pct", row.get("cpu", 0.0)) or 0.0)
            memory = float(row.get("memory_pct", row.get("mem_pct", 0.0)) or 0.0)
            if cpu >= cpu_threshold:
                issues.append(
                    {
                        "type": "high_cpu",
                        "service": service,
                        "cpu_pct": round(cpu, 2),
                        "recommendation": "Scale this service horizontally or reduce per-request compute.",
                    }
                )
            if memory >= memory_threshold:
                issues.append(
                    {
                        "type": "high_memory",
                        "service": service,
                        "memory_pct": round(memory, 2),
                        "recommendation": "Reduce dataframe copies, stream large payloads, or increase memory limits.",
                    }
                )
    return {"issues": issues}


def analyze_pg_statements(path: str | None):
    if not path:
        return {"issues": []}

    issues = []
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            mean_exec = float(row.get("mean_exec_time", 0.0) or 0.0)
            calls = int(float(row.get("calls", 0) or 0))
            if mean_exec < 50.0:
                continue
            issues.append(
                {
                    "type": "slow_query",
                    "query": (row.get("query") or "").strip().replace("\n", " ")[:300],
                    "mean_exec_time_ms": round(mean_exec, 2),
                    "calls": calls,
                    "recommendation": "Add indexes for filter/order columns or reduce scan width and row count.",
                }
            )
    return {"issues": issues[:20]}


def analyze_app_logs(path: str | None, latency_threshold_ms: float):
    if not path:
        return {"issues": []}

    endpoint_stats = defaultdict(lambda: {"count": 0, "errors": 0, "slow": 0, "max_ms": 0.0})
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            endpoint = payload.get("endpoint")
            if not endpoint:
                continue
            duration_ms = float(payload.get("duration_ms", 0.0) or 0.0)
            status_code = int(payload.get("status_code", 0) or 0)
            stats = endpoint_stats[endpoint]
            stats["count"] += 1
            stats["max_ms"] = max(stats["max_ms"], duration_ms)
            if status_code >= 500:
                stats["errors"] += 1
            if duration_ms >= latency_threshold_ms:
                stats["slow"] += 1

    issues = []
    for endpoint, stats in endpoint_stats.items():
        if stats["errors"] > 0:
            issues.append(
                {
                    "type": "log_failures",
                    "endpoint": endpoint,
                    "count": stats["errors"],
                    "recommendation": "Inspect structured logs for request_id-linked failures and fix the dominant exception path.",
                }
            )
        if stats["slow"] > 0:
            issues.append(
                {
                    "type": "log_slow_endpoint",
                    "endpoint": endpoint,
                    "slow_requests": stats["slow"],
                    "max_latency_ms": round(stats["max_ms"], 2),
                    "recommendation": "Profile this route in Grafana and compare API latency against queue, DB, and provider timings.",
                }
            )
    return {"issues": issues}


def build_markdown_report(results: dict):
    lines = ["# Aidssist Bottleneck Report", ""]

    for suite in results["k6"]["overall"]:
        lines.extend(
            [
                f"## {Path(suite['file']).name}",
                "",
                f"- Requests/sec: {suite.get('requests_per_second')}",
                f"- Error rate: {suite.get('error_rate')}",
                f"- Average latency (ms): {suite.get('avg_latency_ms')}",
                f"- p95 latency (ms): {suite.get('p95_latency_ms')}",
                f"- Max latency (ms): {suite.get('max_latency_ms')}",
                "",
            ]
        )

    lines.append("## Issues")
    lines.append("")
    all_issues = (
        results["k6"]["issues"]
        + results["docker"]["issues"]
        + results["pg"]["issues"]
        + results["logs"]["issues"]
    )
    if not all_issues:
        lines.append("- No issues were flagged by the supplied telemetry.")
    else:
        for issue in all_issues:
            title = issue["type"].replace("_", " ").title()
            details = ", ".join(f"{key}={value}" for key, value in issue.items() if key not in {"type", "recommendation"})
            lines.append(f"- {title}: {details}. Fix: {issue['recommendation']}")

    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    results = {
        "k6": analyze_k6_summaries(args.k6_summary, args.latency_threshold_ms, args.error_threshold_rate),
        "docker": analyze_docker_stats(args.docker_stats_csv, args.cpu_threshold, args.memory_threshold),
        "pg": analyze_pg_statements(args.pg_statements_csv),
        "logs": analyze_app_logs(args.app_log, args.latency_threshold_ms),
    }

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(build_markdown_report(results))


if __name__ == "__main__":
    main()
