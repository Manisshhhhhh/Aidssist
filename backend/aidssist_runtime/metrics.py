from __future__ import annotations

import time
from contextlib import contextmanager

try:
    from prometheus_client import CONTENT_TYPE_LATEST, REGISTRY, Counter, Gauge, Histogram, generate_latest
except ImportError:  # pragma: no cover - exercised only when optional deps are absent
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"
    REGISTRY = None

    class _NoOpMetric:
        def labels(self, **kwargs):
            del kwargs
            return self

        def inc(self, amount: float = 1.0):
            del amount

        def observe(self, value: float):
            del value

        def set(self, value: float):
            del value

    def Counter(*args, **kwargs):  # type: ignore[misc]
        del args, kwargs
        return _NoOpMetric()

    def Gauge(*args, **kwargs):  # type: ignore[misc]
        del args, kwargs
        return _NoOpMetric()

    def Histogram(*args, **kwargs):  # type: ignore[misc]
        del args, kwargs
        return _NoOpMetric()

    def generate_latest():
        return b"# prometheus_client is not installed\n"


def _get_registered_metric(metric_name: str):
    if REGISTRY is None:
        return None
    collectors = getattr(REGISTRY, "_names_to_collectors", None)
    if not isinstance(collectors, dict):
        return None
    return collectors.get(metric_name)


def _get_or_create_metric(metric_name: str, factory):
    existing_metric = _get_registered_metric(metric_name)
    if existing_metric is not None:
        return existing_metric

    try:
        return factory()
    except ValueError as error:
        # Streamlit and other dev reload paths can import this module more than once.
        if "Duplicated timeseries in CollectorRegistry" not in str(error):
            raise
        existing_metric = _get_registered_metric(metric_name)
        if existing_metric is None:
            raise
        return existing_metric


REQUEST_COUNT = _get_or_create_metric(
    "aidssist_http_requests",
    lambda: Counter(
        "aidssist_http_requests_total",
        "Total HTTP requests handled by the API.",
        ["method", "endpoint", "status_code"],
    ),
)
REQUEST_LATENCY = _get_or_create_metric(
    "aidssist_http_request_duration_seconds",
    lambda: Histogram(
        "aidssist_http_request_duration_seconds",
        "HTTP request latency in seconds.",
        ["method", "endpoint"],
        buckets=(0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 30),
    ),
)
JOB_COUNT = _get_or_create_metric(
    "aidssist_jobs",
    lambda: Counter(
        "aidssist_jobs_total",
        "Background analysis jobs processed by status.",
        ["status", "intent"],
    ),
)
JOB_DURATION = _get_or_create_metric(
    "aidssist_job_duration_seconds",
    lambda: Histogram(
        "aidssist_job_duration_seconds",
        "Background job processing duration in seconds.",
        ["intent"],
        buckets=(0.1, 0.5, 1, 2, 5, 10, 30, 60, 120, 300, 900),
    ),
)
JOB_QUEUE_WAIT = _get_or_create_metric(
    "aidssist_job_queue_wait_seconds",
    lambda: Histogram(
        "aidssist_job_queue_wait_seconds",
        "Time spent waiting in queue before job execution.",
        ["intent"],
        buckets=(0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 30, 60),
    ),
)
CACHE_OPERATIONS = _get_or_create_metric(
    "aidssist_cache_operations",
    lambda: Counter(
        "aidssist_cache_operations_total",
        "Cache reads and writes grouped by outcome.",
        ["operation", "outcome"],
    ),
)
DB_QUERY_COUNT = _get_or_create_metric(
    "aidssist_db_queries",
    lambda: Counter(
        "aidssist_db_queries_total",
        "Database queries grouped by status.",
        ["status"],
    ),
)
DB_QUERY_DURATION = _get_or_create_metric(
    "aidssist_db_query_duration_seconds",
    lambda: Histogram(
        "aidssist_db_query_duration_seconds",
        "Database query duration in seconds.",
        buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5),
    ),
)
LLM_CALL_COUNT = _get_or_create_metric(
    "aidssist_llm_calls",
    lambda: Counter(
        "aidssist_llm_calls_total",
        "Outbound provider calls grouped by provider and outcome.",
        ["provider", "outcome"],
    ),
)
LLM_CALL_DURATION = _get_or_create_metric(
    "aidssist_llm_call_duration_seconds",
    lambda: Histogram(
        "aidssist_llm_call_duration_seconds",
        "Outbound LLM provider latency in seconds.",
        ["provider"],
        buckets=(0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 30, 60),
    ),
)
QUEUE_DEPTH = _get_or_create_metric(
    "aidssist_queue_depth",
    lambda: Gauge(
        "aidssist_queue_depth",
        "Approximate queue depth for analysis jobs.",
    ),
)


@contextmanager
def time_request(method: str, endpoint: str):
    started = time.perf_counter()
    status_code = "500"
    try:
        yield lambda code: _set_status(code)
        status_code = getattr(_set_status, "status_code", status_code)
    finally:
        duration = time.perf_counter() - started
        REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(duration)
        REQUEST_COUNT.labels(method=method, endpoint=endpoint, status_code=status_code).inc()


def _set_status(code: int | str) -> None:
    _set_status.status_code = str(code)


def observe_cache(operation: str, outcome: str) -> None:
    CACHE_OPERATIONS.labels(operation=operation, outcome=outcome).inc()


def observe_db_query(duration_seconds: float, *, success: bool) -> None:
    DB_QUERY_DURATION.observe(max(duration_seconds, 0))
    DB_QUERY_COUNT.labels(status="success" if success else "failure").inc()


def observe_llm_call(provider: str, duration_seconds: float, *, success: bool) -> None:
    LLM_CALL_DURATION.labels(provider=provider).observe(max(duration_seconds, 0))
    LLM_CALL_COUNT.labels(provider=provider, outcome="success" if success else "failure").inc()


def observe_job_completion(intent: str, duration_seconds: float, *, success: bool) -> None:
    JOB_DURATION.labels(intent=intent).observe(max(duration_seconds, 0))
    JOB_COUNT.labels(status="completed" if success else "failed", intent=intent).inc()


def observe_queue_wait(intent: str, duration_seconds: float) -> None:
    JOB_QUEUE_WAIT.labels(intent=intent).observe(max(duration_seconds, 0))


def set_queue_depth(depth: int) -> None:
    QUEUE_DEPTH.set(max(int(depth), 0))


def render_metrics_payload() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST
