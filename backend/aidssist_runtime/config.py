from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_DIR.parent
APP_STATE_DIR = PROJECT_ROOT / ".aidssist"


DEFAULT_CORS_ORIGINS = ",".join(
    (
        "http://localhost",
        "http://127.0.0.1",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
    )
)


def _default_cors_origins(environment: str) -> str:
    normalized = environment.strip().lower()
    if normalized in {"production", "staging"}:
        return "*"
    return DEFAULT_CORS_ORIGINS


def _env_int(name: str, default: int) -> int:
    try:
        return int(str(os.getenv(name, default)).strip())
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(str(os.getenv(name, default)).strip())
    except (TypeError, ValueError):
        return default


def _env_csv(name: str, default: str = "") -> tuple[str, ...]:
    raw_value = str(os.getenv(name, default)).strip()
    if not raw_value:
        return ()
    return tuple(item.strip() for item in raw_value.split(",") if item.strip())


def _env_str(*names: str, default: str = "") -> str:
    for name in names:
        value = os.getenv(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return default


@dataclass(frozen=True)
class RuntimeSettings:
    environment: str
    api_host: str
    api_port: int
    api_base_url: str
    cors_origins: tuple[str, ...]
    database_url: str
    redis_url: str
    metrics_path: str
    slow_query_ms: int
    cache_ttl_seconds: int
    profile_cache_ttl_seconds: int
    job_timeout_seconds: int
    job_poll_interval_seconds: float
    worker_burst: bool
    celery_broker_url: str
    celery_result_backend: str
    provider_max_concurrency_per_process: int
    pipeline_cache_version: str
    object_store_backend: str
    object_store_dir: Path
    dataset_session_dir: Path
    s3_endpoint_url: str
    s3_access_key_id: str
    s3_secret_access_key: str
    s3_bucket_name: str
    s3_region_name: str
    embedding_model: str
    embedding_dimensions: int
    chunk_size_chars: int
    chunk_overlap_chars: int
    retrieval_top_k: int
    max_solver_retries: int
    max_upload_mb: int
    request_timeout_seconds: int
    db_pool_size: int
    db_max_overflow: int
    session_ttl_hours: int
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    kaggle_config_dir: str
    google_drive_api_base_url: str
    google_drive_download_url: str
    import_sync_threshold_mb: int
    intelligence_sample_rows: int


@lru_cache(maxsize=1)
def get_settings() -> RuntimeSettings:
    app_state_dir = APP_STATE_DIR
    app_state_dir.mkdir(parents=True, exist_ok=True)
    environment = _env_str("AIDSSIST_ENV", "RAILWAY_ENVIRONMENT_NAME", default="development") or "development"

    return RuntimeSettings(
        environment=environment,
        api_host=_env_str("AIDSSIST_API_HOST", default="0.0.0.0") or "0.0.0.0",
        api_port=_env_int("AIDSSIST_API_PORT", _env_int("PORT", 8000)),
        api_base_url=_env_str(
            "AIDSSIST_API_URL",
            "RENDER_EXTERNAL_URL",
            "RAILWAY_PUBLIC_DOMAIN",
            default="",
        ).rstrip("/"),
        cors_origins=_env_csv("AIDSSIST_CORS_ORIGINS", _default_cors_origins(environment)),
        database_url=str(os.getenv("AIDSSIST_DATABASE_URL", "")).strip(),
        redis_url=str(os.getenv("AIDSSIST_REDIS_URL", "redis://redis:6379/0")).strip(),
        metrics_path=str(os.getenv("AIDSSIST_METRICS_PATH", "/metrics")).strip() or "/metrics",
        slow_query_ms=_env_int("AIDSSIST_SLOW_QUERY_MS", 250),
        cache_ttl_seconds=_env_int("AIDSSIST_CACHE_TTL_SECONDS", 3600),
        profile_cache_ttl_seconds=_env_int("AIDSSIST_PROFILE_CACHE_TTL_SECONDS", 1800),
        job_timeout_seconds=_env_int("AIDSSIST_JOB_TIMEOUT_SECONDS", 900),
        job_poll_interval_seconds=_env_float("AIDSSIST_JOB_POLL_INTERVAL_SECONDS", 1.5),
        worker_burst=str(os.getenv("AIDSSIST_WORKER_BURST", "false")).strip().lower() == "true",
        celery_broker_url=_env_str("AIDSSIST_CELERY_BROKER_URL", default=str(os.getenv("AIDSSIST_REDIS_URL", "")).strip()),
        celery_result_backend=_env_str(
            "AIDSSIST_CELERY_RESULT_BACKEND",
            default=str(os.getenv("AIDSSIST_REDIS_URL", "")).strip(),
        ),
        provider_max_concurrency_per_process=_env_int("AIDSSIST_PROVIDER_MAX_CONCURRENCY", 4),
        pipeline_cache_version=str(os.getenv("AIDSSIST_PIPELINE_CACHE_VERSION", "2026-04-01-v1")).strip()
        or "2026-04-01-v1",
        object_store_backend=str(os.getenv("AIDSSIST_OBJECT_STORE_BACKEND", "auto")).strip().lower() or "auto",
        object_store_dir=app_state_dir / "object_store",
        dataset_session_dir=Path(
            _env_str("AIDSSIST_DATASET_SESSION_DIR", default=str(app_state_dir / "datasets"))
        ).expanduser(),
        s3_endpoint_url=str(os.getenv("AIDSSIST_S3_ENDPOINT_URL", "")).strip(),
        s3_access_key_id=str(os.getenv("AIDSSIST_S3_ACCESS_KEY_ID", "")).strip(),
        s3_secret_access_key=str(os.getenv("AIDSSIST_S3_SECRET_ACCESS_KEY", "")).strip(),
        s3_bucket_name=str(os.getenv("AIDSSIST_S3_BUCKET_NAME", "aidssist-artifacts")).strip()
        or "aidssist-artifacts",
        s3_region_name=str(os.getenv("AIDSSIST_S3_REGION_NAME", "us-east-1")).strip() or "us-east-1",
        embedding_model=str(os.getenv("AIDSSIST_EMBEDDING_MODEL", "text-embedding-004")).strip()
        or "text-embedding-004",
        embedding_dimensions=_env_int("AIDSSIST_EMBEDDING_DIMENSIONS", 128),
        chunk_size_chars=_env_int("AIDSSIST_CHUNK_SIZE_CHARS", 1400),
        chunk_overlap_chars=_env_int("AIDSSIST_CHUNK_OVERLAP_CHARS", 160),
        retrieval_top_k=_env_int("AIDSSIST_RETRIEVAL_TOP_K", 8),
        max_solver_retries=_env_int("AIDSSIST_MAX_SOLVER_RETRIES", 2),
        max_upload_mb=_env_int("AIDSSIST_MAX_UPLOAD_MB", 500),
        request_timeout_seconds=_env_int("AIDSSIST_REQUEST_TIMEOUT_SECONDS", 60),
        db_pool_size=_env_int("AIDSSIST_DB_POOL_SIZE", 20),
        db_max_overflow=_env_int("AIDSSIST_DB_MAX_OVERFLOW", 40),
        session_ttl_hours=_env_int("AIDSSIST_SESSION_TTL_HOURS", 720),
        llm_base_url=_env_str("AIDSSIST_LLM_BASE_URL", default="").rstrip("/"),
        llm_api_key=_env_str("AIDSSIST_LLM_API_KEY", default=""),
        llm_model=_env_str("AIDSSIST_LLM_MODEL", default=""),
        kaggle_config_dir=_env_str("AIDSSIST_KAGGLE_CONFIG_DIR", default=""),
        google_drive_api_base_url=_env_str(
            "AIDSSIST_GOOGLE_DRIVE_API_BASE_URL",
            default="https://www.googleapis.com/drive/v3",
        ).rstrip("/"),
        google_drive_download_url=_env_str(
            "AIDSSIST_GOOGLE_DRIVE_DOWNLOAD_URL",
            default="https://www.googleapis.com/drive/v3/files",
        ).rstrip("/"),
        import_sync_threshold_mb=_env_int("AIDSSIST_IMPORT_SYNC_THRESHOLD_MB", 25),
        intelligence_sample_rows=_env_int("AIDSSIST_INTELLIGENCE_SAMPLE_ROWS", 5000),
    )
