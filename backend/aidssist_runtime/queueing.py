from __future__ import annotations

try:
    from redis import Redis
except ImportError:
    Redis = None

try:
    from rq import Queue
except ImportError:
    Queue = None

from .config import get_settings
from .logging_utils import get_logger
from .metrics import set_queue_depth


LOGGER = get_logger(__name__)


def get_redis_connection():
    settings = get_settings()
    if Redis is None or not settings.redis_url:
        return None

    try:
        connection = Redis.from_url(settings.redis_url)
        connection.ping()
        return connection
    except Exception:
        LOGGER.warning("redis connection unavailable; worker queue will fall back to synchronous execution", extra={"component": "queue"})
        return None


def get_analysis_queue():
    if Queue is None:
        return None
    connection = get_redis_connection()
    if connection is None:
        return None
    return Queue("analysis", connection=connection, default_timeout=get_settings().job_timeout_seconds)


def enqueue_analysis_job(job_id: str) -> bool:
    queue = get_analysis_queue()
    if queue is None:
        from .analysis_service import process_analysis_job

        process_analysis_job(job_id)
        return False

    queue.enqueue(
        "backend.aidssist_runtime.analysis_service.process_analysis_job",
        job_id,
        job_id=job_id,
        job_timeout=get_settings().job_timeout_seconds,
    )
    set_queue_depth(queue.count)
    return True


def enqueue_forecast_job(job_id: str) -> bool:
    queue = get_analysis_queue()
    if queue is None:
        from .analysis_service import process_forecast_job

        process_forecast_job(job_id)
        return False

    queue.enqueue(
        "backend.aidssist_runtime.analysis_service.process_forecast_job",
        job_id,
        job_id=job_id,
        job_timeout=get_settings().job_timeout_seconds,
    )
    set_queue_depth(queue.count)
    return True


def enqueue_solve_run(run_id: str) -> bool:
    queue = get_analysis_queue()
    if queue is None:
        from .solver_orchestrator import process_solve_run

        process_solve_run(run_id)
        return False

    queue.enqueue(
        "backend.aidssist_runtime.solver_orchestrator.process_solve_run",
        run_id,
        job_id=f"solve-{run_id}",
        job_timeout=get_settings().job_timeout_seconds,
    )
    set_queue_depth(queue.count)
    return True
