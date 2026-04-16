from __future__ import annotations

try:
    from prometheus_client import start_http_server
except ImportError:  # pragma: no cover - optional local fallback
    def start_http_server(port: int) -> None:
        del port

from .config import get_settings
from .logging_utils import configure_logging
from .queueing import get_analysis_queue, get_redis_connection


def main() -> None:
    configure_logging("worker")
    start_http_server(9109)

    connection = get_redis_connection()
    queue = get_analysis_queue()
    if connection is None or queue is None:
        raise RuntimeError("Redis-backed worker queue is unavailable. Configure AIDSSIST_REDIS_URL before starting the worker.")

    from rq import Connection, Worker

    settings = get_settings()
    with Connection(connection):
        worker = Worker([queue.name])
        worker.work(with_scheduler=True, burst=settings.worker_burst)


if __name__ == "__main__":
    main()
