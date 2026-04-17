from __future__ import annotations

try:
    from celery import Celery
except ImportError:  # pragma: no cover - optional dependency in some local runs
    Celery = None

from .config import get_settings


def get_celery_app():
    if Celery is None:
        return None

    settings = get_settings()
    broker_url = settings.celery_broker_url or settings.redis_url or "memory://"
    result_backend = settings.celery_result_backend or settings.redis_url or "cache+memory://"
    app = Celery("aidssist", broker=broker_url, backend=result_backend)
    app.conf.update(
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        timezone="UTC",
        enable_utc=True,
        task_track_started=True,
    )
    return app


celery_app = get_celery_app()

