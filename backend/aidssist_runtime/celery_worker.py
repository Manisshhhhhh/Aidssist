from __future__ import annotations

from .celery_app import celery_app

# Import task registrations before the worker starts.
from . import celery_tasks as _celery_tasks  # noqa: F401


def main() -> None:
    if celery_app is None:
        raise RuntimeError("Celery is not installed in this Aidssist environment.")
    celery_app.worker_main(["worker", "--loglevel=INFO"])


if __name__ == "__main__":
    main()

