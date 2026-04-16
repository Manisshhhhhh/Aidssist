from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        for attribute in (
            "request_id",
            "job_id",
            "dataset_id",
            "workflow_id",
            "status_code",
            "duration_ms",
            "endpoint",
            "method",
            "component",
        ):
            value = getattr(record, attribute, None)
            if value is not None:
                payload[attribute] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


def configure_logging(component: str) -> None:
    root_logger = logging.getLogger()
    if getattr(root_logger, "_aidssist_json_logging", False):
        return

    log_level = str(os.getenv("AIDSSIST_LOG_LEVEL", "INFO")).upper()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(log_level)
    root_logger._aidssist_json_logging = True
    logging.getLogger(__name__).info("configured structured logging", extra={"component": component})


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
