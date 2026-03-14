"""Logging helpers for the FastAPI application."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone


class JsonFormatter(logging.Formatter):
    """Render structured JSON logs for production use."""

    _RESERVED_FIELDS = {
        "args",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "thread",
        "threadName",
    }

    def format(self, record: logging.LogRecord) -> str:
        """Serialize the log record into a JSON string."""

        payload: dict[str, object] = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        for key, value in record.__dict__.items():
            if key not in self._RESERVED_FIELDS:
                payload[key] = value

        return json.dumps(payload, default=str)


def setup_logging(level: str = "INFO", format_type: str = "pretty") -> None:
    """Configure the root logger once for the application process."""

    handler = logging.StreamHandler(sys.stdout)
    if format_type == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )

    logging.basicConfig(level=level.upper(), handlers=[handler], force=True)

    for logger_name in ("httpx", "urllib3", "pypdf"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)
