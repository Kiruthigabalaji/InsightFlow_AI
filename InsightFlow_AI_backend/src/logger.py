"""
logger.py
---------
Centralised logging configuration for InsightFlow AI.

All agents import `get_logger(__name__)` to obtain a consistently
formatted logger.  A single call to `configure_logging()` at startup
sets up both a human-readable console handler and a JSON file handler
so every run produces a machine-parseable audit trail at
`logs/pipeline_<run_id>.jsonl`.

Usage
-----
    from logger import get_logger
    log = get_logger(__name__)
    log.info("Agent started", extra={"agent": "IngestionAgent", "source": "JLL"})
"""

from __future__ import annotations

import json
import logging
import logging.config
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

_CONFIGURED = False


# ---------------------------------------------------------------------------
# JSON formatter — every log line is a valid JSON object
# ---------------------------------------------------------------------------


class JsonFormatter(logging.Formatter):
    """
    Renders each LogRecord as a single-line JSON object.

    Standard fields (level, logger, message, timestamp) are always present.
    Any keys passed via the `extra` kwarg are merged in at the top level,
    making it easy to filter by agent, source, run_id, etc.
    """

    BUILTIN_KEYS = frozenset(
        {
            "args",
            "created",
            "exc_info",
            "exc_text",
            "filename",
            "funcName",
            "levelname",
            "levelno",
            "lineno",
            "message",
            "module",
            "msecs",
            "msg",
            "name",
            "pathname",
            "process",
            "processName",
            "relativeCreated",
            "stack_info",
            "taskName",
            "thread",
            "threadName",
        }
    )

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        record.message = record.getMessage()
        base: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.message,
        }
        if record.exc_info:
            base["exc_info"] = self.formatException(record.exc_info)
        # Merge any extra keys the caller supplied
        for key, val in record.__dict__.items():
            if key not in self.BUILTIN_KEYS:
                base[key] = val
        return json.dumps(base, default=str)


# ---------------------------------------------------------------------------
# Pretty console formatter
# ---------------------------------------------------------------------------

CONSOLE_FMT = "%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s"
CONSOLE_DATE_FMT = "%H:%M:%S"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def configure_logging(run_id: str = "default", level: str | None = None) -> None:
    """
    Call once at application startup (e.g. inside `main.py` lifespan).

    Parameters
    ----------
    run_id:
        Unique identifier for the current pipeline run.  Used as part of the
        JSONL log filename so each run's logs are isolated.
    level:
        Override the log level (DEBUG / INFO / WARNING / ERROR).
        Falls back to the ``LOG_LEVEL`` environment variable, then INFO.
    """
    global _CONFIGURED  # noqa: PLW0603

    effective_level = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    log_file = LOG_DIR / f"pipeline_{run_id}.jsonl"

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "console": {
                    "()": "logging.Formatter",
                    "fmt": CONSOLE_FMT,
                    "datefmt": CONSOLE_DATE_FMT,
                },
                "json": {
                    "()": f"{__name__}.JsonFormatter",
                },
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",
                    "formatter": "console",
                    "level": effective_level,
                },
                "json_file": {
                    "class": "logging.FileHandler",
                    "filename": str(log_file),
                    "formatter": "json",
                    "level": "DEBUG",
                    "encoding": "utf-8",
                },
            },
            "root": {
                "level": "DEBUG",
                "handlers": ["console", "json_file"],
            },
        }
    )

    _CONFIGURED = True
    logging.getLogger(__name__).info(
        "Logging configured",
        extra={"run_id": run_id, "log_file": str(log_file), "level": effective_level},
    )


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger, configuring with defaults if not yet done.

    Agents should call this at module level::

        log = get_logger(__name__)
    """
    if not _CONFIGURED:
        configure_logging()
    return logging.getLogger(name)
