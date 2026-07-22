"""Loguru setup for the eval API server."""

from __future__ import annotations

import logging
import sys

from loguru import logger

_CONFIGURED = False


class _InterceptHandler(logging.Handler):
    """Route stdlib logging (uvicorn, dspy, …) through loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def configure_logging(*, level: str = "INFO") -> None:
    """Idempotent loguru configuration for the backend process."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    logger.remove()
    logger.add(
        sys.stderr,
        level=level.upper(),
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        enqueue=True,  # safe across worker threads
        backtrace=False,
        diagnose=False,
    )

    logging.root.handlers = [_InterceptHandler()]
    logging.root.setLevel(level.upper())
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "dspy", "httpx", "httpcore"):
        logging.getLogger(name).handlers = []
        logging.getLogger(name).propagate = True

    _CONFIGURED = True
    logger.info("Logging configured (level={})", level.upper())
