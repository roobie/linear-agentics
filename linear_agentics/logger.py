"""Logging configuration for linear-agentics."""

from __future__ import annotations

import logging

import structlog


def _configure_logging() -> None:
    """Configure structlog once."""
    logging.basicConfig(level=logging.WARNING)
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


_configure_logging()


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a logger for the given module name."""
    return structlog.get_logger(name)
