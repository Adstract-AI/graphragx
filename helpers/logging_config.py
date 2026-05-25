"""Configured logging utilities for graphragX."""

from __future__ import annotations

import logging

from helpers.env_variables import GRAPHRAGX_LOG_COLOR, GRAPHRAGX_LOG_LEVEL


class GraphragXLogFormatter(logging.Formatter):
    """Small color-aware formatter for framework logs."""

    reset = "\033[0m"
    colors = {
        "black": "\033[30m",
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
    }
    level_colors = {
        logging.DEBUG: "\033[36m",
        logging.INFO: colors.get(GRAPHRAGX_LOG_COLOR, "\033[36m"),
        logging.WARNING: "\033[33m",
        logging.ERROR: "\033[31m",
        logging.CRITICAL: "\033[35m",
    }

    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        color = self.level_colors.get(
            record.levelno,
            self.colors.get(GRAPHRAGX_LOG_COLOR, "\033[36m"),
        )
        return f"{color}{message}{self.reset}"


def setup_logger() -> None:
    """Configure the root logger once for CLI/framework runs."""
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(GRAPHRAGX_LOG_LEVEL)
    _quiet_noisy_dependency_loggers()

    handler = logging.StreamHandler()
    handler.setLevel(GRAPHRAGX_LOG_LEVEL)
    handler.setFormatter(
        GraphragXLogFormatter(
            "%(asctime)s %(levelname)s %(filename)s: %(message)s"
        )
    )
    root_logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Return a project logger configured through the shared setup."""
    return logging.getLogger(name)


def _quiet_noisy_dependency_loggers() -> None:
    """Suppress successful low-level HTTP logs from chatty client libraries."""
    for logger_name in ["httpx", "httpcore", "qdrant_client", "_client"]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
