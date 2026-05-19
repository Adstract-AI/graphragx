"""Centralized environment variable loading for graphragX."""

from __future__ import annotations

import os

from helpers.constants import (
    DEFAULT_LOGGING_COLOR,
    DEFAULT_LOGGING_LEVEL,
    LOGGING_COLOR_ENV_NAME,
    LOGGING_LEVEL_ENV_NAME,
    OPENAI_API_KEY_ENV_NAME,
)

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()

OPENAI_API_KEY = os.getenv(OPENAI_API_KEY_ENV_NAME)
GRAPHRAGX_LOG_LEVEL = os.getenv(
    LOGGING_LEVEL_ENV_NAME,
    DEFAULT_LOGGING_LEVEL,
).upper()
GRAPHRAGX_LOG_COLOR = os.getenv(
    LOGGING_COLOR_ENV_NAME,
    DEFAULT_LOGGING_COLOR,
).lower()
