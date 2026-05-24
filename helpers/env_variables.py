"""Centralized environment variable loading for graphragX."""

from __future__ import annotations

import os

from helpers.constants import (
    DEFAULT_LOGGING_COLOR,
    DEFAULT_LOGGING_LEVEL,
    DEFAULT_WANDB_MODE,
    DEFAULT_WANDB_PROJECT,
    LOGGING_COLOR_ENV_NAME,
    LOGGING_LEVEL_ENV_NAME,
    OPENAI_API_KEY_ENV_NAME,
    WANDB_ENTITY_ENV_NAME,
    WANDB_MODE_ENV_NAME,
    WANDB_PROJECT_ENV_NAME,
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
WANDB_PROJECT = os.getenv(WANDB_PROJECT_ENV_NAME) or DEFAULT_WANDB_PROJECT
WANDB_ENTITY = os.getenv(WANDB_ENTITY_ENV_NAME) or None
WANDB_MODE = os.getenv(WANDB_MODE_ENV_NAME) or DEFAULT_WANDB_MODE
