"""Helpers for portable path serialization in saved JSON artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_relative_path(value: str | Path) -> str:
    """Return a project-root-relative string when the path lives in this repo."""
    path = Path(value)
    if not path.is_absolute():
        return path.as_posix()

    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def project_absolute_path(value: str | Path) -> Path:
    """Resolve a stored path against the project root when it is relative."""
    path = Path(value)
    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def make_project_paths_relative(value: Any) -> Any:
    """Recursively convert Path values and project-local absolute path strings."""
    if isinstance(value, Path):
        return project_relative_path(value)

    if isinstance(value, str):
        path = Path(value)
        if path.is_absolute():
            return project_relative_path(path)
        return value

    if isinstance(value, list):
        return [make_project_paths_relative(item) for item in value]

    if isinstance(value, tuple):
        return [make_project_paths_relative(item) for item in value]

    if isinstance(value, dict):
        return {
            key: make_project_paths_relative(item)
            for key, item in value.items()
        }

    return value
