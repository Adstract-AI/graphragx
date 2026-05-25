"""Tests for project-portable path serialization."""

from pathlib import Path

from helpers.path_serialization import (
    PROJECT_ROOT,
    make_project_paths_relative,
    project_absolute_path,
)


def test_project_paths_are_serialized_relative_to_repo_root() -> None:
    absolute_path = PROJECT_ROOT / "data" / "webqsp" / "evaluations" / "1_run"

    assert make_project_paths_relative(
        {
            "evaluation_config_path": absolute_path / "evaluation_config.json",
            "nested": {
                "predictions_path": str(absolute_path / "predictions.jsonl"),
            },
        }
    ) == {
        "evaluation_config_path": "data/webqsp/evaluations/1_run/evaluation_config.json",
        "nested": {
            "predictions_path": "data/webqsp/evaluations/1_run/predictions.jsonl",
        },
    }


def test_non_project_absolute_paths_are_left_absolute() -> None:
    path = Path("/tmp/graphragx-test/file.json")

    assert make_project_paths_relative(path) == "/tmp/graphragx-test/file.json"


def test_relative_paths_resolve_against_project_root() -> None:
    assert project_absolute_path("data/webqsp/results/1_run/results_config.json") == (
        PROJECT_ROOT / "data" / "webqsp" / "results" / "1_run" / "results_config.json"
    )
