"""Tests for the lightweight TOML experiment runner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import main

from scripts.run_experiments import (
    ExperimentManifestError,
    load_manifest,
    resolve_execution_order,
    run_experiments,
)


def test_experiment_zero_contains_complete_retriever_matrix() -> None:
    project_root = Path(__file__).resolve().parents[1]
    manifest = load_manifest(
        project_root / "experiments" / "experiment_0_gnn_architectures.toml"
    )
    parser = main.build_parser()
    expected_options = {
        "graphsage": {
            "gnn_layer_count": 2,
            "gnn_hidden_dimension": 256,
            "node_classifier": "mlp",
            "dropout": 0.1,
        },
        "aa-graphsage": {
            "gnn_layer_count": 2,
            "gnn_hidden_dimension": 256,
            "node_classifier": "mlp",
            "dropout": 0.1,
            "use_edge_mlp": True,
            "use_reverse_edges": True,
            "question_aware_classifier": True,
            "add_layer_normalization": True,
            "edge_mlp_hidden_dim": 256,
        },
        "rgcn": {
            "gnn_layer_count": 2,
            "gnn_hidden_dimension": 256,
            "dropout": 0.1,
            "num_bases": 30,
        },
        "hgt": {
            "gnn_layer_count": 2,
            "gnn_hidden_dimension": 256,
            "dropout": 0.1,
            "attention_heads": 8,
        },
        "rearev": {
            "gnn_hidden_dimension": 50,
            "dropout": 0.1,
            "num_instructions": 2,
            "reasoning_steps": 2,
            "adaptive_iterations": 3,
        },
        "nbfnet": {
            "gnn_layer_count": 3,
            "gnn_hidden_dimension": 32,
        },
    }

    observed: set[tuple[str, int]] = set()
    for run in manifest.runs:
        parsed = parser.parse_args([*manifest.default_args, *run.args])
        assert parsed.run_mode == "retriever-only"
        assert parsed.training_epochs == 10
        assert parsed.training_batch_size == 1
        assert parsed.answer_threshold == 0.7
        assert parsed.candidate_top_k == 10
        assert parsed.candidate_limit == 15
        for option_name, expected_value in expected_options[
            parsed.gnn_architecture
        ].items():
            assert getattr(parsed, option_name) == expected_value
        observed.add((parsed.gnn_architecture, parsed.random_seed))

    assert observed == {
        (architecture, seed)
        for architecture in (
            "graphsage",
            "aa-graphsage",
            "rgcn",
            "hgt",
            "rearev",
            "nbfnet",
        )
        for seed in (42, 1337, 2026)
    }


def _write_manifest(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


def _write_fake_main(project_root: Path) -> None:
    (project_root / "main.py").write_text(
        """\
import json
import pathlib
import sys

path = pathlib.Path("invocations.jsonl")
with path.open("a", encoding="utf-8") as output:
    output.write(json.dumps(sys.argv[1:]) + "\\n")
print("fake pipeline", *sys.argv[1:])
raise SystemExit(7 if "--fail" in sys.argv else 0)
""",
        encoding="utf-8",
    )


def test_manifest_order_includes_dependencies_and_defaults(tmp_path: Path) -> None:
    manifest = load_manifest(
        _write_manifest(
            tmp_path / "experiments.toml",
            """
version = 1
[defaults]
args = ["--default", "--seed", "9"]
[[runs]]
id = "train"
args = ["--retriever-only"]
[[runs]]
id = "infer"
after = ["train"]
args = ["--inference-only"]
""",
        )
    )

    assert manifest.default_args == ("--default", "--seed", "9")
    assert [run.run_id for run in resolve_execution_order(manifest, ["infer"])] == [
        "train",
        "infer",
    ]


def test_manifest_rejects_cycles_and_unknown_dependencies(tmp_path: Path) -> None:
    with pytest.raises(ExperimentManifestError, match="unknown dependencies"):
        load_manifest(
            _write_manifest(
                tmp_path / "unknown.toml",
                'version = 1\n[[runs]]\nid = "a"\nafter = ["missing"]\nargs = []\n',
            )
        )

    with pytest.raises(ExperimentManifestError, match="Dependency cycle"):
        load_manifest(
            _write_manifest(
                tmp_path / "cycle.toml",
                """
version = 1
[[runs]]
id = "a"
after = ["b"]
args = []
[[runs]]
id = "b"
after = ["a"]
args = []
""",
            )
        )


def test_runner_executes_in_order_logs_and_skips_successful_resume(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    _write_fake_main(project_root)
    manifest = load_manifest(
        _write_manifest(
            tmp_path / "run.toml",
            """
version = 1
[defaults]
args = ["--seed", "42"]
[[runs]]
id = "first"
args = ["--one"]
[[runs]]
id = "second"
after = ["first"]
args = ["--two"]
""",
        )
    )
    state_root = tmp_path / "state"

    assert run_experiments(
        manifest,
        project_root=project_root,
        state_root=state_root,
    ) == 0
    assert run_experiments(
        manifest,
        project_root=project_root,
        state_root=state_root,
    ) == 0

    invocations = [
        json.loads(line)
        for line in (project_root / "invocations.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert invocations == [
        ["--seed", "42", "--one"],
        ["--seed", "42", "--two"],
    ]
    state = json.loads((state_root / "state.json").read_text(encoding="utf-8"))
    assert state["runs"]["first"]["status"] == "succeeded"
    assert state["runs"]["second"]["status"] == "succeeded"
    assert "fake pipeline --seed 42 --one" in (
        state_root / "logs" / "first.log"
    ).read_text(encoding="utf-8")


def test_continue_on_error_runs_independent_work_and_blocks_dependents(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    _write_fake_main(project_root)
    manifest = load_manifest(
        _write_manifest(
            tmp_path / "failures.toml",
            """
version = 1
[[runs]]
id = "failure"
args = ["--fail"]
[[runs]]
id = "dependent"
after = ["failure"]
args = ["--dependent"]
[[runs]]
id = "independent"
args = ["--independent"]
""",
        )
    )
    state_root = tmp_path / "state"

    assert run_experiments(
        manifest,
        project_root=project_root,
        continue_on_error=True,
        state_root=state_root,
    ) == 1

    invocations = [
        json.loads(line)
        for line in (project_root / "invocations.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert invocations == [["--fail"], ["--independent"]]
    state = json.loads((state_root / "state.json").read_text(encoding="utf-8"))
    assert state["runs"]["failure"]["status"] == "failed"
    assert state["runs"]["dependent"]["status"] == "blocked"
    assert state["runs"]["independent"]["status"] == "succeeded"


def test_dry_run_does_not_create_state_or_launch_commands(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    _write_fake_main(project_root)
    manifest = load_manifest(
        _write_manifest(
            tmp_path / "dry.toml",
            'version = 1\n[[runs]]\nid = "only"\nargs = ["--one"]\n',
        )
    )
    state_root = tmp_path / "state"

    assert run_experiments(
        manifest,
        project_root=project_root,
        dry_run=True,
        state_root=state_root,
    ) == 0
    assert not state_root.exists()
    assert not (project_root / "invocations.jsonl").exists()
