"""Tests for the resumable stage-aware W&B experiment lifecycle."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from pipeline.evaluation.services.wandb_experiment import (
    WandbExperimentCoordinator,
)
from pipeline.exceptions import PipelineException
from pipeline.abstract import StepContext
from pipeline.evaluation.models import SavedLlmInferenceRun
from pipeline.evaluation.steps.wandb_stages import LogInferenceToWandbStep


class FakeRun:
    def __init__(self) -> None:
        self.id = "wandb-run-id"
        self.url = "https://wandb.test/run"
        self.logged: list[tuple[dict, int | None]] = []
        self.artifacts = []
        self.finished = False

    def log(self, payload, step=None) -> None:
        self.logged.append((payload, step))

    def log_artifact(self, artifact) -> None:
        self.artifacts.append(artifact)

    def finish(self) -> None:
        self.finished = True


class FakeArtifact:
    def __init__(self, *, name: str, type: str) -> None:
        self.name = name
        self.type = type
        self.files: list[str] = []

    def add_file(self, path: str) -> None:
        self.files.append(path)


class FakeWandb:
    Artifact = FakeArtifact

    def __init__(self) -> None:
        self.run = FakeRun()
        self.init_calls: list[dict] = []

    def init(self, **kwargs):
        self.init_calls.append(kwargs)
        return self.run


def test_coordinator_uses_one_run_and_persists_lineage(tmp_path) -> None:
    fake_wandb = FakeWandb()
    config_path = tmp_path / "model_config.json"
    config_path.write_text("{}", encoding="utf-8")
    coordinator = WandbExperimentCoordinator(project="project", entity="team")

    with patch(
        "pipeline.evaluation.services.wandb_experiment.importlib.import_module",
        return_value=fake_wandb,
    ):
        coordinator.log_training_progress(
            {"epoch": 1, "instance": 3, "global_step": 3, "loss": 0.4}
        )
        coordinator.log({"Retriever/hits_at_1": 0.8})
        coordinator.persist_metadata(config_path)
        coordinator.finish()

    assert len(fake_wandb.init_calls) == 1
    assert fake_wandb.run.logged[0][0]["Training/global_step"] == 3
    assert fake_wandb.run.finished is True
    tracking = json.loads(config_path.read_text(encoding="utf-8"))["wandb"]
    assert tracking["run_id"] == "wandb-run-id"
    assert tracking["project"] == "project"


def test_coordinator_resumes_persisted_run(tmp_path) -> None:
    fake_wandb = FakeWandb()
    config_path = tmp_path / "evaluation_config.json"
    config_path.write_text(
        json.dumps(
            {
                "wandb": {
                    "status": "logged",
                    "run_id": "existing-id",
                    "run_url": "https://wandb.test/existing",
                    "project": "project",
                    "entity": "team",
                }
            }
        ),
        encoding="utf-8",
    )
    coordinator = WandbExperimentCoordinator(project="project", entity="team")

    with patch(
        "pipeline.evaluation.services.wandb_experiment.importlib.import_module",
        return_value=fake_wandb,
    ):
        coordinator.ensure_run(source_config_path=config_path)

    assert fake_wandb.init_calls[0]["id"] == "existing-id"
    assert fake_wandb.init_calls[0]["resume"] == "allow"


def test_coordinator_rejects_conflicting_persisted_project(tmp_path) -> None:
    config_path = tmp_path / "evaluation_config.json"
    config_path.write_text(
        json.dumps(
            {
                "wandb": {
                    "status": "logged",
                    "run_id": "existing-id",
                    "project": "old-project",
                    "entity": None,
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(PipelineException, match="project"):
        WandbExperimentCoordinator(project="new-project").ensure_run(
            source_config_path=config_path
        )


def test_operational_initialization_failure_is_non_fatal() -> None:
    coordinator = WandbExperimentCoordinator(project="project")

    with patch(
        "pipeline.evaluation.services.wandb_experiment.importlib.import_module",
        side_effect=RuntimeError("offline"),
    ):
        coordinator.log({"Retriever/hits_at_1": 0.5})

    assert coordinator.metadata.status == "failed"
    assert coordinator.metadata.error_message == "offline"


def test_multiple_inference_runs_are_namespaced_on_one_wandb_run(tmp_path) -> None:
    fake_wandb = FakeWandb()
    coordinator = WandbExperimentCoordinator(project="project")
    step = LogInferenceToWandbStep(coordinator=coordinator)

    with patch(
        "pipeline.evaluation.services.wandb_experiment.importlib.import_module",
        return_value=fake_wandb,
    ):
        for run_number, run_name in [(1, "first"), (2, "second")]:
            run_dir = tmp_path / run_name
            run_dir.mkdir()
            config_path = run_dir / "inference_config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "evaluation_config": {},
                        "inference": {"total_tokens": run_number * 10},
                    }
                ),
                encoding="utf-8",
            )
            answers_path = run_dir / "answers.jsonl"
            reasoning_path = run_dir / "reasoning.jsonl"
            answers_path.write_text("", encoding="utf-8")
            reasoning_path.write_text("", encoding="utf-8")
            step.execute_default(
                StepContext(
                    result=SavedLlmInferenceRun(
                        dataset_id="WebQSP",
                        evaluation_run_name="1_eval",
                        inference_run_directory=run_dir,
                        inference_run_name=run_name,
                        inference_run_number=run_number,
                        model_id="gpt-test",
                        total_instances=1,
                        successful_answers=1,
                        failed_answers=0,
                        reasoning_path=reasoning_path,
                        answers_path=answers_path,
                        inference_config_path=config_path,
                    )
                )
            )

    assert len(fake_wandb.init_calls) == 1
    logged_keys = {
        key
        for payload, _ in fake_wandb.run.logged
        for key in payload
    }
    assert "Inference/first/total_tokens" in logged_keys
    assert "Inference/second/total_tokens" in logged_keys
