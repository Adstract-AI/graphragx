"""Tests for optional WandB final result logging."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from pipeline.abstract import StepContext
from pipeline.evaluation.models import FinalResultsEvaluationResult
from pipeline.evaluation.services import (
    WandbFinalResultsConfig,
    WandbFinalResultsLoggingService,
    WandbFinalResultsLogResult,
)
from pipeline.evaluation.steps.wandb_final_results import LogFinalResultsToWandbStep


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{json.dumps(row)}\n" for row in rows),
        encoding="utf-8",
    )


def _fake_final_result(tmp_path: Path) -> FinalResultsEvaluationResult:
    results_dir = tmp_path / "data" / "webqsp" / "results" / "1_test"
    sources_dir = tmp_path / "sources"
    answers_path = sources_dir / "answers.jsonl"
    reasoning_path = sources_dir / "reasoning.jsonl"
    predictions_path = sources_dir / "predictions.jsonl"
    inference_config_path = sources_dir / "inference_config.json"
    evaluation_config_path = sources_dir / "evaluation_config.json"
    for path in [
        reasoning_path,
        predictions_path,
        inference_config_path,
        evaluation_config_path,
    ]:
        _write_json(path, {"source": path.name})
    _write_jsonl(
        answers_path,
        [
            {
                "instance_index": 0,
                "explanation": "Moon -> orbits -> Earth",
            }
        ],
    )
    results_config_path = results_dir / "results_config.json"
    retrieval_metrics_path = results_dir / "retrieval_metrics.json"
    reasoning_metrics_path = results_dir / "reasoning_metrics.json"
    per_instance_results_path = results_dir / "per_instance_results.jsonl"
    _write_json(
        results_config_path,
        {
            "dataset_id": "webqsp",
            "model_id": "test-model",
            "evaluation_run_name": "1_eval",
            "inference_run_name": "1_inference",
            "answers_path": str(answers_path),
            "reasoning_path": str(reasoning_path),
            "predictions_path": str(predictions_path),
            "inference_config_path": str(inference_config_path),
            "evaluation_config_path": str(evaluation_config_path),
        },
    )
    _write_json(
        retrieval_metrics_path,
        {
            "hits_at_1": 0.5,
            "hit_at_k": 1.0,
            "average_candidate_count": 2.0,
            "missing_gold_in_graph_count": 0,
        },
    )
    _write_json(
        reasoning_metrics_path,
        {
            "accuracy": 0.5,
            "hit_rate": 0.5,
            "hits_at_1": 0.5,
            "precision": 1.0,
            "recall": 0.5,
            "f1": 2 / 3,
            "grounded_explanation_rate": 0.5,
            "fully_grounded_explanation_rate": 0.5,
            "ndcg_at_1": 0.5,
            "ndcg_at_5": 0.75,
            "ndcg_at_10": 0.75,
            "ndcg_at_candidate_limit": 0.75,
        },
    )
    _write_jsonl(
        per_instance_results_path,
        [
            {
                "instance_index": 0,
                "question": "What does the Moon orbit?",
                "q_entity": ["Moon"],
                "gold_answers": ["Earth"],
                "predicted_answers": ["Earth"],
                "exact_match": True,
                "hit": True,
                "hits_at_1": True,
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
                "mentioned_triple_count": 1,
                "grounded_mentioned_triple_count": 1,
                "grounded_explanation": True,
                "fully_grounded_explanation": True,
                "ndcg_at_1": 1.0,
                "ndcg_at_5": 1.0,
                "ndcg_at_10": 1.0,
                "ndcg_at_candidate_limit": 1.0,
                "answer_error_message": None,
            }
        ],
    )
    return FinalResultsEvaluationResult(
        dataset_id="webqsp",
        results_run_directory=results_dir,
        results_run_name="1_test",
        results_run_number=1,
        results_config_path=results_config_path,
        retrieval_metrics_path=retrieval_metrics_path,
        reasoning_metrics_path=reasoning_metrics_path,
        per_instance_results_path=per_instance_results_path,
        evaluated_instances=1,
        accuracy=0.5,
        hit_rate=0.5,
        hits_at_1=0.5,
        precision=1.0,
        recall=0.5,
        f1=2 / 3,
        grounded_explanation_rate=0.5,
        ndcg_at_10=0.75,
    )


def test_wandb_payload_construction(tmp_path: Path) -> None:
    final_result = _fake_final_result(tmp_path)
    service = WandbFinalResultsLoggingService()
    retrieval_metrics = service._load_json_object(final_result.retrieval_metrics_path)
    reasoning_metrics = service._load_json_object(final_result.reasoning_metrics_path)
    results_config = service._load_json_object(final_result.results_config_path)
    per_instance_rows = service._load_jsonl_objects(
        final_result.per_instance_results_path
    )

    scalars = service.build_scalar_metrics(retrieval_metrics, reasoning_metrics)
    table_rows = service.build_table_rows(results_config, per_instance_rows)

    assert scalars["retrieval_hits_at_1"] == 0.5
    assert scalars["answer_f1"] == 2 / 3
    assert scalars["ranking_ndcg_at_10"] == 0.75
    assert service.table_columns[0] == "instance_index"
    assert len(table_rows) == 1
    assert table_rows[0][5] == "Moon -> orbits -> Earth"


def test_wandb_logging_success_with_fake_module(
    tmp_path: Path,
    monkeypatch,
) -> None:
    final_result = _fake_final_result(tmp_path)
    captured: dict[str, object] = {}

    class FakeRun:
        id = "run-1"
        url = "https://wandb.test/run-1"

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def log(self, payload):
            captured.setdefault("logs", []).append(payload)

        def log_artifact(self, artifact):
            captured["artifact_files"] = artifact.files

    class FakeArtifact:
        def __init__(self, name, type):
            self.name = name
            self.type = type
            self.files = []

        def add_file(self, path, name=None):
            self.files.append((path, name))

    class FakeTable:
        def __init__(self, columns, data):
            self.columns = columns
            self.data = data

    def fake_init(**kwargs):
        captured["init"] = kwargs
        return FakeRun()

    fake_wandb = SimpleNamespace(
        init=fake_init,
        Table=FakeTable,
        Artifact=FakeArtifact,
    )
    monkeypatch.setitem(__import__("sys").modules, "wandb", fake_wandb)

    result = WandbFinalResultsLoggingService().log_final_results(
        final_result=final_result,
        config=WandbFinalResultsConfig(
            project="project",
            entity="entity",
            mode="disabled",
        ),
    )

    assert result.status == "logged"
    assert result.run_id == "run-1"
    assert captured["init"]["project"] == "project"
    assert captured["init"]["entity"] == "entity"
    assert captured["init"]["mode"] == "disabled"
    assert any("answer_f1" in payload for payload in captured["logs"])
    assert any(service_name == "results/results_config.json" for _, service_name in captured["artifact_files"])
    assert any(service_name == "sources/answers.jsonl" for _, service_name in captured["artifact_files"])


def test_wandb_step_handles_service_failure(tmp_path: Path) -> None:
    final_result = _fake_final_result(tmp_path)

    class FailingService:
        def log_final_results(self, final_result, config):
            raise RuntimeError("boom")

    result = LogFinalResultsToWandbStep(
        project="project",
        mode="disabled",
        logging_service=FailingService(),
    ).execute(StepContext(result=final_result))

    assert result.wandb_status == "failed"
    assert result.wandb_error_message == "boom"


def test_wandb_step_records_success(tmp_path: Path) -> None:
    final_result = _fake_final_result(tmp_path)

    class SuccessfulService:
        def log_final_results(self, final_result, config):
            return WandbFinalResultsLogResult(
                status="logged",
                run_id="run-1",
                run_url="https://wandb.test/run-1",
            )

    result = LogFinalResultsToWandbStep(
        project="project",
        mode="disabled",
        logging_service=SuccessfulService(),
    ).execute(StepContext(result=final_result))

    assert result.wandb_status == "logged"
    assert result.wandb_run_id == "run-1"
    assert result.wandb_run_url == "https://wandb.test/run-1"
