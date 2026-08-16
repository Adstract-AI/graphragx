"""Stage-aware W&B logging steps for training and retrieval artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from helpers.logging_config import get_logger
from helpers.constants import GNN_ANSWER_RETRIEVER_WEIGHTS_FILENAME
from pipeline.abstract import AbstractStep, StepContext
from pipeline.exceptions import PipelineException
from pipeline.evaluation.models import (
    GnnAnswerRetrieverEvaluationResult,
    SavedLlmInferenceRun,
)
from helpers.path_serialization import project_absolute_path
from pipeline.evaluation.services.wandb_experiment import WandbExperimentCoordinator
from pipeline.preparation.steps.gnn_answer_retriever_training import (
    TrainedGnnAnswerRetriever,
)

logger = get_logger(__name__)
LEGACY_GNN_CONFIG_FILENAME = "gnn_answer_retriever_config.json"


class LogTrainingToWandbStep(
    AbstractStep[TrainedGnnAnswerRetriever, TrainedGnnAnswerRetriever]
):
    """Log and annotate one completed GNN training artifact."""

    def __init__(
        self,
        coordinator: WandbExperimentCoordinator,
        force_default: bool = False,
    ) -> None:
        super().__init__(force_default=force_default)
        self.coordinator = coordinator

    def execute_default(
        self,
        context: StepContext[TrainedGnnAnswerRetriever],
    ) -> TrainedGnnAnswerRetriever:
        result = context.result
        if result is None:
            raise PipelineException("Training W&B logging requires a training result.")
        self.coordinator.ensure_run(run_name=result.model_run_name)
        for point in result.loss_history:
            epoch = int(point["epoch"])
            self.coordinator.log(
                {
                    "Training/epoch_average_loss": float(point["average_loss"]),
                    "Training/epoch": epoch,
                }
            )
        self.coordinator.persist_metadata(result.model_config_path)
        self.coordinator.log_artifact(
            name=f"model-{result.model_run_name}",
            artifact_type="gnn-model",
            paths=[result.model_config_path, result.model_artifact_path],
        )
        self.coordinator.persist_metadata(result.model_config_path)
        metadata = self.coordinator.metadata
        logger.info(
            f"Recorded training W&B stage: run={result.model_run_name} "
            f"status={metadata.status}"
        )
        return result.model_copy(
            update={
                "wandb_status": metadata.status,
                "wandb_run_id": metadata.run_id,
                "wandb_run_url": metadata.run_url,
                "wandb_error_message": metadata.error_message,
            }
        )


class LogRetrieverToWandbStep(
    AbstractStep[GnnAnswerRetrieverEvaluationResult, GnnAnswerRetrieverEvaluationResult]
):
    """Log retriever metrics and predictions to the shared experiment."""

    def __init__(
        self,
        coordinator: WandbExperimentCoordinator,
        force_default: bool = False,
    ) -> None:
        super().__init__(force_default=force_default)
        self.coordinator = coordinator

    def execute_default(
        self,
        context: StepContext[GnnAnswerRetrieverEvaluationResult],
    ) -> GnnAnswerRetrieverEvaluationResult:
        result = context.result
        if result is None:
            raise PipelineException("Retriever W&B logging requires an evaluation result.")
        had_active_run = self.coordinator.has_active_run
        model_config_path = result.model_run_directory / "model_config.json"
        if not model_config_path.exists():
            model_config_path = (
                result.model_run_directory / LEGACY_GNN_CONFIG_FILENAME
            )
        source_config_path = (
            result.evaluation_config_path
            if self._has_lineage(result.evaluation_config_path)
            else model_config_path
        )
        self.coordinator.ensure_run(
            source_config_path=source_config_path,
            run_name=result.evaluation_run_name,
        )
        if not had_active_run and result.wandb_run_id is None:
            self._backfill_training(model_config_path)
        metrics = {
            "Retriever/evaluated_instances": result.evaluated_instances,
            "Retriever/hits_at_1": result.hits_at_1,
            "Retriever/hits_at_5": result.hits_at_5,
            "Retriever/hits_at_10": result.hits_at_10,
            "Retriever/hits_at_candidate_limit": result.hits_at_candidate_limit,
            "Retriever/average_candidate_count": result.average_candidate_count,
            "Retriever/missing_gold_in_graph_count": result.missing_gold_in_graph_count,
        }
        self.coordinator.log(metrics)
        self.coordinator.persist_metadata(result.evaluation_config_path)
        artifact_paths = [result.evaluation_config_path, result.predictions_path]
        if result.retrieval_metrics_path is not None:
            artifact_paths.append(result.retrieval_metrics_path)
        self.coordinator.log_artifact(
            name=f"retriever-{result.evaluation_run_name}",
            artifact_type="retriever-results",
            paths=artifact_paths,
        )
        self.coordinator.persist_metadata(result.evaluation_config_path)
        metadata = self.coordinator.metadata
        return result.model_copy(
            update={
                "wandb_status": metadata.status,
                "wandb_run_id": metadata.run_id,
                "wandb_run_url": metadata.run_url,
                "wandb_error_message": metadata.error_message,
            }
        )

    def _backfill_training(self, model_config_path: Path) -> None:
        """Log saved training history when an upstream run has no W&B lineage."""
        try:
            payload: dict[str, Any] = json.loads(
                model_config_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            return
        for point in payload.get("loss_history", []):
            if not isinstance(point, dict):
                continue
            epoch = point.get("epoch")
            average_loss = point.get("average_loss")
            if isinstance(epoch, int) and isinstance(average_loss, int | float):
                self.coordinator.log(
                    {
                        "Training/epoch": epoch,
                        "Training/epoch_average_loss": float(average_loss),
                    }
                )
        self.coordinator.log_artifact(
            name=f"model-{model_config_path.parent.name}",
            artifact_type="gnn-model",
            paths=[
                model_config_path,
                model_config_path.parent / GNN_ANSWER_RETRIEVER_WEIGHTS_FILENAME,
            ],
        )

    @staticmethod
    def _has_lineage(config_path: Path) -> bool:
        """Return whether a JSON config contains a resumable W&B run id."""
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        tracking = payload.get("wandb") if isinstance(payload, dict) else None
        return isinstance(tracking, dict) and bool(tracking.get("run_id"))


class LogInferenceToWandbStep(
    AbstractStep[SavedLlmInferenceRun, SavedLlmInferenceRun]
):
    """Log a persisted inference run before combined metric computation."""

    def __init__(
        self,
        coordinator: WandbExperimentCoordinator,
        force_default: bool = False,
    ) -> None:
        super().__init__(force_default=force_default)
        self.coordinator = coordinator

    def execute_default(
        self,
        context: StepContext[SavedLlmInferenceRun],
    ) -> SavedLlmInferenceRun:
        result = context.result
        if result is None:
            raise PipelineException("Inference W&B logging requires an inference run.")
        try:
            config = json.loads(result.inference_config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            config = {}
        evaluation_ref = config.get("evaluation_config", {})
        evaluation_config_value = (
            evaluation_ref.get("full_config_path")
            if isinstance(evaluation_ref, dict)
            else None
        )
        source_config_path = (
            project_absolute_path(evaluation_config_value)
            if isinstance(evaluation_config_value, str)
            else None
        )
        inference = config.get("inference", {})
        prefix = f"Inference/{result.inference_run_name}"
        payload: dict[str, float | int | str] = {
            f"{prefix}/total_instances": result.total_instances,
            f"{prefix}/successful_answers": result.successful_answers,
            f"{prefix}/failed_answers": result.failed_answers,
            f"{prefix}/model_id": result.model_id,
        }
        if isinstance(inference, dict):
            for source_key in [
                "total_prompt_tokens",
                "total_completion_tokens",
                "total_tokens",
                "total_cost_usd",
            ]:
                value = inference.get(source_key)
                if isinstance(value, int | float):
                    payload[f"{prefix}/{source_key}"] = value
        self.coordinator.log(
            payload,
            source_config_path=source_config_path,
            run_name=result.inference_run_name,
        )
        self.coordinator.persist_metadata(result.inference_config_path)
        self.coordinator.log_artifact(
            name=f"inference-raw-{result.inference_run_name}",
            artifact_type="inference-predictions",
            paths=[
                result.inference_config_path,
                result.answers_path,
                result.reasoning_path,
            ],
            source_config_path=source_config_path,
        )
        self.coordinator.persist_metadata(result.inference_config_path)
        metadata = self.coordinator.metadata
        return result.model_copy(
            update={
                "wandb_status": metadata.status,
                "wandb_run_id": metadata.run_id,
                "wandb_run_url": metadata.run_url,
                "wandb_error_message": metadata.error_message,
            }
        )
