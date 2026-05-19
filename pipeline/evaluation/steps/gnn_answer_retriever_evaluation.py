"""Evaluation step for saved GNN answer-retriever runs."""

from __future__ import annotations

from pydantic import Field

from constants import DEFAULT_ANSWER_THRESHOLD, DEFAULT_CANDIDATE_TOP_K
from logging_config import get_logger
from pipeline.abstract import AbstractStep, StepContext, StepResult
from pipeline.evaluation.models import (
    GnnAnswerRetrieverEvaluationConfig,
    GnnAnswerRetrieverEvaluationResult,
)
from pipeline.exceptions import InvalidInteractiveConfigurationInputException
from pipeline.preparation.models.webqsp_local_graph import PreparedWebQSPGraphDataset
from pipeline.preparation.steps.configuration_building import BuiltPipelineConfiguration
from pipeline.preparation.steps.gnn_answer_retriever_training import (
    TrainedGnnAnswerRetriever,
)
from pipeline.services.gnn_answer_retriever_evaluation import (
    GnnAnswerRetrieverEvaluationService,
)

logger = get_logger(__name__)


class EvaluateGnnAnswerRetrieverContext(StepContext[StepResult]):
    """Specialized context for evaluating a saved GNN answer-retriever run."""

    prepared_dataset: PreparedWebQSPGraphDataset = Field(
        ...,
        description="Prepared WebQSP graph dataset used for evaluation.",
    )
    pipeline_configuration: BuiltPipelineConfiguration = Field(
        ...,
        description="Pipeline configuration used for embedding cache choices.",
    )


class EvaluateGnnAnswerRetrieverStep(AbstractStep[GnnAnswerRetrieverEvaluationResult, StepResult]):
    """Evaluate a saved trained GNN answer-retriever run over WebQSP test graphs."""

    def __init__(
        self,
        model_run_name: str | None = None,
        model_run_number: int | None = None,
        answer_threshold: float = DEFAULT_ANSWER_THRESHOLD,
        candidate_top_k: int = DEFAULT_CANDIDATE_TOP_K,
        evaluation_run_name: str | None = None,
        evaluation_max_instances: int | None = None,
        evaluation_service: GnnAnswerRetrieverEvaluationService | None = None,
        force_default: bool = False,
    ):
        super().__init__(force_default=force_default)
        self.evaluation_config = GnnAnswerRetrieverEvaluationConfig(
            model_run_name=model_run_name,
            model_run_number=model_run_number,
            answer_threshold=answer_threshold,
            candidate_top_k=candidate_top_k,
            run_name=evaluation_run_name,
            max_instances=evaluation_max_instances,
        )
        self.evaluation_service = (
            evaluation_service or GnnAnswerRetrieverEvaluationService()
        )

    def execute_default(
        self,
        context: EvaluateGnnAnswerRetrieverContext,
    ) -> GnnAnswerRetrieverEvaluationResult:
        evaluation_config = self.evaluation_config
        if isinstance(context.result, TrainedGnnAnswerRetriever):
            evaluation_config = evaluation_config.model_copy(
                update={
                    "model_run_name": evaluation_config.model_run_name
                    or context.result.model_run_name,
                    "model_run_number": evaluation_config.model_run_number
                    or context.result.model_run_number,
                }
            )

        if (
            evaluation_config.model_run_name is None
            and evaluation_config.model_run_number is None
        ):
            raise InvalidInteractiveConfigurationInputException(
                "GNN answer-retriever evaluation requires a saved model run name "
                "or number."
            )

        logger.info(
            f"Starting EvaluateGnnAnswerRetrieverStep: "
            f"requested_model_run_name={evaluation_config.model_run_name} "
            f"requested_model_run_number={evaluation_config.model_run_number} "
            f"threshold={evaluation_config.answer_threshold} "
            f"candidate_top_k={evaluation_config.candidate_top_k}"
        )
        outcome = self.evaluation_service.evaluate(
            prepared_dataset=context.prepared_dataset,
            pipeline_configuration=context.pipeline_configuration,
            evaluation_config=evaluation_config,
        )
        logger.info(
            f"Finished EvaluateGnnAnswerRetrieverStep: "
            f"evaluation_run={outcome.storage_result.evaluation_run_name} "
            f"evaluated_instances={outcome.evaluated_instances}"
        )
        return GnnAnswerRetrieverEvaluationResult(
            dataset_id=context.prepared_dataset.dataset_id,
            model_run_directory=outcome.loaded_model_run.run_directory,
            model_run_name=outcome.loaded_model_run.run_name,
            model_run_number=outcome.loaded_model_run.run_number,
            evaluation_run_directory=outcome.storage_result.evaluation_run_directory,
            evaluation_run_name=outcome.storage_result.evaluation_run_name,
            evaluation_run_number=outcome.storage_result.evaluation_run_number,
            evaluated_instances=outcome.evaluated_instances,
            hits_at_1=outcome.hits_at_1,
            hits_at_1_count=outcome.hits_at_1_count,
            hit_at_k=outcome.hit_at_k,
            hit_at_k_count=outcome.hit_at_k_count,
            average_candidate_count=outcome.average_candidate_count,
            missing_gold_in_graph_count=outcome.missing_gold_in_graph_count,
            predictions_path=outcome.storage_result.predictions_path,
            summary_metrics_path=outcome.storage_result.summary_metrics_path,
            evaluation_config_path=outcome.storage_result.evaluation_config_path,
        )
