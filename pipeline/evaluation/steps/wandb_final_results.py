"""Optional WandB logging step for final results."""

from __future__ import annotations

from helpers.env_variables import WANDB_ENTITY, WANDB_MODE, WANDB_PROJECT
from helpers.logging_config import get_logger
from pipeline.abstract import AbstractStep, StepContext
from pipeline.evaluation.models import FinalResultsEvaluationResult
from pipeline.evaluation.services import (
    WandbFinalResultsConfig,
    WandbFinalResultsLoggingService,
    WandbFinalResultsLogResult,
)

logger = get_logger(__name__)


class LogFinalResultsToWandbStep(
    AbstractStep[FinalResultsEvaluationResult, FinalResultsEvaluationResult]
):
    """Best-effort WandB upload for final local result artifacts."""

    def __init__(
        self,
        project: str | None = None,
        entity: str | None = None,
        mode: str | None = None,
        logging_service: WandbFinalResultsLoggingService | None = None,
        force_default: bool = False,
    ):
        super().__init__(force_default=force_default)
        self.project = project or WANDB_PROJECT
        self.entity = entity if entity is not None else WANDB_ENTITY
        self.mode = mode or WANDB_MODE
        self.logging_service = logging_service or WandbFinalResultsLoggingService()

    def execute_default(
        self,
        context: StepContext[FinalResultsEvaluationResult],
    ) -> FinalResultsEvaluationResult:
        final_result = context.result
        if final_result is None:
            logger.warning("Skipping WandB logging because final results are missing.")
            return FinalResultsEvaluationResult(
                dataset_id="",
                results_run_directory=".",
                results_run_name="",
                results_run_number=0,
                results_config_path=".",
                retrieval_metrics_path=".",
                reasoning_metrics_path=".",
                per_instance_results_path=".",
                evaluated_instances=0,
                accuracy=0.0,
                hit_rate=0.0,
                hits_at_1=0.0,
                precision=0.0,
                recall=0.0,
                f1=0.0,
                grounded_explanation_rate=0.0,
                ndcg_at_10=0.0,
                wandb_status="skipped",
                wandb_error_message="Final results are missing.",
            )

        logger.info(
            f"Logging final results to WandB: project={self.project} "
            f"entity={self.entity} mode={self.mode} "
            f"results_run={final_result.results_run_name}"
        )
        try:
            outcome = self.logging_service.log_final_results(
                final_result=final_result,
                config=WandbFinalResultsConfig(
                    project=self.project,
                    entity=self.entity,
                    mode=self.mode,
                ),
            )
        except Exception as error:
            logger.warning(f"WandB final results logging failed: {error}")
            outcome = WandbFinalResultsLogResult(
                status="failed",
                error_message=str(error),
            )
        return final_result.model_copy(
            update={
                "wandb_status": outcome.status,
                "wandb_run_id": outcome.run_id,
                "wandb_run_url": outcome.run_url,
                "wandb_error_message": outcome.error_message,
            }
        )
