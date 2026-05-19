"""Training step for the GNN answer retriever."""

from __future__ import annotations

from pathlib import Path

from pydantic import ConfigDict, Field

from constants import (
    DEFAULT_TRAINING_DEVICE,
    DEFAULT_TRAINING_EPOCHS,
    DEFAULT_TRAINING_LEARNING_RATE,
    DEFAULT_TRAINING_LOG_EVERY,
    DEFAULT_TRAINING_WEIGHT_DECAY,
)
from logging_config import get_logger
from pipeline.abstract import AbstractStep, StepContext, StepResult
from pipeline.exceptions import InvalidInteractiveConfigurationInputException
from pipeline.preparation.models.interfaces import AnswerRetrieverModel
from pipeline.preparation.models.webqsp_local_graph import PreparedWebQSPGraphDataset
from pipeline.preparation.steps.configuration_building import BuiltPipelineConfiguration
from pipeline.preparation.steps.gnn_model_building import BuiltGnnAnswerRetriever
from pipeline.services.gnn_answer_retriever_training import (
    GnnAnswerRetrieverTrainingConfig,
    GnnAnswerRetrieverTrainingService,
)

logger = get_logger(__name__)


class TrainedGnnAnswerRetriever(StepResult):
    """Fully trained GNN answer-retriever artifact."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    dataset_id: str = Field(..., description="Selected dataset identifier.")
    hidden_dimension: int = Field(
        ...,
        description="Shared hidden dimension used by the trained retriever.",
    )
    gnn_layer_count: int = Field(..., description="Number of trained GNN layers.")
    node_classifier: str = Field(..., description="Trained node classifier id.")
    training_epochs: int = Field(..., description="Number of training epochs used.")
    training_learning_rate: float = Field(..., description="Training learning rate.")
    training_weight_decay: float = Field(..., description="Training weight decay.")
    training_max_instances: int | None = Field(
        default=None,
        description="Optional maximum number of training instances used.",
    )
    training_log_every: int = Field(..., description="Training progress log interval.")
    training_device: str = Field(..., description="Requested training device.")
    training_run_name: str | None = Field(
        default=None,
        description="Optional user-provided training run label.",
    )
    selected_device: str = Field(..., description="Resolved PyTorch training device.")
    final_loss: float = Field(..., description="Final average epoch loss.")
    trained_instances: int = Field(..., description="Number of instances trained on.")
    model: AnswerRetrieverModel = Field(..., description="Trained retriever model.")
    model_artifact_path: Path = Field(..., description="Saved model weights path.")
    model_config_path: Path = Field(..., description="Saved model config path.")
    model_run_directory: Path = Field(..., description="Versioned training run directory.")
    model_run_name: str = Field(..., description="Resolved training run folder name.")
    model_run_number: int = Field(..., description="Incremental training run number.")
    embedding_cache_directory: Path = Field(
        ...,
        description="Directory containing cached OpenAI embeddings.",
    )


class TrainGnnAnswerRetrieverContext(StepContext[BuiltGnnAnswerRetriever]):
    """Specialized context for training the GNN answer retriever."""

    prepared_dataset: PreparedWebQSPGraphDataset = Field(
        ...,
        description="Prepared WebQSP graph dataset used for training.",
    )
    pipeline_configuration: BuiltPipelineConfiguration = Field(
        ...,
        description="Pipeline configuration used for training-time embeddings.",
    )


class TrainGnnAnswerRetrieverStep(
    AbstractStep[TrainedGnnAnswerRetriever, BuiltGnnAnswerRetriever]
):
    """Train the built GNN answer retriever on prepared WebQSP graphs."""

    def __init__(
        self,
        training_epochs: int = DEFAULT_TRAINING_EPOCHS,
        training_learning_rate: float = DEFAULT_TRAINING_LEARNING_RATE,
        training_weight_decay: float = DEFAULT_TRAINING_WEIGHT_DECAY,
        training_max_instances: int | None = None,
        training_log_every: int = DEFAULT_TRAINING_LOG_EVERY,
        training_device: str = DEFAULT_TRAINING_DEVICE,
        training_run_name: str | None = None,
        training_service: GnnAnswerRetrieverTrainingService | None = None,
        force_default: bool = False,
    ):
        super().__init__(force_default=force_default)
        self.training_config = GnnAnswerRetrieverTrainingConfig(
            epochs=training_epochs,
            learning_rate=training_learning_rate,
            weight_decay=training_weight_decay,
            max_instances=training_max_instances,
            log_every=training_log_every,
            device=training_device,
            run_name=training_run_name,
        )
        self.training_service = training_service or GnnAnswerRetrieverTrainingService()

    def execute_default(
        self,
        context: TrainGnnAnswerRetrieverContext,
    ) -> TrainedGnnAnswerRetriever:
        built_retriever = context.result
        if built_retriever is None:
            raise InvalidInteractiveConfigurationInputException(
                "GNN answer-retriever training requires a built retriever."
            )

        logger.info(
            f"Starting TrainGnnAnswerRetrieverStep: dataset={built_retriever.dataset_id} "
            f"epochs={self.training_config.epochs} "
            f"max_instances={self.training_config.max_instances} "
            f"run_name={self.training_config.run_name}"
        )
        outcome = self.training_service.train(
            built_retriever=built_retriever,
            prepared_dataset=context.prepared_dataset,
            configuration=context.pipeline_configuration,
            training_config=self.training_config,
        )
        logger.info(
            f"Finished TrainGnnAnswerRetrieverStep: "
            f"run={outcome.model_run_name} final_loss={outcome.final_loss:.6f} "
            f"trained_instances={outcome.trained_instances}"
        )
        return TrainedGnnAnswerRetriever(
            dataset_id=built_retriever.dataset_id,
            hidden_dimension=built_retriever.hidden_dimension,
            gnn_layer_count=built_retriever.gnn_layer_count,
            node_classifier=built_retriever.node_classifier,
            training_epochs=self.training_config.epochs,
            training_learning_rate=self.training_config.learning_rate,
            training_weight_decay=self.training_config.weight_decay,
            training_max_instances=self.training_config.max_instances,
            training_log_every=self.training_config.log_every,
            training_device=self.training_config.device,
            training_run_name=self.training_config.run_name,
            selected_device=outcome.selected_device,
            final_loss=outcome.final_loss,
            trained_instances=outcome.trained_instances,
            model=built_retriever.model,
            model_artifact_path=outcome.model_artifact_path,
            model_config_path=outcome.model_config_path,
            model_run_directory=outcome.model_run_directory,
            model_run_name=outcome.model_run_name,
            model_run_number=outcome.model_run_number,
            embedding_cache_directory=outcome.embedding_cache_directory,
        )
