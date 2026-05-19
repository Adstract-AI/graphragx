"""GNN answer-retriever construction step."""

from __future__ import annotations

from pydantic import ConfigDict, Field

from helpers.logging_config import get_logger
from pipeline.abstract import AbstractStep, StepContext, StepResult
from pipeline.exceptions import InvalidInteractiveConfigurationInputException
from pipeline.preparation.helpers.configuration_definitions import OPENAI_EMBEDDING_MODELS
from pipeline.preparation.models.interfaces import AnswerRetrieverModel
from pipeline.preparation.models.webqsp_local_graph import PreparedWebQSPGraphDataset
from pipeline.preparation.steps.configuration_building import BuiltPipelineConfiguration

logger = get_logger(__name__)


class BuiltGnnAnswerRetriever(StepResult):
    """Constructed PyTorch answer-retriever architecture artifact."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    dataset_id: str = Field(..., description="Selected dataset identifier.")
    entity_embedding_model: str = Field(
        ...,
        description="OpenAI embedding model used for entity text.",
    )
    entity_embedding_dimension: int = Field(
        ...,
        description="Entity text embedding dimension before projection.",
    )
    hidden_dimension: int = Field(
        ...,
        description="Shared hidden dimension used by projection, GNN, and classifier.",
    )
    gnn_layer_count: int = Field(..., description="Number of weighted GNN layers.")
    node_classifier: str = Field(..., description="Node classifier architecture id.")
    model: AnswerRetrieverModel = Field(
        ...,
        description="Constructed PyTorch answer-retriever model.",
    )


class BuildGnnAnswerRetrieverContext(StepContext[PreparedWebQSPGraphDataset]):
    """Specialized context for constructing the GNN answer retriever."""

    pipeline_configuration: BuiltPipelineConfiguration = Field(
        ...,
        description="Pipeline configuration required to build the retriever.",
    )


class BuildGnnAnswerRetrieverStep(
    AbstractStep[BuiltGnnAnswerRetriever, PreparedWebQSPGraphDataset]
):
    """Build the PyTorch GNN retriever and node classifier."""

    def __init__(
        self,
        force_default: bool = False,
    ):
        super().__init__(force_default=force_default)

    def execute_default(
        self,
        context: BuildGnnAnswerRetrieverContext,
    ) -> BuiltGnnAnswerRetriever:
        prepared_dataset = context.result
        if prepared_dataset is None:
            raise InvalidInteractiveConfigurationInputException(
                "GNN answer-retriever building requires a prepared WebQSP dataset."
            )

        configuration = context.pipeline_configuration
        from pipeline.preparation.models.gnn_answer_retriever import GnnAnswerRetriever

        entity_embedding_definition = OPENAI_EMBEDDING_MODELS[
            configuration.entity_embedding_model
        ]

        logger.info(
            f"Building GNN answer retriever: dataset={prepared_dataset.dataset_id} "
            f"entity_embedding_dimension={entity_embedding_definition.dimensions} "
            f"hidden_dimension={configuration.gnn_hidden_dimension} "
            f"gnn_layers={configuration.gnn_layer_count} "
            f"node_classifier={configuration.node_classifier}"
        )
        model = GnnAnswerRetriever(
            entity_embedding_dimension=entity_embedding_definition.dimensions,
            hidden_dimension=configuration.gnn_hidden_dimension,
            gnn_layer_count=configuration.gnn_layer_count,
            node_classifier=configuration.node_classifier,
        )

        logger.info(f"Built GNN answer retriever architecture")
        return BuiltGnnAnswerRetriever(
            dataset_id=prepared_dataset.dataset_id,
            entity_embedding_model=configuration.entity_embedding_model,
            entity_embedding_dimension=entity_embedding_definition.dimensions,
            hidden_dimension=configuration.gnn_hidden_dimension,
            gnn_layer_count=configuration.gnn_layer_count,
            node_classifier=configuration.node_classifier,
            model=model,
        )
