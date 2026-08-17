"""GNN answer-retriever construction step."""

from __future__ import annotations

from typing import Any

from pydantic import ConfigDict, Field

from helpers.logging_config import get_logger
from pipeline.abstract import AbstractStep, StepContext, StepResult
from pipeline.preparation.exceptions import InvalidInteractiveConfigurationInputException
from pipeline.preparation.helpers.configuration_definitions import OPENAI_EMBEDDING_MODELS
from pipeline.preparation.helpers.configuration_definitions import GNN_ARCHITECTURES
from pipeline.preparation.models.interfaces import AnswerRetrieverModel
from pipeline.preparation.models.webqsp_local_graph import PreparedWebQSPGraphDataset
from pipeline.preparation.steps.configuration_building import BuiltPipelineConfiguration

logger = get_logger(__name__)


class BuiltGnnAnswerRetriever(StepResult):
    """Constructed PyTorch answer-retriever architecture artifact."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    dataset_id: str = Field(..., description="Selected dataset identifier.")
    gnn_architecture: str = Field(default="graphsage")
    gnn_architecture_options: dict[str, Any] = Field(default_factory=dict)
    entity_embedding_model: str = Field(
        ...,
        description="OpenAI embedding model used for entity text.",
    )
    entity_embedding_dimension: int = Field(
        ...,
        description="Entity text embedding dimension before projection.",
    )
    hidden_dimension: int | None = Field(
        default=None,
        description="Shared hidden dimension used by projection, GNN, and classifier.",
    )
    gnn_layer_count: int | None = Field(default=None, description="Number of weighted GNN layers.")
    node_classifier: str | None = Field(default=None, description="Node classifier architecture id.")
    question_embedding_dimension: int = Field(
        ...,
        description="Question text embedding dimension before projection.",
    )
    relation_embedding_dimension: int = Field(
        ...,
        description="Relation text embedding dimension before projection.",
    )
    use_edge_mlp: bool = Field(default=False)
    question_aware_classifier: bool = Field(default=False)
    use_reverse_edges: bool = Field(default=False)
    add_layer_normalization: bool = Field(default=False)
    edge_mlp_hidden_dim: int | None = Field(default=None, description="Hidden dimension for edge MLP.")
    dropout: float = Field(default=0.1)
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
        from pipeline.preparation.models.gnn_answer_retriever import build_gnn_answer_retriever

        embedding_model = (
            configuration.embedding_model
            or configuration.entity_embedding_model
        )
        embedding_definition = OPENAI_EMBEDDING_MODELS[embedding_model]
        architecture_options = dict(configuration.gnn_architecture_options)
        supported_option_ids = set(
            GNN_ARCHITECTURES[configuration.gnn_architecture].option_map
        )
        for option_id, value in {
            "gnn_layer_count": configuration.gnn_layer_count,
            "gnn_hidden_dimension": configuration.gnn_hidden_dimension,
            "node_classifier": configuration.node_classifier,
            "dropout": configuration.dropout,
            "use_edge_mlp": configuration.use_edge_mlp,
            "question_aware_classifier": configuration.question_aware_classifier,
            "use_reverse_edges": configuration.use_reverse_edges,
            "add_layer_normalization": configuration.add_layer_normalization,
            "edge_mlp_hidden_dim": configuration.edge_mlp_hidden_dim,
        }.items():
            if option_id in supported_option_ids and value is not None:
                architecture_options.setdefault(option_id, value)

        logger.info(
            f"Building GNN answer retriever: dataset={prepared_dataset.dataset_id} "
            f"gnn_architecture={configuration.gnn_architecture} "
            f"embedding_model={embedding_model} "
            f"embedding_dimension={embedding_definition.dimensions} "
            f"hidden_dimension={configuration.gnn_hidden_dimension} "
            f"gnn_layers={configuration.gnn_layer_count} "
            f"node_classifier={configuration.node_classifier} "
            f"use_edge_mlp={configuration.use_edge_mlp} "
            f"question_aware_classifier={configuration.question_aware_classifier} "
            f"add_layer_normalization={configuration.add_layer_normalization} "
            f"dropout={configuration.dropout}"
        )
        model = build_gnn_answer_retriever(
            gnn_architecture=configuration.gnn_architecture,
            architecture_options=architecture_options,
            entity_embedding_dimension=embedding_definition.dimensions,
            question_embedding_dimension=embedding_definition.dimensions,
            relation_embedding_dimension=embedding_definition.dimensions,
        )

        logger.info(f"Built GNN answer retriever architecture")
        return BuiltGnnAnswerRetriever(
            dataset_id=prepared_dataset.dataset_id,
            gnn_architecture=configuration.gnn_architecture,
            gnn_architecture_options=architecture_options,
            entity_embedding_model=embedding_model,
            entity_embedding_dimension=embedding_definition.dimensions,
            question_embedding_dimension=embedding_definition.dimensions,
            relation_embedding_dimension=embedding_definition.dimensions,
            hidden_dimension=configuration.gnn_hidden_dimension,
            gnn_layer_count=configuration.gnn_layer_count,
            node_classifier=configuration.node_classifier,
            use_edge_mlp=configuration.use_edge_mlp,
            question_aware_classifier=configuration.question_aware_classifier,
            use_reverse_edges=configuration.use_reverse_edges,
            add_layer_normalization=configuration.add_layer_normalization,
            edge_mlp_hidden_dim=configuration.edge_mlp_hidden_dim,
            dropout=configuration.dropout,
            model=model,
        )
