"""Pipeline configuration building step for the preparation pipeline."""

from __future__ import annotations

from pydantic import Field, BaseModel

from helpers.logging_config import get_logger
from pipeline.abstract import AbstractStep, StepContext, StepResult
from pipeline.preparation.exceptions import (
    InvalidContextConstructionSelectionException,
    InvalidEntityEmbeddingModelSelectionException,
    InvalidGnnHiddenDimensionSelectionException,
    InvalidGnnArchitectureConfigurationException,
    InvalidGnnArchitectureSelectionException,
    InvalidGnnLayerCountSelectionException,
    InvalidInteractiveConfigurationInputException,
    InvalidMainLlmSelectionException,
    InvalidNodeClassifierSelectionException,
    InvalidQuestionEmbeddingModelSelectionException,
    InvalidRelationEmbeddingModelSelectionException,
    InvalidSubgraphConstructionSelectionException,
)
from pipeline.preparation.services.selection import SelectionService
from pipeline.preparation.helpers.configuration_definitions import (
    CONTEXT_CONSTRUCTION_STRATEGIES,
    GNN_ARCHITECTURES,
    GNN_HIDDEN_DIMENSION_OPTIONS,
    GNN_LAYER_COUNT_OPTIONS,
    GRAPH_SAGE_ARCHITECTURE_ID,
    NODE_CLASSIFIERS,
    OPENAI_EMBEDDING_MODELS,
    RECOMMENDED_CONTEXT_CONSTRUCTION_STRATEGY_ID,
    RECOMMENDED_ENTITY_EMBEDDING_MODEL_ID,
    RECOMMENDED_GNN_HIDDEN_DIMENSION,
    RECOMMENDED_GNN_ARCHITECTURE_ID,
    RECOMMENDED_GNN_LAYER_COUNT,
    RECOMMENDED_MAIN_LLM_MODEL_ID,
    RECOMMENDED_NODE_CLASSIFIER_ID,
    RECOMMENDED_QUESTION_EMBEDDING_MODEL_ID,
    RECOMMENDED_RELATION_EMBEDDING_MODEL_ID,
    RECOMMENDED_SUBGRAPH_CONSTRUCTION_ALGORITHM_ID,
    SHARED_LLM_MODELS,
    SUBGRAPH_CONSTRUCTION_ALGORITHMS,
)
from pipeline.preparation.steps.dataset_selection import SelectedDataset

logger = get_logger(__name__)


class PipelineConfigurationInput(BaseModel):
    """Optional programmatic input for pipeline configuration building."""

    main_llm_model: str | None = Field(default=None)
    subgraph_construction_algorithm: str | None = Field(default=None)
    context_construction_strategy: str | None = Field(default=None)
    gnn_architecture: str | None = Field(default=None)
    gnn_layer_count: int | None = Field(default=None)
    gnn_hidden_dimension: int | None = Field(default=None)
    node_classifier: str | None = Field(default=None)
    question_embedding_model: str | None = Field(default=None)
    relation_embedding_model: str | None = Field(default=None)
    entity_embedding_model: str | None = Field(default=None)
    use_edge_mlp: bool | None = Field(default=None)
    question_aware_classifier: bool | None = Field(default=None)
    use_reverse_edges: bool | None = Field(default=None)
    add_layer_normalization: bool | None = Field(default=None)
    edge_mlp_hidden_dim: int | None = Field(default=None)
    dropout: float | None = Field(default=None)


class BuiltPipelineConfiguration(StepResult):
    """Unified pipeline configuration artifact."""

    dataset_id: str = Field(..., description="Selected dataset identifier.")
    gnn_architecture: str = Field(default="graphsage", description="Selected GNN architecture id.")
    main_llm_model: str = Field(..., description="Selected main LLM model id.")
    subgraph_construction_algorithm: str = Field(
        ..., description="Selected subgraph construction algorithm id."
    )
    context_construction_strategy: str = Field(
        ..., description="Selected context construction strategy id."
    )
    gnn_layer_count: int = Field(..., description="Selected number of GNN layers.")
    gnn_hidden_dimension: int = Field(
        ...,
        description="Selected hidden dimension for projected GNN node states.",
    )
    node_classifier: str = Field(..., description="Selected node classifier id.")
    question_embedding_model: str = Field(
        ..., description="OpenAI embedding model for question text."
    )
    relation_embedding_model: str = Field(
        ..., description="OpenAI embedding model for relation text."
    )
    entity_embedding_model: str = Field(
        ..., description="OpenAI embedding model for entity text."
    )
    use_edge_mlp: bool = Field(default=False)
    question_aware_classifier: bool = Field(default=False)
    use_reverse_edges: bool = Field(default=False)
    add_layer_normalization: bool = Field(default=False)
    edge_mlp_hidden_dim: int | None = Field(default=None)
    dropout: float = Field(default=0.1)


class BuildPipelineConfigurationStep(
    AbstractStep[BuiltPipelineConfiguration, SelectedDataset]
):
    """Build the core pipeline configuration after dataset selection."""

    def __init__(
        self,
        main_llm_model: str | None = None,
        subgraph_algorithm: str | None = None,
        context_strategy: str | None = None,
        gnn_architecture: str | None = None,
        gnn_layer_count: int | None = None,
        gnn_hidden_dimension: int | None = None,
        node_classifier: str | None = None,
        question_embedding_model: str | None = None,
        relation_embedding_model: str | None = None,
        entity_embedding_model: str | None = None,
        use_edge_mlp: bool | None = None,
        question_aware_classifier: bool | None = None,
        use_reverse_edges: bool | None = None,
        add_layer_normalization: bool | None = None,
        edge_mlp_hidden_dim: int | None = None,
        dropout: float | None = None,
        input_func=None,
        force_default: bool = False,
    ):
        super().__init__(force_default=force_default)
        self.configuration_input = PipelineConfigurationInput(
            main_llm_model=main_llm_model,
            subgraph_construction_algorithm=subgraph_algorithm,
            context_construction_strategy=context_strategy,
            gnn_architecture=gnn_architecture,
            gnn_layer_count=gnn_layer_count,
            gnn_hidden_dimension=gnn_hidden_dimension,
            node_classifier=node_classifier,
            question_embedding_model=question_embedding_model,
            relation_embedding_model=relation_embedding_model,
            entity_embedding_model=entity_embedding_model,
            use_edge_mlp=use_edge_mlp,
            question_aware_classifier=question_aware_classifier,
            use_reverse_edges=use_reverse_edges,
            add_layer_normalization=add_layer_normalization,
            edge_mlp_hidden_dim=edge_mlp_hidden_dim,
            dropout=dropout,
        )
        self.selection_service = SelectionService(input_func=input_func)

    def execute_default(
        self,
        context: StepContext[SelectedDataset],
    ) -> BuiltPipelineConfiguration:
        selected_dataset = context.result
        if selected_dataset is None:
            raise InvalidInteractiveConfigurationInputException(
                "Configuration building requires a selected dataset in the incoming context."
            )

        logger.info(f"Building pipeline configuration for dataset={selected_dataset.dataset_id}")
        implicit_graphsage = (
            GRAPH_SAGE_ARCHITECTURE_ID
            if self.configuration_input.gnn_architecture is None
            and any(
                value is not None
                for value in (
                    self.configuration_input.gnn_layer_count,
                    self.configuration_input.gnn_hidden_dimension,
                    self.configuration_input.node_classifier,
                    self.configuration_input.dropout,
                    self.configuration_input.use_edge_mlp,
                    self.configuration_input.question_aware_classifier,
                    self.configuration_input.use_reverse_edges,
                    self.configuration_input.add_layer_normalization,
                    self.configuration_input.edge_mlp_hidden_dim,
                )
            )
            else None
        )
        gnn_architecture = self.selection_service.resolve_choice(
            provided_value=self.configuration_input.gnn_architecture or implicit_graphsage,
            options=GNN_ARCHITECTURES,
            prompt_title="GNN Architecture",
            prompt_help="Select the GNN architecture before configuring its options.",
            recommended_id=RECOMMENDED_GNN_ARCHITECTURE_ID,
            invalid_exception_type=InvalidGnnArchitectureSelectionException,
            value_getter=lambda item: item.architecture_id,
            label_getter=lambda item: item.display_name,
        )
        architecture = GNN_ARCHITECTURES[gnn_architecture]
        if not architecture.supports_advanced_options:
            explicitly_advanced = [
                name
                for name in (
                    "use_edge_mlp",
                    "question_aware_classifier",
                    "use_reverse_edges",
                    "add_layer_normalization",
                    "edge_mlp_hidden_dim",
                )
                if getattr(self.configuration_input, name) is not None
            ]
            if explicitly_advanced:
                raise InvalidGnnArchitectureConfigurationException(
                    f"Architecture {gnn_architecture} does not support: "
                    + ", ".join(explicitly_advanced)
                )
        selected_gnn_layer_count = self.selection_service.resolve_choice(
            provided_value=(
                str(self.configuration_input.gnn_layer_count)
                if self.configuration_input.gnn_layer_count is not None
                else None
            ),
            options=GNN_LAYER_COUNT_OPTIONS,
            prompt_title="GNN Layer Count",
            prompt_help="Select the number of GNN message-passing layers.",
            recommended_id=str(RECOMMENDED_GNN_LAYER_COUNT),
            invalid_exception_type=InvalidGnnLayerCountSelectionException,
            value_getter=lambda item: str(item.layer_count),
            label_getter=lambda item: item.display_name,
        )
        selected_gnn_hidden_dimension = self.selection_service.resolve_choice(
            provided_value=(
                str(self.configuration_input.gnn_hidden_dimension)
                if self.configuration_input.gnn_hidden_dimension is not None
                else None
            ),
            options=GNN_HIDDEN_DIMENSION_OPTIONS,
            prompt_title="GNN Hidden Dimension",
            prompt_help="Select the width of projected node states inside the GNN.",
            recommended_id=str(RECOMMENDED_GNN_HIDDEN_DIMENSION),
            invalid_exception_type=InvalidGnnHiddenDimensionSelectionException,
            value_getter=lambda item: str(item.hidden_dimension),
            label_getter=lambda item: item.display_name,
        )
        node_classifier = self.selection_service.resolve_choice(
            provided_value=self.configuration_input.node_classifier,
            options=NODE_CLASSIFIERS,
            prompt_title="Node Classifier",
            prompt_help="Select the classifier used after the final GNN layer.",
            recommended_id=RECOMMENDED_NODE_CLASSIFIER_ID,
            invalid_exception_type=InvalidNodeClassifierSelectionException,
            value_getter=lambda item: item.classifier_id,
            label_getter=lambda item: item.display_name,
        )
        shared_options_complete = all(
            value is not None
            for value in (
                self.configuration_input.gnn_layer_count,
                self.configuration_input.gnn_hidden_dimension,
                self.configuration_input.node_classifier,
            )
        )
        provided_dropout = self.configuration_input.dropout
        if provided_dropout is None and shared_options_complete:
            provided_dropout = architecture.default_dropout
        selected_dropout = self.selection_service.resolve_choice(
            provided_value=(
                str(provided_dropout)
                if provided_dropout is not None
                else None
            ),
            options={str(value): value for value in architecture.supported_dropouts},
            prompt_title="GNN Dropout",
            prompt_help=f"Select dropout for {architecture.display_name}.",
            recommended_id=str(architecture.default_dropout),
            invalid_exception_type=InvalidGnnArchitectureConfigurationException,
            value_getter=str,
            label_getter=lambda value: f"{value:g}",
        )

        use_edge_mlp = False
        question_aware_classifier = False
        use_reverse_edges = False
        add_layer_normalization = False
        edge_mlp_hidden_dim = None
        if architecture.supports_advanced_options:
            boolean_options = {"yes": True, "no": False}

            def resolve_boolean(field_name: str, title: str) -> bool:
                supplied = getattr(self.configuration_input, field_name)
                selected = self.selection_service.resolve_choice(
                    provided_value=("yes" if supplied else "no") if supplied is not None else None,
                    options=boolean_options,
                    prompt_title=title,
                    prompt_help=f"Enable this option for {architecture.display_name}?",
                    recommended_id="yes",
                    invalid_exception_type=InvalidGnnArchitectureConfigurationException,
                    value_getter=lambda value: "yes" if value else "no",
                    label_getter=lambda value: "Yes" if value else "No",
                )
                return selected == "yes"

            use_edge_mlp = resolve_boolean("use_edge_mlp", "Use Edge MLP")
            use_reverse_edges = resolve_boolean("use_reverse_edges", "Use Reverse Edges")
            question_aware_classifier = resolve_boolean(
                "question_aware_classifier", "Question-Aware Classifier"
            )
            add_layer_normalization = resolve_boolean(
                "add_layer_normalization", "Add Layer Normalization"
            )
            if use_edge_mlp:
                selected_edge_width = self.selection_service.resolve_choice(
                    provided_value=(
                        str(self.configuration_input.edge_mlp_hidden_dim)
                        if self.configuration_input.edge_mlp_hidden_dim is not None
                        else None
                    ),
                    options={
                        str(value): value
                        for value in architecture.supported_edge_mlp_hidden_dimensions
                    },
                    prompt_title="Edge MLP Hidden Dimension",
                    prompt_help="Select the hidden width of the relation-aware edge MLP.",
                    recommended_id=str(architecture.default_edge_mlp_hidden_dimension),
                    invalid_exception_type=InvalidGnnArchitectureConfigurationException,
                    value_getter=str,
                    label_getter=lambda value: str(value),
                )
                edge_mlp_hidden_dim = int(selected_edge_width)
            elif self.configuration_input.edge_mlp_hidden_dim is not None:
                raise InvalidGnnArchitectureConfigurationException(
                    "--edge-mlp-hidden-dim cannot be used when edge MLP is disabled."
                )

            if node_classifier == "linear" and question_aware_classifier:
                if self.configuration_input.node_classifier is not None:
                    raise InvalidGnnArchitectureConfigurationException(
                        "AA-GraphSAGE linear classification requires "
                        "--no-question-aware-classifier."
                    )
                self.selection_service.output_func(
                    "Linear classification is incompatible with a question-aware head. "
                    "Please select the MLP classifier."
                )
                node_classifier = self.selection_service.prompt_for_choice(
                    options={"mlp": NODE_CLASSIFIERS["mlp"]},
                    prompt_title="Node Classifier",
                    prompt_help="Question-aware AA-GraphSAGE requires an MLP classifier.",
                    recommended_id="mlp",
                    invalid_exception_type=InvalidNodeClassifierSelectionException,
                    value_getter=lambda item: item.classifier_id,
                    label_getter=lambda item: item.display_name,
                )
        main_llm_model = self.selection_service.resolve_choice(
            provided_value=self.configuration_input.main_llm_model,
            options=SHARED_LLM_MODELS,
            prompt_title="Main LLM Model",
            prompt_help="Select the primary model used for final question answering.",
            recommended_id=RECOMMENDED_MAIN_LLM_MODEL_ID,
            invalid_exception_type=InvalidMainLlmSelectionException,
            value_getter=lambda item: item.model_id,
            label_getter=lambda item: item.display_name,
        )
        subgraph_algorithm = self.selection_service.resolve_choice(
            provided_value=self.configuration_input.subgraph_construction_algorithm,
            options=SUBGRAPH_CONSTRUCTION_ALGORITHMS,
            prompt_title="Subgraph Construction Algorithm",
            prompt_help="Select the algorithm used to build question-specific subgraphs.",
            recommended_id=RECOMMENDED_SUBGRAPH_CONSTRUCTION_ALGORITHM_ID,
            invalid_exception_type=InvalidSubgraphConstructionSelectionException,
            value_getter=lambda item: item.algorithm_id,
            label_getter=lambda item: item.display_name,
        )
        context_strategy = self.selection_service.resolve_choice(
            provided_value=self.configuration_input.context_construction_strategy,
            options=CONTEXT_CONSTRUCTION_STRATEGIES,
            prompt_title="Context Construction Strategy",
            prompt_help="Select how the retrieved subgraph will be represented for the LLM.",
            recommended_id=RECOMMENDED_CONTEXT_CONSTRUCTION_STRATEGY_ID,
            invalid_exception_type=InvalidContextConstructionSelectionException,
            value_getter=lambda item: item.strategy_id,
            label_getter=lambda item: item.display_name,
        )
        question_embedding_model = self.selection_service.resolve_choice(
            provided_value=self.configuration_input.question_embedding_model,
            options=OPENAI_EMBEDDING_MODELS,
            prompt_title="Question Embedding Model",
            prompt_help="Select the OpenAI embedding model used for question text.",
            recommended_id=RECOMMENDED_QUESTION_EMBEDDING_MODEL_ID,
            invalid_exception_type=InvalidQuestionEmbeddingModelSelectionException,
            value_getter=lambda item: item.model_id,
            label_getter=lambda item: item.display_name,
        )
        relation_embedding_model = self.selection_service.resolve_choice(
            provided_value=self.configuration_input.relation_embedding_model,
            options=OPENAI_EMBEDDING_MODELS,
            prompt_title="Relation Embedding Model",
            prompt_help="Select the OpenAI embedding model used for relation text.",
            recommended_id=RECOMMENDED_RELATION_EMBEDDING_MODEL_ID,
            invalid_exception_type=InvalidRelationEmbeddingModelSelectionException,
            value_getter=lambda item: item.model_id,
            label_getter=lambda item: item.display_name,
        )
        entity_embedding_model = self.selection_service.resolve_choice(
            provided_value=self.configuration_input.entity_embedding_model,
            options=OPENAI_EMBEDDING_MODELS,
            prompt_title="Entity Embedding Model",
            prompt_help="Select the OpenAI embedding model used for entity text.",
            recommended_id=RECOMMENDED_ENTITY_EMBEDDING_MODEL_ID,
            invalid_exception_type=InvalidEntityEmbeddingModelSelectionException,
            value_getter=lambda item: item.model_id,
            label_getter=lambda item: item.display_name,
        )

        logger.info(
            f"Built pipeline configuration: gnn_architecture={gnn_architecture} "
            f"gnn_layers={selected_gnn_layer_count} "
            f"gnn_hidden_dimension={selected_gnn_hidden_dimension} "
            f"node_classifier={node_classifier} "
            f"question_embedding_model={question_embedding_model} "
            f"relation_embedding_model={relation_embedding_model} "
            f"entity_embedding_model={entity_embedding_model} "
            f"use_edge_mlp={use_edge_mlp} "
            f"question_aware_classifier="
            f"{question_aware_classifier} "
            f"use_reverse_edges={use_reverse_edges} "
            f"add_layer_normalization="
            f"{add_layer_normalization}"
        )
        return BuiltPipelineConfiguration(
            dataset_id=selected_dataset.dataset_id,
            gnn_architecture=gnn_architecture,
            main_llm_model=main_llm_model,
            subgraph_construction_algorithm=subgraph_algorithm,
            context_construction_strategy=context_strategy,
            gnn_layer_count=int(selected_gnn_layer_count),
            gnn_hidden_dimension=int(selected_gnn_hidden_dimension),
            node_classifier=node_classifier,
            question_embedding_model=question_embedding_model,
            relation_embedding_model=relation_embedding_model,
            entity_embedding_model=entity_embedding_model,
            use_edge_mlp=use_edge_mlp,
            question_aware_classifier=question_aware_classifier,
            use_reverse_edges=use_reverse_edges,
            add_layer_normalization=add_layer_normalization,
            edge_mlp_hidden_dim=edge_mlp_hidden_dim,
            dropout=float(selected_dropout),
        )
