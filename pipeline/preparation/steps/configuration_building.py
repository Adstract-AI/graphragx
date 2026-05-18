"""Pipeline configuration building step for the preparation pipeline."""

from __future__ import annotations

from pydantic import Field, BaseModel

from pipeline.abstract import AbstractStep, StepContext, StepResult
from pipeline.exceptions import (
    InvalidAssistantLlmSelectionException,
    InvalidContextConstructionSelectionException,
    InvalidEntityEmbeddingModelSelectionException,
    InvalidGnnLayerCountSelectionException,
    InvalidInteractiveConfigurationInputException,
    InvalidMainLlmSelectionException,
    InvalidNodeClassifierSelectionException,
    InvalidQuestionEmbeddingModelSelectionException,
    InvalidRelationEmbeddingModelSelectionException,
    InvalidSubgraphConstructionSelectionException,
)
from pipeline.services.selection import SelectionService
from pipeline.preparation.helpers.configuration_definitions import (
    CONTEXT_CONSTRUCTION_STRATEGIES,
    GNN_LAYER_COUNT_OPTIONS,
    NODE_CLASSIFIERS,
    OPENAI_EMBEDDING_MODELS,
    RECOMMENDED_ASSISTANT_LLM_MODEL_ID,
    RECOMMENDED_CONTEXT_CONSTRUCTION_STRATEGY_ID,
    RECOMMENDED_ENTITY_EMBEDDING_MODEL_ID,
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


class PipelineConfigurationInput(BaseModel):
    """Optional programmatic input for pipeline configuration building."""

    main_llm_model: str | None = Field(default=None)
    assistant_llm_model: str | None = Field(default=None)
    subgraph_construction_algorithm: str | None = Field(default=None)
    context_construction_strategy: str | None = Field(default=None)
    gnn_layer_count: int | None = Field(default=None)
    node_classifier: str | None = Field(default=None)
    question_embedding_model: str | None = Field(default=None)
    relation_embedding_model: str | None = Field(default=None)
    entity_embedding_model: str | None = Field(default=None)


class BuiltPipelineConfiguration(StepResult):
    """Unified pipeline configuration artifact."""

    dataset_id: str = Field(..., description="Selected dataset identifier.")
    main_llm_model: str = Field(..., description="Selected main LLM model id.")
    assistant_llm_model: str = Field(..., description="Selected assistant LLM model id.")
    subgraph_construction_algorithm: str = Field(
        ..., description="Selected subgraph construction algorithm id."
    )
    context_construction_strategy: str = Field(
        ..., description="Selected context construction strategy id."
    )
    gnn_layer_count: int = Field(..., description="Selected number of GNN layers.")
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


class BuildPipelineConfigurationStep(
    AbstractStep[BuiltPipelineConfiguration, SelectedDataset]
):
    """Build the core pipeline configuration after dataset selection."""

    def __init__(
        self,
        main_llm_model: str | None = None,
        assistant_llm_model: str | None = None,
        subgraph_algorithm: str | None = None,
        context_strategy: str | None = None,
        gnn_layer_count: int | None = None,
        node_classifier: str | None = None,
        question_embedding_model: str | None = None,
        relation_embedding_model: str | None = None,
        entity_embedding_model: str | None = None,
        input_func=None,
        force_default: bool = False,
    ):
        super().__init__(force_default=force_default)
        self.configuration_input = PipelineConfigurationInput(
            main_llm_model=main_llm_model,
            assistant_llm_model=assistant_llm_model,
            subgraph_construction_algorithm=subgraph_algorithm,
            context_construction_strategy=context_strategy,
            gnn_layer_count=gnn_layer_count,
            node_classifier=node_classifier,
            question_embedding_model=question_embedding_model,
            relation_embedding_model=relation_embedding_model,
            entity_embedding_model=entity_embedding_model,
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
        assistant_llm_model = self.selection_service.resolve_choice(
            provided_value=self.configuration_input.assistant_llm_model,
            options=SHARED_LLM_MODELS,
            prompt_title="Assistant LLM Model",
            prompt_help="Select the support model used for intermediate reasoning tasks.",
            recommended_id=RECOMMENDED_ASSISTANT_LLM_MODEL_ID,
            invalid_exception_type=InvalidAssistantLlmSelectionException,
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

        return BuiltPipelineConfiguration(
            dataset_id=selected_dataset.dataset_id,
            main_llm_model=main_llm_model,
            assistant_llm_model=assistant_llm_model,
            subgraph_construction_algorithm=subgraph_algorithm,
            context_construction_strategy=context_strategy,
            gnn_layer_count=int(selected_gnn_layer_count),
            node_classifier=node_classifier,
            question_embedding_model=question_embedding_model,
            relation_embedding_model=relation_embedding_model,
            entity_embedding_model=entity_embedding_model,
        )
