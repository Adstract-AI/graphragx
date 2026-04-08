"""Pipeline configuration building step for the preparation pipeline."""

from __future__ import annotations

from pydantic import Field, BaseModel

from pipeline.abstract import AbstractStep, StepContext, StepResult
from pipeline.exceptions import (
    InvalidAssistantLlmSelectionException,
    InvalidContextConstructionSelectionException,
    InvalidGnnArchitectureSelectionException,
    InvalidInteractiveConfigurationInputException,
    InvalidMainLlmSelectionException,
    InvalidSubgraphConstructionSelectionException,
)
from pipeline.services.selection import SelectionService
from pipeline.preparation.helpers.configuration_definitions import (
    CONTEXT_CONSTRUCTION_STRATEGIES,
    GNN_ARCHITECTURES,
    RECOMMENDED_ASSISTANT_LLM_MODEL_ID,
    RECOMMENDED_CONTEXT_CONSTRUCTION_STRATEGY_ID,
    RECOMMENDED_GNN_ARCHITECTURE_ID,
    RECOMMENDED_MAIN_LLM_MODEL_ID,
    RECOMMENDED_SUBGRAPH_CONSTRUCTION_ALGORITHM_ID,
    SHARED_LLM_MODELS,
    SUBGRAPH_CONSTRUCTION_ALGORITHMS,
)
from pipeline.preparation.steps.dataset_selection import SelectedKnowledgeGraphDataset


class PipelineConfigurationInput(BaseModel):
    """Optional programmatic input for pipeline configuration building."""

    main_llm_model: str | None = Field(default=None)
    assistant_llm_model: str | None = Field(default=None)
    subgraph_construction_algorithm: str | None = Field(default=None)
    context_construction_strategy: str | None = Field(default=None)
    gnn_architecture: str | None = Field(default=None)


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
    gnn_architecture: str = Field(..., description="Selected GNN architecture id.")


class BuildPipelineConfigurationStep(
    AbstractStep[BuiltPipelineConfiguration, SelectedKnowledgeGraphDataset]
):
    """Build the core pipeline configuration after dataset selection."""

    def __init__(
        self,
        main_llm_model: str | None = None,
        assistant_llm_model: str | None = None,
        subgraph_algorithm: str | None = None,
        context_strategy: str | None = None,
        gnn_architecture: str | None = None,
        input_func=None,
        force_default: bool = False,
    ):
        super().__init__(force_default=force_default)
        self.configuration_input = PipelineConfigurationInput(
            main_llm_model=main_llm_model,
            assistant_llm_model=assistant_llm_model,
            subgraph_construction_algorithm=subgraph_algorithm,
            context_construction_strategy=context_strategy,
            gnn_architecture=gnn_architecture,
        )
        self.selection_service = SelectionService(input_func=input_func)

    def execute_default(
        self,
        context: StepContext[SelectedKnowledgeGraphDataset],
    ) -> BuiltPipelineConfiguration:
        selected_dataset = context.result
        if selected_dataset is None:
            raise InvalidInteractiveConfigurationInputException(
                "Configuration building requires a selected dataset in the incoming context."
            )

        gnn_architecture = self.selection_service.resolve_choice(
            provided_value=self.configuration_input.gnn_architecture,
            options=GNN_ARCHITECTURES,
            prompt_title="GNN Architecture",
            prompt_help="Select the graph neural network architecture used for knowledge graph modeling.",
            recommended_id=RECOMMENDED_GNN_ARCHITECTURE_ID,
            invalid_exception_type=InvalidGnnArchitectureSelectionException,
            value_getter=lambda item: item.architecture_id,
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

        return BuiltPipelineConfiguration(
            dataset_id=selected_dataset.dataset_id,
            main_llm_model=main_llm_model,
            assistant_llm_model=assistant_llm_model,
            subgraph_construction_algorithm=subgraph_algorithm,
            context_construction_strategy=context_strategy,
            gnn_architecture=gnn_architecture,
        )
