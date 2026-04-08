"""Knowledge graph dataset selection for the preparation pipeline."""

from __future__ import annotations

from collections.abc import Callable

from pydantic import Field

from pipeline.abstract import AbstractStep, StepContext, StepResult
from pipeline.exceptions import UnsupportedKnowledgeGraphDatasetException
from pipeline.models import InitialStepResult
from pipeline.preparation.helpers.dataset_definitions import (
    FB15K_237_DATASET_ID,
    KNOWLEDGE_GRAPH_DATASETS,
)
from pipeline.services.selection import SelectionService


class SelectedKnowledgeGraphDataset(StepResult):
    """Metadata-only result describing the selected KG dataset."""

    dataset_id: str = Field(..., description="Stable identifier of the selected dataset.")
    display_name: str = Field(..., description="Human-readable dataset name.")
    dataset_family: str = Field(..., description="High-level dataset family.")
    task_domain: str = Field(..., description="Reasoning task or domain classification.")
    description: str = Field(..., description="Short dataset description.")
    supported: bool = Field(..., description="Whether the dataset is currently supported.")


class SelectKnowledgeGraphDatasetStep(
    AbstractStep[SelectedKnowledgeGraphDataset, InitialStepResult]
):
    """Select the knowledge graph dataset for the preparation pipeline."""

    def __init__(
        self,
        requested_dataset: str | None = None,
        input_func: Callable[[str], str] | None = None,
        force_default: bool = False,
    ):
        super().__init__(force_default=force_default)
        self.requested_dataset = requested_dataset
        self.selection_service = SelectionService(input_func=input_func)

    def execute_default(
        self,
        context: StepContext[InitialStepResult],
    ) -> SelectedKnowledgeGraphDataset:
        requested_dataset = self.selection_service.resolve_choice(
            provided_value=self.requested_dataset,
            options=KNOWLEDGE_GRAPH_DATASETS,
            prompt_title="Knowledge Graph Dataset",
            prompt_help="Select the dataset used as the knowledge graph world for the pipeline.",
            recommended_id=FB15K_237_DATASET_ID,
            invalid_exception_type=UnsupportedKnowledgeGraphDatasetException,
            value_getter=lambda item: item.dataset_id,
            label_getter=lambda item: item.display_name,
        )
        dataset_definition = KNOWLEDGE_GRAPH_DATASETS.get(requested_dataset)
        if dataset_definition is None or dataset_definition.dataset_id != FB15K_237_DATASET_ID:
            raise UnsupportedKnowledgeGraphDatasetException(
                f"Unsupported knowledge graph dataset: {requested_dataset}"
            )

        return SelectedKnowledgeGraphDataset(
            dataset_id=dataset_definition.dataset_id,
            display_name=dataset_definition.display_name,
            dataset_family=dataset_definition.dataset_family,
            task_domain=dataset_definition.task_domain,
            description=dataset_definition.description,
            supported=dataset_definition.supported,
        )
