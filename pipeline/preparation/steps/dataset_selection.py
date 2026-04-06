"""Knowledge graph dataset selection for the preparation pipeline."""

from __future__ import annotations

from pydantic import Field

from pipeline.abstract import AbstractStep, StepContext, StepResult
from pipeline.exceptions import UnsupportedKnowledgeGraphDatasetException
from pipeline.models import InitialStepResult
from pipeline.preparation.helpers.dataset_definitions import (
    FB15K_237_DATASET_ID,
    KNOWLEDGE_GRAPH_DATASETS,
    KnowledgeGraphDatasetDefinition,
)


class KnowledgeGraphDatasetSelection(InitialStepResult):
    """Bootstrap input for selecting the KG dataset used in preparation."""

    requested_dataset: str = Field(
        default=FB15K_237_DATASET_ID,
        description="Dataset identifier requested for the preparation pipeline.",
    )


class SelectedKnowledgeGraphDataset(StepResult):
    """Metadata-only result describing the selected KG dataset."""

    dataset_id: str = Field(..., description="Stable identifier of the selected dataset.")
    display_name: str = Field(..., description="Human-readable dataset name.")
    dataset_family: str = Field(..., description="High-level dataset family.")
    task_domain: str = Field(..., description="Reasoning task or domain classification.")
    description: str = Field(..., description="Short dataset description.")
    supported: bool = Field(..., description="Whether the dataset is currently supported.")


class SelectKnowledgeGraphDatasetStep(
    AbstractStep[SelectedKnowledgeGraphDataset, KnowledgeGraphDatasetSelection]
):
    """Select the knowledge graph dataset for the preparation pipeline."""

    def execute_default(
        self,
        context: StepContext[KnowledgeGraphDatasetSelection],
    ) -> SelectedKnowledgeGraphDataset:
        requested_dataset = self._get_requested_dataset(context)
        dataset_definition = KNOWLEDGE_GRAPH_DATASETS.get(requested_dataset)
        if dataset_definition is None or dataset_definition.dataset_id != FB15K_237_DATASET_ID:
            raise UnsupportedKnowledgeGraphDatasetException(
                f"Unsupported knowledge graph dataset: {requested_dataset}"
            )

        return self._build_selected_dataset(dataset_definition)

    @staticmethod
    def _get_requested_dataset(context: StepContext[KnowledgeGraphDatasetSelection]) -> str:
        """Resolve the requested dataset from the incoming step context."""
        if context.result is None:
            return FB15K_237_DATASET_ID

        return context.result.requested_dataset

    @staticmethod
    def _build_selected_dataset(
        dataset_definition: KnowledgeGraphDatasetDefinition,
    ) -> SelectedKnowledgeGraphDataset:
        """Create the step result from the typed dataset definition."""
        return SelectedKnowledgeGraphDataset(**dataset_definition.model_dump())
