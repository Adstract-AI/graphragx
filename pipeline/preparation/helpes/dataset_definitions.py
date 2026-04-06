"""Single source of truth for built-in knowledge graph dataset definitions."""

from __future__ import annotations

from typing import Final

from pydantic import BaseModel, ConfigDict, Field


class KnowledgeGraphDatasetDefinition(BaseModel):
    """Typed definition of a built-in knowledge graph dataset."""

    model_config = ConfigDict(frozen=True)

    dataset_id: str = Field(..., description="Stable identifier of the dataset.")
    display_name: str = Field(..., description="Human-readable dataset name.")
    dataset_family: str = Field(..., description="High-level dataset family.")
    task_domain: str = Field(..., description="Reasoning task or domain classification.")
    description: str = Field(..., description="Short dataset description.")
    supported: bool = Field(..., description="Whether the dataset is currently supported.")

FB15K_237_DATASET_ID: Final[str] = "FB15K-237"

KNOWLEDGE_GRAPH_DATASETS: Final[dict[str, KnowledgeGraphDatasetDefinition]] = {
    FB15K_237_DATASET_ID: KnowledgeGraphDatasetDefinition(
        dataset_id=FB15K_237_DATASET_ID,
        display_name="FB15K-237",
        dataset_family="knowledge_graph",
        task_domain="multi_hop_reasoning",
        description=(
            "A Freebase-derived knowledge graph benchmark commonly used for "
            "link prediction and graph reasoning experiments."
        ),
        supported=True,
    ),
}
