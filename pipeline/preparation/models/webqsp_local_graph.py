"""Prepared WebQSP local graph models."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from pipeline.abstract import StepResult

if TYPE_CHECKING:
    from torch import Tensor as TorchTensor
else:
    TorchTensor = Any


class WebQSPLocalGraphExample(BaseModel):
    """Prepared local graph representation for one WebQSP example."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    question: str = Field(..., description="Natural language question.")
    q_entity: list[str] = Field(..., description="Topic/question entities.")
    a_entity: list[str] = Field(..., description="Gold answer entities.")
    nodes: list[str] = Field(..., description="Local graph node names.")
    node2id: dict[str, int] = Field(..., description="Local node-to-id mapping.")
    edge_index: TorchTensor = Field(..., description="Directed edge tensor.")
    edge_relations: list[str] = Field(..., description="Relation text for each edge.")
    node_labels: TorchTensor = Field(..., description="Binary answer-node labels.")


class WebQSPVocabularyStore(BaseModel):
    """Reusable node and relation vocabularies collected from WebQSP graphs."""

    nodes: dict[str, int] = Field(default_factory=dict)
    relations: dict[str, int] = Field(default_factory=dict)


class PreparedWebQSPLocalGraphDataset(StepResult):
    """Prepared WebQSP graph dataset artifact."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    dataset_id: str = Field(..., description="Selected dataset identifier.")
    processing_version: str = Field(..., description="Processing version.")
    train_examples: list[WebQSPLocalGraphExample] = Field(default_factory=list)
    test_examples: list[WebQSPLocalGraphExample] = Field(default_factory=list)
    vocabulary_store: WebQSPVocabularyStore = Field(...)
    cache_directory: Path = Field(..., description="Processed dataset cache directory.")

    @property
    def train_size(self) -> int:
        """Return the number of prepared training examples."""
        return len(self.train_examples)

    @property
    def test_size(self) -> int:
        """Return the number of prepared test examples."""
        return len(self.test_examples)
