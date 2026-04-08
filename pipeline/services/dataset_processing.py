"""Dataset processing services for loaded knowledge graph artifacts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from pipeline.exceptions import (
    KnowledgeGraphDatasetLoadingException,
    MalformedKnowledgeGraphDatasetException,
)
from pipeline.services.abstract import AbstractService

if TYPE_CHECKING:
    from torch_geometric.data import Data
    from pipeline.preparation.steps.dataset_loading import KnowledgeGraphRawTriple


class AbstractKnowledgeGraphDatasetProcessingService(AbstractService, ABC):
    """Base service for processing loaded knowledge graph data objects."""

    @abstractmethod
    def extract_raw_triples(
        self,
        dataset_id: str,
        data: Data,
    ) -> list[KnowledgeGraphRawTriple]:
        """Extract the typed raw triple view from the loaded graph data."""


class KnowledgeGraphDatasetProcessingService(
    AbstractKnowledgeGraphDatasetProcessingService
):
    """Process loaded graph data into preparation-ready triple artifacts."""

    def extract_raw_triples(
        self,
        dataset_id: str,
        data: Data,
    ) -> list[KnowledgeGraphRawTriple]:
        from pipeline.preparation.steps.dataset_loading import KnowledgeGraphRawTriple

        edge_index = data.edge_index
        edge_type = data.edge_type

        try:
            head_ids = edge_index[0].tolist()
            tail_ids = edge_index[1].tolist()
            relation_ids = edge_type.tolist()
        except Exception as error:
            raise KnowledgeGraphDatasetLoadingException(
                f"Failed to extract raw triples from dataset {dataset_id}: {error}"
            ) from error

        if not (len(head_ids) == len(tail_ids) == len(relation_ids)):
            raise MalformedKnowledgeGraphDatasetException(
                f"Loaded dataset {dataset_id} has inconsistent triple tensor lengths."
            )

        return [
            KnowledgeGraphRawTriple(
                head_id=int(head_id),
                relation_id=int(relation_id),
                tail_id=int(tail_id),
            )
            for head_id, relation_id, tail_id in zip(
                head_ids,
                relation_ids,
                tail_ids,
                strict=True,
            )
        ]
