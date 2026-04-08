"""Dataset loader services for preparation-time knowledge graph ingestion."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from pipeline.exceptions import (
    KnowledgeGraphDatasetLoadingException,
    MalformedKnowledgeGraphDatasetException,
    MissingTorchDependencyException,
    MissingTorchGeometricDependencyException,
    UnsupportedKnowledgeGraphDatasetLoaderException,
)
from pipeline.preparation.helpers.dataset_definitions import (
    FB15K_237_DATASET_ID,
    KNOWLEDGE_GRAPH_DATASET_LOADERS,
)
from pipeline.services.abstract import AbstractService

if TYPE_CHECKING:
    from torch_geometric.data import Data, Dataset


class AbstractDatasetLoaderService(AbstractService, ABC):
    """Base service for loading built-in knowledge graph datasets."""

    @abstractmethod
    def load_dataset(self, dataset_id: str) -> tuple[Dataset, Data]:
        """Load the configured dataset and return the dataset wrapper plus its data object."""


class TorchGeometricKnowledgeGraphLoaderService(AbstractDatasetLoaderService):
    """Load supported knowledge graph datasets through Torch Geometric."""

    def load_dataset(self, dataset_id: str) -> tuple[Dataset, Data]:
        loader_definition = KNOWLEDGE_GRAPH_DATASET_LOADERS.get(dataset_id)
        if loader_definition is None:
            raise UnsupportedKnowledgeGraphDatasetLoaderException(
                f"Unsupported dataset loader configuration for dataset: {dataset_id}"
            )

        if dataset_id != FB15K_237_DATASET_ID:
            raise UnsupportedKnowledgeGraphDatasetLoaderException(
                f"Unsupported dataset loader configuration for dataset: {dataset_id}"
            )

        try:
            from torch_geometric.datasets import FB15k_237
        except ModuleNotFoundError as error:
            raise MissingTorchGeometricDependencyException(
                "Missing required dependency: torch_geometric"
            ) from error

        loader_definition.cache_root.mkdir(parents=True, exist_ok=True)

        try:
            dataset = FB15k_237(root=str(loader_definition.cache_root))
        except Exception as error:
            raise KnowledgeGraphDatasetLoadingException(
                f"Failed to load knowledge graph dataset {dataset_id}: {error}"
            ) from error

        try:
            data = dataset[0]
        except Exception as error:
            raise MalformedKnowledgeGraphDatasetException(
                f"Loaded dataset {dataset_id} does not expose an indexable graph object."
            ) from error

        if not hasattr(data, "edge_index") or not hasattr(data, "edge_type"):
            raise MalformedKnowledgeGraphDatasetException(
                f"Loaded dataset {dataset_id} is missing edge_index or edge_type."
            )

        return dataset, data
