"""Dataset loader services for preparation-time dataset ingestion."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from pipeline.exceptions import (
    DatasetLoadingException,
    MissingHuggingFaceDatasetsDependencyException,
    UnsupportedDatasetLoaderException,
)
from pipeline.preparation.helpers.dataset_definitions import (
    DATASET_LOADERS,
    DatasetLoaderDefinition,
    WEBQSP_DATASET_ID,
)
from pipeline.services.abstract import AbstractService

if TYPE_CHECKING:
    from datasets import DatasetDict


class AbstractDatasetLoaderService(AbstractService, ABC):
    """Base service for loading built-in datasets."""

    def get_loader_definition(self, dataset_id: str) -> DatasetLoaderDefinition:
        """Return the loader definition for a supported dataset."""
        loader_definition = DATASET_LOADERS.get(dataset_id)
        if loader_definition is None:
            raise UnsupportedDatasetLoaderException(
                f"Unsupported dataset loader configuration for dataset: {dataset_id}"
            )

        return loader_definition

    @abstractmethod
    def load_dataset(self, dataset_id: str) -> DatasetDict:
        """Load the configured dataset and return its Hugging Face dataset dictionary."""


class HuggingFaceWebQSPDatasetLoaderService(AbstractDatasetLoaderService):
    """Load WebQSP from Hugging Face datasets."""

    def load_dataset(self, dataset_id: str) -> DatasetDict:
        loader_definition = self.get_loader_definition(dataset_id)
        if dataset_id != WEBQSP_DATASET_ID:
            raise UnsupportedDatasetLoaderException(
                f"Unsupported dataset loader configuration for dataset: {dataset_id}"
            )

        try:
            from datasets import load_dataset
        except ModuleNotFoundError as error:
            raise MissingHuggingFaceDatasetsDependencyException(
                "Missing required dependency: datasets"
            ) from error

        loader_definition.cache_root.mkdir(parents=True, exist_ok=True)

        try:
            return load_dataset(
                loader_definition.hugging_face_dataset_name,
                cache_dir=str(loader_definition.cache_root),
            )
        except Exception as error:
            raise DatasetLoadingException(
                f"Failed to load dataset {dataset_id}: {error}"
            ) from error
