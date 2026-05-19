"""Storage service for cached WebQSP local graph artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from pipeline.exceptions import (
    MissingTorchDependencyException,
    ProcessedDatasetStorageException,
    UnsupportedDatasetProcessorException,
)
from pipeline.preparation.helpers.dataset_definitions import (
    DATASET_LOADERS,
    WEBQSP_DATASET_ID,
)
from pipeline.preparation.models.webqsp_local_graph import (
    PreparedWebQSPLocalGraphDataset,
    WebQSPLocalGraphExample,
    WebQSPVocabularyStore,
)
from pipeline.services.abstract import AbstractService


class WebQSPLocalGraphStorageService(AbstractService):
    """Persist and load processed WebQSP local graph artifacts."""

    metadata_filename = "metadata.json"
    train_examples_filename = "train_examples.pt"
    test_examples_filename = "test_examples.pt"
    nodes_filename = "nodes.json"
    relations_filename = "relations.json"

    def get_cache_directory(self, dataset_id: str) -> Path:
        """Return the processed cache directory for a supported dataset."""
        if dataset_id != WEBQSP_DATASET_ID:
            raise UnsupportedDatasetProcessorException(
                f"Unsupported processed dataset cache for dataset: {dataset_id}"
            )

        return DATASET_LOADERS[dataset_id].cache_root / "processed"

    def load_if_available(
        self,
        dataset_id: str,
        processing_version: str,
    ) -> PreparedWebQSPLocalGraphDataset | None:
        """Load cached processed artifacts when cache metadata is valid."""
        cache_directory = self.get_cache_directory(dataset_id)
        metadata_path = cache_directory / self.metadata_filename
        train_examples_path = cache_directory / self.train_examples_filename
        test_examples_path = cache_directory / self.test_examples_filename
        nodes_path = cache_directory / self.nodes_filename
        relations_path = cache_directory / self.relations_filename

        required_paths = [
            metadata_path,
            train_examples_path,
            test_examples_path,
            nodes_path,
            relations_path,
        ]
        if any(not path.exists() for path in required_paths):
            return None

        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if not self._is_valid_metadata(
                metadata=metadata,
                dataset_id=dataset_id,
                processing_version=processing_version,
            ):
                return None

            train_examples = self._load_examples(train_examples_path)
            test_examples = self._load_examples(test_examples_path)
            vocabulary_store = WebQSPVocabularyStore(
                nodes=json.loads(nodes_path.read_text(encoding="utf-8")),
                relations=json.loads(relations_path.read_text(encoding="utf-8")),
            )

            if metadata["train_size"] != len(train_examples):
                return None

            if metadata["test_size"] != len(test_examples):
                return None

            return PreparedWebQSPLocalGraphDataset(
                dataset_id=dataset_id,
                processing_version=processing_version,
                train_examples=train_examples,
                test_examples=test_examples,
                vocabulary_store=vocabulary_store,
                cache_directory=cache_directory,
            )
        except MissingTorchDependencyException:
            raise
        except Exception as error:
            raise ProcessedDatasetStorageException(
                f"Failed to load processed WebQSP dataset cache: {error}"
            ) from error

    def save(self, dataset: PreparedWebQSPLocalGraphDataset) -> None:
        """Save processed examples, vocabularies, and metadata."""
        cache_directory = dataset.cache_directory
        cache_directory.mkdir(parents=True, exist_ok=True)

        try:
            self._save_examples(
                path=cache_directory / self.train_examples_filename,
                examples=dataset.train_examples,
            )
            self._save_examples(
                path=cache_directory / self.test_examples_filename,
                examples=dataset.test_examples,
            )
            (cache_directory / self.nodes_filename).write_text(
                json.dumps(dataset.vocabulary_store.nodes, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            (cache_directory / self.relations_filename).write_text(
                json.dumps(dataset.vocabulary_store.relations, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            (cache_directory / self.metadata_filename).write_text(
                json.dumps(self._build_metadata(dataset), indent=2, sort_keys=True),
                encoding="utf-8",
            )
        except MissingTorchDependencyException:
            raise
        except Exception as error:
            raise ProcessedDatasetStorageException(
                f"Failed to save processed WebQSP dataset cache: {error}"
            ) from error

    @staticmethod
    def _is_valid_metadata(
        metadata: dict[str, str | int],
        dataset_id: str,
        processing_version: str,
    ) -> bool:
        """Return whether metadata matches the requested processed dataset."""
        return (
            metadata.get("dataset_id") == dataset_id
            and metadata.get("processing_version") == processing_version
            and isinstance(metadata.get("train_size"), int)
            and isinstance(metadata.get("test_size"), int)
        )

    @staticmethod
    def _build_metadata(dataset: PreparedWebQSPLocalGraphDataset) -> dict[str, str | int]:
        """Build persisted metadata for a processed dataset."""
        return {
            "dataset_id": dataset.dataset_id,
            "processing_version": dataset.processing_version,
            "train_size": dataset.train_size,
            "test_size": dataset.test_size,
            "node_count": len(dataset.vocabulary_store.nodes),
            "relation_count": len(dataset.vocabulary_store.relations),
        }

    def _load_examples(self, path: Path) -> list[WebQSPLocalGraphExample]:
        """Load processed graph examples from a torch cache file."""
        torch_module = self._load_torch()
        try:
            return torch_module.load(path, weights_only=False)
        except TypeError:
            return torch_module.load(path)

    def _save_examples(
        self,
        path: Path,
        examples: list[WebQSPLocalGraphExample],
    ) -> None:
        """Save processed graph examples into a torch cache file."""
        torch_module = self._load_torch()
        torch_module.save(examples, path)

    @staticmethod
    def _load_torch():
        """Import PyTorch for processed tensor cache I/O."""
        try:
            import torch
        except ModuleNotFoundError as error:
            raise MissingTorchDependencyException(
                "PyTorch is required to load or save processed WebQSP graph tensors."
            ) from error

        return torch
