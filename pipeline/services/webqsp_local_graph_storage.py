"""Storage service for cached WebQSP processed graph artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from pipeline.exceptions import (
    ProcessedDatasetStorageException,
    UnsupportedDatasetProcessorException,
)
from pipeline.preparation.helpers.dataset_definitions import (
    DATASET_LOADERS,
    WEBQSP_DATASET_ID,
)
from constants import WEBQSP_QUESTION_VOCABULARY_FILENAME
from pipeline.preparation.models.webqsp_local_graph import (
    PreparedWebQSPGraphDataset,
    WebQSPProcessedInstance,
    WebQSPVocabularyStore,
)
from pipeline.services.abstract import AbstractService


class WebQSPLocalGraphStorageService(AbstractService):
    """Persist and load processed WebQSP graph artifacts."""

    metadata_filename = "metadata.json"
    train_instances_filename = "train_instances.pt"
    test_instances_filename = "test_instances.pt"
    nodes_filename = "nodes.json"
    relations_filename = "relations.json"
    questions_filename = WEBQSP_QUESTION_VOCABULARY_FILENAME

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
    ) -> PreparedWebQSPGraphDataset | None:
        """Load cached processed artifacts when cache metadata is valid."""
        cache_directory = self.get_cache_directory(dataset_id)
        metadata_path = cache_directory / self.metadata_filename
        train_instances_path = cache_directory / self.train_instances_filename
        test_instances_path = cache_directory / self.test_instances_filename
        nodes_path = cache_directory / self.nodes_filename
        relations_path = cache_directory / self.relations_filename
        questions_path = cache_directory / self.questions_filename

        required_paths = [
            metadata_path,
            train_instances_path,
            test_instances_path,
            nodes_path,
            relations_path,
            questions_path,
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

            train_instances = self._load_instances(train_instances_path)
            test_instances = self._load_instances(test_instances_path)
            vocabulary_store = WebQSPVocabularyStore(
                nodes=json.loads(nodes_path.read_text(encoding="utf-8")),
                relations=json.loads(relations_path.read_text(encoding="utf-8")),
                questions=json.loads(questions_path.read_text(encoding="utf-8")),
            )

            if metadata["train_size"] != len(train_instances):
                return None

            if metadata["test_size"] != len(test_instances):
                return None

            return PreparedWebQSPGraphDataset(
                dataset_id=dataset_id,
                processing_version=processing_version,
                train_instances=train_instances,
                test_instances=test_instances,
                vocabulary_store=vocabulary_store,
                cache_directory=cache_directory,
            )
        except Exception as error:
            raise ProcessedDatasetStorageException(
                f"Failed to load processed WebQSP dataset cache: {error}"
            ) from error

    def save(self, dataset: PreparedWebQSPGraphDataset) -> None:
        """Save processed instances, vocabularies, and metadata."""
        cache_directory = dataset.cache_directory
        cache_directory.mkdir(parents=True, exist_ok=True)

        try:
            self._save_instances(
                path=cache_directory / self.train_instances_filename,
                instances=dataset.train_instances,
            )
            self._save_instances(
                path=cache_directory / self.test_instances_filename,
                instances=dataset.test_instances,
            )
            (cache_directory / self.nodes_filename).write_text(
                json.dumps(dataset.vocabulary_store.nodes, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            (cache_directory / self.relations_filename).write_text(
                json.dumps(dataset.vocabulary_store.relations, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            (cache_directory / self.questions_filename).write_text(
                json.dumps(dataset.vocabulary_store.questions, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            (cache_directory / self.metadata_filename).write_text(
                json.dumps(self._build_metadata(dataset), indent=2, sort_keys=True),
                encoding="utf-8",
            )
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
    def _build_metadata(dataset: PreparedWebQSPGraphDataset) -> dict[str, str | int]:
        """Build persisted metadata for a processed dataset."""
        return {
            "dataset_id": dataset.dataset_id,
            "processing_version": dataset.processing_version,
            "train_size": dataset.train_size,
            "test_size": dataset.test_size,
            "node_count": len(dataset.vocabulary_store.nodes),
            "relation_count": len(dataset.vocabulary_store.relations),
            "question_count": len(dataset.vocabulary_store.questions),
        }

    def _load_instances(self, path: Path) -> list[WebQSPProcessedInstance]:
        """Load processed graph instances from a torch cache file."""
        import torch

        try:
            return torch.load(path, weights_only=False)
        except TypeError:
            return torch.load(path)

    def _save_instances(
        self,
        path: Path,
        instances: list[WebQSPProcessedInstance],
    ) -> None:
        """Save processed graph instances into a torch cache file."""
        import torch

        torch.save(instances, path)
