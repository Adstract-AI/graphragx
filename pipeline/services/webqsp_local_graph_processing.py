"""Services for converting WebQSP examples into local graph tensors."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pipeline.exceptions import (
    MalformedWebQSPExampleException,
    MissingTorchDependencyException,
    UnsupportedDatasetProcessorException,
)
from pipeline.preparation.helpers.dataset_definitions import WEBQSP_DATASET_ID
from pipeline.preparation.models.webqsp_local_graph import (
    PreparedWebQSPLocalGraphDataset,
    WebQSPLocalGraphExample,
    WebQSPVocabularyStore,
)
from pipeline.services.abstract import AbstractService

if TYPE_CHECKING:
    from torch import Tensor as TorchTensor
    from pipeline.preparation.steps.dataset_loading import LoadedDataset
else:
    TorchTensor = Any


class WebQSPLocalGraphProcessorService(AbstractService):
    """Convert loaded WebQSP rows into trainable local graph examples."""

    def process_loaded_dataset(
        self,
        loaded_dataset: LoadedDataset,
        processing_version: str,
        cache_directory: Path,
    ) -> PreparedWebQSPLocalGraphDataset:
        """Process WebQSP train, validation, and test splits."""
        if loaded_dataset.dataset_id != WEBQSP_DATASET_ID:
            raise UnsupportedDatasetProcessorException(
                f"Unsupported dataset processor for dataset: {loaded_dataset.dataset_id}"
            )

        vocabulary_store = WebQSPVocabularyStore()
        train_examples = self._process_split(
            rows=loaded_dataset.hugging_face_dataset["train"],
            vocabulary_store=vocabulary_store,
        )
        test_rows = self._combined_test_rows(loaded_dataset.hugging_face_dataset)
        test_examples = self._process_split(
            rows=test_rows,
            vocabulary_store=vocabulary_store,
        )

        return PreparedWebQSPLocalGraphDataset(
            dataset_id=loaded_dataset.dataset_id,
            processing_version=processing_version,
            train_examples=train_examples,
            test_examples=test_examples,
            vocabulary_store=vocabulary_store,
            cache_directory=cache_directory,
        )

    def _process_split(
        self,
        rows: Iterable[Mapping[str, Any]],
        vocabulary_store: WebQSPVocabularyStore,
    ) -> list[WebQSPLocalGraphExample]:
        """Process every row from one logical split."""
        return [
            self.process_row(row=row, vocabulary_store=vocabulary_store)
            for row in rows
        ]

    def process_row(
        self,
        row: Mapping[str, Any],
        vocabulary_store: WebQSPVocabularyStore,
    ) -> WebQSPLocalGraphExample:
        """Convert one WebQSP row into a local graph example."""
        self._validate_row(row)

        question = str(row["question"])
        q_entity = self._normalize_string_list(row["q_entity"], "q_entity")
        a_entity = self._normalize_string_list(row["a_entity"], "a_entity")
        triples = self._normalize_triples(row["graph"])

        nodes: list[str] = []
        node2id: dict[str, int] = {}
        edge_relations: list[str] = []
        edge_sources: list[int] = []
        edge_targets: list[int] = []

        for head, relation, tail in triples:
            head_id = self._get_or_add_local_node(head, nodes, node2id)
            tail_id = self._get_or_add_local_node(tail, nodes, node2id)
            self._get_or_add_vocabulary_item(head, vocabulary_store.nodes)
            self._get_or_add_vocabulary_item(tail, vocabulary_store.nodes)
            self._get_or_add_vocabulary_item(relation, vocabulary_store.relations)

            edge_sources.append(head_id)
            edge_targets.append(tail_id)
            edge_relations.append(relation)

        torch_module = self._load_torch()
        edge_index = torch_module.tensor(
            [edge_sources, edge_targets],
            dtype=torch_module.long,
        )
        answer_entities = set(a_entity)
        node_labels = torch_module.tensor(
            [1.0 if node in answer_entities else 0.0 for node in nodes],
            dtype=torch_module.float,
        )

        return WebQSPLocalGraphExample(
            question=question,
            q_entity=q_entity,
            a_entity=a_entity,
            nodes=nodes,
            node2id=node2id,
            edge_index=edge_index,
            edge_relations=edge_relations,
            node_labels=node_labels,
        )

    @staticmethod
    def _combined_test_rows(dataset: Mapping[str, Iterable[Mapping[str, Any]]]) -> list[Mapping[str, Any]]:
        """Return validation and test rows as one evaluation split."""
        rows: list[Mapping[str, Any]] = []
        rows.extend(dataset["validation"])
        rows.extend(dataset["test"])
        return rows

    @staticmethod
    def _validate_row(row: Mapping[str, Any]) -> None:
        """Validate the required WebQSP row fields."""
        required_fields = ["question", "q_entity", "a_entity", "graph"]
        missing_fields = [field for field in required_fields if field not in row]
        if missing_fields:
            raise MalformedWebQSPExampleException(
                f"WebQSP row is missing fields: {', '.join(missing_fields)}"
            )

    @staticmethod
    def _normalize_string_list(value: Any, field_name: str) -> list[str]:
        """Normalize WebQSP entity fields into string lists."""
        if isinstance(value, str):
            return [value]

        if not isinstance(value, Sequence):
            raise MalformedWebQSPExampleException(
                f"WebQSP field {field_name} must be a string or sequence of strings."
            )

        return [str(item) for item in value]

    @staticmethod
    def _normalize_triples(value: Any) -> list[tuple[str, str, str]]:
        """Normalize WebQSP graph triples into typed string tuples."""
        if not isinstance(value, Sequence):
            raise MalformedWebQSPExampleException(
                "WebQSP field graph must be a sequence of triples."
            )

        triples: list[tuple[str, str, str]] = []
        for triple in value:
            if not isinstance(triple, Sequence) or len(triple) != 3:
                raise MalformedWebQSPExampleException(
                    "Every WebQSP graph item must be a triple."
                )

            triples.append((str(triple[0]), str(triple[1]), str(triple[2])))

        return triples

    @staticmethod
    def _get_or_add_local_node(
        node: str,
        nodes: list[str],
        node2id: dict[str, int],
    ) -> int:
        """Return a local node id, adding the node when needed."""
        if node not in node2id:
            node2id[node] = len(nodes)
            nodes.append(node)

        return node2id[node]

    @staticmethod
    def _get_or_add_vocabulary_item(item: str, vocabulary: dict[str, int]) -> int:
        """Return a vocabulary id, adding the item when needed."""
        if item not in vocabulary:
            vocabulary[item] = len(vocabulary)

        return vocabulary[item]

    @staticmethod
    def _load_torch():
        """Import PyTorch for tensor construction."""
        try:
            import torch
        except ModuleNotFoundError as error:
            raise MissingTorchDependencyException(
                "PyTorch is required to prepare WebQSP local graph tensors."
            ) from error

        return torch
