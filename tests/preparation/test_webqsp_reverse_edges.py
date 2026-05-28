"""Tests for optional WebQSP reverse-edge graph processing."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    torch = None

from pipeline.preparation.models.webqsp_local_graph import (
    PreparedWebQSPGraphDataset,
    WebQSPEntityMappingSummary,
    WebQSPVocabularyStore,
)
from pipeline.preparation.services.webqsp_local_graph_processing import (
    WebQSPLocalGraphProcessorService,
)
from pipeline.preparation.services.webqsp_local_graph_storage import (
    WebQSPLocalGraphStorageService,
)


class FakeEntityNameMappingService:
    def resolve_entity(self, entity: str) -> str:
        return entity

    def reset_summary(self) -> None:
        pass

    def build_summary(self) -> WebQSPEntityMappingSummary:
        return WebQSPEntityMappingSummary()


@unittest.skipIf(torch is None, "PyTorch is not installed.")
class WebQSPReverseEdgeProcessingTests(unittest.TestCase):
    def make_processor(self) -> WebQSPLocalGraphProcessorService:
        return WebQSPLocalGraphProcessorService(
            entity_name_mapping_service=FakeEntityNameMappingService()
        )

    def test_without_reverse_edges_keeps_original_edges(self) -> None:
        vocabulary = WebQSPVocabularyStore()

        instance = self.make_processor().process_row(
            row={
                "question": "q",
                "q_entity": ["A"],
                "a_entity": ["B"],
                "graph": [["A", "people.person.parents", "B"]],
            },
            vocabulary_store=vocabulary,
            use_reverse_edges=False,
        )

        self.assertEqual(instance.edge_relations, ["people.person.parents"])
        self.assertEqual(instance.edge_index.tolist(), [[0], [1]])
        self.assertNotIn("reverse__people.person.parents", vocabulary.relations)

    def test_reverse_edges_double_edges_and_add_reverse_relation(self) -> None:
        vocabulary = WebQSPVocabularyStore()

        instance = self.make_processor().process_row(
            row={
                "question": "q",
                "q_entity": ["A"],
                "a_entity": ["B"],
                "graph": [["A", "people.person.parents", "B"]],
            },
            vocabulary_store=vocabulary,
            use_reverse_edges=True,
        )

        self.assertEqual(
            instance.edge_relations,
            ["people.person.parents", "reverse__people.person.parents"],
        )
        self.assertEqual(instance.edge_index.tolist(), [[0, 1], [1, 0]])
        self.assertIn("people.person.parents", vocabulary.relations)
        self.assertIn("reverse__people.person.parents", vocabulary.relations)

    def test_reverse_edge_cache_variant_does_not_match_normal_cache(self) -> None:
        storage = WebQSPLocalGraphStorageService()
        with tempfile.TemporaryDirectory() as temporary_directory:
            cache_directory = Path(temporary_directory)
            dataset = PreparedWebQSPGraphDataset(
                dataset_id="WebQSP",
                processing_version="test",
                use_reverse_edges=True,
                train_instances=[],
                test_instances=[],
                vocabulary_store=WebQSPVocabularyStore(),
                cache_directory=cache_directory,
            )
            storage.save(dataset)
            metadata = {
                "dataset_id": "WebQSP",
                "processing_version": "test",
                "use_reverse_edges": True,
                "train_size": 0,
                "test_size": 0,
            }

        self.assertTrue(
            storage._is_valid_metadata(
                metadata=metadata,
                dataset_id="WebQSP",
                processing_version="test",
                use_reverse_edges=True,
            )
        )
        self.assertFalse(
            storage._is_valid_metadata(
                metadata=metadata,
                dataset_id="WebQSP",
                processing_version="test",
                use_reverse_edges=False,
            )
        )


if __name__ == "__main__":
    unittest.main()
