"""Tests for the legacy .pt-to-Qdrant embedding migration helpers."""

import unittest

from pipeline.preparation.services.embedding_cache import (
    TextEmbeddingCache,
    WebQSPEmbeddingCacheService,
)
from scripts.migrate_pt_embeddings_to_qdrant import invert_vocabulary, migrate_kind
from tests.test_embedding_cache import FakeEmbeddingService, FakeQdrantClient


class MigrationTests(unittest.TestCase):
    def test_invert_vocabulary(self) -> None:
        self.assertEqual(invert_vocabulary({"Paris": 3}), {3: "Paris"})

    def test_migrate_kind_skips_existing_and_upserts_missing(self) -> None:
        qdrant_client = FakeQdrantClient()
        service = WebQSPEmbeddingCacheService(
            embedding_service=FakeEmbeddingService(),
            qdrant_client=qdrant_client,
        )
        cache = TextEmbeddingCache(
            dataset_id="WebQSP",
            model_id="text-embedding-3-small",
            cache_kind="nodes",
            vocabulary={"old": 0, "new": 1},
            collection_name="graphragx_embeddings_text_embedding_3_small",
            vector_size=1536,
        )
        existing_id = service.point_id(cache=cache, text="old")
        qdrant_client.points[existing_id] = {
            "id": existing_id,
            "vector": [9.0, 9.0],
            "payload": {"text": "old"},
        }

        result = migrate_kind(
            service=service,
            cache=cache,
            legacy_embeddings={0: [9.0, 9.0], 1: [1.0, 1.0], 99: [0.0, 0.0]},
            id_to_text={0: "old", 1: "new"},
            preprocess=False,
        )

        self.assertEqual(result["legacy_embeddings"], 3)
        self.assertEqual(result["matched_texts"], 2)
        self.assertEqual(result["already_present"], 1)
        self.assertEqual(result["migrated"], 1)
        self.assertEqual(result["skipped_missing_text"], 1)
        self.assertEqual(len(qdrant_client.upserts), 1)
        self.assertEqual(qdrant_client.upserts[0][0]["payload"]["text"], "new")


if __name__ == "__main__":
    unittest.main()
