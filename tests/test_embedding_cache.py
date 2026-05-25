"""Tests for persistent embedding cache behavior."""

from pathlib import Path
import tempfile
import unittest

from pipeline.preparation.services.embedding_cache import (
    TextEmbeddingCache,
    WebQSPEmbeddingCacheService,
)


class FakeEmbeddingService:
    def embed_texts(self, texts: list[str], model_id: str) -> dict[str, list[float]]:
        return {
            text: [float(index), 1.0]
            for index, text in enumerate(texts)
        }


class RecordingEmbeddingCacheService(WebQSPEmbeddingCacheService):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.saved_embedding_counts: list[int] = []

    def save_cache(self, cache: TextEmbeddingCache) -> None:
        self.saved_embedding_counts.append(len(cache.embeddings))


class EmbeddingCacheTests(unittest.TestCase):
    def test_embedding_cache_saves_every_configured_batches_and_final_batch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            service = RecordingEmbeddingCacheService(
                embedding_service=FakeEmbeddingService(),
                batch_size=2,
                save_every_batches=3,
            )
            cache = TextEmbeddingCache(
                model_id="text-embedding-3-small",
                cache_kind="nodes",
                vocabulary={},
                embeddings={},
                embedding_path=Path(temporary_directory) / "nodes.pt",
            )

            service.ensure_embeddings(
                cache=cache,
                texts=[f"node {index}" for index in range(10)],
            )

        self.assertEqual(len(cache.embeddings), 10)
        self.assertEqual(service.saved_embedding_counts, [6, 10])

    def test_embedding_cache_rejects_non_positive_save_interval(self) -> None:
        with self.assertRaisesRegex(ValueError, "save interval"):
            WebQSPEmbeddingCacheService(save_every_batches=0)


if __name__ == "__main__":
    unittest.main()
