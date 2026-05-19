"""Persistent embedding cache services for WebQSP training artifacts."""

from __future__ import annotations

import json
import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from constants import (
    DEFAULT_EMBEDDING_BATCH_SIZE,
    WEBQSP_NODE_EMBEDDINGS_FILENAME,
    WEBQSP_QUESTION_EMBEDDINGS_FILENAME,
    WEBQSP_RELATION_EMBEDDINGS_FILENAME,
)
from logging_config import get_logger
from pipeline.services.abstract import AbstractService
from pipeline.services.openai_text_embedding import LangChainOpenAiTextEmbeddingService

logger = get_logger(__name__)


class TextEmbeddingCache(BaseModel):
    """Loaded embedding cache for one text category and embedding model."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_id: str = Field(..., description="Embedding model id used for this cache.")
    cache_kind: str = Field(..., description="Text category stored in this cache.")
    vocabulary: dict[str, int] = Field(default_factory=dict)
    embeddings: dict[int, list[float]] = Field(default_factory=dict)
    embedding_path: Path = Field(..., description="Torch cache path for vectors.")
    vocabulary_path: Path | None = Field(
        default=None,
        description="JSON vocabulary path when the vocabulary is cache-owned.",
    )


class WebQSPEmbeddingCacheService(AbstractService):
    """Load, populate, and save WebQSP embedding caches."""

    nodes_filename = WEBQSP_NODE_EMBEDDINGS_FILENAME
    relations_filename = WEBQSP_RELATION_EMBEDDINGS_FILENAME
    questions_filename = WEBQSP_QUESTION_EMBEDDINGS_FILENAME
    default_batch_size = DEFAULT_EMBEDDING_BATCH_SIZE

    def __init__(
        self,
        embedding_service: LangChainOpenAiTextEmbeddingService | None = None,
        batch_size: int = default_batch_size,
    ):
        self.embedding_service = embedding_service or LangChainOpenAiTextEmbeddingService()
        if batch_size <= 0:
            raise ValueError("Embedding cache batch size must be greater than zero.")

        self.batch_size = batch_size

    def load_node_cache(
        self,
        cache_root: Path,
        model_id: str,
        vocabulary: dict[str, int],
    ) -> TextEmbeddingCache:
        """Load the entity/node embedding cache for a model."""
        return self._load_cache(
            cache_root=cache_root,
            model_id=model_id,
            cache_kind="nodes",
            embedding_filename=self.nodes_filename,
            vocabulary=vocabulary,
            vocabulary_path=None,
        )

    def load_relation_cache(
        self,
        cache_root: Path,
        model_id: str,
        vocabulary: dict[str, int],
    ) -> TextEmbeddingCache:
        """Load the relation embedding cache for a model."""
        return self._load_cache(
            cache_root=cache_root,
            model_id=model_id,
            cache_kind="relations",
            embedding_filename=self.relations_filename,
            vocabulary=vocabulary,
            vocabulary_path=None,
        )

    def load_question_cache(
        self,
        cache_root: Path,
        model_id: str,
        vocabulary: dict[str, int],
    ) -> TextEmbeddingCache:
        """Load the training-question embedding cache for a model."""
        return self._load_cache(
            cache_root=cache_root,
            model_id=model_id,
            cache_kind="questions",
            embedding_filename=self.questions_filename,
            vocabulary=vocabulary,
            vocabulary_path=None,
        )

    def ensure_embeddings(
        self,
        cache: TextEmbeddingCache,
        texts: list[str],
        preprocess: bool = False,
    ) -> None:
        """Populate missing embeddings for the requested texts."""
        missing_texts = self._missing_texts(cache, texts)
        if not missing_texts:
            logger.info(
                f"Embedding cache hit for {cache.cache_kind}/{cache.model_id}: "
                f"requested={len(list(dict.fromkeys(texts)))} "
                f"cached={len(cache.embeddings)} missing=0"
            )
            return

        total_batches = (len(missing_texts) + self.batch_size - 1) // self.batch_size
        logger.info(
            f"Embedding cache fill for {cache.cache_kind}/{cache.model_id}: "
            f"requested={len(list(dict.fromkeys(texts)))} "
            f"cached={len(cache.embeddings)} missing={len(missing_texts)} "
            f"batch_size={self.batch_size} preview={self._preview_texts(missing_texts)}"
        )
        for batch_index, start_index in enumerate(
            range(0, len(missing_texts), self.batch_size),
            start=1,
        ):
            batch_texts = missing_texts[start_index : start_index + self.batch_size]
            embedding_inputs = [
                self.preprocess_relation_text(text) if preprocess else text
                for text in batch_texts
            ]
            logger.info(
                f"Embedding {cache.cache_kind}/{cache.model_id} batch "
                f"{batch_index}/{total_batches}: "
                f"original_text_count={len(batch_texts)} "
                f"endpoint_text_count={len(embedding_inputs)} "
                f"preview={self._preview_texts(batch_texts)}"
            )
            embedded_inputs = self.embedding_service.embed_texts(
                texts=embedding_inputs,
                model_id=cache.model_id,
            )
            for original_text, embedding_input in zip(
                batch_texts,
                embedding_inputs,
                strict=True,
            ):
                text_id = self._get_or_add_text_id(cache, original_text)
                cache.embeddings[text_id] = embedded_inputs[embedding_input]

            self.save_cache(cache)
            logger.info(
                f"Saved {cache.cache_kind}/{cache.model_id} embedding cache after "
                f"batch {batch_index}/{total_batches}: "
                f"cached={len(cache.embeddings)} path={cache.embedding_path}"
            )

    def embedding_for_text(
        self,
        cache: TextEmbeddingCache,
        text: str,
    ) -> list[float]:
        """Return an already-cached embedding for a text value."""
        return cache.embeddings[cache.vocabulary[text]]

    def save_cache(self, cache: TextEmbeddingCache) -> None:
        """Persist one embedding cache."""
        import torch

        cache.embedding_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_embedding_path = cache.embedding_path.with_suffix(
            f"{cache.embedding_path.suffix}.tmp"
        )
        torch.save(cache.embeddings, temporary_embedding_path)
        temporary_embedding_path.replace(cache.embedding_path)
        if cache.vocabulary_path is not None:
            temporary_vocabulary_path = cache.vocabulary_path.with_suffix(
                f"{cache.vocabulary_path.suffix}.tmp"
            )
            temporary_vocabulary_path.write_text(
                json.dumps(cache.vocabulary, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            temporary_vocabulary_path.replace(cache.vocabulary_path)

    @staticmethod
    def preprocess_relation_text(relation: str) -> str:
        """Normalize relation identifiers before embedding."""
        normalized = re.sub(r"[._/]+", " ", relation)
        return " ".join(normalized.split())

    def _load_cache(
        self,
        cache_root: Path,
        model_id: str,
        cache_kind: str,
        embedding_filename: str,
        vocabulary: dict[str, int],
        vocabulary_path: Path | None,
    ) -> TextEmbeddingCache:
        embedding_path = self._model_cache_directory(cache_root, model_id) / embedding_filename
        return TextEmbeddingCache(
            model_id=model_id,
            cache_kind=cache_kind,
            vocabulary=dict(vocabulary),
            embeddings=self._load_embeddings(embedding_path),
            embedding_path=embedding_path,
            vocabulary_path=vocabulary_path,
        )

    @staticmethod
    def _load_embeddings(path: Path) -> dict[int, list[float]]:
        if not path.exists():
            return {}

        import torch

        try:
            loaded_embeddings = torch.load(path, weights_only=False)
        except Exception as error:
            logger.warning(
                f"Ignoring unreadable embedding cache at {path}. "
                f"It will be rebuilt. Error: {error}"
            )
            return {}

        return {
            int(text_id): list(vector)
            for text_id, vector in loaded_embeddings.items()
        }

    @staticmethod
    def _model_cache_directory(cache_root: Path, model_id: str) -> Path:
        return cache_root / "embeddings" / model_id

    def _missing_texts(
        self,
        cache: TextEmbeddingCache,
        texts: list[str],
    ) -> list[str]:
        unique_texts = list(dict.fromkeys(texts))
        return [
            text
            for text in unique_texts
            if self._get_or_add_text_id(cache, text) not in cache.embeddings
        ]

    @staticmethod
    def _get_or_add_text_id(cache: TextEmbeddingCache, text: str) -> int:
        if text not in cache.vocabulary:
            cache.vocabulary[text] = len(cache.vocabulary)

        return cache.vocabulary[text]

    @staticmethod
    def _preview_texts(texts: list[str]) -> list[str]:
        return [
            text[:117] + "..." if len(text) > 120 else text
            for text in texts[:3]
        ]
