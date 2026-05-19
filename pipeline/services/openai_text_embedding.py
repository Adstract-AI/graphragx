"""OpenAI text embedding service used by training-time preparation steps."""

from __future__ import annotations

import os
import logging

from pipeline.exceptions import OpenAiEmbeddingConfigurationException
from pipeline.services.abstract import AbstractService

logger = logging.getLogger(__name__)


class LangChainOpenAiTextEmbeddingService(AbstractService):
    """Create text embeddings through LangChain's OpenAI embedding integration."""

    def embed_texts(self, texts: list[str], model_id: str) -> dict[str, list[float]]:
        """Embed a list of unique texts and return vectors keyed by input text."""
        if not texts:
            return {}

        from dotenv import load_dotenv

        load_dotenv()
        if not os.getenv("OPENAI_API_KEY"):
            raise OpenAiEmbeddingConfigurationException(
                "OPENAI_API_KEY must be set in .env before creating OpenAI embeddings."
            )

        from langchain_openai import OpenAIEmbeddings

        logger.info(
            "Calling OpenAI embedding endpoint: model=%s text_count=%s preview=%s",
            model_id,
            len(texts),
            self._preview_texts(texts),
        )
        embedding_client = OpenAIEmbeddings(model=model_id)
        vectors = embedding_client.embed_documents(texts)
        logger.info(
            "Received OpenAI embeddings: model=%s vector_count=%s",
            model_id,
            len(vectors),
        )
        return dict(zip(texts, vectors, strict=True))

    @staticmethod
    def _preview_texts(texts: list[str]) -> list[str]:
        """Return a short log-safe preview of the texts being embedded."""
        return [
            text[:117] + "..." if len(text) > 120 else text
            for text in texts[:3]
        ]
