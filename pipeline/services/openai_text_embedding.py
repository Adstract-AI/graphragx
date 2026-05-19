"""OpenAI text embedding service used by training-time preparation steps."""

from __future__ import annotations

from constants import OPENAI_API_KEY_ENV_NAME
from env_variables import OPENAI_API_KEY
from logging_config import get_logger
from pipeline.exceptions import OpenAiEmbeddingConfigurationException
from pipeline.services.abstract import AbstractService

logger = get_logger(__name__)


class LangChainOpenAiTextEmbeddingService(AbstractService):
    """Create text embeddings through LangChain's OpenAI embedding integration."""

    def embed_texts(self, texts: list[str], model_id: str) -> dict[str, list[float]]:
        """Embed a list of unique texts and return vectors keyed by input text."""
        if not texts:
            return {}

        if not OPENAI_API_KEY:
            raise OpenAiEmbeddingConfigurationException(
                f"{OPENAI_API_KEY_ENV_NAME} must be set in .env before creating OpenAI embeddings."
            )

        from langchain_openai import OpenAIEmbeddings

        logger.info(
            f"Calling OpenAI embedding endpoint: model={model_id} "
            f"text_count={len(texts)} preview={self._preview_texts(texts)}"
        )
        embedding_client = OpenAIEmbeddings(model=model_id)
        vectors = embedding_client.embed_documents(texts)
        logger.info(
            f"Received OpenAI embeddings: model={model_id} vector_count={len(vectors)}"
        )
        return dict(zip(texts, vectors, strict=True))

    @staticmethod
    def _preview_texts(texts: list[str]) -> list[str]:
        """Return a short log-safe preview of the texts being embedded."""
        return [
            text[:117] + "..." if len(text) > 120 else text
            for text in texts[:3]
        ]
