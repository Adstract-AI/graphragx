"""OpenAI text embedding service used by training-time preparation steps."""

from __future__ import annotations

import time
from typing import Any

from helpers.constants import OPENAI_API_KEY_ENV_NAME
from helpers.env_variables import OPENAI_API_KEY
from helpers.logging_config import get_logger
from helpers.openai_rate_limit_logging import (
    create_rate_limit_logging_http_client,
    format_rate_limit_retry_message,
    is_openai_rate_limit_error,
    rate_limit_wait_seconds,
)
from pipeline.preparation.exceptions import OpenAiEmbeddingConfigurationException
from pipeline.services import AbstractService

logger = get_logger(__name__)


class LangChainOpenAiTextEmbeddingService(AbstractService):
    """Create text embeddings through LangChain's OpenAI embedding integration."""

    max_rate_limit_retries = 8
    default_rate_limit_wait_seconds = 30.0
    max_rate_limit_wait_seconds = 120.0

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
        embedding_client = self._create_embedding_client(
            OpenAIEmbeddings,
            model_id,
            text_count=len(texts),
        )
        vectors = self._embed_documents_with_visible_rate_limit_retries(
            embedding_client=embedding_client,
            texts=texts,
            model_id=model_id,
        )
        logger.info(
            f"Received OpenAI embeddings: model={model_id} vector_count={len(vectors)}"
        )
        return dict(zip(texts, vectors, strict=True))

    def _embed_documents_with_visible_rate_limit_retries(
        self,
        embedding_client: Any,
        texts: list[str],
        model_id: str,
    ) -> list[list[float]]:
        for attempt_number in range(1, self.max_rate_limit_retries + 1):
            try:
                return embedding_client.embed_documents(texts)
            except Exception as error:
                if not is_openai_rate_limit_error(error):
                    raise

                wait_seconds = rate_limit_wait_seconds(
                    error=error,
                    attempt_number=attempt_number,
                    default_wait_seconds=self.default_rate_limit_wait_seconds,
                    max_wait_seconds=self.max_rate_limit_wait_seconds,
                )
                logger.warning(
                    format_rate_limit_retry_message(
                        operation="embeddings",
                        model_id=model_id,
                        item_count=len(texts),
                        attempt_number=attempt_number,
                        max_attempts=self.max_rate_limit_retries,
                        wait_seconds=wait_seconds,
                        error=error,
                    )
                )
                time.sleep(wait_seconds)

        return embedding_client.embed_documents(texts)

    @staticmethod
    def _create_embedding_client(
        openai_embeddings_type: Any,
        model_id: str,
        text_count: int,
    ) -> Any:
        http_client = create_rate_limit_logging_http_client(
            logger=logger,
            operation="embeddings",
            model_id=model_id,
            item_count=text_count,
        )
        try:
            return openai_embeddings_type(
                model=model_id,
                max_retries=0,
                http_client=http_client,
            )
        except TypeError:
            logger.warning(
                "Current LangChain OpenAIEmbeddings does not support max_retries=0 "
                "or custom http_client; OpenAI SDK retries/rate-limit headers may "
                "remain hidden in logs."
            )
            return openai_embeddings_type(model=model_id)

    @staticmethod
    def _preview_texts(texts: list[str]) -> list[str]:
        """Return a short log-safe preview of the texts being embedded."""
        return [
            text[:117] + "..." if len(text) > 120 else text
            for text in texts[:3]
        ]
