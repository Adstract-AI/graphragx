"""Tests for OpenAI embedding service retry behavior."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from pipeline.preparation.services.openai_text_embedding import (
    LangChainOpenAiTextEmbeddingService,
)
from helpers.openai_rate_limit_logging import log_openai_rate_limit_response
from helpers.logging_config import get_logger


class FakeResponse:
    status_code = 429
    headers = {"retry-after": "0"}


class FakeRateLimitError(Exception):
    response = FakeResponse()


class FakeEmbeddingClient:
    def __init__(self):
        self.calls = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        if self.calls == 1:
            raise FakeRateLimitError("rate limited")
        return [[1.0, 2.0] for _ in texts]


class OpenAiTextEmbeddingTests(unittest.TestCase):
    def test_rate_limit_is_logged_and_retried(self) -> None:
        service = LangChainOpenAiTextEmbeddingService()
        client = FakeEmbeddingClient()

        with patch("time.sleep") as sleep, self.assertLogs(
            "pipeline.preparation.services.openai_text_embedding",
            level="WARNING",
        ) as logs:
            vectors = service._embed_documents_with_visible_rate_limit_retries(
                embedding_client=client,
                texts=["one", "two"],
                model_id="text-embedding-3-small",
            )

        self.assertEqual(vectors, [[1.0, 2.0], [1.0, 2.0]])
        self.assertEqual(client.calls, 2)
        sleep.assert_called_once_with(0.0)
        self.assertTrue(
            any(
                "OpenAI rate limit hit" in message
                and "\033[91m" in message
                and "cooldown=0s" in message
                for message in logs.output
            )
        )

    def test_rate_limit_headers_are_quiet_for_successful_response_with_remaining_quota(
        self,
    ) -> None:
        response = SimpleNamespace(
            status_code=200,
            headers={
                "x-ratelimit-limit-requests": "60",
                "x-ratelimit-remaining-requests": "59",
                "x-ratelimit-reset-requests": "1s",
                "x-ratelimit-limit-tokens": "150000",
                "x-ratelimit-remaining-tokens": "149984",
                "x-ratelimit-reset-tokens": "6m0s",
            },
        )

        with self.assertNoLogs(
            "pipeline.preparation.services.openai_text_embedding",
            level="INFO",
        ):
            log_openai_rate_limit_response(
                response=response,
                logger=get_logger("pipeline.preparation.services.openai_text_embedding"),
                operation="embeddings",
                model_id="text-embedding-3-small",
                item_count=1024,
            )

    def test_rate_limit_headers_are_logged_from_exhausted_successful_response(
        self,
    ) -> None:
        response = SimpleNamespace(
            status_code=200,
            headers={
                "x-ratelimit-limit-requests": "60",
                "x-ratelimit-remaining-requests": "0",
                "x-ratelimit-reset-requests": "1s",
                "x-ratelimit-limit-tokens": "150000",
                "x-ratelimit-remaining-tokens": "42",
                "x-ratelimit-reset-tokens": "6m0s",
            },
        )

        with self.assertLogs(
            "pipeline.preparation.services.openai_text_embedding",
            level="WARNING",
        ) as logs:
            log_openai_rate_limit_response(
                response=response,
                logger=get_logger("pipeline.preparation.services.openai_text_embedding"),
                operation="embeddings",
                model_id="text-embedding-3-small",
                item_count=1024,
            )

        self.assertTrue(
            any(
                "OpenAI rate limit hit" in message
                and "cooldown=requests=1s, tokens=6m0s" in message
                for message in logs.output
            )
        )

    def test_rate_limit_headers_are_logged_from_429_response(self) -> None:
        response = SimpleNamespace(
            status_code=429,
            headers={
                "x-ratelimit-remaining-requests": "0",
                "x-ratelimit-reset-requests": "20s",
                "x-ratelimit-remaining-tokens": "0",
                "x-ratelimit-reset-tokens": "20s",
                "retry-after": "20",
            },
        )

        with self.assertLogs(
            "pipeline.preparation.services.openai_text_embedding",
            level="WARNING",
        ) as logs:
            log_openai_rate_limit_response(
                response=response,
                logger=get_logger("pipeline.preparation.services.openai_text_embedding"),
                operation="embeddings",
                model_id="text-embedding-3-small",
                item_count=1024,
            )

        self.assertTrue(
            any(
                "OpenAI rate limit hit" in message
                and "cooldown=20s" in message
                for message in logs.output
            )
        )


if __name__ == "__main__":
    unittest.main()
