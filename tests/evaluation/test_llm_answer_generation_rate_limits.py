"""Tests for visible LLM answer-generation rate-limit retry logs."""

import unittest
from unittest.mock import patch

from pipeline.evaluation.services.llm_answer_generation import (
    LangChainOpenAiAnswerGenerationService,
)


class FakeResponse:
    status_code = 429
    headers = {"retry-after": "0"}


class FakeRateLimitError(Exception):
    response = FakeResponse()


class FakeChatModel:
    def __init__(self):
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        if self.calls == 1:
            raise FakeRateLimitError("chat rate limited")
        return "ok"


class LlmAnswerGenerationRateLimitTests(unittest.TestCase):
    def test_rate_limit_is_logged_and_retried(self) -> None:
        service = LangChainOpenAiAnswerGenerationService()
        chat_model = FakeChatModel()

        with patch("time.sleep") as sleep, self.assertLogs(
            "pipeline.evaluation.services.llm_answer_generation",
            level="WARNING",
        ) as logs:
            result = service._invoke_with_visible_rate_limit_retries(
                chat_model=chat_model,
                messages=[],
                model_id="gpt-4.1-nano",
                prompt="question",
            )

        self.assertEqual(result, "ok")
        self.assertEqual(chat_model.calls, 2)
        sleep.assert_called_once_with(0.0)
        self.assertTrue(
            any(
                "OpenAI rate limit hit" in message
                and "operation=llm_answer_generation" in message
                and "\033[91m" in message
                for message in logs.output
            )
        )


if __name__ == "__main__":
    unittest.main()
