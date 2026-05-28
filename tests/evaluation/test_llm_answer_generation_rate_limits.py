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


class CapturingChatOpenAI:
    captured_kwargs = None

    def __init__(self, **kwargs):
        self.__class__.captured_kwargs = kwargs


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

    def test_deepseek_chat_model_uses_deepseek_key_and_base_url(self) -> None:
        service = LangChainOpenAiAnswerGenerationService()

        model = service._create_chat_model(
            chat_openai_type=CapturingChatOpenAI,
            model_id="deepseek-v4-flash",
            prompt="question",
            api_key="deepseek-key",
            base_url="https://api.deepseek.com",
        )

        self.assertIsInstance(model, CapturingChatOpenAI)
        self.assertEqual(
            CapturingChatOpenAI.captured_kwargs["model"],
            "deepseek-v4-flash",
        )
        self.assertEqual(CapturingChatOpenAI.captured_kwargs["api_key"], "deepseek-key")
        self.assertEqual(
            CapturingChatOpenAI.captured_kwargs["base_url"],
            "https://api.deepseek.com",
        )
        self.assertEqual(CapturingChatOpenAI.captured_kwargs["timeout"], 45.0)
        self.assertNotIn("model_kwargs", CapturingChatOpenAI.captured_kwargs)

    def test_openai_chat_model_uses_timeout_and_json_response_format(self) -> None:
        service = LangChainOpenAiAnswerGenerationService()

        model = service._create_chat_model(
            chat_openai_type=CapturingChatOpenAI,
            model_id="gpt-4.1-nano",
            prompt="question",
            api_key="openai-key",
        )

        self.assertIsInstance(model, CapturingChatOpenAI)
        self.assertEqual(CapturingChatOpenAI.captured_kwargs["timeout"], 45.0)
        self.assertEqual(
            CapturingChatOpenAI.captured_kwargs["model_kwargs"],
            {"response_format": {"type": "json_object"}},
        )

    def test_deepseek_missing_api_key_mentions_deepseek_env_name(self) -> None:
        service = LangChainOpenAiAnswerGenerationService()

        with patch(
            "pipeline.evaluation.services.llm_answer_generation.DEEPSEEK_API_KEY",
            None,
        ), self.assertRaisesRegex(Exception, "DEEPSEEK_API_KEY"):
            service.generate_answer_with_explanation(
                question="q",
                reasoning_paths_text="paths",
                model_id="deepseek-v4-pro",
            )

    def test_deepseek_cost_estimation_uses_deepseek_prices(self) -> None:
        cost = LangChainOpenAiAnswerGenerationService.estimate_cost_usd(
            model_id="deepseek-v4-pro",
            prompt_tokens=1_000_000,
            completion_tokens=1_000_000,
        )

        self.assertEqual(cost, 1.305)


if __name__ == "__main__":
    unittest.main()
