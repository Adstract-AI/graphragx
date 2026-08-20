"""LangChain-backed final answer generation from reasoning paths."""

from __future__ import annotations

import json
import re
import time
from typing import Any

from helpers.constants import (
    DEEPSEEK_API_KEY_ENV_NAME,
    DEFAULT_VEZILKA_BASE_URL,
    OPENAI_API_KEY_ENV_NAME,
    VEZILKA_API_KEY_ENV_NAME,
)
from helpers.env_variables import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    OPENAI_API_KEY,
    VEZILKA_API_KEY,
)
from helpers.logging_config import get_logger
from helpers.openai_rate_limit_logging import (
    create_rate_limit_logging_http_client,
    format_rate_limit_retry_message,
    is_openai_rate_limit_error,
    rate_limit_wait_seconds,
)
from pipeline.evaluation.exceptions import LlmAnswerGenerationException
from pipeline.services import AbstractService

logger = get_logger(__name__)


class LangChainOpenAiAnswerGenerationService(AbstractService):
    """Generate final QA answers with a simple OpenAI chat model."""

    max_rate_limit_retries = 8
    default_rate_limit_wait_seconds = 30.0
    max_rate_limit_wait_seconds = 120.0
    request_timeout_seconds = 45.0
    slow_request_warning_seconds = 30.0
    max_completion_tokens = 1024
    deepseek_model_ids = {"deepseek-v4-flash", "deepseek-v4-pro"}

    # USD per 1M tokens. Unknown models fall back to 0-cost accounting.
    model_token_prices = {
        "deepseek-v4-flash": {"input": 0.14, "output": 0.28},
        "deepseek-v4-pro": {"input": 0.435, "output": 0.87},
        "gpt-5-mini": {"input": 0.25, "output": 2.0},
        "gpt-5-nano": {"input": 0.05, "output": 0.4},
        "gpt-4.1": {"input": 2.0, "output": 8.0},
        "gpt-4.1-mini": {"input": 0.4, "output": 1.6},
        "gpt-4.1-nano": {"input": 0.1, "output": 0.4},
        "gpt-4o": {"input": 2.5, "output": 10.0},
        "gpt-4o-mini": {"input": 0.15, "output": 0.6},
    }

    system_prompt = (
        "You answer questions using only the provided reasoning paths. "
        "Return only valid JSON with the keys answer and explanation. "
        "If the paths do not support an answer, set answer to Unknown."
    )

    def generate_answer(
        self,
        question: str,
        reasoning_paths_text: str,
        model_id: str,
        provider_id: str = "openai",
    ) -> tuple[str, str]:
        """Call the LLM and return the generated answer with the prompt."""
        result = self.generate_answer_with_explanation(
            question=question,
            reasoning_paths_text=reasoning_paths_text,
            model_id=model_id,
            provider_id=provider_id,
        )
        return result["answer"], result["prompt"]

    def generate_answer_with_explanation(
        self,
        question: str,
        reasoning_paths_text: str,
        model_id: str,
        provider_id: str = "openai",
    ) -> dict[str, str]:
        """Call the LLM and return parsed answer, explanation, and raw response."""
        api_key, api_key_env_name, base_url = self._model_api_settings(
            model_id,
            provider_id,
        )
        if not api_key:
            raise LlmAnswerGenerationException(
                f"{api_key_env_name} must be set in .env before LLM inference."
            )

        prompt = self.build_prompt(
            question=question,
            reasoning_paths_text=reasoning_paths_text,
        )

        try:
            started_at = time.monotonic()
            if provider_id == "vezilka":
                response = self._invoke_vezilka_completion(
                    model_id=model_id,
                    prompt=prompt,
                    api_key=api_key,
                    base_url=base_url or DEFAULT_VEZILKA_BASE_URL,
                )
                raw_response = self.extract_completion_content(response).strip()
            else:
                from langchain_core.messages import HumanMessage, SystemMessage
                from langchain_openai import ChatOpenAI

                messages = [
                    SystemMessage(content=self.system_prompt),
                    HumanMessage(content=prompt),
                ]
                chat_model = self._create_chat_model(
                    chat_openai_type=ChatOpenAI,
                    model_id=model_id,
                    prompt=prompt,
                    api_key=api_key,
                    base_url=base_url,
                )
                response = self._invoke_with_visible_rate_limit_retries(
                    chat_model=chat_model,
                    messages=messages,
                    model_id=model_id,
                    prompt=prompt,
                )
                raw_response = self.extract_response_content(response.content).strip()
            elapsed_seconds = time.monotonic() - started_at
            if elapsed_seconds >= self.slow_request_warning_seconds:
                logger.warning(
                    f"Slow LLM answer generation call: model={model_id} "
                    f"elapsed_seconds={elapsed_seconds:.2f} "
                    f"prompt_chars={len(prompt)}"
                )
        except Exception as error:
            raise LlmAnswerGenerationException(
                f"LLM answer generation failed: {error}"
            ) from error

        parsed_response = self.parse_json_response(raw_response)
        usage = self.extract_token_usage(response)
        estimated_cost = self.estimate_cost_usd(
            model_id=model_id,
            prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"],
        )
        return {
            "answer": parsed_response["answer"],
            "explanation": parsed_response["explanation"],
            "raw_response": raw_response,
            "prompt": prompt,
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "total_tokens": usage["total_tokens"],
            "estimated_cost_usd": estimated_cost,
        }

    def _invoke_with_visible_rate_limit_retries(
        self,
        chat_model: Any,
        messages: list[Any],
        model_id: str,
        prompt: str,
    ) -> Any:
        for attempt_number in range(1, self.max_rate_limit_retries + 1):
            try:
                return chat_model.invoke(messages)
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
                        operation="llm_answer_generation",
                        model_id=model_id,
                        item_count=len(prompt),
                        attempt_number=attempt_number,
                        max_attempts=self.max_rate_limit_retries,
                        wait_seconds=wait_seconds,
                        error=error,
                    )
                )
                time.sleep(wait_seconds)

        return chat_model.invoke(messages)

    def _invoke_vezilka_completion(
        self,
        *,
        model_id: str,
        prompt: str,
        api_key: str,
        base_url: str,
    ) -> Any:
        """Invoke Vezilka's OpenAI-compatible ``/v1/completions`` endpoint."""
        from openai import OpenAI

        http_client = create_rate_limit_logging_http_client(
            logger=logger,
            operation="llm_answer_generation",
            model_id=model_id,
            item_count=len(prompt),
        )
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=self.request_timeout_seconds,
            max_retries=0,
            http_client=http_client,
        )
        completion_prompt = (
            f"System:\n{self.system_prompt}\n\n"
            f"User:\n{prompt}\n\nAssistant:\n"
        )
        for attempt_number in range(1, self.max_rate_limit_retries + 1):
            try:
                return client.completions.create(
                    model=model_id,
                    prompt=completion_prompt,
                    temperature=0,
                    max_tokens=self.max_completion_tokens,
                )
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
                        operation="llm_answer_generation",
                        model_id=model_id,
                        item_count=len(prompt),
                        attempt_number=attempt_number,
                        max_attempts=self.max_rate_limit_retries,
                        wait_seconds=wait_seconds,
                        error=error,
                    )
                )
                time.sleep(wait_seconds)
        return client.completions.create(
            model=model_id,
            prompt=completion_prompt,
            temperature=0,
            max_tokens=self.max_completion_tokens,
        )

    @staticmethod
    def _create_chat_model(
        chat_openai_type: Any,
        model_id: str,
        prompt: str,
        api_key: str,
        base_url: str | None = None,
    ) -> Any:
        http_client = create_rate_limit_logging_http_client(
            logger=logger,
            operation="llm_answer_generation",
            model_id=model_id,
            item_count=len(prompt),
        )
        model_kwargs = {
            "model": model_id,
            "api_key": api_key,
            "temperature": 0,
            "max_retries": 0,
            "timeout": LangChainOpenAiAnswerGenerationService.request_timeout_seconds,
            "http_client": http_client,
        }
        if base_url is None:
            model_kwargs["model_kwargs"] = {
                "response_format": {"type": "json_object"}
            }
        if base_url is not None:
            model_kwargs["base_url"] = base_url

        try:
            return chat_openai_type(**model_kwargs)
        except TypeError:
            try:
                fallback_kwargs = dict(model_kwargs)
                fallback_kwargs.pop("model_kwargs", None)
                return chat_openai_type(**fallback_kwargs)
            except TypeError:
                pass
            if base_url is not None:
                try:
                    fallback_kwargs = dict(model_kwargs)
                    fallback_kwargs.pop("base_url", None)
                    fallback_kwargs["openai_api_base"] = base_url
                    return chat_openai_type(**fallback_kwargs)
                except TypeError:
                    pass
            logger.warning(
                "Current LangChain ChatOpenAI does not support max_retries=0 "
                "or custom http_client; OpenAI SDK retries/rate-limit headers may "
                "remain hidden in logs."
            )
            fallback_kwargs = {
                "model": model_id,
                "api_key": api_key,
                "temperature": 0,
                "timeout": LangChainOpenAiAnswerGenerationService.request_timeout_seconds,
            }
            if base_url is not None:
                fallback_kwargs["base_url"] = base_url
            else:
                fallback_kwargs["model_kwargs"] = {
                    "response_format": {"type": "json_object"}
                }
            try:
                return chat_openai_type(**fallback_kwargs)
            except TypeError:
                legacy_kwargs = dict(fallback_kwargs)
                legacy_kwargs.pop("timeout", None)
                legacy_kwargs.pop("model_kwargs", None)
                return chat_openai_type(**legacy_kwargs)

    @classmethod
    def _model_api_settings(
        cls,
        model_id: str,
        provider_id: str = "openai",
    ) -> tuple[str | None, str, str | None]:
        if provider_id == "vezilka":
            return VEZILKA_API_KEY, VEZILKA_API_KEY_ENV_NAME, DEFAULT_VEZILKA_BASE_URL
        if provider_id == "deepseek" or model_id in cls.deepseek_model_ids:
            return DEEPSEEK_API_KEY, DEEPSEEK_API_KEY_ENV_NAME, DEEPSEEK_BASE_URL

        return OPENAI_API_KEY, OPENAI_API_KEY_ENV_NAME, None

    @classmethod
    def extract_completion_content(cls, response: Any) -> str:
        choices = getattr(response, "choices", None)
        if not choices:
            raise LlmAnswerGenerationException(
                "Vezilka completion response did not contain any choices."
            )
        return cls.extract_response_content(getattr(choices[0], "text", ""))

    @classmethod
    def extract_token_usage(cls, response: Any) -> dict[str, int]:
        """Extract token counts from LangChain/OpenAI response metadata."""
        usage_metadata = getattr(response, "usage_metadata", None)
        if isinstance(usage_metadata, dict):
            input_tokens = cls._int_value(usage_metadata.get("input_tokens"))
            output_tokens = cls._int_value(usage_metadata.get("output_tokens"))
            total_tokens = cls._int_value(usage_metadata.get("total_tokens"))
            return {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": total_tokens or input_tokens + output_tokens,
            }

        direct_usage = getattr(response, "usage", None)
        if direct_usage is not None:
            prompt_tokens = cls._int_value(getattr(direct_usage, "prompt_tokens", 0))
            completion_tokens = cls._int_value(
                getattr(direct_usage, "completion_tokens", 0)
            )
            total_tokens = cls._int_value(getattr(direct_usage, "total_tokens", 0))
            return {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens or prompt_tokens + completion_tokens,
            }

        response_metadata = getattr(response, "response_metadata", None)
        token_usage = {}
        if isinstance(response_metadata, dict):
            raw_token_usage = response_metadata.get("token_usage", {})
            if isinstance(raw_token_usage, dict):
                token_usage = raw_token_usage

        prompt_tokens = cls._int_value(token_usage.get("prompt_tokens"))
        completion_tokens = cls._int_value(token_usage.get("completion_tokens"))
        total_tokens = cls._int_value(token_usage.get("total_tokens"))
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens or prompt_tokens + completion_tokens,
        }

    @classmethod
    def estimate_cost_usd(
        cls,
        model_id: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> float:
        prices = cls.model_token_prices.get(model_id)
        if prices is None:
            return 0.0
        return round(
            (
                (prompt_tokens * prices["input"])
                + (completion_tokens * prices["output"])
            )
            / 1_000_000,
            8,
        )

    @staticmethod
    def _int_value(value: Any) -> int:
        return value if isinstance(value, int) else 0

    @staticmethod
    def build_prompt(question: str, reasoning_paths_text: str) -> str:
        """Build the final-answer prompt."""
        return (
            "Question:\n"
            f"{question}\n\n"
            "Reasoning paths:\n"
            f"{reasoning_paths_text}\n\n"
            "Return only valid JSON in this exact shape:\n"
            "{\"answer\": \"...\", \"explanation\": \"...\"}\n"
            "The answer must be only the answer entity or entities. "
            "If multiple answers are supported, use a comma-separated string. "
            "The explanation must briefly name the reasoning path triples used. "
            "Use only the reasoning paths."
        )

    @classmethod
    def parse_json_response(cls, response_text: str) -> dict[str, str]:
        """Parse the model JSON answer and explanation."""
        cleaned_response = cls._strip_json_code_fence(response_text)
        try:
            parsed_response = json.loads(cleaned_response)
        except json.JSONDecodeError as error:
            raise LlmAnswerGenerationException(
                f"LLM response was not valid JSON: {error}"
            ) from error

        if not isinstance(parsed_response, dict):
            raise LlmAnswerGenerationException("LLM response JSON must be an object.")

        answer = parsed_response.get("answer")
        explanation = parsed_response.get("explanation")
        if not isinstance(answer, str) or not isinstance(explanation, str):
            raise LlmAnswerGenerationException(
                "LLM response JSON must contain string fields 'answer' and "
                "'explanation'."
            )

        return {
            "answer": answer.strip(),
            "explanation": explanation.strip(),
        }

    @staticmethod
    def _strip_json_code_fence(response_text: str) -> str:
        stripped_text = response_text.strip()
        code_fence_match = re.fullmatch(
            r"```(?:json)?\s*(.*?)\s*```",
            stripped_text,
            flags=re.DOTALL,
        )
        if code_fence_match is None:
            return stripped_text

        return code_fence_match.group(1).strip()

    @staticmethod
    def extract_response_content(content: Any) -> str:
        """Normalize LangChain/provider response content into plain text."""
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            text_parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(str(item.get("text", "")))
                elif isinstance(item, str):
                    text_parts.append(item)
            return "\n".join(part for part in text_parts if part)

        return str(content)
