"""LangChain-backed final answer generation from reasoning paths."""

from __future__ import annotations

import json
import re
from typing import Any

from helpers.constants import OPENAI_API_KEY_ENV_NAME
from helpers.env_variables import OPENAI_API_KEY
from pipeline.evaluation.exceptions import LlmAnswerGenerationException
from pipeline.services.abstract import AbstractService


class LangChainOpenAiAnswerGenerationService(AbstractService):
    """Generate final QA answers with a simple OpenAI chat model."""

    # USD per 1M tokens. Unknown models fall back to 0-cost accounting.
    model_token_prices = {
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
    ) -> tuple[str, str]:
        """Call the LLM and return the generated answer with the prompt."""
        result = self.generate_answer_with_explanation(
            question=question,
            reasoning_paths_text=reasoning_paths_text,
            model_id=model_id,
        )
        return result["answer"], result["prompt"]

    def generate_answer_with_explanation(
        self,
        question: str,
        reasoning_paths_text: str,
        model_id: str,
    ) -> dict[str, str]:
        """Call the LLM and return parsed answer, explanation, and raw response."""
        if not OPENAI_API_KEY:
            raise LlmAnswerGenerationException(
                f"{OPENAI_API_KEY_ENV_NAME} must be set in .env before LLM inference."
            )

        prompt = self.build_prompt(
            question=question,
            reasoning_paths_text=reasoning_paths_text,
        )

        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            from langchain_openai import ChatOpenAI

            chat_model = ChatOpenAI(
                model=model_id,
                api_key=OPENAI_API_KEY,
                temperature=0,
            )
            response = chat_model.invoke(
                [
                    SystemMessage(content=self.system_prompt),
                    HumanMessage(content=prompt),
                ]
            )
        except Exception as error:
            raise LlmAnswerGenerationException(
                f"LLM answer generation failed: {error}"
            ) from error

        raw_response = self.extract_response_content(response.content).strip()
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
