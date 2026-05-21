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
        return {
            "answer": parsed_response["answer"],
            "explanation": parsed_response["explanation"],
            "raw_response": raw_response,
            "prompt": prompt,
        }

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
