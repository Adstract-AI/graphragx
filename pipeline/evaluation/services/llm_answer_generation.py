"""LangChain-backed final answer generation from reasoning paths."""

from __future__ import annotations

from typing import Any

from helpers.constants import OPENAI_API_KEY_ENV_NAME
from helpers.env_variables import OPENAI_API_KEY
from pipeline.evaluation.exceptions import LlmAnswerGenerationException
from pipeline.services.abstract import AbstractService


class LangChainOpenAiAnswerGenerationService(AbstractService):
    """Generate final QA answers with a simple OpenAI chat model."""

    system_prompt = (
        "You answer questions using only the provided reasoning paths. "
        "Return only the final answer. If the paths do not support an answer, "
        "return Unknown."
    )

    def generate_answer(
        self,
        question: str,
        reasoning_paths_text: str,
        model_id: str,
    ) -> tuple[str, str]:
        """Call the LLM and return the generated answer with the prompt."""
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

        return self.extract_response_content(response.content).strip(), prompt

    @staticmethod
    def build_prompt(question: str, reasoning_paths_text: str) -> str:
        """Build the final-answer prompt."""
        return (
            "Question:\n"
            f"{question}\n\n"
            "Reasoning paths:\n"
            f"{reasoning_paths_text}\n\n"
            "Return only the answer entity or entities. "
            "Do not explain. "
            "Do not write \"Final answer:\". "
            "If multiple answers are supported, return them as a comma-separated list. "
            "Use only the reasoning paths."
        )

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
