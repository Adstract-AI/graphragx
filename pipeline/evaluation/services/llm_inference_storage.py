"""Storage service for post-retrieval LLM inference runs."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from helpers.constants import (
    LLM_INFERENCE_ANSWERS_FILENAME,
    LLM_INFERENCE_PROMPTS_FILENAME,
    LLM_INFERENCE_REASONING_PATHS_FILENAME,
    LLM_INFERENCE_REASONING_SUBGRAPHS_FILENAME,
    LLM_INFERENCE_SUMMARY_FILENAME,
)
from pipeline.evaluation.models import GeneratedFinalAnswersBatch
from pipeline.exceptions import PipelineException
from pipeline.services.abstract import AbstractService


class LlmInferenceStoragePayload(BaseModel):
    """Data persisted for one LLM inference run."""

    answers: GeneratedFinalAnswersBatch


class LlmInferenceStorageResult(BaseModel):
    """Paths and version metadata produced by inference storage."""

    inference_run_directory: Path
    inference_run_name: str
    inference_run_number: int
    reasoning_paths_path: Path
    reasoning_subgraphs_path: Path
    prompts_path: Path
    answers_path: Path
    summary_path: Path


class LlmInferenceStorageService(AbstractService):
    """Persist numbered LLM inference runs."""

    reasoning_paths_filename = LLM_INFERENCE_REASONING_PATHS_FILENAME
    reasoning_subgraphs_filename = LLM_INFERENCE_REASONING_SUBGRAPHS_FILENAME
    prompts_filename = LLM_INFERENCE_PROMPTS_FILENAME
    answers_filename = LLM_INFERENCE_ANSWERS_FILENAME
    summary_filename = LLM_INFERENCE_SUMMARY_FILENAME

    def save_inference_run(
        self,
        inference_root: Path,
        run_name: str | None,
        payload: LlmInferenceStoragePayload,
    ) -> LlmInferenceStorageResult:
        """Create a numbered inference run directory and persist all outputs."""
        try:
            inference_run_directory = self._create_inference_run_directory(
                inference_root=inference_root,
                run_name=run_name,
            )
            reasoning_paths_path = inference_run_directory / self.reasoning_paths_filename
            reasoning_subgraphs_path = (
                inference_run_directory / self.reasoning_subgraphs_filename
            )
            prompts_path = inference_run_directory / self.prompts_filename
            answers_path = inference_run_directory / self.answers_filename
            summary_path = inference_run_directory / self.summary_filename

            self._write_jsonl(reasoning_paths_path, self._iter_reasoning_paths(payload))
            self._write_jsonl(
                reasoning_subgraphs_path,
                self._iter_reasoning_subgraphs(payload),
            )
            self._write_jsonl(prompts_path, self._iter_prompts(payload))
            self._write_jsonl(answers_path, self._iter_answers(payload))
            summary_path.write_text(
                json.dumps(self._build_summary(payload), indent=2, sort_keys=True),
                encoding="utf-8",
            )
        except OSError as error:
            raise PipelineException(f"Could not save LLM inference run: {error}") from error

        return LlmInferenceStorageResult(
            inference_run_directory=inference_run_directory,
            inference_run_name=inference_run_directory.name,
            inference_run_number=self._extract_run_number(inference_run_directory.name),
            reasoning_paths_path=reasoning_paths_path,
            reasoning_subgraphs_path=reasoning_subgraphs_path,
            prompts_path=prompts_path,
            answers_path=answers_path,
            summary_path=summary_path,
        )

    @staticmethod
    def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8") as output_file:
            for row in rows:
                output_file.write(json.dumps(row, sort_keys=True, default=str))
                output_file.write("\n")

    @staticmethod
    def _iter_reasoning_paths(
        payload: LlmInferenceStoragePayload,
    ) -> list[dict[str, Any]]:
        return [
            {
                "instance_index": item.instance_index,
                "question": item.question,
                "reasoning_paths_text": item.reasoning_paths_text,
            }
            for item in payload.answers.items
        ]

    @staticmethod
    def _iter_reasoning_subgraphs(
        payload: LlmInferenceStoragePayload,
    ) -> list[dict[str, Any]]:
        return [
            {
                "instance_index": item.instance_index,
                "question": item.question,
                "triples": [
                    triple.model_dump(mode="json")
                    for triple in item.reasoning_subgraph_triples
                ],
            }
            for item in payload.answers.items
        ]

    @staticmethod
    def _iter_prompts(payload: LlmInferenceStoragePayload) -> list[dict[str, Any]]:
        return [
            {
                "instance_index": item.instance_index,
                "question": item.question,
                "model_id": item.model_id,
                "prompt": item.prompt,
                "error_message": item.error_message,
            }
            for item in payload.answers.items
        ]

    @staticmethod
    def _iter_answers(payload: LlmInferenceStoragePayload) -> list[dict[str, Any]]:
        return [
            {
                "instance_index": item.instance_index,
                "question": item.question,
                "q_entity": item.q_entity,
                "gold_answers": item.a_entity,
                "answer_candidates": item.answer_candidates,
                "model_id": item.model_id,
                "answer": item.answer,
                "error_message": item.error_message,
            }
            for item in payload.answers.items
        ]

    @staticmethod
    def _build_summary(payload: LlmInferenceStoragePayload) -> dict[str, Any]:
        answers = payload.answers
        return {
            "dataset_id": answers.dataset_id,
            "evaluation_run_name": answers.evaluation_run_name,
            "model_id": answers.model_id,
            "total_instances": len(answers.items),
            "successful_answers": answers.successful_answers,
            "failed_answers": answers.failed_answers,
        }

    def _create_inference_run_directory(
        self,
        inference_root: Path,
        run_name: str | None,
    ) -> Path:
        inference_root.mkdir(parents=True, exist_ok=True)
        run_number = self._next_run_number(inference_root)
        run_label = self._resolve_run_label(run_name)
        inference_run_directory = inference_root / f"{run_number}_{run_label}"
        inference_run_directory.mkdir(parents=True, exist_ok=False)
        return inference_run_directory

    @classmethod
    def _next_run_number(cls, inference_root: Path) -> int:
        existing_run_numbers = [
            cls._extract_run_number(path.name)
            for path in inference_root.iterdir()
            if path.is_dir() and cls._extract_run_number(path.name) > 0
        ]
        return max(existing_run_numbers, default=0) + 1

    @classmethod
    def _resolve_run_label(cls, run_name: str | None) -> str:
        if run_name is None or not run_name.strip():
            return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        return cls._sanitize_run_name(run_name)

    @staticmethod
    def _sanitize_run_name(run_name: str) -> str:
        sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", run_name.strip())
        sanitized = sanitized.strip("._-")
        return sanitized or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    @staticmethod
    def _extract_run_number(run_directory_name: str) -> int:
        run_number_match = re.match(r"^(\d+)_", run_directory_name)
        if run_number_match is None:
            return 0

        return int(run_number_match.group(1))
