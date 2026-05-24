"""Storage service for saved GNN answer-retriever evaluation runs."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from helpers.constants import (
    GNN_ANSWER_RETRIEVER_EVALUATION_CONFIG_FILENAME,
    GNN_ANSWER_RETRIEVER_EVALUATION_PREDICTIONS_FILENAME,
)
from pipeline.evaluation.models import EvaluatedAnswerRetrievalInstance
from pipeline.exceptions import GnnAnswerRetrieverEvaluationException
from pipeline.services.abstract import AbstractService

JsonScalar = str | int | float | bool | None
JsonObjectLevel1 = dict[str, JsonScalar]
JsonObjectLevel2 = dict[str, JsonScalar | JsonObjectLevel1]
JsonObjectLevel3 = dict[str, JsonScalar | JsonObjectLevel1 | JsonObjectLevel2]


class GnnAnswerRetrieverEvaluationStoragePayload(BaseModel):
    """Data persisted for one evaluation run."""

    evaluation_config: JsonObjectLevel3 = Field(default_factory=dict)
    predictions: list[EvaluatedAnswerRetrievalInstance] = Field(default_factory=list)


class GnnAnswerRetrieverEvaluationStorageResult(BaseModel):
    """Paths and version metadata produced by evaluation storage."""

    evaluation_run_directory: Path
    evaluation_run_name: str
    evaluation_run_number: int
    evaluation_config_path: Path
    predictions_path: Path


class GnnAnswerRetrieverEvaluationStorageService(AbstractService):
    """Persist numbered evaluation runs."""

    config_filename = GNN_ANSWER_RETRIEVER_EVALUATION_CONFIG_FILENAME
    predictions_filename = GNN_ANSWER_RETRIEVER_EVALUATION_PREDICTIONS_FILENAME

    def save_evaluation_run(
        self,
        evaluation_root: Path,
        run_name: str | None,
        payload: GnnAnswerRetrieverEvaluationStoragePayload,
    ) -> GnnAnswerRetrieverEvaluationStorageResult:
        """Create a numbered evaluation run directory and persist all outputs."""
        try:
            evaluation_run_directory = self._create_evaluation_run_directory(
                evaluation_root=evaluation_root,
                run_name=run_name,
            )
            evaluation_config_path = evaluation_run_directory / self.config_filename
            predictions_path = evaluation_run_directory / self.predictions_filename

            evaluation_config_path.write_text(
                json.dumps(payload.evaluation_config, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            with predictions_path.open("w", encoding="utf-8") as predictions_file:
                for prediction in payload.predictions:
                    predictions_file.write(prediction.model_dump_json())
                    predictions_file.write("\n")
        except OSError as error:
            raise GnnAnswerRetrieverEvaluationException(
                f"Could not save GNN answer-retriever evaluation run: {error}"
            ) from error

        return GnnAnswerRetrieverEvaluationStorageResult(
            evaluation_run_directory=evaluation_run_directory,
            evaluation_run_name=evaluation_run_directory.name,
            evaluation_run_number=self._extract_run_number(
                evaluation_run_directory.name
            ),
            evaluation_config_path=evaluation_config_path,
            predictions_path=predictions_path,
        )

    def _create_evaluation_run_directory(
        self,
        evaluation_root: Path,
        run_name: str | None,
    ) -> Path:
        evaluation_root.mkdir(parents=True, exist_ok=True)
        run_number = self._next_run_number(evaluation_root)
        run_label = self._resolve_run_label(run_name)
        evaluation_run_directory = evaluation_root / f"{run_number}_{run_label}"
        evaluation_run_directory.mkdir(parents=True, exist_ok=False)
        return evaluation_run_directory

    @classmethod
    def _next_run_number(cls, evaluation_root: Path) -> int:
        existing_run_numbers = [
            cls._extract_run_number(path.name)
            for path in evaluation_root.iterdir()
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
