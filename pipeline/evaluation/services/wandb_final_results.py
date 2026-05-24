"""Optional WandB logging for final results runs."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from helpers.logging_config import get_logger
from pipeline.evaluation.models import FinalResultsEvaluationResult
from pipeline.services.abstract import AbstractService

logger = get_logger(__name__)


class WandbFinalResultsConfig(BaseModel):
    """Runtime WandB settings for final results logging."""

    project: str = Field(..., description="WandB project name.")
    entity: str | None = Field(default=None, description="Optional WandB entity.")
    mode: str = Field(default="online", description="WandB mode.")


class WandbFinalResultsLogResult(BaseModel):
    """Outcome of optional WandB logging."""

    status: Literal["logged", "skipped", "failed"] = Field(
        ...,
        description="WandB logging status.",
    )
    run_id: str | None = Field(default=None, description="WandB run id.")
    run_url: str | None = Field(default=None, description="WandB run URL.")
    error_message: str | None = Field(default=None, description="Logging error.")


class WandbFinalResultsLoggingService(AbstractService):
    """Upload final result metrics, tables, and artifacts to WandB."""

    table_key = "per_instance_results"
    artifact_type = "evaluation-results"
    source_path_keys = {
        "answers_path",
        "reasoning_path",
        "predictions_path",
        "inference_config_path",
        "evaluation_config_path",
    }
    table_columns = [
        "instance_index",
        "question",
        "q_entity",
        "gold_answers",
        "predicted_answers",
        "explanation",
        "exact_match",
        "hit",
        "hits_at_1",
        "precision",
        "recall",
        "f1",
        "mentioned_triple_count",
        "grounded_mentioned_triple_count",
        "grounded_explanation",
        "fully_grounded_explanation",
        "ndcg_at_1",
        "ndcg_at_5",
        "ndcg_at_10",
        "ndcg_at_candidate_limit",
        "answer_error_message",
    ]

    def log_final_results(
        self,
        final_result: FinalResultsEvaluationResult,
        config: WandbFinalResultsConfig,
    ) -> WandbFinalResultsLogResult:
        """Best-effort upload of local final results to WandB."""
        try:
            wandb = importlib.import_module("wandb")
            results_config = self._load_json_object(final_result.results_config_path)
            retrieval_metrics = self._load_json_object(
                final_result.retrieval_metrics_path
            )
            reasoning_metrics = self._load_json_object(
                final_result.reasoning_metrics_path
            )
            per_instance_rows = self._load_jsonl_objects(
                final_result.per_instance_results_path
            )
            scalar_metrics = self.build_scalar_metrics(
                retrieval_metrics=retrieval_metrics,
                reasoning_metrics=reasoning_metrics,
            )
            table_rows = self.build_table_rows(
                results_config=results_config,
                per_instance_rows=per_instance_rows,
            )
            run_name = f"graphragx_{final_result.results_run_name}"
            tags = self._build_tags(results_config)

            with wandb.init(
                project=config.project,
                entity=config.entity,
                mode=config.mode,
                name=run_name,
                tags=tags,
                config=self._stringify_paths(results_config),
                job_type="final-results",
            ) as run:
                run.log(scalar_metrics)
                table = wandb.Table(columns=self.table_columns, data=table_rows)
                run.log({self.table_key: table})
                artifact = wandb.Artifact(
                    self._artifact_name(final_result.results_run_name),
                    type=self.artifact_type,
                )
                self._add_artifact_files(
                    artifact=artifact,
                    final_result=final_result,
                    results_config=results_config,
                )
                run.log_artifact(artifact)
                return WandbFinalResultsLogResult(
                    status="logged",
                    run_id=getattr(run, "id", None),
                    run_url=getattr(run, "url", None),
                )
        except Exception as error:
            logger.warning(f"WandB final results logging failed: {error}")
            return WandbFinalResultsLogResult(
                status="failed",
                error_message=str(error),
            )

    @classmethod
    def build_scalar_metrics(
        cls,
        retrieval_metrics: dict[str, Any],
        reasoning_metrics: dict[str, Any],
    ) -> dict[str, float | int]:
        """Build WandB-safe scalar metric names."""
        mappings = {
            "retrieval_hits_at_1": retrieval_metrics.get("hits_at_1"),
            "retrieval_hit_at_k": retrieval_metrics.get("hit_at_k"),
            "retrieval_average_candidate_count": retrieval_metrics.get(
                "average_candidate_count"
            ),
            "retrieval_missing_gold_in_graph_count": retrieval_metrics.get(
                "missing_gold_in_graph_count"
            ),
            "answer_accuracy": reasoning_metrics.get("accuracy"),
            "answer_hit_rate": reasoning_metrics.get("hit_rate"),
            "answer_hits_at_1": reasoning_metrics.get("hits_at_1"),
            "answer_precision": reasoning_metrics.get("precision"),
            "answer_recall": reasoning_metrics.get("recall"),
            "answer_f1": reasoning_metrics.get("f1"),
            "grounding_grounded_explanation_rate": reasoning_metrics.get(
                "grounded_explanation_rate"
            ),
            "grounding_fully_grounded_explanation_rate": reasoning_metrics.get(
                "fully_grounded_explanation_rate"
            ),
            "ranking_ndcg_at_1": reasoning_metrics.get("ndcg_at_1"),
            "ranking_ndcg_at_5": reasoning_metrics.get("ndcg_at_5"),
            "ranking_ndcg_at_10": reasoning_metrics.get("ndcg_at_10"),
            "ranking_ndcg_at_candidate_limit": reasoning_metrics.get(
                "ndcg_at_candidate_limit"
            ),
        }
        return {
            key: value
            for key, value in mappings.items()
            if isinstance(value, int | float)
        }

    def build_table_rows(
        self,
        results_config: dict[str, Any],
        per_instance_rows: list[dict[str, Any]],
    ) -> list[list[Any]]:
        """Build WandB table rows, adding explanations from answers.jsonl."""
        answers_by_index = self._load_answers_by_index(results_config)
        table_rows: list[list[Any]] = []
        for row in per_instance_rows:
            instance_index = row.get("instance_index")
            answer_row = answers_by_index.get(instance_index, {})
            table_rows.append(
                [
                    instance_index,
                    row.get("question", ""),
                    row.get("q_entity", []),
                    row.get("gold_answers", []),
                    row.get("predicted_answers", []),
                    answer_row.get("explanation", ""),
                    row.get("exact_match", False),
                    row.get("hit", False),
                    row.get("hits_at_1", False),
                    row.get("precision", 0.0),
                    row.get("recall", 0.0),
                    row.get("f1", 0.0),
                    row.get("mentioned_triple_count", 0),
                    row.get("grounded_mentioned_triple_count", 0),
                    row.get("grounded_explanation", False),
                    row.get("fully_grounded_explanation", False),
                    row.get("ndcg_at_1", 0.0),
                    row.get("ndcg_at_5", 0.0),
                    row.get("ndcg_at_10", 0.0),
                    row.get("ndcg_at_candidate_limit", 0.0),
                    row.get("answer_error_message"),
                ]
            )
        return table_rows

    def _load_answers_by_index(
        self,
        results_config: dict[str, Any],
    ) -> dict[int, dict[str, Any]]:
        answers_path = results_config.get("answers_path")
        if not isinstance(answers_path, str):
            return {}
        path = Path(answers_path)
        if not path.exists():
            return {}
        return {
            row["instance_index"]: row
            for row in self._load_jsonl_objects(path)
            if isinstance(row.get("instance_index"), int)
        }

    @classmethod
    def _build_tags(cls, results_config: dict[str, Any]) -> list[str]:
        tags = ["graphragx"]
        for key in [
            "dataset_id",
            "model_id",
            "evaluation_run_name",
            "inference_run_name",
        ]:
            value = results_config.get(key)
            if value:
                tags.append(str(value))
        return tags

    @classmethod
    def _add_artifact_files(
        cls,
        artifact: Any,
        final_result: FinalResultsEvaluationResult,
        results_config: dict[str, Any],
    ) -> None:
        for path in sorted(final_result.results_run_directory.iterdir()):
            if path.is_file():
                artifact.add_file(str(path), name=f"results/{path.name}")

        for key in sorted(cls.source_path_keys):
            value = results_config.get(key)
            if not isinstance(value, str):
                continue
            path = Path(value)
            if path.exists() and path.is_file():
                artifact.add_file(str(path), name=f"sources/{path.name}")

    @staticmethod
    def _artifact_name(results_run_name: str) -> str:
        sanitized = "".join(
            character if character.isalnum() or character in "._-" else "_"
            for character in results_run_name
        ).strip("._-")
        return f"graphragx-results-{sanitized or 'run'}"

    @classmethod
    def _stringify_paths(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {key: cls._stringify_paths(item) for key, item in value.items()}
        if isinstance(value, list):
            return [cls._stringify_paths(item) for item in value]
        if isinstance(value, Path):
            return str(value)
        return value

    @staticmethod
    def _load_json_object(path: Path) -> dict[str, Any]:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"JSON file {path} must contain an object.")
        return value

    @staticmethod
    def _load_jsonl_objects(path: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row {line_number} in {path} must be an object.")
            rows.append(value)
        return rows
