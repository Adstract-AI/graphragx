"""Final results computation and storage for retrieval plus reasoning runs."""

from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from string import punctuation
from typing import Any

from pydantic import BaseModel, Field

from helpers.constants import (
    FINAL_RESULTS_CONFIG_FILENAME,
    FINAL_RESULTS_PER_INSTANCE_FILENAME,
    FINAL_RESULTS_REASONING_METRICS_FILENAME,
    FINAL_RESULTS_RETRIEVAL_METRICS_FILENAME,
)
from pipeline.evaluation.exceptions import FinalResultsEvaluationException
from pipeline.evaluation.models import (
    EvaluatedAnswerRetrievalInstance,
    FinalAnswerMetrics,
    FinalReasoningMetrics,
    FinalResultsConfig,
    GnnAnswerRetrieverEvaluationResult,
    PerInstanceFinalResult,
    RankingMetrics,
    SavedLlmInferenceRun,
)
from pipeline.evaluation.models.final_results import ExplanationGroundingMetrics
from pipeline.services.abstract import AbstractService


class FinalResultsStorageResult(BaseModel):
    """Paths and version metadata produced by final results storage."""

    results_run_directory: Path
    results_run_name: str
    results_run_number: int
    results_config_path: Path
    retrieval_metrics_path: Path
    reasoning_metrics_path: Path
    per_instance_results_path: Path


class FinalResultsEvaluationOutcome(BaseModel):
    """Computed final metrics and persisted results metadata."""

    storage_result: FinalResultsStorageResult
    reasoning_metrics: FinalReasoningMetrics
    per_instance_results: list[PerInstanceFinalResult] = Field(default_factory=list)


class FinalResultsEvaluationService(AbstractService):
    """Compute and persist final WebQSP retrieval and reasoning metrics."""

    results_config_filename = FINAL_RESULTS_CONFIG_FILENAME
    retrieval_metrics_filename = FINAL_RESULTS_RETRIEVAL_METRICS_FILENAME
    reasoning_metrics_filename = FINAL_RESULTS_REASONING_METRICS_FILENAME
    per_instance_results_filename = FINAL_RESULTS_PER_INSTANCE_FILENAME
    unknown_answer_values = {"", "unknown", "n/a", "none", "null"}
    article_pattern = re.compile(r"\b(a|an|the)\b")
    whitespace_pattern = re.compile(r"\s+")
    arrow_triple_pattern = re.compile(
        r"([^.;\n]+?)\s*->\s*([A-Za-z0-9_./-]+)\s*->\s*([^.;\n]+)"
    )

    def evaluate(
        self,
        gnn_evaluation_result: GnnAnswerRetrieverEvaluationResult,
        llm_inference_run: SavedLlmInferenceRun,
    ) -> FinalResultsEvaluationOutcome:
        """Compute final results from saved retrieval and inference artifacts."""
        answers = self._load_jsonl_objects(llm_inference_run.answers_path)
        reasoning_rows = self._load_jsonl_objects(llm_inference_run.reasoning_path)
        predictions = [
            EvaluatedAnswerRetrievalInstance.model_validate(item)
            for item in self._load_jsonl_objects(gnn_evaluation_result.predictions_path)
        ]
        evaluation_config = self._load_json_object(
            gnn_evaluation_result.evaluation_config_path
        )
        candidate_limit = self._extract_candidate_limit(evaluation_config)

        answers_by_index = self._index_rows(answers, "answers")
        reasoning_by_index = self._index_rows(reasoning_rows, "reasoning")
        predictions_by_index = {
            prediction.instance_index: prediction
            for prediction in predictions
        }
        self._validate_index_sets(
            answers_by_index=answers_by_index,
            reasoning_by_index=reasoning_by_index,
            predictions_by_index=predictions_by_index,
        )

        per_instance_results = [
            self._build_per_instance_result(
                instance_index=instance_index,
                answer_row=answers_by_index[instance_index],
                reasoning_row=reasoning_by_index[instance_index],
                prediction=predictions_by_index[instance_index],
                candidate_limit=candidate_limit,
            )
            for instance_index in sorted(answers_by_index)
        ]
        retrieval_metrics = self._build_retrieval_metrics(
            gnn_evaluation_result=gnn_evaluation_result,
            predictions=predictions,
        )
        reasoning_metrics = self._build_reasoning_metrics(
            gnn_evaluation_result=gnn_evaluation_result,
            llm_inference_run=llm_inference_run,
            per_instance_results=per_instance_results,
            candidate_limit=candidate_limit,
        )
        results_config = FinalResultsConfig(
            dataset_id=llm_inference_run.dataset_id,
            evaluation_run_name=gnn_evaluation_result.evaluation_run_name,
            inference_run_name=llm_inference_run.inference_run_name,
            model_run_name=gnn_evaluation_result.model_run_name,
            model_id=llm_inference_run.model_id,
            model_run_directory=gnn_evaluation_result.model_run_directory,
            evaluation_run_directory=gnn_evaluation_result.evaluation_run_directory,
            inference_run_directory=llm_inference_run.inference_run_directory,
            predictions_path=gnn_evaluation_result.predictions_path,
            evaluation_config_path=gnn_evaluation_result.evaluation_config_path,
            answers_path=llm_inference_run.answers_path,
            reasoning_path=llm_inference_run.reasoning_path,
            inference_config_path=llm_inference_run.inference_config_path,
        )
        storage_result = self._save_results_run(
            results_root=llm_inference_run.inference_run_directory.parent.parent / "results",
            results_config=results_config,
            retrieval_metrics=retrieval_metrics,
            reasoning_metrics=reasoning_metrics,
            per_instance_results=per_instance_results,
        )
        return FinalResultsEvaluationOutcome(
            storage_result=storage_result,
            reasoning_metrics=reasoning_metrics,
            per_instance_results=per_instance_results,
        )

    @staticmethod
    def _build_retrieval_metrics(
        gnn_evaluation_result: GnnAnswerRetrieverEvaluationResult,
        predictions: list[EvaluatedAnswerRetrievalInstance],
    ) -> dict[str, Any]:
        evaluated_instances = len(predictions)
        hits_at_1_count = sum(1 for prediction in predictions if prediction.hit_at_1)
        hits_at_5_count = sum(
            1
            for prediction in predictions
            if FinalResultsEvaluationService._retrieval_hit_at_k(prediction, 5)
        )
        hits_at_10_count = sum(
            1
            for prediction in predictions
            if FinalResultsEvaluationService._retrieval_hit_at_k(prediction, 10)
        )
        total_candidate_count = sum(
            len(prediction.answer_candidates)
            for prediction in predictions
        )
        missing_gold_in_graph_count = sum(
            1 for prediction in predictions if prediction.missing_gold_in_graph
        )
        return {
            "dataset_id": gnn_evaluation_result.dataset_id,
            "model_run_name": gnn_evaluation_result.model_run_name,
            "model_run_number": gnn_evaluation_result.model_run_number,
            "evaluation_run_name": gnn_evaluation_result.evaluation_run_name,
            "evaluation_run_number": gnn_evaluation_result.evaluation_run_number,
            "evaluated_instances": evaluated_instances,
            "hits_at_1": FinalResultsEvaluationService._safe_divide(
                hits_at_1_count,
                evaluated_instances,
            ),
            "hits_at_1_count": hits_at_1_count,
            "hits_at_5": FinalResultsEvaluationService._safe_divide(
                hits_at_5_count,
                evaluated_instances,
            ),
            "hits_at_5_count": hits_at_5_count,
            "hits_at_10": FinalResultsEvaluationService._safe_divide(
                hits_at_10_count,
                evaluated_instances,
            ),
            "hits_at_10_count": hits_at_10_count,
            "average_candidate_count": FinalResultsEvaluationService._safe_divide(
                total_candidate_count,
                evaluated_instances,
            ),
            "missing_gold_in_graph_count": missing_gold_in_graph_count,
        }

    @staticmethod
    def _retrieval_hit_at_k(
        prediction: EvaluatedAnswerRetrievalInstance,
        k: int,
    ) -> bool:
        return any(
            candidate.is_gold_answer
            for candidate in prediction.answer_candidates[:k]
        )

    def _build_per_instance_result(
        self,
        instance_index: int,
        answer_row: dict[str, Any],
        reasoning_row: dict[str, Any],
        prediction: EvaluatedAnswerRetrievalInstance,
        candidate_limit: int,
    ) -> PerInstanceFinalResult:
        gold_answers = [str(item) for item in answer_row.get("gold_answers", [])]
        predicted_answers = self._prediction_answers(answer_row)
        normalized_gold_answers = self._normalize_answer_set(gold_answers)
        normalized_predicted_answers = self._normalize_answer_list(predicted_answers)
        gold_set = set(normalized_gold_answers)
        predicted_set = set(normalized_predicted_answers)
        true_positive_count = len(gold_set & predicted_set)
        false_positive_count = len(predicted_set - gold_set)
        false_negative_count = len(gold_set - predicted_set)
        precision = self._safe_divide(true_positive_count, true_positive_count + false_positive_count)
        recall = self._safe_divide(true_positive_count, true_positive_count + false_negative_count)
        f1 = self._f1(precision, recall)
        grounding = self._compute_grounding(
            explanation=str(answer_row.get("explanation", "")),
            reasoning_row=reasoning_row,
        )
        return PerInstanceFinalResult(
            instance_index=instance_index,
            question=str(answer_row.get("question", "")),
            q_entity=[str(item) for item in answer_row.get("q_entity", [])],
            gold_answers=gold_answers,
            predicted_answers=predicted_answers,
            normalized_gold_answers=normalized_gold_answers,
            normalized_predicted_answers=normalized_predicted_answers,
            exact_match=gold_set == predicted_set,
            hit=true_positive_count > 0,
            hits_at_1=(
                bool(normalized_predicted_answers)
                and normalized_predicted_answers[0] in gold_set
            ),
            true_positive_count=true_positive_count,
            false_positive_count=false_positive_count,
            false_negative_count=false_negative_count,
            precision=precision,
            recall=recall,
            f1=f1,
            answer_error_message=answer_row.get("error_message"),
            mentioned_triple_count=grounding["mentioned_triple_count"],
            grounded_mentioned_triple_count=grounding["grounded_mentioned_triple_count"],
            grounded_explanation=grounding["grounded_explanation"],
            fully_grounded_explanation=grounding["fully_grounded_explanation"],
            ndcg_at_1=self._ndcg_at_k(prediction, 1),
            ndcg_at_5=self._ndcg_at_k(prediction, 5),
            ndcg_at_10=self._ndcg_at_k(prediction, 10),
            ndcg_at_candidate_limit=self._ndcg_at_k(prediction, candidate_limit),
        )

    def _build_reasoning_metrics(
        self,
        gnn_evaluation_result: GnnAnswerRetrieverEvaluationResult,
        llm_inference_run: SavedLlmInferenceRun,
        per_instance_results: list[PerInstanceFinalResult],
        candidate_limit: int,
    ) -> FinalReasoningMetrics:
        evaluated_instances = len(per_instance_results)
        exact_match_count = sum(1 for item in per_instance_results if item.exact_match)
        hit_count = sum(1 for item in per_instance_results if item.hit)
        hits_at_1_count = sum(1 for item in per_instance_results if item.hits_at_1)
        true_positive_count = sum(item.true_positive_count for item in per_instance_results)
        false_positive_count = sum(item.false_positive_count for item in per_instance_results)
        false_negative_count = sum(item.false_negative_count for item in per_instance_results)
        precision = self._safe_divide(true_positive_count, true_positive_count + false_positive_count)
        recall = self._safe_divide(true_positive_count, true_positive_count + false_negative_count)
        grounded_count = sum(1 for item in per_instance_results if item.grounded_explanation)
        fully_grounded_count = sum(
            1 for item in per_instance_results if item.fully_grounded_explanation
        )
        mentioned_triple_count = sum(item.mentioned_triple_count for item in per_instance_results)
        grounded_mentioned_triple_count = sum(
            item.grounded_mentioned_triple_count for item in per_instance_results
        )
        return FinalReasoningMetrics(
            dataset_id=llm_inference_run.dataset_id,
            evaluation_run_name=gnn_evaluation_result.evaluation_run_name,
            inference_run_name=llm_inference_run.inference_run_name,
            model_run_name=gnn_evaluation_result.model_run_name,
            model_id=llm_inference_run.model_id,
            answer_metrics=FinalAnswerMetrics(
                evaluated_instances=evaluated_instances,
                successful_answers=sum(
                    1 for item in per_instance_results if item.answer_error_message is None
                ),
                failed_answers=sum(
                    1 for item in per_instance_results if item.answer_error_message is not None
                ),
                exact_match_count=exact_match_count,
                accuracy=self._safe_divide(exact_match_count, evaluated_instances),
                hit_count=hit_count,
                hit_rate=self._safe_divide(hit_count, evaluated_instances),
                hits_at_1_count=hits_at_1_count,
                hits_at_1=self._safe_divide(hits_at_1_count, evaluated_instances),
                true_positive_count=true_positive_count,
                false_positive_count=false_positive_count,
                false_negative_count=false_negative_count,
                precision=precision,
                recall=recall,
                f1=self._f1(precision, recall),
            ),
            explanation_grounding_metrics=ExplanationGroundingMetrics(
                grounded_explanation_count=grounded_count,
                fully_grounded_explanation_count=fully_grounded_count,
                grounded_explanation_rate=self._safe_divide(
                    grounded_count,
                    evaluated_instances,
                ),
                fully_grounded_explanation_rate=self._safe_divide(
                    fully_grounded_count,
                    evaluated_instances,
                ),
                mentioned_triple_count=mentioned_triple_count,
                grounded_mentioned_triple_count=grounded_mentioned_triple_count,
            ),
            ranking_metrics=RankingMetrics(
                ndcg_at_1=self._mean([item.ndcg_at_1 for item in per_instance_results]),
                ndcg_at_5=self._mean([item.ndcg_at_5 for item in per_instance_results]),
                ndcg_at_10=self._mean([item.ndcg_at_10 for item in per_instance_results]),
                ndcg_at_candidate_limit=self._mean(
                    [item.ndcg_at_candidate_limit for item in per_instance_results]
                ),
                candidate_limit=candidate_limit,
            ),
        )

    def _save_results_run(
        self,
        results_root: Path,
        results_config: FinalResultsConfig,
        retrieval_metrics: dict[str, Any],
        reasoning_metrics: FinalReasoningMetrics,
        per_instance_results: list[PerInstanceFinalResult],
    ) -> FinalResultsStorageResult:
        try:
            results_run_directory = self._create_results_run_directory(results_root)
            results_config_path = results_run_directory / self.results_config_filename
            retrieval_metrics_path = results_run_directory / self.retrieval_metrics_filename
            reasoning_metrics_path = results_run_directory / self.reasoning_metrics_filename
            per_instance_results_path = (
                results_run_directory / self.per_instance_results_filename
            )
            results_config_path.write_text(
                json.dumps(results_config.model_dump(mode="json"), indent=2, sort_keys=True),
                encoding="utf-8",
            )
            retrieval_metrics_path.write_text(
                json.dumps(retrieval_metrics, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            reasoning_metrics_path.write_text(
                json.dumps(reasoning_metrics.flattened_payload(), indent=2, sort_keys=True),
                encoding="utf-8",
            )
            with per_instance_results_path.open("w", encoding="utf-8") as output_file:
                for item in per_instance_results:
                    output_file.write(item.model_dump_json())
                    output_file.write("\n")
        except OSError as error:
            raise FinalResultsEvaluationException(
                f"Could not save final results run: {error}"
            ) from error

        return FinalResultsStorageResult(
            results_run_directory=results_run_directory,
            results_run_name=results_run_directory.name,
            results_run_number=self._extract_run_number(results_run_directory.name),
            results_config_path=results_config_path,
            retrieval_metrics_path=retrieval_metrics_path,
            reasoning_metrics_path=reasoning_metrics_path,
            per_instance_results_path=per_instance_results_path,
        )

    def _compute_grounding(
        self,
        explanation: str,
        reasoning_row: dict[str, Any],
    ) -> dict[str, Any]:
        subgraph_triples = self._normalized_subgraph_triples(reasoning_row)
        mentioned_triples = self._extract_explanation_triples(explanation)
        if not mentioned_triples:
            mentioned_triples = {
                triple
                for triple in subgraph_triples
                if self._triple_to_text(triple) in self._normalize_text(explanation)
            }

        grounded_triples = mentioned_triples & subgraph_triples
        mentioned_count = len(mentioned_triples)
        grounded_count = len(grounded_triples)
        return {
            "mentioned_triple_count": mentioned_count,
            "grounded_mentioned_triple_count": grounded_count,
            "grounded_explanation": grounded_count > 0,
            "fully_grounded_explanation": mentioned_count > 0 and grounded_count == mentioned_count,
        }

    def _normalized_subgraph_triples(
        self,
        reasoning_row: dict[str, Any],
    ) -> set[tuple[str, str, str]]:
        subgraph = reasoning_row.get("subgraph")
        if not isinstance(subgraph, list):
            raise FinalResultsEvaluationException(
                f"Reasoning row {reasoning_row.get('instance_index')} is missing subgraph."
            )

        triples: set[tuple[str, str, str]] = set()
        for triple in subgraph:
            if not isinstance(triple, dict):
                continue
            triples.add(
                (
                    self._normalize_text(str(triple.get("source", ""))),
                    self._normalize_text(str(triple.get("relation", ""))),
                    self._normalize_text(str(triple.get("target", ""))),
                )
            )
        return triples

    def _extract_explanation_triples(self, explanation: str) -> set[tuple[str, str, str]]:
        triples: set[tuple[str, str, str]] = set()
        for match in self.arrow_triple_pattern.finditer(explanation):
            triples.add(
                (
                    self._normalize_text(match.group(1)),
                    self._normalize_text(match.group(2)),
                    self._normalize_text(match.group(3)),
                )
            )
        return triples

    @staticmethod
    def _triple_to_text(triple: tuple[str, str, str]) -> str:
        return " ".join(triple)

    def _prediction_answers(self, answer_row: dict[str, Any]) -> list[str]:
        if answer_row.get("error_message") is not None:
            return []

        raw_answer = str(answer_row.get("answer", ""))
        if self._normalize_text(raw_answer) in self.unknown_answer_values:
            return []

        return [
            item.strip()
            for item in raw_answer.split(",")
            if item.strip()
        ]

    def _normalize_answer_set(self, answers: list[str]) -> list[str]:
        return sorted(set(self._normalize_answer_list(answers)))

    def _normalize_answer_list(self, answers: list[str]) -> list[str]:
        normalized_answers: list[str] = []
        seen_answers: set[str] = set()
        for answer in self._split_answer_values(answers):
            normalized_answer = self._normalize_answer(answer)
            if not normalized_answer or normalized_answer in seen_answers:
                continue
            normalized_answers.append(normalized_answer)
            seen_answers.add(normalized_answer)
        return normalized_answers

    @staticmethod
    def _split_answer_values(answers: list[str]) -> list[str]:
        return [
            part.strip()
            for answer in answers
            for part in answer.split(",")
            if part.strip()
        ]

    def _normalize_answer(self, answer: str) -> str:
        normalized = answer.lower().strip()
        normalized = normalized.translate(str.maketrans("", "", punctuation))
        normalized = self.article_pattern.sub(" ", normalized)
        return self.whitespace_pattern.sub(" ", normalized).strip()

    def _normalize_text(self, value: str) -> str:
        normalized = value.lower().strip()
        normalized = normalized.translate(str.maketrans("", "", punctuation.replace(".", "")))
        return self.whitespace_pattern.sub(" ", normalized).strip()

    @staticmethod
    def _ndcg_at_k(
        prediction: EvaluatedAnswerRetrievalInstance,
        k: int,
    ) -> float:
        if k <= 0 or not prediction.answer_candidates:
            return 0.0

        candidates = sorted(
            prediction.answer_candidates,
            key=lambda candidate: candidate.probability,
            reverse=True,
        )[:k]
        relevances = [1.0 if candidate.is_gold_answer else 0.0 for candidate in candidates]
        dcg = sum(
            relevance / math.log2(rank + 2)
            for rank, relevance in enumerate(relevances)
        )
        ideal_relevant_count = min(
            sum(1 for candidate in prediction.answer_candidates if candidate.is_gold_answer),
            k,
        )
        if ideal_relevant_count <= 0:
            return 0.0

        idcg = sum(
            1.0 / math.log2(rank + 2)
            for rank in range(ideal_relevant_count)
        )
        return dcg / idcg

    @staticmethod
    def _safe_divide(numerator: int | float, denominator: int | float) -> float:
        if denominator == 0:
            return 0.0

        return numerator / denominator

    @classmethod
    def _f1(cls, precision: float, recall: float) -> float:
        return cls._safe_divide(2 * precision * recall, precision + recall)

    @staticmethod
    def _mean(values: list[float]) -> float:
        if not values:
            return 0.0

        return sum(values) / len(values)

    @staticmethod
    def _load_jsonl_objects(path: Path) -> list[dict[str, Any]]:
        try:
            rows: list[dict[str, Any]] = []
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise FinalResultsEvaluationException(
                        f"JSONL row {line_number} in {path} must be an object."
                    )
                rows.append(value)
            return rows
        except OSError as error:
            raise FinalResultsEvaluationException(
                f"Could not read JSONL file {path}: {error}"
            ) from error
        except json.JSONDecodeError as error:
            raise FinalResultsEvaluationException(
                f"Invalid JSONL content in {path}: {error}"
            ) from error

    @staticmethod
    def _load_json_object(path: Path) -> dict[str, Any]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except OSError as error:
            raise FinalResultsEvaluationException(
                f"Could not read JSON file {path}: {error}"
            ) from error
        except json.JSONDecodeError as error:
            raise FinalResultsEvaluationException(
                f"Invalid JSON content in {path}: {error}"
            ) from error

        if not isinstance(value, dict):
            raise FinalResultsEvaluationException(f"JSON file {path} must contain an object.")
        return value

    @staticmethod
    def _index_rows(
        rows: list[dict[str, Any]],
        row_kind: str,
    ) -> dict[int, dict[str, Any]]:
        indexed_rows: dict[int, dict[str, Any]] = {}
        for row in rows:
            instance_index = row.get("instance_index")
            if not isinstance(instance_index, int):
                raise FinalResultsEvaluationException(
                    f"{row_kind} row is missing integer instance_index."
                )
            if instance_index in indexed_rows:
                raise FinalResultsEvaluationException(
                    f"{row_kind} contains duplicate instance_index={instance_index}."
                )
            indexed_rows[instance_index] = row
        return indexed_rows

    @staticmethod
    def _validate_index_sets(
        answers_by_index: dict[int, dict[str, Any]],
        reasoning_by_index: dict[int, dict[str, Any]],
        predictions_by_index: dict[int, EvaluatedAnswerRetrievalInstance],
    ) -> None:
        answer_indexes = set(answers_by_index)
        reasoning_indexes = set(reasoning_by_index)
        prediction_indexes = set(predictions_by_index)
        if answer_indexes != reasoning_indexes or answer_indexes != prediction_indexes:
            raise FinalResultsEvaluationException(
                "Final results source files must contain the same instance indexes. "
                f"answers={sorted(answer_indexes)} reasoning={sorted(reasoning_indexes)} "
                f"predictions={sorted(prediction_indexes)}"
            )

    @staticmethod
    def _extract_candidate_limit(evaluation_config: dict[str, Any]) -> int:
        raw_candidate_limit = evaluation_config.get("evaluation", {}).get("candidate_limit")
        if isinstance(raw_candidate_limit, int) and raw_candidate_limit > 0:
            return raw_candidate_limit

        return 10

    def _create_results_run_directory(self, results_root: Path) -> Path:
        results_root.mkdir(parents=True, exist_ok=True)
        run_number = self._next_run_number(results_root)
        run_label = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        results_run_directory = results_root / f"{run_number}_{run_label}"
        results_run_directory.mkdir(parents=True, exist_ok=False)
        return results_run_directory

    @classmethod
    def _next_run_number(cls, results_root: Path) -> int:
        existing_run_numbers = [
            cls._extract_run_number(path.name)
            for path in results_root.iterdir()
            if path.is_dir() and cls._extract_run_number(path.name) > 0
        ]
        return max(existing_run_numbers, default=0) + 1

    @staticmethod
    def _extract_run_number(run_directory_name: str) -> int:
        run_number_match = re.match(r"^(\d+)_", run_directory_name)
        if run_number_match is None:
            return 0

        return int(run_number_match.group(1))
