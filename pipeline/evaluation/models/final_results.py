"""Models for final retrieval and reasoning evaluation results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from pipeline.abstract import StepResult


class FinalResultsConfig(BaseModel):
    """Source artifact paths and run metadata for one final results run."""

    dataset_id: str = Field(..., description="Evaluated dataset identifier.")
    model_id: str = Field(..., description="LLM model used for answer generation.")
    gnn_id: str = Field(..., description="Compact GNN architecture id.")
    run_name: str | None = Field(default=None, description="Created results run name.")
    run_number: int | None = Field(default=None, description="Created results run number.")
    configs: dict[str, Path] = Field(default_factory=dict)
    artifacts: dict[str, Any] = Field(default_factory=dict)


class FinalAnswerMetrics(BaseModel):
    """Aggregate final answer metrics."""

    evaluated_instances: int = Field(..., description="Number of evaluated answers.")
    successful_answers: int = Field(..., description="Answers without generation errors.")
    failed_answers: int = Field(..., description="Answers with generation errors.")
    exact_match_count: int = Field(..., description="Exact set match count.")
    accuracy: float = Field(..., description="Exact set match rate.")
    hit_count: int = Field(
        ...,
        description="Questions where any predicted answer matches a gold answer.",
    )
    hit_rate: float = Field(..., description="Share of questions with any hit.")
    hits_at_1_count: int = Field(
        ...,
        description="Questions where the first predicted answer is gold.",
    )
    hits_at_1: float = Field(
        ...,
        description="Share of questions where the first predicted answer is gold.",
    )
    true_positive_count: int = Field(..., description="Micro true positives.")
    false_positive_count: int = Field(..., description="Micro false positives.")
    false_negative_count: int = Field(..., description="Micro false negatives.")
    precision: float = Field(..., description="Micro precision.")
    recall: float = Field(..., description="Micro recall.")
    f1: float = Field(..., description="Micro F1 score.")


class ExplanationGroundingMetrics(BaseModel):
    """Aggregate explanation grounding metrics."""

    grounded_explanation_count: int = Field(
        ...,
        description="Instances with at least one grounded explanation triple.",
    )
    fully_grounded_explanation_count: int = Field(
        ...,
        description="Instances where every mentioned explanation triple is grounded.",
    )
    grounded_explanation_rate: float = Field(
        ...,
        description="Share of instances with any grounded explanation triple.",
    )
    fully_grounded_explanation_rate: float = Field(
        ...,
        description="Share of instances with fully grounded explanation triples.",
    )
    mentioned_triple_count: int = Field(
        ...,
        description="Total explanation-mentioned triples.",
    )
    grounded_mentioned_triple_count: int = Field(
        ...,
        description="Total explanation-mentioned triples found in the subgraph.",
    )


class RankingMetrics(BaseModel):
    """Aggregate retrieval ranking metrics."""

    ndcg_at_1: float = Field(..., description="Mean nDCG@1.")
    ndcg_at_5: float = Field(..., description="Mean nDCG@5.")
    ndcg_at_10: float = Field(..., description="Mean nDCG@10.")
    ndcg_at_candidate_limit: float = Field(
        ...,
        description="Mean nDCG at the configured candidate limit.",
    )
    candidate_limit: int = Field(..., description="Configured candidate limit.")


class PerInstanceFinalResult(BaseModel):
    """Per-question final metrics row."""

    instance_index: int = Field(..., description="Instance index in the evaluation set.")
    question: str = Field(..., description="Question text.")
    q_entity: list[str] = Field(default_factory=list)
    gold_answers: list[str] = Field(default_factory=list)
    predicted_answers: list[str] = Field(default_factory=list)
    normalized_gold_answers: list[str] = Field(default_factory=list)
    normalized_predicted_answers: list[str] = Field(default_factory=list)
    exact_match: bool = Field(..., description="Whether normalized answer sets match.")
    hit: bool = Field(
        ...,
        description="Whether any predicted answer matches a gold answer.",
    )
    hits_at_1: bool = Field(
        ...,
        description="Whether the first predicted answer matches a gold answer.",
    )
    true_positive_count: int = Field(..., description="Answer true positives.")
    false_positive_count: int = Field(..., description="Answer false positives.")
    false_negative_count: int = Field(..., description="Answer false negatives.")
    precision: float = Field(..., description="Per-instance precision.")
    recall: float = Field(..., description="Per-instance recall.")
    f1: float = Field(..., description="Per-instance F1.")
    answer_error_message: str | None = Field(
        default=None,
        description="LLM generation error for the answer row.",
    )
    mentioned_triple_count: int = Field(
        ...,
        description="Explanation-mentioned triples.",
    )
    grounded_mentioned_triple_count: int = Field(
        ...,
        description="Explanation-mentioned triples found in the subgraph.",
    )
    grounded_explanation: bool = Field(
        ...,
        description="Whether at least one mentioned explanation triple is grounded.",
    )
    fully_grounded_explanation: bool = Field(
        ...,
        description="Whether every mentioned explanation triple is grounded.",
    )
    ndcg_at_1: float = Field(..., description="Instance nDCG@1.")
    ndcg_at_5: float = Field(..., description="Instance nDCG@5.")
    ndcg_at_10: float = Field(..., description="Instance nDCG@10.")
    ndcg_at_candidate_limit: float = Field(
        ...,
        description="Instance nDCG at the configured candidate limit.",
    )


class FinalReasoningMetrics(BaseModel):
    """Complete aggregate final metrics payload."""

    dataset_id: str = Field(..., description="Evaluated dataset identifier.")
    evaluation_run_name: str = Field(..., description="Source GNN evaluation run name.")
    inference_run_name: str = Field(..., description="Source LLM inference run name.")
    model_run_name: str = Field(..., description="Source GNN model run name.")
    model_id: str = Field(..., description="LLM model id.")
    answer_metrics: FinalAnswerMetrics
    explanation_grounding_metrics: ExplanationGroundingMetrics
    ranking_metrics: RankingMetrics

    def flattened_payload(self) -> dict[str, Any]:
        """Return a flat JSON payload for easy report consumption."""
        return {
            "dataset_id": self.dataset_id,
            "evaluation_run_name": self.evaluation_run_name,
            "inference_run_name": self.inference_run_name,
            "model_run_name": self.model_run_name,
            "model_id": self.model_id,
            **self.answer_metrics.model_dump(mode="json"),
            **self.explanation_grounding_metrics.model_dump(mode="json"),
            **self.ranking_metrics.model_dump(mode="json"),
        }


class FinalResultsEvaluationResult(StepResult):
    """Saved final results run metadata."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    dataset_id: str = Field(..., description="Evaluated dataset identifier.")
    results_run_directory: Path = Field(..., description="Created results directory.")
    results_run_name: str = Field(..., description="Created results run folder name.")
    results_run_number: int = Field(..., description="Created results run number.")
    results_config_path: Path = Field(..., description="Saved results config path.")
    retrieval_metrics_path: Path = Field(..., description="Copied retrieval metrics path.")
    reasoning_metrics_path: Path = Field(..., description="Saved reasoning metrics path.")
    per_instance_results_path: Path = Field(..., description="Saved per-instance rows path.")
    evaluated_instances: int = Field(..., description="Number of evaluated instances.")
    accuracy: float = Field(..., description="Final answer exact-match accuracy.")
    hit_rate: float = Field(..., description="Final answer hit rate.")
    hits_at_1: float = Field(..., description="Final answer Hits@1.")
    precision: float = Field(..., description="Final answer micro precision.")
    recall: float = Field(..., description="Final answer micro recall.")
    f1: float = Field(..., description="Final answer micro F1.")
    grounded_explanation_rate: float = Field(
        ...,
        description="Share of instances with grounded explanations.",
    )
    ndcg_at_10: float = Field(..., description="Mean nDCG@10.")
    wandb_status: str | None = Field(
        default=None,
        description="Optional WandB logging status.",
    )
    wandb_run_id: str | None = Field(
        default=None,
        description="Optional WandB run id.",
    )
    wandb_run_url: str | None = Field(
        default=None,
        description="Optional WandB run URL.",
    )
    wandb_error_message: str | None = Field(
        default=None,
        description="Optional WandB logging error.",
    )
