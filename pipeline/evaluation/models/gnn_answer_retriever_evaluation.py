"""Models for GNN answer-retriever evaluation."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from helpers.constants import (
    DEFAULT_ANSWER_THRESHOLD,
    DEFAULT_CANDIDATE_LIMIT,
    DEFAULT_CANDIDATE_TOP_K,
    DEFAULT_EVALUATION_LOG_EVERY,
)
from pipeline.abstract import StepResult


class GnnAnswerRetrieverEvaluationConfig(BaseModel):
    """Runtime settings for evaluating a saved GNN answer-retriever run."""

    model_run_name: str | None = Field(default=None)
    model_run_number: int | None = Field(default=None)
    answer_threshold: float = Field(default=DEFAULT_ANSWER_THRESHOLD)
    candidate_top_k: int = Field(default=DEFAULT_CANDIDATE_TOP_K)
    candidate_limit: int = Field(default=DEFAULT_CANDIDATE_LIMIT)
    run_name: str | None = Field(default=None)
    max_instances: int | None = Field(default=None)
    log_every: int = Field(default=DEFAULT_EVALUATION_LOG_EVERY)


class AnswerCandidateScore(BaseModel):
    """Score for one selected answer-candidate node."""

    node: str = Field(..., description="Candidate node text.")
    local_node_id: int = Field(..., description="Candidate local graph node id.")
    global_node_id: int = Field(..., description="Global nodes.json vocabulary id.")
    logit: float = Field(..., description="Raw classifier logit.")
    probability: float = Field(..., description="Sigmoid classifier probability.")
    is_gold_answer: bool = Field(..., description="Whether the candidate is gold.")
    selection_reason: str = Field(
        ...,
        description="Whether the node was selected by threshold or fallback top-k.",
    )


class GoldAnswerScore(BaseModel):
    """Score for a gold answer node, even when it was not selected."""

    node: str = Field(..., description="Gold answer node text.")
    local_node_id: int | None = Field(
        default=None,
        description="Local node id when the gold answer is present in the graph.",
    )
    global_node_id: int | None = Field(
        default=None,
        description="Global nodes.json vocabulary id when the node is known.",
    )
    logit: float | None = Field(default=None, description="Raw classifier logit.")
    probability: float | None = Field(default=None, description="Sigmoid probability.")
    present_in_graph: bool = Field(
        ...,
        description="Whether the gold answer appears in the local graph.",
    )


class EvaluatedAnswerRetrievalInstance(BaseModel):
    """Persisted retrieval evaluation output for one WebQSP test instance."""

    instance_index: int = Field(..., description="Index inside the prepared test split.")
    question: str = Field(..., description="Natural language question.")
    q_entity: list[str] = Field(..., description="Question/topic entities.")
    a_entity: list[str] = Field(..., description="Gold answer entities.")
    answer_candidates: list[AnswerCandidateScore] = Field(default_factory=list)
    gold_answer_scores: list[GoldAnswerScore] = Field(default_factory=list)
    hit_at_1: bool = Field(..., description="Whether top scored node is gold.")
    hit_at_5: bool = Field(
        default=False,
        description="Whether any top-5 selected answer candidate is gold.",
    )
    hit_at_10: bool = Field(
        default=False,
        description="Whether any top-10 selected answer candidate is gold.",
    )
    hit_at_candidate_limit: bool = Field(
        default=False,
        description="Whether any selected answer candidate is gold.",
    )
    missing_gold_in_graph: bool = Field(
        ...,
        description="Whether no gold answer appears in the local graph.",
    )


class GnnAnswerRetrieverEvaluationResult(StepResult):
    """Saved evaluation result for a GNN answer-retriever run."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    dataset_id: str = Field(..., description="Evaluated dataset identifier.")
    model_run_directory: Path = Field(..., description="Selected model run directory.")
    model_run_name: str = Field(..., description="Selected model run folder name.")
    model_run_number: int = Field(..., description="Selected model run number.")
    evaluation_run_directory: Path = Field(
        ...,
        description="Created evaluation run directory.",
    )
    evaluation_run_name: str = Field(..., description="Created evaluation run folder name.")
    evaluation_run_number: int = Field(..., description="Created evaluation run number.")
    evaluated_instances: int = Field(..., description="Number of evaluated instances.")
    hits_at_1: float = Field(..., description="Hits@1 rate.")
    hits_at_1_count: int = Field(..., description="Hits@1 count.")
    hits_at_5: float = Field(default=0.0, description="Hits@5 rate.")
    hits_at_5_count: int = Field(default=0, description="Hits@5 count.")
    hits_at_10: float = Field(default=0.0, description="Hits@10 rate.")
    hits_at_10_count: int = Field(default=0, description="Hits@10 count.")
    hits_at_candidate_limit: float = Field(
        default=0.0,
        description="Hits at configured candidate limit rate.",
    )
    hits_at_candidate_limit_count: int = Field(
        default=0,
        description="Hits at configured candidate limit count.",
    )
    average_candidate_count: float = Field(
        ...,
        description="Average number of selected candidates.",
    )
    missing_gold_in_graph_count: int = Field(
        ...,
        description="Instances where no gold answer appears in the local graph.",
    )
    predictions_path: Path = Field(..., description="Saved JSONL predictions path.")
    evaluation_config_path: Path = Field(..., description="Saved evaluation config path.")
