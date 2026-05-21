"""Batch models for post-retrieval LLM inference runs."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from pipeline.abstract import StepResult
from pipeline.evaluation.models.gnn_answer_retriever_evaluation import (
    EvaluatedAnswerRetrievalInstance,
)
from pipeline.evaluation.models.path_extraction import (
    CandidateNodeScores,
    ExtractedReasoningPaths,
    GraphTriple,
)


class ReasoningSampleForPrediction(BaseModel):
    """One GNN prediction converted into a graph reasoning sample."""

    instance_index: int = Field(..., description="Index inside the prepared test split.")
    prediction: EvaluatedAnswerRetrievalInstance = Field(
        ...,
        description="GNN answer-retriever prediction for the instance.",
    )
    candidate_scores: CandidateNodeScores = Field(
        ...,
        description="Candidate scores and local graph sample for path extraction.",
    )


class BuiltReasoningSamples(StepResult):
    """Batch of GNN predictions converted into path-extraction inputs."""

    dataset_id: str = Field(..., description="Dataset identifier.")
    evaluation_run_name: str = Field(..., description="Source GNN evaluation run name.")
    evaluation_run_directory: Path = Field(
        ...,
        description="Source GNN evaluation run directory.",
    )
    samples: list[ReasoningSampleForPrediction] = Field(default_factory=list)


class ReasoningPathsForPrediction(BaseModel):
    """Extracted shortest paths for one GNN prediction."""

    instance_index: int = Field(..., description="Index inside the prepared test split.")
    prediction: EvaluatedAnswerRetrievalInstance = Field(
        ...,
        description="Source GNN answer-retriever prediction.",
    )
    extracted_paths: ExtractedReasoningPaths = Field(
        ...,
        description="Shortest paths and deduplicated reasoning subgraph.",
    )


class ExtractedReasoningPathsBatch(StepResult):
    """Batch output of shortest-path extraction over GNN candidates."""

    dataset_id: str = Field(..., description="Dataset identifier.")
    evaluation_run_name: str = Field(..., description="Source GNN evaluation run name.")
    items: list[ReasoningPathsForPrediction] = Field(default_factory=list)


class GeneratedAnswerForPrediction(BaseModel):
    """Generated LLM answer for one path-extracted prediction."""

    instance_index: int = Field(..., description="Index inside the prepared test split.")
    question: str = Field(..., description="Natural language question.")
    q_entity: list[str] = Field(default_factory=list)
    a_entity: list[str] = Field(default_factory=list)
    answer_candidates: list[str] = Field(default_factory=list)
    reasoning_subgraph_triples: list[GraphTriple] = Field(default_factory=list)
    reasoning_paths_text: str = Field(default="")
    model_id: str = Field(..., description="LLM model used for answer generation.")
    prompt: str = Field(default="", description="Prompt sent to the LLM.")
    answer: str = Field(default="", description="Generated final answer.")
    error_message: str | None = Field(
        default=None,
        description="Generation error when this instance failed.",
    )


class GeneratedFinalAnswersBatch(StepResult):
    """Batch of LLM-generated answers for extracted reasoning paths."""

    dataset_id: str = Field(..., description="Dataset identifier.")
    evaluation_run_name: str = Field(..., description="Source GNN evaluation run name.")
    model_id: str = Field(..., description="LLM model used for answer generation.")
    items: list[GeneratedAnswerForPrediction] = Field(default_factory=list)

    @property
    def successful_answers(self) -> int:
        """Return the number of successfully generated answers."""
        return sum(1 for item in self.items if item.error_message is None)

    @property
    def failed_answers(self) -> int:
        """Return the number of failed answer generations."""
        return sum(1 for item in self.items if item.error_message is not None)


class SavedLlmInferenceRun(StepResult):
    """Persisted inference run metadata."""

    dataset_id: str = Field(..., description="Dataset identifier.")
    evaluation_run_name: str = Field(..., description="Source GNN evaluation run name.")
    inference_run_directory: Path = Field(..., description="Created inference directory.")
    inference_run_name: str = Field(..., description="Created inference run folder name.")
    inference_run_number: int = Field(..., description="Created inference run number.")
    model_id: str = Field(..., description="LLM model used for answer generation.")
    total_instances: int = Field(..., description="Number of instances processed.")
    successful_answers: int = Field(..., description="Number of successful generations.")
    failed_answers: int = Field(..., description="Number of failed generations.")
    reasoning_paths_path: Path = Field(..., description="Saved reasoning paths path.")
    reasoning_subgraphs_path: Path = Field(..., description="Saved subgraphs path.")
    prompts_path: Path = Field(..., description="Saved prompts path.")
    answers_path: Path = Field(..., description="Saved answers path.")
    summary_path: Path = Field(..., description="Saved summary path.")
