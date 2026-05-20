"""Evaluation phase components for graphragX."""

from typing import Any

from pipeline.evaluation.models import (
    AnswerCandidateScore,
    CandidateNodeScore,
    CandidateNodeScores,
    EvaluationSample,
    EvaluatedAnswerRetrievalInstance,
    ExtractedReasoningPaths,
    GeneratedFinalAnswer,
    GnnAnswerRetrieverEvaluationConfig,
    GnnAnswerRetrieverEvaluationResult,
    GoldAnswerScore,
    GraphTriple,
    ReasoningPath,
)
from pipeline.evaluation.exceptions import (
    InvalidEvaluationSampleException,
    LlmAnswerGenerationException,
    ShortestPathExtractionException,
)

_LAZY_EXPORT_MODULES: dict[str, str] = {
    "EvaluateGnnAnswerRetrieverContext": "pipeline.evaluation.steps.gnn_answer_retriever_evaluation",
    "EvaluateGnnAnswerRetrieverStep": "pipeline.evaluation.steps.gnn_answer_retriever_evaluation",
    "ExtractShortestPathsStep": "pipeline.evaluation.steps.path_extraction",
    "GnnPredictionCandidateScoringStep": "pipeline.evaluation.steps.gnn_prediction_candidate_scoring",
    "GenerateFinalAnswerStep": "pipeline.evaluation.steps.llm_answer_generation",
    "LangChainOpenAiAnswerGenerationService": "pipeline.evaluation.services.llm_answer_generation",
    "MockCandidateNodeScoringStep": "pipeline.evaluation.steps.mock_candidate_scoring",
    "ShortestPathExtractionService": "pipeline.evaluation.services.shortest_path_extraction",
}


def __getattr__(name: str) -> Any:
    """Lazy-load evaluation steps and services to avoid package import cycles."""
    module_name = _LAZY_EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


__all__ = [
    "AnswerCandidateScore",
    "CandidateNodeScore",
    "CandidateNodeScores",
    "EvaluationSample",
    "EvaluateGnnAnswerRetrieverContext",
    "EvaluateGnnAnswerRetrieverStep",
    "EvaluatedAnswerRetrievalInstance",
    "ExtractedReasoningPaths",
    "ExtractShortestPathsStep",
    "GeneratedFinalAnswer",
    "GenerateFinalAnswerStep",
    "GnnAnswerRetrieverEvaluationConfig",
    "GnnAnswerRetrieverEvaluationResult",
    "GnnPredictionCandidateScoringStep",
    "GoldAnswerScore",
    "GraphTriple",
    "InvalidEvaluationSampleException",
    "LangChainOpenAiAnswerGenerationService",
    "LlmAnswerGenerationException",
    "MockCandidateNodeScoringStep",
    "ReasoningPath",
    "ShortestPathExtractionException",
    "ShortestPathExtractionService",
]
