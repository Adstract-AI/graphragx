"""Evaluation phase components for graphragX."""

from typing import Any

from pipeline.evaluation.models import (
    AnswerCandidateScore,
    BuiltReasoningSamples,
    CandidateNodeScore,
    CandidateNodeScores,
    EvaluationSample,
    EvaluatedAnswerRetrievalInstance,
    ExtractedReasoningPathsBatch,
    ExtractedReasoningPaths,
    GeneratedAnswerForPrediction,
    GeneratedFinalAnswer,
    GeneratedFinalAnswersBatch,
    GnnAnswerRetrieverEvaluationConfig,
    GnnAnswerRetrieverEvaluationResult,
    GoldAnswerScore,
    GraphTriple,
    ReasoningPathsForPrediction,
    ReasoningPath,
    ReasoningSampleForPrediction,
    SavedLlmInferenceRun,
)
from pipeline.evaluation.exceptions import (
    InvalidEvaluationSampleException,
    LlmAnswerGenerationException,
    ShortestPathExtractionException,
)

_LAZY_EXPORT_MODULES: dict[str, str] = {
    "BuildReasoningSamplesFromGnnEvaluationContext": "pipeline.evaluation.steps.llm_inference",
    "BuildReasoningSamplesFromGnnEvaluationStep": "pipeline.evaluation.steps.llm_inference",
    "EvaluateGnnAnswerRetrieverContext": "pipeline.evaluation.steps.gnn_answer_retriever_evaluation",
    "EvaluateGnnAnswerRetrieverStep": "pipeline.evaluation.steps.gnn_answer_retriever_evaluation",
    "ExtractShortestPathsBatchStep": "pipeline.evaluation.steps.llm_inference",
    "ExtractShortestPathsStep": "pipeline.evaluation.steps.path_extraction",
    "GnnPredictionCandidateScoringStep": "pipeline.evaluation.steps.gnn_prediction_candidate_scoring",
    "GenerateAndSaveFinalAnswersBatchesStep": "pipeline.evaluation.steps.llm_inference",
    "GenerateAndSaveFinalAnswersBatchesContext": "pipeline.evaluation.steps.llm_inference",
    "GenerateFinalAnswersBatchStep": "pipeline.evaluation.steps.llm_inference",
    "GenerateFinalAnswerStep": "pipeline.evaluation.steps.llm_answer_generation",
    "LangChainOpenAiAnswerGenerationService": "pipeline.evaluation.services.llm_answer_generation",
    "LlmInferenceStoragePayload": "pipeline.evaluation.services.llm_inference_storage",
    "LlmInferenceStorageResult": "pipeline.evaluation.services.llm_inference_storage",
    "LlmInferenceStorageService": "pipeline.evaluation.services.llm_inference_storage",
    "MockCandidateNodeScoringStep": "pipeline.evaluation.steps.mock_candidate_scoring",
    "SaveInferenceRunStep": "pipeline.evaluation.steps.llm_inference",
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
    "BuildReasoningSamplesFromGnnEvaluationContext",
    "BuildReasoningSamplesFromGnnEvaluationStep",
    "BuiltReasoningSamples",
    "CandidateNodeScore",
    "CandidateNodeScores",
    "EvaluationSample",
    "EvaluateGnnAnswerRetrieverContext",
    "EvaluateGnnAnswerRetrieverStep",
    "EvaluatedAnswerRetrievalInstance",
    "ExtractedReasoningPathsBatch",
    "ExtractedReasoningPaths",
    "ExtractShortestPathsBatchStep",
    "ExtractShortestPathsStep",
    "GeneratedAnswerForPrediction",
    "GeneratedFinalAnswer",
    "GeneratedFinalAnswersBatch",
    "GenerateAndSaveFinalAnswersBatchesStep",
    "GenerateAndSaveFinalAnswersBatchesContext",
    "GenerateFinalAnswersBatchStep",
    "GenerateFinalAnswerStep",
    "GnnAnswerRetrieverEvaluationConfig",
    "GnnAnswerRetrieverEvaluationResult",
    "GnnPredictionCandidateScoringStep",
    "GoldAnswerScore",
    "GraphTriple",
    "InvalidEvaluationSampleException",
    "LangChainOpenAiAnswerGenerationService",
    "LlmInferenceStoragePayload",
    "LlmInferenceStorageResult",
    "LlmInferenceStorageService",
    "LlmAnswerGenerationException",
    "MockCandidateNodeScoringStep",
    "ReasoningPathsForPrediction",
    "ReasoningPath",
    "ReasoningSampleForPrediction",
    "SaveInferenceRunStep",
    "SavedLlmInferenceRun",
    "ShortestPathExtractionException",
    "ShortestPathExtractionService",
]
