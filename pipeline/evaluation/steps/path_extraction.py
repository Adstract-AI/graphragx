"""Pipeline step for extracting reasoning paths from candidate nodes."""

from __future__ import annotations

from pipeline.abstract import AbstractStep, StepContext
from pipeline.evaluation.exceptions import ShortestPathExtractionException
from pipeline.evaluation.models import CandidateNodeScores, ExtractedReasoningPaths
from pipeline.evaluation.services import ShortestPathExtractionService


class ExtractShortestPathsStep(
    AbstractStep[ExtractedReasoningPaths, CandidateNodeScores]
):
    """Extract shortest reasoning paths for ranked candidate answer nodes."""

    def __init__(
        self,
        shortest_path_service: ShortestPathExtractionService | None = None,
        force_default: bool = False,
    ):
        super().__init__(force_default=force_default)
        self.shortest_path_service = shortest_path_service or ShortestPathExtractionService()

    def execute_default(
        self,
        context: StepContext[CandidateNodeScores],
    ) -> ExtractedReasoningPaths:
        candidate_scores = context.result
        if candidate_scores is None:
            raise ShortestPathExtractionException(
                "Shortest path extraction requires candidate node scores."
            )

        return self.shortest_path_service.extract_paths(
            sample=candidate_scores.sample,
            candidates=candidate_scores.candidates,
        )
