"""Temporary mock candidate scoring for evaluation path extraction."""

from __future__ import annotations

from pipeline.abstract import AbstractStep, StepContext
from pipeline.evaluation.exceptions import InvalidEvaluationSampleException
from pipeline.evaluation.models import (
    CandidateNodeScore,
    CandidateNodeScores,
    EvaluationSample,
)


class MockCandidateNodeScoringStep(AbstractStep[CandidateNodeScores, EvaluationSample]):
    """Create deterministic mock answer candidates until trained GNN scores are available."""

    def __init__(self, top_k: int = 5, force_default: bool = False):
        super().__init__(force_default=force_default)
        self.top_k = top_k

    def execute_default(
        self,
        context: StepContext[EvaluationSample],
    ) -> CandidateNodeScores:
        sample = context.result
        if sample is None:
            raise InvalidEvaluationSampleException(
                "Mock candidate scoring requires an evaluation sample."
            )

        candidates = self._build_mock_candidates(sample)
        return CandidateNodeScores(
            sample=sample,
            candidates=candidates,
            top_k=self.top_k,
        )

    def _build_mock_candidates(self, sample: EvaluationSample) -> list[CandidateNodeScore]:
        """Rank gold answer entities first, then deterministic graph distractors."""
        seen_nodes: set[str] = set()
        candidates: list[CandidateNodeScore] = []

        for answer_index, answer_entity in enumerate(sample.a_entities):
            if answer_entity in seen_nodes:
                continue
            seen_nodes.add(answer_entity)
            candidates.append(
                CandidateNodeScore(
                    node_id=answer_entity,
                    score=1.0 - (answer_index * 0.01),
                    source="mock_gold",
                )
            )

        distractor_nodes = sorted(
            {
                node
                for triple in sample.graph_triples
                for node in (triple.source, triple.target)
                if node not in seen_nodes
            }
        )
        for distractor_index, distractor_node in enumerate(distractor_nodes):
            if len(candidates) >= self.top_k:
                break
            seen_nodes.add(distractor_node)
            candidates.append(
                CandidateNodeScore(
                    node_id=distractor_node,
                    score=max(0.0, 0.5 - (distractor_index * 0.01)),
                    source="mock_distractor",
                )
            )

        return candidates[: self.top_k]
