"""Shortest path extraction and reasoning subgraph construction."""

from __future__ import annotations

from collections import deque

from pipeline.evaluation.models import (
    CandidateNodeScore,
    EvaluationSample,
    ExtractedReasoningPaths,
    GraphTriple,
    ReasoningPath,
)
from pipeline.services import AbstractService


class ShortestPathExtractionService(AbstractService):
    """Find shortest paths and merge them into a compact reasoning subgraph."""

    def extract_paths(
        self,
        sample: EvaluationSample,
        candidates: list[CandidateNodeScore],
    ) -> ExtractedReasoningPaths:
        """
        Extract candidate paths and a readable text block for LLM context.

        Args:
            sample: Evaluation sample containing q-entities and local graph triples.
            candidates: Ranked candidate answer nodes.

        Returns:
            ExtractedReasoningPaths: Structured paths and deduplicated subgraph text.
        """
        adjacency = self._build_undirected_adjacency(sample.graph_triples)
        paths = [
            self._build_candidate_path(sample, candidate, adjacency)
            for candidate in candidates
        ]
        reasoning_subgraph_triples = self.build_reasoning_subgraph(paths)
        reasoning_paths_text = self.verbalize_subgraph(reasoning_subgraph_triples)
        found_paths = sum(1 for path in paths if path.path_found)

        return ExtractedReasoningPaths(
            sample=sample,
            paths=paths,
            reasoning_subgraph_triples=reasoning_subgraph_triples,
            reasoning_paths_text=reasoning_paths_text,
            found_paths=found_paths,
            missing_paths=len(paths) - found_paths,
        )

    @staticmethod
    def build_reasoning_subgraph(paths: list[ReasoningPath]) -> list[GraphTriple]:
        """Merge found shortest-path triples into one deduplicated subgraph."""
        seen_triples: set[tuple[str, str, str]] = set()
        subgraph_triples: list[GraphTriple] = []
        for path in paths:
            if not path.path_found:
                continue
            for shortest_path in path.shortest_paths:
                for triple in shortest_path:
                    triple_key = (triple.source, triple.relation, triple.target)
                    if triple_key in seen_triples:
                        continue
                    seen_triples.add(triple_key)
                    subgraph_triples.append(triple)

        return subgraph_triples

    @staticmethod
    def verbalize_subgraph(triples: list[GraphTriple]) -> str:
        """Convert the deduplicated reasoning subgraph into LLM context text."""
        if not triples:
            return "No reasoning subgraph found."

        triple_lines = [
            f"{triple.source} -> {triple.relation} -> {triple.target}"
            for triple in triples
        ]
        return "\n".join(["Reasoning subgraph:", *triple_lines])

    def _build_candidate_path(
        self,
        sample: EvaluationSample,
        candidate: CandidateNodeScore,
        adjacency: dict[str, list[tuple[str, GraphTriple]]],
    ) -> ReasoningPath:
        """Return all equal-length shortest paths from any q-entity to one candidate."""
        shortest_paths = self._find_shortest_path_triples(
            start_nodes=sample.q_entities,
            target_node=candidate.node_id,
            adjacency=adjacency,
        )
        return ReasoningPath(
            candidate_node=candidate.node_id,
            candidate_score=candidate.score,
            path_found=bool(shortest_paths),
            triples=shortest_paths[0] if shortest_paths else [],
            shortest_paths=shortest_paths,
        )

    @staticmethod
    def _build_undirected_adjacency(
        triples: list[GraphTriple],
    ) -> dict[str, list[tuple[str, GraphTriple]]]:
        """Build undirected traversal adjacency while preserving original triples."""
        adjacency: dict[str, list[tuple[str, GraphTriple]]] = {}
        for triple in triples:
            adjacency.setdefault(triple.source, []).append((triple.target, triple))
            adjacency.setdefault(triple.target, []).append((triple.source, triple))

        for neighbors in adjacency.values():
            neighbors.sort(key=lambda item: (item[0], item[1].relation))

        return adjacency

    def _find_shortest_path_triples(
        self,
        start_nodes: list[str],
        target_node: str,
        adjacency: dict[str, list[tuple[str, GraphTriple]]],
    ) -> list[list[GraphTriple]]:
        """Run deterministic BFS and return every equal-length shortest path."""
        if target_node in start_nodes:
            return [[]]

        shortest_paths: list[list[GraphTriple]] = []
        shortest_length: int | None = None
        seen_path_keys: set[tuple[tuple[str, str, str], ...]] = set()
        queue = deque(
            (start_node, [], (start_node,)) for start_node in sorted(start_nodes)
        )

        while queue:
            current_node, path_triples, path_nodes = queue.popleft()
            if shortest_length is not None and len(path_triples) >= shortest_length:
                continue

            for next_node, triple in adjacency.get(current_node, []):
                if next_node in path_nodes:
                    continue

                next_path = [*path_triples, triple]
                next_path_length = len(next_path)
                if shortest_length is not None and next_path_length > shortest_length:
                    continue

                if next_node == target_node:
                    if shortest_length is None:
                        shortest_length = next_path_length
                    path_key = tuple(
                        (item.source, item.relation, item.target)
                        for item in next_path
                    )
                    if path_key not in seen_path_keys:
                        seen_path_keys.add(path_key)
                        shortest_paths.append(next_path)
                    continue

                queue.append((next_node, next_path, (*path_nodes, next_node)))

        return shortest_paths
