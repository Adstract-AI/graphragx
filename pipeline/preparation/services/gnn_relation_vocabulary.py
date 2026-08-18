"""Stable relation-vocabulary helpers for categorical GNN architectures."""

from __future__ import annotations

import hashlib
import json
from typing import Any


RGCN_RELATION_VOCABULARY_FILENAME = "relation_vocabulary.json"
RGCN_ARCHITECTURE_CONTEXT_VERSION = 1


def validate_relation_vocabulary(vocabulary: dict[str, int]) -> None:
    """Validate a deterministic contiguous relation-to-id mapping."""
    if not vocabulary:
        raise ValueError("Relation-aware architectures require a non-empty vocabulary.")
    if any(
        not isinstance(key, str) or not isinstance(value, int)
        for key, value in vocabulary.items()
    ):
        raise ValueError("Relation vocabulary entries must map strings to integers.")
    expected_ids = list(range(len(vocabulary)))
    if sorted(vocabulary.values()) != expected_ids:
        raise ValueError("Relation vocabulary ids must be unique and contiguous from zero.")


def relation_vocabulary_sha256(vocabulary: dict[str, int]) -> str:
    """Return a stable hash for one relation mapping."""
    validate_relation_vocabulary(vocabulary)
    canonical = json.dumps(
        vocabulary,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def build_relation_architecture_context(
    vocabulary: dict[str, int],
) -> dict[str, Any]:
    """Build persisted structural metadata for an R-GCN model."""
    return {
        "version": RGCN_ARCHITECTURE_CONTEXT_VERSION,
        "relation_type_count": len(vocabulary),
        "relation_vocabulary_sha256": relation_vocabulary_sha256(vocabulary),
    }


def validate_relation_architecture_context(
    context: dict[str, Any],
    vocabulary: dict[str, int],
) -> None:
    """Ensure saved structural metadata matches its vocabulary artifact."""
    expected = build_relation_architecture_context(vocabulary)
    for key, expected_value in expected.items():
        if context.get(key) != expected_value:
            raise ValueError(
                f"R-GCN architecture context {key}={context.get(key)!r} does not "
                f"match relation vocabulary value {expected_value!r}."
            )


def relation_ids_for_edges(
    edge_relations: list[str],
    vocabulary: dict[str, int],
) -> list[int]:
    """Resolve edge relation strings through an authoritative saved mapping."""
    missing = sorted(
        {relation for relation in edge_relations if relation not in vocabulary}
    )
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(
            f"Graph contains {len(missing)} relations missing from the saved R-GCN "
            f"vocabulary: {preview}"
        )
    return [vocabulary[relation] for relation in edge_relations]


def build_sorted_typed_edges(
    *,
    edge_index,
    edge_relations: list[str],
    vocabulary: dict[str, int],
    torch,
):
    """Build aligned categorical edge types sorted by relation ID."""
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, edge_count]")
    if edge_index.shape[1] != len(edge_relations):
        raise ValueError(
            "edge_index and edge_relations must contain the same number of edges"
        )
    edge_type = torch.tensor(
        relation_ids_for_edges(edge_relations, vocabulary),
        dtype=torch.long,
    )
    if edge_type.numel() == 0:
        return edge_index, edge_type
    order = torch.argsort(edge_type, stable=True)
    return edge_index.index_select(1, order), edge_type.index_select(0, order)
