"""Architecture resolution helpers shared by configuration and artifact loaders."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pipeline.preparation.helpers.configuration_definitions import (
    AA_GRAPH_SAGE_ARCHITECTURE_ID,
    GNN_ARCHITECTURES,
    GRAPH_SAGE_ARCHITECTURE_ID,
)

ADVANCED_GNN_BOOLEAN_FIELDS = (
    "use_edge_mlp",
    "use_reverse_edges",
    "question_aware_classifier",
    "add_layer_normalization",
)


def infer_gnn_architecture(config: Mapping[str, Any]) -> str:
    """Resolve an explicit architecture or infer one from a legacy config."""
    explicit = config.get("gnn_architecture")
    if explicit in GNN_ARCHITECTURES:
        return str(explicit)

    training = config.get("training")
    training_config = training if isinstance(training, Mapping) else {}
    training_explicit = training_config.get("gnn_architecture")
    if training_explicit in GNN_ARCHITECTURES:
        return str(training_explicit)
    if any(bool(config.get(field) or training_config.get(field)) for field in ADVANCED_GNN_BOOLEAN_FIELDS):
        return AA_GRAPH_SAGE_ARCHITECTURE_ID
    return GRAPH_SAGE_ARCHITECTURE_ID


def architecture_defaults(architecture_id: str) -> dict[str, Any]:
    """Return canonical runtime defaults for an architecture."""
    definition = GNN_ARCHITECTURES[architecture_id]
    return {
        "gnn_architecture": architecture_id,
        "gnn_layer_count": definition.default_layer_count,
        "gnn_hidden_dimension": definition.default_hidden_dimension,
        "node_classifier": definition.default_classifier,
        "dropout": definition.default_dropout,
        "use_edge_mlp": definition.default_use_edge_mlp,
        "use_reverse_edges": definition.default_use_reverse_edges,
        "question_aware_classifier": definition.default_question_aware_classifier,
        "add_layer_normalization": definition.default_add_layer_normalization,
        "edge_mlp_hidden_dim": definition.default_edge_mlp_hidden_dimension,
    }
