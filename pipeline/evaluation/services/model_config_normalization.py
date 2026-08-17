"""Canonical model-config views used for persistence compatibility and W&B."""

from __future__ import annotations

from typing import Any

from pipeline.preparation.helpers.configuration_definitions import GNN_ARCHITECTURES
from pipeline.preparation.helpers.gnn_architecture import (
    architecture_defaults,
    infer_gnn_architecture,
)


_TRAINING_KEYS = (
    "epochs",
    "learning_rate",
    "weight_decay",
    "max_instances",
    "start_instance",
    "log_every",
    "device",
    "profile",
    "embedding_cache_device",
    "embedding_cache_dtype",
    "loss_function",
)

_MODEL_TRAINING_OUTCOME_KEYS = (
    "training_start_instance",
    "training_end_instance",
    "trained_instance_range",
    "loss_history",
    "final_loss",
    "trained_instances",
)


def normalize_model_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return one canonical, de-duplicated model configuration.

    New model files already use this shape. Older files are projected into it
    without being modified, so loading historical runs remains compatible.
    """
    if not isinstance(config, dict):
        return {}

    raw = {key: value for key, value in config.items() if key != "wandb"}
    raw_training = raw.get("training")
    training = raw_training if isinstance(raw_training, dict) else {}

    try:
        architecture = infer_gnn_architecture(raw)
    except (KeyError, ValueError):
        architecture = raw.get("gnn_architecture") or "graphsage"
    architecture = str(architecture)

    persisted_options = raw.get("gnn_architecture_options")
    if not isinstance(persisted_options, dict):
        persisted_options = training.get("gnn_architecture_options", {})
    if not isinstance(persisted_options, dict):
        persisted_options = {}

    legacy_values = {
        "gnn_layer_count": raw.get("gnn_layer_count", training.get("gnn_layer_count")),
        "gnn_hidden_dimension": raw.get(
            "gnn_hidden_dimension",
            training.get("gnn_hidden_dimension", training.get("hidden_dimension")),
        ),
        "node_classifier": raw.get("node_classifier", training.get("node_classifier")),
        "dropout": raw.get("dropout", training.get("dropout")),
        "use_edge_mlp": raw.get("use_edge_mlp", training.get("use_edge_mlp")),
        "use_reverse_edges": raw.get(
            "use_reverse_edges", training.get("use_reverse_edges")
        ),
        "question_aware_classifier": raw.get(
            "question_aware_classifier", training.get("question_aware_classifier")
        ),
        "add_layer_normalization": raw.get(
            "add_layer_normalization", training.get("add_layer_normalization")
        ),
        "edge_mlp_hidden_dim": raw.get(
            "edge_mlp_hidden_dim", training.get("edge_mlp_hidden_dim")
        ),
    }
    try:
        defaults = architecture_defaults(architecture)
    except KeyError:
        defaults = {}
    defaults.pop("gnn_architecture", None)
    supported = GNN_ARCHITECTURES.get(architecture)
    supported_ids = set(supported.option_map) if supported is not None else set(defaults)
    architecture_options = {
        option_id: persisted_options.get(
            option_id,
            legacy_values.get(option_id, default),
        )
        for option_id, default in defaults.items()
        if option_id in supported_ids
    }
    if architecture_options.get("use_edge_mlp") is False:
        architecture_options["edge_mlp_hidden_dim"] = None
    architecture_options = {
        key: value for key, value in architecture_options.items() if value is not None
    }

    embedding_model = (
        raw.get("embedding_model")
        or raw.get("entity_embedding_model")
        or raw.get("question_embedding_model")
        or raw.get("relation_embedding_model")
    )
    embedding_dimension = (
        raw.get("embedding_dimension")
        or raw.get("entity_embedding_dimension")
        or raw.get("question_embedding_dimension")
        or raw.get("relation_embedding_dimension")
    )

    canonical: dict[str, Any] = {}
    for key in (
        "dataset_id",
        "run_name",
        "run_number",
        "is_fine_tuned_model",
        "continued_from_model_run_name",
        "continued_from_model_run_number",
    ):
        if key in raw and raw[key] is not None:
            canonical[key] = raw[key]
    canonical.update(
        {
            "gnn_architecture": architecture,
            "gnn_architecture_options": architecture_options,
        }
    )
    if embedding_model is not None:
        canonical["embedding_model"] = embedding_model
    if embedding_dimension is not None:
        canonical["embedding_dimension"] = embedding_dimension
    for key in _MODEL_TRAINING_OUTCOME_KEYS:
        if key in raw:
            canonical[key] = raw[key]
        elif key in training:
            canonical[key] = training[key]

    canonical_training: dict[str, Any] = {}
    for key in _TRAINING_KEYS:
        if key in training:
            canonical_training[key] = training[key]
        elif key in raw:
            canonical_training[key] = raw[key]
    if canonical_training:
        canonical["training"] = canonical_training
    return canonical
