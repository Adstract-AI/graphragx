"""Tests for architecture registry defaults and legacy inference."""

from pipeline.preparation.helpers.gnn_architecture import (
    architecture_defaults,
    infer_gnn_architecture,
)


def test_architecture_defaults_are_stable() -> None:
    baseline = architecture_defaults("graphsage")
    advanced = architecture_defaults("aa-graphsage")

    assert baseline["gnn_layer_count"] == 2
    assert baseline["gnn_hidden_dimension"] == 256
    assert baseline["node_classifier"] == "mlp"
    assert baseline["dropout"] == 0.1
    assert not baseline["use_edge_mlp"]
    assert advanced["use_edge_mlp"]
    assert advanced["use_reverse_edges"]
    assert advanced["question_aware_classifier"]
    assert advanced["add_layer_normalization"]
    assert advanced["edge_mlp_hidden_dim"] == 256


def test_legacy_architecture_inference_ignores_dropout_and_unused_edge_width() -> None:
    assert infer_gnn_architecture(
        {"dropout": 0.1, "edge_mlp_hidden_dim": 512}
    ) == "graphsage"


def test_legacy_architecture_inference_detects_any_advanced_boolean() -> None:
    assert infer_gnn_architecture(
        {"training": {"use_reverse_edges": True}}
    ) == "aa-graphsage"


def test_explicit_architecture_takes_precedence_over_legacy_fields() -> None:
    assert infer_gnn_architecture(
        {"gnn_architecture": "graphsage", "use_edge_mlp": True}
    ) == "graphsage"
