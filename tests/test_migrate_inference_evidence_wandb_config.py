"""Tests for the one-time inference evidence W&B config migration."""

from __future__ import annotations

import copy

import pytest

from scripts.migrate_inference_evidence_wandb_config import migrate_remote_config


def test_migration_moves_evidence_out_of_inference_without_duplication() -> None:
    original = {
        "configs": {
            "inference": {
                "model_id": "deepseek-v4-flash",
                "evidence_subgraph": {
                    "algorithm": "pcst",
                    "pcst": {
                        "edge_cost_strategy": "constant",
                        "edge_cost_lambda": 1.0,
                    },
                },
            }
        }
    }
    untouched = copy.deepcopy(original)

    repaired, changed = migrate_remote_config(
        original,
        local_evidence=original["configs"]["inference"]["evidence_subgraph"],
    )

    assert changed
    assert original == untouched
    assert repaired["configs"]["evidence"]["algorithm"] == "pcst"
    assert "evidence_subgraph" not in repaired["configs"]["inference"]
    assert repaired["configs"]["inference"]["model_id"] == "deepseek-v4-flash"


def test_migration_is_idempotent_when_remote_config_is_already_fixed() -> None:
    evidence = {"algorithm": "shortest_path"}
    original = {
        "configs": {
            "evidence": evidence,
            "inference": {"model_id": "deepseek-v4-flash"},
        }
    }

    repaired, changed = migrate_remote_config(
        original,
        local_evidence=evidence,
    )

    assert not changed
    assert repaired == original


def test_migration_rejects_conflicting_existing_evidence() -> None:
    with pytest.raises(ValueError, match="conflicts"):
        migrate_remote_config(
            {
                "configs": {
                    "evidence": {"algorithm": "shortest_path"},
                    "inference": {
                        "evidence_subgraph": {"algorithm": "pcst"},
                    },
                }
            },
            local_evidence={"algorithm": "pcst"},
        )
