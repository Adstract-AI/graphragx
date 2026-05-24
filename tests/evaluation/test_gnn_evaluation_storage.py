"""Tests for GNN evaluation storage payload shapes."""

from pipeline.services.gnn_answer_retriever_evaluation_storage import (
    GnnAnswerRetrieverEvaluationStoragePayload,
)


def test_evaluation_storage_payload_accepts_nested_model_config_reference() -> None:
    payload = GnnAnswerRetrieverEvaluationStoragePayload(
        evaluation_config={
            "dataset_id": "WebQSP",
            "model_config": {
                "model_run_name": "1_model",
                "model_run_number": 1,
                "full_config_path": "/tmp/models/1_model/model_config.json",
                "weights_path": "/tmp/models/1_model/gnn_answer_retriever.pt",
            },
            "evaluation": {
                "candidate_limit": 15,
            },
        }
    )

    assert payload.evaluation_config["model_config"]["model_run_number"] == 1
