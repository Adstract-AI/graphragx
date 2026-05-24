"""Tests for GNN evaluation storage payload shapes."""

from pipeline.services.gnn_answer_retriever_evaluation_storage import (
    GnnAnswerRetrieverEvaluationStoragePayload,
)


def test_evaluation_storage_payload_accepts_model_config_loss_history() -> None:
    payload = GnnAnswerRetrieverEvaluationStoragePayload(
        evaluation_config={
            "dataset_id": "WebQSP",
            "model_run": {
                "name": "1_model",
                "number": 1,
            },
            "model_configuration": {
                "dataset_id": "WebQSP",
                "training": {
                    "epochs": 3,
                    "learning_rate": 0.001,
                    "run_name": None,
                },
                "loss_function": "BCEWithLogitsLoss",
                "loss_history": [
                    {
                        "epoch": 1,
                        "average_loss": 1.4,
                    },
                    {
                        "epoch": 2,
                        "average_loss": 1.1,
                    },
                ],
                "final_loss": 1.1,
                "trained_instances": 10,
            },
            "evaluation": {
                "candidate_limit": 15,
            },
        }
    )

    assert payload.evaluation_config["model_configuration"]["loss_history"][0][
        "average_loss"
    ] == 1.4
