"""Entry point for running the graphragX framework."""

from __future__ import annotations

import argparse
import json
from typing import Any, Literal, Sequence

from pydantic import BaseModel

from helpers.constants import (
    DEFAULT_ANSWER_THRESHOLD,
    DEFAULT_CANDIDATE_TOP_K,
    DEFAULT_TRAINING_DEVICE,
    DEFAULT_TRAINING_EPOCHS,
    DEFAULT_TRAINING_LEARNING_RATE,
    DEFAULT_TRAINING_LOG_EVERY,
    DEFAULT_TRAINING_WEIGHT_DECAY,
)
from helpers.logging_config import setup_logger
from pipeline import (
    BuildGnnAnswerRetrieverStep,
    BuildPipelineConfigurationStep,
    BuildWebQSPLocalGraphsStep,
    InitialStepResult,
    LoadDatasetStep,
    Pipeline,
    PipelineExecutionResult,
    SelectDatasetStep,
    StepContext,
    TrainGnnAnswerRetrieverStep,
    EvaluateGnnAnswerRetrieverStep,
    BuildReasoningSamplesFromGnnEvaluationStep,
    ExtractShortestPathsBatchStep,
    GenerateAndSaveFinalAnswersBatchesStep,
    PipelineException,
)
from pipeline.preparation.helpers.configuration_definitions import (
    RECOMMENDED_ASSISTANT_LLM_MODEL_ID,
    RECOMMENDED_CONTEXT_CONSTRUCTION_STRATEGY_ID,
    RECOMMENDED_ENTITY_EMBEDDING_MODEL_ID,
    RECOMMENDED_GNN_HIDDEN_DIMENSION,
    RECOMMENDED_GNN_LAYER_COUNT,
    RECOMMENDED_MAIN_LLM_MODEL_ID,
    RECOMMENDED_NODE_CLASSIFIER_ID,
    RECOMMENDED_QUESTION_EMBEDDING_MODEL_ID,
    RECOMMENDED_RELATION_EMBEDDING_MODEL_ID,
    RECOMMENDED_SUBGRAPH_CONSTRUCTION_ALGORITHM_ID,
)
from pipeline.preparation.helpers.dataset_definitions import WEBQSP_DATASET_ID


class PipelineRuntimeConfig(BaseModel):
    """Runtime configuration used to initialize the current framework pipeline."""

    run_mode: Literal["full", "train-only", "evaluation-only"] = "full"
    dataset: str | None = None
    main_llm_model: str | None = None
    assistant_llm_model: str | None = None
    subgraph_algorithm: str | None = None
    context_strategy: str | None = None
    gnn_layer_count: int | None = None
    gnn_hidden_dimension: int | None = None
    node_classifier: str | None = None
    question_embedding_model: str | None = None
    relation_embedding_model: str | None = None
    entity_embedding_model: str | None = None
    training_epochs: int = DEFAULT_TRAINING_EPOCHS
    training_learning_rate: float = DEFAULT_TRAINING_LEARNING_RATE
    training_weight_decay: float = DEFAULT_TRAINING_WEIGHT_DECAY
    training_max_instances: int | None = None
    training_log_every: int = DEFAULT_TRAINING_LOG_EVERY
    training_device: str = DEFAULT_TRAINING_DEVICE
    training_run_name: str | None = None
    evaluation_model_run_name: str | None = None
    evaluation_model_run_number: int | None = None
    answer_threshold: float = DEFAULT_ANSWER_THRESHOLD
    candidate_top_k: int = DEFAULT_CANDIDATE_TOP_K
    evaluation_run_name: str | None = None
    evaluation_max_instances: int | None = None
    with_llm_inference: bool = False
    inference_run_name: str | None = None
    llm_inference_batch_size: int = 10
    use_default_config_values: bool = False
    force_all_default: bool = False

    def with_defaulted_user_inputs(self) -> "PipelineRuntimeConfig":
        """Fill all user-provided selections with recommended defaults when requested."""
        if not self.use_default_config_values:
            return self

        return self.model_copy(
            update={
                "dataset": self.dataset or WEBQSP_DATASET_ID,
                "main_llm_model": self.main_llm_model or RECOMMENDED_MAIN_LLM_MODEL_ID,
                "assistant_llm_model": self.assistant_llm_model
                or RECOMMENDED_ASSISTANT_LLM_MODEL_ID,
                "subgraph_algorithm": self.subgraph_algorithm
                or RECOMMENDED_SUBGRAPH_CONSTRUCTION_ALGORITHM_ID,
                "context_strategy": self.context_strategy
                or RECOMMENDED_CONTEXT_CONSTRUCTION_STRATEGY_ID,
                "gnn_layer_count": self.gnn_layer_count
                or RECOMMENDED_GNN_LAYER_COUNT,
                "gnn_hidden_dimension": self.gnn_hidden_dimension
                or RECOMMENDED_GNN_HIDDEN_DIMENSION,
                "node_classifier": self.node_classifier
                or RECOMMENDED_NODE_CLASSIFIER_ID,
                "question_embedding_model": self.question_embedding_model
                or RECOMMENDED_QUESTION_EMBEDDING_MODEL_ID,
                "relation_embedding_model": self.relation_embedding_model
                or RECOMMENDED_RELATION_EMBEDDING_MODEL_ID,
                "entity_embedding_model": self.entity_embedding_model
                or RECOMMENDED_ENTITY_EMBEDDING_MODEL_ID,
            }
        )


def build_pipeline(config: PipelineRuntimeConfig) -> Pipeline:
    """Build the current runnable graphragX pipeline."""
    resolved_config = config.with_defaulted_user_inputs()
    setup_steps = [
        SelectDatasetStep(
            requested_dataset=resolved_config.dataset,
        ),
        BuildPipelineConfigurationStep(
            main_llm_model=resolved_config.main_llm_model,
            assistant_llm_model=resolved_config.assistant_llm_model,
            subgraph_algorithm=resolved_config.subgraph_algorithm,
            context_strategy=resolved_config.context_strategy,
            gnn_layer_count=resolved_config.gnn_layer_count,
            gnn_hidden_dimension=resolved_config.gnn_hidden_dimension,
            node_classifier=resolved_config.node_classifier,
            question_embedding_model=resolved_config.question_embedding_model,
            relation_embedding_model=resolved_config.relation_embedding_model,
            entity_embedding_model=resolved_config.entity_embedding_model,
        ),
        LoadDatasetStep(),
        BuildWebQSPLocalGraphsStep(),
    ]
    training_steps = [
        BuildGnnAnswerRetrieverStep(),
        TrainGnnAnswerRetrieverStep(
            training_epochs=resolved_config.training_epochs,
            training_learning_rate=resolved_config.training_learning_rate,
            training_weight_decay=resolved_config.training_weight_decay,
            training_max_instances=resolved_config.training_max_instances,
            training_log_every=resolved_config.training_log_every,
            training_device=resolved_config.training_device,
            training_run_name=resolved_config.training_run_name,
        ),
    ]
    evaluation_steps = [
        EvaluateGnnAnswerRetrieverStep(
            model_run_name=resolved_config.evaluation_model_run_name,
            model_run_number=resolved_config.evaluation_model_run_number,
            answer_threshold=resolved_config.answer_threshold,
            candidate_top_k=resolved_config.candidate_top_k,
            evaluation_run_name=resolved_config.evaluation_run_name,
            evaluation_max_instances=resolved_config.evaluation_max_instances,
        ),
    ]
    if resolved_config.with_llm_inference:
        evaluation_steps.extend(
            [
                BuildReasoningSamplesFromGnnEvaluationStep(),
                ExtractShortestPathsBatchStep(),
                GenerateAndSaveFinalAnswersBatchesStep(
                    model_id=resolved_config.main_llm_model,
                    inference_run_name=resolved_config.inference_run_name,
                    inference_batch_size=resolved_config.llm_inference_batch_size,
                ),
            ]
        )

    if resolved_config.run_mode == "train-only":
        preparation_steps = [*setup_steps, *training_steps]
        selected_evaluation_steps = []
    elif resolved_config.run_mode == "evaluation-only":
        preparation_steps = setup_steps
        selected_evaluation_steps = evaluation_steps
    else:
        preparation_steps = [*setup_steps, *training_steps]
        selected_evaluation_steps = evaluation_steps

    return Pipeline(
        preparation_steps=preparation_steps,
        evaluation_steps=selected_evaluation_steps,
        force_all_default=resolved_config.force_all_default,
    )


def run_pipeline(config: PipelineRuntimeConfig) -> PipelineExecutionResult:
    """Run the full graphragX pipeline."""
    if (
        config.run_mode == "evaluation-only"
        and config.evaluation_model_run_name is None
        and config.evaluation_model_run_number is None
    ):
        raise PipelineException(
            "Evaluation-only mode requires --evaluation-model-run-name or "
            "--evaluation-model-run-number."
        )

    pipeline = build_pipeline(config=config)
    initial_context = StepContext(result=InitialStepResult())
    if config.run_mode == "train-only":
        return pipeline.prepare(initial_context)

    return pipeline.run(initial_context)


def _serialize_value(value: Any) -> Any:
    """Convert nested Pydantic objects into JSON-serializable values."""
    if isinstance(value, BaseModel):
        return {key: _serialize_value(item) for key, item in value.model_dump().items()}

    if isinstance(value, list):
        return [_serialize_value(item) for item in value]

    if isinstance(value, dict):
        return {key: _serialize_value(item) for key, item in value.items()}

    return value


def serialize_execution_result(result: PipelineExecutionResult) -> dict[str, Any]:
    """Convert a pipeline execution result into a JSON-serializable dictionary."""
    payload = result.model_dump()
    payload["final_result"] = _serialize_value(result.final_result)
    return payload


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the framework entry point."""
    parser = argparse.ArgumentParser(description="Run graphragX.")
    run_mode_group = parser.add_mutually_exclusive_group()
    run_mode_group.add_argument(
        "--full",
        dest="run_mode",
        action="store_const",
        const="full",
        default="full",
        help="Run training and evaluation. This is the default.",
    )
    run_mode_group.add_argument(
        "--train-only",
        dest="run_mode",
        action="store_const",
        const="train-only",
        help="Run setup and GNN training, then stop successfully.",
    )
    run_mode_group.add_argument(
        "--evaluation-only",
        dest="run_mode",
        action="store_const",
        const="evaluation-only",
        help="Run setup and evaluate a saved model run.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset choice for the current run.",
    )
    parser.add_argument(
        "--main-llm-model",
        default=None,
        help="Optional main LLM model id for non-interactive configuration.",
    )
    parser.add_argument(
        "--assistant-llm-model",
        default=None,
        help="Optional assistant LLM model id for non-interactive configuration.",
    )
    parser.add_argument(
        "--subgraph-algorithm",
        default=None,
        help="Optional subgraph construction algorithm id for non-interactive configuration.",
    )
    parser.add_argument(
        "--context-strategy",
        default=None,
        help="Optional context construction strategy id for non-interactive configuration.",
    )
    parser.add_argument(
        "--gnn-layers",
        type=int,
        default=None,
        help="Optional number of GNN layers for non-interactive configuration.",
    )
    parser.add_argument(
        "--gnn-hidden-dim",
        type=int,
        default=None,
        help="Optional GNN hidden dimension for non-interactive configuration.",
    )
    parser.add_argument(
        "--node-classifier",
        default=None,
        help="Optional node classifier id for non-interactive configuration.",
    )
    parser.add_argument(
        "--question-embedding-model",
        default=None,
        help="Optional OpenAI embedding model id for question text.",
    )
    parser.add_argument(
        "--relation-embedding-model",
        default=None,
        help="Optional OpenAI embedding model id for relation text.",
    )
    parser.add_argument(
        "--entity-embedding-model",
        default=None,
        help="Optional OpenAI embedding model id for entity text.",
    )
    parser.add_argument(
        "--training-epochs",
        type=int,
        default=DEFAULT_TRAINING_EPOCHS,
        help="Number of GNN answer-retriever training epochs.",
    )
    parser.add_argument(
        "--training-learning-rate",
        type=float,
        default=DEFAULT_TRAINING_LEARNING_RATE,
        help="Learning rate for GNN answer-retriever training.",
    )
    parser.add_argument(
        "--training-weight-decay",
        type=float,
        default=DEFAULT_TRAINING_WEIGHT_DECAY,
        help="Weight decay for GNN answer-retriever training.",
    )
    parser.add_argument(
        "--training-max-instances",
        type=int,
        default=None,
        help="Optional maximum number of WebQSP training instances to use.",
    )
    parser.add_argument(
        "--training-log-every",
        type=int,
        default=DEFAULT_TRAINING_LOG_EVERY,
        help="Log training progress after this many instances.",
    )
    parser.add_argument(
        "--training-device",
        default=DEFAULT_TRAINING_DEVICE,
        choices=["auto", "cpu", "cuda", "mps"],
        help="PyTorch device used for GNN answer-retriever training.",
    )
    parser.add_argument(
        "--training-run-name",
        default=None,
        help="Optional label for the versioned training run folder.",
    )
    parser.add_argument(
        "--evaluation-model-run-name",
        default=None,
        help="Saved model run folder name or suffix to evaluate.",
    )
    parser.add_argument(
        "--evaluation-model-run-number",
        type=int,
        default=None,
        help="Saved model run numeric prefix to evaluate.",
    )
    parser.add_argument(
        "--answer-threshold",
        type=float,
        default=DEFAULT_ANSWER_THRESHOLD,
        help="Minimum answer-node probability for threshold candidate selection.",
    )
    parser.add_argument(
        "--candidate-top-k",
        type=int,
        default=DEFAULT_CANDIDATE_TOP_K,
        help="Fallback top-k candidate count when no node passes the threshold.",
    )
    parser.add_argument(
        "--evaluation-run-name",
        default=None,
        help="Optional label for the versioned evaluation run folder.",
    )
    parser.add_argument(
        "--evaluation-max-instances",
        type=int,
        default=None,
        help="Optional maximum number of WebQSP test instances to evaluate.",
    )
    parser.add_argument(
        "--with-llm-inference",
        action="store_true",
        help="Continue after GNN candidate retrieval with paths, LLM answers, and saved inference outputs.",
    )
    parser.add_argument(
        "--inference-run-name",
        default=None,
        help="Optional label for the versioned LLM inference run folder.",
    )
    parser.add_argument(
        "--llm-inference-batch-size",
        type=int,
        default=10,
        help="Number of samples to generate and save per LLM inference batch.",
    )
    parser.add_argument(
        "--default",
        dest="use_default_config_values",
        action="store_true",
        help="Use default values for all configurable user selections.",
    )
    parser.add_argument(
        "--force-default",
        dest="force_all_default",
        action="store_true",
        help="Force every pipeline step to use its execute_default path.",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for graphragX."""
    setup_logger()
    parser = build_parser()
    args = parser.parse_args(argv)
    runtime_config = PipelineRuntimeConfig(
        run_mode=args.run_mode,
        dataset=args.dataset,
        main_llm_model=args.main_llm_model,
        assistant_llm_model=args.assistant_llm_model,
        subgraph_algorithm=args.subgraph_algorithm,
        context_strategy=args.context_strategy,
        gnn_layer_count=args.gnn_layers,
        gnn_hidden_dimension=args.gnn_hidden_dim,
        node_classifier=args.node_classifier,
        question_embedding_model=args.question_embedding_model,
        relation_embedding_model=args.relation_embedding_model,
        entity_embedding_model=args.entity_embedding_model,
        training_epochs=args.training_epochs,
        training_learning_rate=args.training_learning_rate,
        training_weight_decay=args.training_weight_decay,
        training_max_instances=args.training_max_instances,
        training_log_every=args.training_log_every,
        training_device=args.training_device,
        training_run_name=args.training_run_name,
        evaluation_model_run_name=args.evaluation_model_run_name,
        evaluation_model_run_number=args.evaluation_model_run_number,
        answer_threshold=args.answer_threshold,
        candidate_top_k=args.candidate_top_k,
        evaluation_run_name=args.evaluation_run_name,
        evaluation_max_instances=args.evaluation_max_instances,
        with_llm_inference=args.with_llm_inference,
        inference_run_name=args.inference_run_name,
        llm_inference_batch_size=args.llm_inference_batch_size,
        use_default_config_values=args.use_default_config_values,
        force_all_default=args.force_all_default,
    )

    try:
        result = run_pipeline(config=runtime_config)
    except Exception as error:
        error_payload = {
            "success": False,
            "error_message": str(error),
            "exception_type": error.__class__.__name__,
        }
        print(json.dumps(error_payload, indent=2))
        return 1

    print(json.dumps(serialize_execution_result(result), indent=2, default=str))
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
