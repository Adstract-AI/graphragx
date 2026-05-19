"""Entry point for running the graphragX framework."""

from __future__ import annotations

import argparse
import json
from typing import Any, Sequence

from pydantic import BaseModel

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
    return Pipeline(
        preparation_steps=[
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
            BuildGnnAnswerRetrieverStep(),
        ],
        evaluation_steps=[],
        force_all_default=resolved_config.force_all_default,
    )


def run_pipeline(config: PipelineRuntimeConfig) -> PipelineExecutionResult:
    """Run the full graphragX pipeline."""
    pipeline = build_pipeline(config=config)
    initial_context = StepContext(result=InitialStepResult())
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
    parser = build_parser()
    args = parser.parse_args(argv)
    runtime_config = PipelineRuntimeConfig(
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
