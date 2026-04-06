"""Entry point for running the graphragX framework."""

from __future__ import annotations

import argparse
import json
from typing import Any, Sequence

from pydantic import BaseModel

from pipeline import (
    KnowledgeGraphDatasetSelection,
    Pipeline,
    PipelineExecutionResult,
    SelectKnowledgeGraphDatasetStep,
    StepContext,
)


def build_pipeline(force_all_default: bool = False) -> Pipeline:
    """Build the current runnable graphragX pipeline."""
    return Pipeline(
        preparation_steps=[SelectKnowledgeGraphDatasetStep()],
        evaluation_steps=[],
        force_all_default=force_all_default,
    )


def run_pipeline(
    dataset: str,
    force_all_default: bool = False,
) -> PipelineExecutionResult:
    """Run the full graphragX pipeline."""
    pipeline = build_pipeline(force_all_default=force_all_default)
    initial_context = StepContext(
        result=KnowledgeGraphDatasetSelection(requested_dataset=dataset),
    )
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
        default="FB15K-237",
        help="Knowledge graph dataset choice for the current run.",
    )
    parser.add_argument(
        "--force-default",
        action="store_true",
        help="Force steps to use their default execution path.",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for graphragX."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        result = run_pipeline(
            dataset=args.dataset,
            force_all_default=args.force_default,
        )
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
