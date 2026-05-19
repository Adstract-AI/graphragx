"""Centralized utilities for building step contexts between pipeline stages."""

from __future__ import annotations

from collections.abc import Callable
from typing import Optional

from pipeline.abstract import AbstractStep, StepContext, StepResult
from pipeline.exceptions import PipelineException
from pipeline.models import PipelineResultBank
from pipeline.preparation.steps.configuration_building import BuiltPipelineConfiguration
from pipeline.preparation.steps.gnn_model_building import (
    BuildGnnAnswerRetrieverContext,
    BuildGnnAnswerRetrieverStep,
)


class StepContextBuilder:
    """Creates step contexts from prior step outputs."""

    def __init__(self) -> None:
        self.result_bank = PipelineResultBank()
        self.context_factories: dict[
            type[AbstractStep],
            Callable[[StepResult | None, bool, PipelineException], StepContext],
        ] = {
            BuildGnnAnswerRetrieverStep: self._create_gnn_answer_retriever_context,
        }

    def create_context(
        self,
        result: Optional[StepResult],
        outcome: bool = True,
        exception=None,
        next_step: AbstractStep | None = None,
    ) -> StepContext:
        """Wrap a previous result into the context required by the next step."""
        if result is not None:
            self.store_result(result)

        if next_step is not None:
            for step_type, context_factory in self.context_factories.items():
                if isinstance(next_step, step_type):
                    return context_factory(result, outcome, exception)

        return StepContext(
            result=result,
            outcome=outcome,
            exception=exception,
        )

    def _create_gnn_answer_retriever_context(
        self,
        result: StepResult | None,
        outcome: bool,
        exception: PipelineException,
    ) -> BuildGnnAnswerRetrieverContext:
        """Create the specialized context required by the GNN builder step."""
        return BuildGnnAnswerRetrieverContext(
            result=result,
            outcome=outcome,
            exception=exception,
            pipeline_configuration=self.get_required_result(
                BuiltPipelineConfiguration
            ),
        )

    def store_result(self, result: StepResult) -> None:
        """Store a result in the shared in-memory bank."""
        self.result_bank.store(result)

    def get_stored_result(self, result_type: type[StepResult]) -> StepResult | None:
        """Return the latest stored result for a given result type."""
        return self.result_bank.get(result_type)

    def get_required_result(self, result_type: type[StepResult]) -> StepResult:
        """Return the latest stored result or raise when it is missing."""
        return self.result_bank.get_required(result_type)

    def has_stored_result(self, result_type: type[StepResult]) -> bool:
        """Check whether the given result type exists in the bank."""
        return self.result_bank.has(result_type)

    def clear_result_bank(self) -> None:
        """Clear the shared result bank."""
        self.result_bank.clear()
