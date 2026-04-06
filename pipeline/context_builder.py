"""Centralized utilities for building step contexts between pipeline stages."""

from __future__ import annotations

from typing import Optional

from pipeline.abstract import StepContext, StepResult
from pipeline.models import PipelineResultBank


class StepContextBuilder:
    """Creates step contexts from prior step outputs."""

    def __init__(self) -> None:
        self.result_bank = PipelineResultBank()

    def create_context(
            self,
            result: Optional[StepResult],
            outcome: bool = True,
            exception=None,
    ) -> StepContext:
        """Wrap a previous result into a context for the next step."""
        if result is not None:
            self.store_result(result)

        return StepContext(
            result=result,
            outcome=outcome,
            exception=exception)

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
