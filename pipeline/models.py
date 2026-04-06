"""Shared models for pipeline execution and neutral bootstrap artifacts."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from pipeline.abstract import StepResult
from pipeline.exceptions import PipelineException


class PipelineExecutionResult(BaseModel):
    """Result of a complete pipeline execution."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    success: bool = Field(..., description="Whether the pipeline completed successfully.")
    final_result: Optional[object] = Field(
        default=None,
        description="Final step result produced by the pipeline.",
    )
    execution_time_ms: Optional[float] = Field(
        default=None,
        description="Total execution time in milliseconds.",
    )
    steps_executed: int = Field(default=0, description="Number of steps executed.")
    total_steps: int = Field(default=0, description="Total configured steps.")
    error_message: Optional[str] = Field(
        default=None,
        description="Final error message if execution failed.",
    )
    exception_type: Optional[str] = Field(
        default=None,
        description="Final exception type if execution failed.",
    )
    executed_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When the pipeline execution finished.",
    )

    @classmethod
    def success_result(
        cls,
        final_result: object,
        execution_time_ms: float,
        steps_executed: int,
        total_steps: int,
    ) -> "PipelineExecutionResult":
        """Create a successful execution result."""
        return cls(
            success=True,
            final_result=final_result,
            execution_time_ms=execution_time_ms,
            steps_executed=steps_executed,
            total_steps=total_steps,
        )

    @classmethod
    def error_result(
        cls,
        error: Exception,
        execution_time_ms: float,
        steps_executed: int,
        total_steps: int,
        final_result: object | None = None,
    ) -> "PipelineExecutionResult":
        """Create a failed execution result."""
        return cls(
            success=False,
            final_result=final_result,
            execution_time_ms=execution_time_ms,
            steps_executed=steps_executed,
            total_steps=total_steps,
            error_message=str(error),
            exception_type=error.__class__.__name__,
        )


class InitialStepResult(StepResult):
    """Neutral bootstrap artifact for the first pipeline context."""


class PipelineResultBank(BaseModel):
    """In-memory bank of latest step results keyed by result type."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    stored_results: dict[str, StepResult] = Field(
        default_factory=dict,
        description="Latest stored result for each result type.",
    )

    def store(self, result: StepResult) -> None:
        """Store the latest result for its concrete result type."""
        self.stored_results[self._get_result_key(type(result))] = result

    def get(self, result_type: type[StepResult]) -> StepResult | None:
        """Return the latest stored result for the given type."""
        return self.stored_results.get(self._get_result_key(result_type))

    def get_required(self, result_type: type[StepResult]) -> StepResult:
        """Return a stored result or raise when the result type is missing."""
        result = self.get(result_type)
        if result is None:
            raise PipelineException(
                f"No stored result found for type {result_type.__name__}."
            )

        return result

    def has(self, result_type: type[StepResult]) -> bool:
        """Check whether a result type is stored in the bank."""
        return self._get_result_key(result_type) in self.stored_results

    def clear(self) -> None:
        """Clear all stored results."""
        self.stored_results.clear()

    @staticmethod
    def _get_result_key(result_type: type[StepResult]) -> str:
        """Build a stable internal key for a result type."""
        return f"{result_type.__module__}.{result_type.__qualname__}"
