"""Shared models for pipeline execution and neutral bootstrap artifacts."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from pipeline.abstract import StepResult


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
