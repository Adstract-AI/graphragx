"""Pipeline orchestrator for shared graphragX step execution."""

from __future__ import annotations

import time
from typing import Iterable

from pipeline.abstract import AbstractStep, StepContext
from pipeline.context_builder import StepContextBuilder
from pipeline.models import PipelineExecutionResult


class Pipeline:
    """Execute preparation and evaluation phases with shared pipeline primitives."""

    def __init__(
        self,
        preparation_steps: Iterable[AbstractStep] | None = None,
        evaluation_steps: Iterable[AbstractStep] | None = None,
        context_builder: StepContextBuilder | None = None,
        force_all_default: bool = False,
    ):
        self.preparation_steps = list(preparation_steps or [])
        self.evaluation_steps = list(evaluation_steps or [])
        self.context_builder = context_builder or StepContextBuilder()
        self.force_all_default = force_all_default

        if self.force_all_default:
            for step in [*self.preparation_steps, *self.evaluation_steps]:
                step.force_default = True

    def prepare(self, initial_context: StepContext) -> PipelineExecutionResult:
        """Run the preparation phase."""
        return self._run_steps(
            self.preparation_steps,
            initial_context,
            reset_result_bank=True,
        )

    def evaluate(self, initial_context: StepContext) -> PipelineExecutionResult:
        """Run the evaluation phase."""
        return self._run_steps(
            self.evaluation_steps,
            initial_context,
            reset_result_bank=True,
        )

    def run(self, initial_context: StepContext) -> PipelineExecutionResult:
        """Run the full pipeline by executing preparation and then evaluation."""
        self.context_builder.clear_result_bank()

        preparation_result = self._run_steps(
            self.preparation_steps,
            initial_context,
            reset_result_bank=False,
        )
        if not preparation_result.success:
            return preparation_result

        evaluation_context = self.context_builder.create_context(
            result=preparation_result.final_result,
        )
        evaluation_result = self._run_steps(
            self.evaluation_steps,
            evaluation_context,
            reset_result_bank=False,
        )
        if not evaluation_result.success:
            return evaluation_result

        total_steps_executed = (
            preparation_result.steps_executed + evaluation_result.steps_executed
        )
        total_steps = preparation_result.total_steps + evaluation_result.total_steps
        total_execution_time_ms = (
            (preparation_result.execution_time_ms or 0.0)
            + (evaluation_result.execution_time_ms or 0.0)
        )

        return PipelineExecutionResult.success_result(
            final_result=evaluation_result.final_result,
            execution_time_ms=total_execution_time_ms,
            steps_executed=total_steps_executed,
            total_steps=total_steps,
        )

    def _run_steps(
        self,
        steps: list[AbstractStep],
        initial_context: StepContext,
        reset_result_bank: bool,
    ) -> PipelineExecutionResult:
        """Run a configured phase in order and return success/failure metadata."""
        if reset_result_bank:
            self.context_builder.clear_result_bank()

        start_time = time.perf_counter()
        steps_executed = 0
        current_context = initial_context
        last_result = current_context.result
        total_steps = len(steps)

        try:
            for index, step in enumerate(steps):
                last_result = step.execute(current_context)
                steps_executed += 1
                next_step = steps[index + 1] if index + 1 < total_steps else None
                current_context = self.context_builder.create_context(
                    result=last_result,
                    next_step=next_step,
                )

            execution_time_ms = (time.perf_counter() - start_time) * 1000
            return PipelineExecutionResult.success_result(
                final_result=last_result,
                execution_time_ms=execution_time_ms,
                steps_executed=steps_executed,
                total_steps=total_steps,
            )
        except Exception as error:
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            return PipelineExecutionResult.error_result(
                error=error,
                execution_time_ms=execution_time_ms,
                steps_executed=steps_executed,
                total_steps=total_steps,
                final_result=last_result,
            )
