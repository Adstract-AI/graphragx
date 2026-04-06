"""
Ad Injection Pipeline Implementation.

This module contains the main Pipeline class that orchestrates the execution
of all steps in the ad injection process, from initial processing to final
AdResponse creation.
"""
import time
from typing import List, Optional

from ad_injection.pipeline.abstract import AbstractStep, StepContext
from ad_injection.pipeline.models import ExecutionResult
from ad_injection.pipeline.steps.finishing_step import FinishingStep
from ad_injection.pipeline.step_context_builder import StepContextBuilder
from ad_injection.pipeline.step_context_enrichment import StepContextEnrichment
from ad_injection.pipeline.exceptions import (
    PipelineCriticalFailureException,
    PipelineExecutionPersistedErrorException,
)
from ad_injection.pipeline.ad_injection_exception_resolver import AdInjectionExceptionResolver
from ad_request.models import AdRequestStatus
from ad_response.models import AdResponseStatus
from ad_response.services.ad_response_service import AdResponseService
from helper_functions.logger import get_logger


logger = get_logger(__file__)


class Pipeline:
    """
    Main ad injection pipeline that orchestrates the execution of all steps.

    The pipeline takes a list of processing steps and one finishing step,
    executes them in sequence, and always returns an ExecutionResult containing
    an AdResponse (either successful or error).

    The pipeline is designed to be fault-tolerant - it will always produce
    an AdResponse, even if exceptions occur during execution.
    """

    def __init__(
        self,
        steps: List[AbstractStep],
        finishing_step: FinishingStep,
        ad_request,
        enrichment: Optional[StepContextEnrichment] = None,
        force_all_default: bool = False
    ):
        """
        Initialize the pipeline with steps and configuration.

        Args:
            steps: List of processing steps to execute before finishing
            finishing_step: The final step that creates the AdResponse
            ad_request: The AdRequest instance for this pipeline execution
            enrichment: Optional enrichment data for future extensibility.
                        Pass a StepContextEnrichment subclass when additional
                        context-creation data is needed.
            force_all_default: If True, forces all steps to use default implementation
        """
        self.steps = steps
        self.finishing_step = finishing_step
        self.ad_request = ad_request
        self.force_all_default = force_all_default
        self.ad_response_service = AdResponseService()
        self.context_builder = StepContextBuilder(enrichment)
        self.exception_resolver = AdInjectionExceptionResolver()

        # Apply force_default to all steps if requested
        if self.force_all_default:
            for step in self.steps:
                step.force_default = True
            self.finishing_step.force_default = True

        logger.info(f"Pipeline initialized with {len(self.steps)} processing steps and finishing step")


    def run(self, initial_context: StepContext) -> ExecutionResult:
        """
        Execute the complete ad injection pipeline.

        This method runs all steps in sequence and always returns an ExecutionResult
        containing an AdResponse.

        Step Behavior:
        - Critical steps: If they fail, they throw exceptions that stop pipeline execution
        - Passable steps: Handle their own failures internally and always return a result

        The finishing step is always passable and will always produce an AdResponse.

        Args:
            initial_context: Initial context for the first step

        Returns:
            ExecutionResult: Always contains an AdResponse and execution metadata
        """
        start_time = time.time()
        total_steps = len(self.steps) + 1  # +1 for finishing step
        steps_executed = 0
        current_context = initial_context  # Initialize outside try block to avoid reference issues

        logger.info(f"Starting pipeline execution with {total_steps} total steps")

        try:
            # Execute processing steps in sequence
            for i, step in enumerate(self.steps):
                logger.info(f"Executing step {i+1}/{len(self.steps)}: {step.__class__.__name__}")

                # Execute the step and get its result
                # Critical steps will throw exceptions that stop pipeline execution
                # Passable steps handle their own failures internally and always return a result
                step_result = step.execute(current_context)
                steps_executed += 1

                # Create new context for next step using ContextBuilder
                current_context = self.context_builder.create_context(
                    result=step_result,
                    ad_request=self.ad_request,
                    outcome=True,
                    exception=None
                )

                logger.debug(f"Step {step.__class__.__name__} completed successfully")


            # Execute finishing step (this is a passable step that always returns a FinishingResult)
            logger.info(f"Executing finishing step: {self.finishing_step.__class__.__name__}")

            finishing_result = self.finishing_step.execute(current_context)
            steps_executed += 1

            # Calculate execution time
            execution_time_ms = (time.time() - start_time) * 1000

            logger.info(f"Pipeline completed successfully in {execution_time_ms:.2f}ms")

            # Pipeline ran without error — mark the request as processed
            self.ad_request.status = AdRequestStatus.PROCESSED
            self.ad_request.save()

            # Return success result
            return ExecutionResult.success_result(
                ad_response=finishing_result.ad_response,
                execution_time_ms=execution_time_ms,
                steps_executed=steps_executed,
                total_steps=total_steps
            )

        except Exception as e:
            logger.warning(f"Pipeline exception ({e.__class__.__name__}): {e}")
            return self._handle_pipeline_failure(
                error=e,
                context=current_context,
                start_time=start_time,
                steps_executed=steps_executed,
                total_steps=total_steps
            )

    def _handle_pipeline_failure(
        self,
        error: Exception,
        context: StepContext,
        start_time: float,
        steps_executed: int,
        total_steps: int
    ) -> ExecutionResult:
        """
        Handle pipeline failures by creating an unsuccessful AdResponse.

        The ``AdInjectionExceptionResolver`` determines the correct
        ``AdResponseStatus`` and ``AdResponseErrorCodes`` for the given
        exception — no caller needs to pass those values explicitly.

        Args:
            error: The exception that caused the failure
            context: The current step context
            start_time: Pipeline start time for calculating execution time
            steps_executed: Number of steps executed before failure
            total_steps: Total number of steps in pipeline

        Returns:
            ExecutionResult: Contains the unsuccessful AdResponse and failure metadata
        """
        try:
            execution_time_ms = (time.time() - start_time) * 1000

            auction = None
            if context.result and hasattr(context.result, 'auction'):
                auction = context.result.auction

            error_message = f"Pipeline execution failed: {str(error)}"
            exception_type = error.__class__.__name__

            response_status = self.exception_resolver.resolve_status(error)

            error_ad_response = self.ad_response_service.create_unsuccessful_ad_response(
                error_message=error_message,
                ad_request=self.ad_request,
                auction=auction,
                status=response_status,
            )

            if response_status == AdResponseStatus.ERROR:
                self.ad_request.status = AdRequestStatus.FAILED
                error_code = self.exception_resolver.resolve_error_code(error)

                self.ad_response_service.create_error_tracking(
                    ad_response=error_ad_response,
                    error_code=error_code,
                    error_message=error_message,
                )
            else:
                self.ad_request.status = AdRequestStatus.PROCESSED
            self.ad_request.save()

            logger.warning(
                f"Created unsuccessful AdResponse {error_ad_response.pk} "
                f"with status={response_status} for {exception_type}"
            )

            execution_result = ExecutionResult.error_result(
                ad_response=error_ad_response,
                error_message=error_message,
                exception_type=exception_type,
                execution_time_ms=execution_time_ms,
                steps_executed=steps_executed,
                total_steps=total_steps
            )

            if response_status == AdResponseStatus.ERROR:
                raise PipelineExecutionPersistedErrorException(
                    detail=error_message
                )

            return execution_result

        except PipelineExecutionPersistedErrorException as e:
            raise e

        except Exception as critical_error:
            logger.critical(f"Critical failure in error handling: {critical_error}")

            # As an absolute last resort, create a minimal error response
            try:
                from ad_response.models import AdResponse

                minimal_error_response = AdResponse.objects.create(
                    ad_request=self.ad_request,
                    status=AdResponseStatus.ERROR,
                )

                self.ad_response_service.create_error_tracking(
                    ad_response=minimal_error_response,
                    error_code=self.exception_resolver.resolve_error_code(error),
                    error_message=f"Critical pipeline failure: {str(error)}. Error handling failed: {str(critical_error)}",
                )

                execution_time_ms = (time.time() - start_time) * 1000

                return ExecutionResult.error_result(
                    ad_response=minimal_error_response,
                    error_message=f"Critical pipeline failure: {str(error)}",
                    exception_type=error.__class__.__name__,
                    execution_time_ms=execution_time_ms,
                    steps_executed=steps_executed,
                    total_steps=total_steps
                )

            except Exception as final_error:
                # This should never happen, but if it does, we must raise
                logger.critical(f"Complete failure to create any response: {final_error}")
                raise PipelineCriticalFailureException(
                    original_error=error,
                    error_handling_error=critical_error,
                    final_error=final_error
                ) from error

    def get_step_count(self) -> int:
        """
        Get the total number of steps in this pipeline.

        Returns:
            int: Total number of steps (processing steps + finishing step)
        """
        return len(self.steps) + 1

    def get_step_names(self) -> List[str]:
        """
        Get the names of all steps in this pipeline.

        Returns:
            List[str]: List of step class names
        """
        step_names = [step.__class__.__name__ for step in self.steps]
        step_names.append(self.finishing_step.__class__.__name__)
        return step_names

    def clear_context_data(self) -> None:
        """
        Clear all additional context data (filtering queries, ad_request, etc.).

        This can be called between pipeline runs to ensure clean state.
        """
        self.context_builder.clear_additional_data()

    def is_force_default_enabled(self) -> bool:
        """
        Check if force_default is enabled for all steps.

        Returns:
            bool: True if all steps are forced to use default implementation
        """
        return self.force_all_default
