"""
Step Context Builder for Ad Injection Pipeline.

This module contains the StepContextBuilder class that handles the conversion
from step results to appropriate contexts for the next step in the pipeline.
"""
from typing import Dict, Type, Callable, Optional

from ad_injection.pipeline.abstract import StepContext, StepResult
from ad_injection.pipeline.models import (
    InitialResult, ConversationModerationResult, FraudResult, ContextResult, EmbeddingResult,
    VectorSearchResult, AuctionResult,
)
from ad_injection.pipeline.step_context_enrichment import StepContextEnrichment
from ad_request.models import AdRequest
from helper_functions.logger import get_logger

logger = get_logger(__file__)


class StepContextBuilder:
    """
    Handles the creation of appropriate StepContext objects from step results.
    """

    def __init__(self, enrichment: Optional[StepContextEnrichment] = None):
        """
        Initialize the context builder.

        Args:
            enrichment: Optional enrichment data for future extensibility.
        """
        # Mapping of result types to context creation functions
        self._context_creators: Dict[Type[StepResult], Callable] = {
            InitialResult: self._create_basic_context,
            ConversationModerationResult: self._create_basic_context,
            FraudResult: self._create_basic_context,
            ContextResult: self._create_basic_context,
            EmbeddingResult: self._create_basic_context,
            VectorSearchResult: self._create_basic_context,
            AuctionResult: self._create_basic_context,
        }

        self._enrichment = enrichment

    def create_context(
        self,
        result: StepResult,
        ad_request: AdRequest,
        outcome: bool = True,
        exception=None,
    ) -> StepContext:
        """
        Create an appropriate StepContext from a step result.

        Args:
            result: The result from the previous step.
            ad_request: The AdRequest for the current pipeline execution.
            outcome: Whether the step was successful.
            exception: Any exception that occurred.

        Returns:
            StepContext with ad_request propagated.

        Raises:
            ValueError: If no context creator is found for the result type.
        """
        if result is None:
            return StepContext(ad_request=ad_request, result=None, outcome=outcome, exception=exception)

        result_type = type(result)
        context_creator = self._context_creators.get(result_type)

        if context_creator is None:
            logger.warning(f"No context creator found for result type {result_type.__name__}, using basic context")
            return self._create_basic_context(result, ad_request, outcome, exception)

        logger.debug(f"Creating context for result type {result_type.__name__}")
        return context_creator(result, ad_request, outcome, exception)

    def _create_basic_context(
        self,
        result: StepResult,
        ad_request: AdRequest,
        outcome: bool,
        exception,
    ) -> StepContext:
        """
        Create a basic StepContext with the result and ad_request.

        Args:
            result: The step result.
            ad_request: The pipeline's ad request.
            outcome: Whether the step was successful.
            exception: Any exception that occurred.

        Returns:
            StepContext with ad_request propagated.
        """
        return StepContext(ad_request=ad_request, result=result, outcome=outcome, exception=exception)

    def register_context_creator(self, result_type: Type[StepResult], creator_func: Callable) -> None:
        """
        Register a custom context creator for a specific result type.

        Args:
            result_type: The type of step result.
            creator_func: Function that creates the context.
        """
        self._context_creators[result_type] = creator_func
        logger.debug(f"Registered custom context creator for {result_type.__name__}")

    def clear_additional_data(self) -> None:
        """
        Clear all additional context data.

        This can be called between pipeline runs to ensure clean state.
        """
        if self._enrichment:
            self._enrichment.clear()
        logger.debug("Cleared all additional context data")
