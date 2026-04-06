
from rest_framework import status
from rest_framework.exceptions import APIException


class AdInjectionPipelineException(Exception):
    """Base exception for Ad Injection Pipeline errors."""
    pass


class CriticalStepException(AdInjectionPipelineException):
    """Exception raised when a critical step fails and pipeline should stop."""
    pass


class StepNotImplementedException(AdInjectionPipelineException):
    """Exception raised when a required step method is not implemented."""
    pass


class MissingContractException(AdInjectionPipelineException):
    """Raised at step construction when no contract is registered for the step's result type."""
    pass


class PassableStepException(AdInjectionPipelineException):
    """Exception raised by passable steps to trigger fallback execution."""
    pass


class ConversationModerationStepFailedException(CriticalStepException):
    """Exception raised when the conversation and moderation step fails."""
    pass


class ConversationModerationRejectedException(AdInjectionPipelineException):
    """Exception raised when the conversation moderation rejects the prompt as unsuitable for ads."""
    pass


class FraudStepFailedException(PassableStepException):
    """Exception raised when the fraud detection step fails its contract signing."""
    pass


class ContextBuildingStepFailedException(PassableStepException):
    """Exception raised when the context building step fails its contract signing."""
    pass


class EmbeddingStepFailedException(CriticalStepException):
    """Exception raised when the embedding step fails to generate embeddings."""
    pass


class VectorSearchStepFailedException(CriticalStepException):
    """Exception raised when the critical vector search step fails."""
    pass


class AuctionStepFailedException(CriticalStepException):
    """Exception raised when the auction step fails to create or run an auction."""
    pass


class FinishingStepException(CriticalStepException):
    """Exception raised when the finishing step fails."""
    pass


class NoFillPipelineException(AdInjectionPipelineException):
    """Base exception for no-fill outcomes — pipeline completed but had nothing to serve."""
    pass


class NoSearchPointsException(NoFillPipelineException):
    """Exception raised when the vector search returns no results."""
    pass


class NoValidBiddersException(NoFillPipelineException):
    """Exception raised when the auction has no valid bidders to participate."""
    pass


class PipelineCriticalFailureException(APIException):
    """
    Exception raised when the pipeline fails completely and cannot create any response.

    This is the final fallback exception that will be automatically handled by Django
    REST framework and will stop endpoint execution.
    """
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    default_detail = 'Critical pipeline failure - unable to process request.'
    default_code = 'pipeline_critical_failure'

    def __init__(self, original_error=None, error_handling_error=None, final_error=None, detail=None):
        if detail is None:
            error_parts = []
            if original_error:
                error_parts.append(f"Original: {str(original_error)}")
            if error_handling_error:
                error_parts.append(f"Error handling: {str(error_handling_error)}")
            if final_error:
                error_parts.append(f"Final: {str(final_error)}")

            if error_parts:
                detail = f"Complete pipeline failure. {', '.join(error_parts)}"
            else:
                detail = self.default_detail

        super().__init__(detail)


class PipelineExecutionPersistedErrorException(APIException):
    """
    Exception raised after the pipeline persisted an internal-error outcome.

    The pipeline already created the unsuccessful AdResponse and related error
    tracking at this point. This exception only changes the HTTP response shape
    so callers receive a 500 for true internal pipeline failures.
    """

    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    default_detail = "Internal pipeline failure occurred after persisting the error response."
    default_code = "pipeline_execution_persisted_error"

    def __init__(self, detail: str | None = None):
        super().__init__(detail or self.default_detail)
