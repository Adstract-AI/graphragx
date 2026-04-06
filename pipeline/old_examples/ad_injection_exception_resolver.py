"""
Ad Injection Exception Resolver.

Centralises the logic that maps a pipeline exception to the outcome values
needed by `_handle_pipeline_failure`:

  - ``resolve_status``     — returns the ``AdResponseStatus`` for the exception
  - ``resolve_error_code`` — returns the ``AdResponseErrorCodes`` for the exception;
                             only called when ``resolve_status`` returned ERROR
"""
from ad_injection.pipeline.exceptions import (
    NoFillPipelineException,
    ConversationModerationStepFailedException,
    ConversationModerationRejectedException,
    FraudStepFailedException,
    ContextBuildingStepFailedException,
    EmbeddingStepFailedException,
    VectorSearchStepFailedException,
    AuctionStepFailedException,
    FinishingStepException,
    CriticalStepException,
)
from ad_response.models import AdResponseStatus, AdResponseErrorCodes


class AdInjectionExceptionResolver:
    """
    Resolves pipeline exceptions to the correct ``AdResponseStatus`` and
    ``AdResponseErrorCodes`` values.

    Status rules
    ------------
    - ``NoSearchPointsException``           → NO_FILL  (subclass of NoFillPipelineException)
    - ``NoValidBiddersException``           → NO_FILL  (subclass of NoFillPipelineException)
    - Any other exception                   → ERROR

    Error code rules (only relevant when status is ERROR)
    -----------------------------------------------------
    - ``ConversationModerationStepFailedException`` → MODERATION_FAILED
    - ``FraudStepFailedException``          → FRAUD_DETECTION_FAILED
    - ``ContextBuildingStepFailedException``→ PIPELINE_ERROR
    - ``EmbeddingStepFailedException``      → EMBEDDING_FAILED
    - ``VectorSearchStepFailedException``   → RETRIEVAL_FAILED
    - ``AuctionStepFailedException``        → AUCTION_FAILED
    - ``FinishingStepException``            → FINISHING_FAILED
    - Other ``CriticalStepException``       → PIPELINE_ERROR
    - Any other ``Exception``               → UNKNOWN
    """

    def resolve_status(self, error: Exception) -> str:
        """
        Return the ``AdResponseStatus`` that should be stamped on the
        ``AdResponse`` for the given exception.
        """
        if isinstance(error, NoFillPipelineException):
            return AdResponseStatus.NO_FILL

        if isinstance(error, ConversationModerationRejectedException):
            return AdResponseStatus.REJECTED

        return AdResponseStatus.ERROR

    def resolve_error_code(self, error: Exception) -> str:
        """
        Return the ``AdResponseErrorCodes`` value for the given exception.

        This should only be called when ``resolve_status`` has already
        returned ``AdResponseStatus.ERROR`` — non-error outcomes (e.g.
        NO_FILL, REJECTED) must not produce an error tracking record.
        """
        if isinstance(error, EmbeddingStepFailedException):
            return AdResponseErrorCodes.EMBEDDING_FAILED

        if isinstance(error, ConversationModerationStepFailedException):
            return AdResponseErrorCodes.MODERATION_FAILED

        if isinstance(error, FraudStepFailedException):
            return AdResponseErrorCodes.FRAUD_DETECTION_FAILED

        if isinstance(error, ContextBuildingStepFailedException):
            return AdResponseErrorCodes.PIPELINE_ERROR

        if isinstance(error, VectorSearchStepFailedException):
            return AdResponseErrorCodes.RETRIEVAL_FAILED

        if isinstance(error, AuctionStepFailedException):
            return AdResponseErrorCodes.AUCTION_FAILED

        if isinstance(error, FinishingStepException):
            return AdResponseErrorCodes.FINISHING_FAILED

        if isinstance(error, CriticalStepException):
            return AdResponseErrorCodes.PIPELINE_ERROR

        return AdResponseErrorCodes.UNKNOWN
