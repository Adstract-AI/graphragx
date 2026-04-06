from typing import List, Optional
from datetime import datetime

from pydantic import BaseModel, Field

from ad_injection.pipeline.abstract import StepResult, StepContext, AbstractStep
from ad_injection.services.vector_store_service import SearchedPoint
from auction.models import Auction
from ad_response.models import AdResponse
from publisher.models import Message


# === Pipeline Execution Result ===

class ExecutionResult(BaseModel):
    """
    Result of the complete ad injection pipeline execution.
    
    This object always contains an AdResponse (success or error) and provides
    metadata about the pipeline execution.
    """
    ad_response: AdResponse = Field(..., description="The final AdResponse object (success or error)")
    success: bool = Field(..., description="Whether the pipeline executed successfully")
    execution_time_ms: Optional[float] = Field(None, description="Total pipeline execution time in milliseconds")
    steps_executed: int = Field(0, description="Number of steps that were executed")
    total_steps: int = Field(0, description="Total number of steps in the pipeline")
    error_message: Optional[str] = Field(None, description="Error message if pipeline failed")
    exception_type: Optional[str] = Field(None, description="Type of exception that caused failure")
    executed_at: datetime = Field(default_factory=datetime.now, description="When the pipeline was executed")
    
    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def success_result(
        cls,
        ad_response: AdResponse,
        execution_time_ms: float,
        steps_executed: int,
        total_steps: int
    ) -> 'ExecutionResult':
        """
        Create a successful execution result.
        
        Args:
            ad_response: The successful AdResponse
            execution_time_ms: Total execution time
            steps_executed: Number of steps executed
            total_steps: Total steps in pipeline
            
        Returns:
            ExecutionResult with success status
        """
        return cls(
            ad_response=ad_response,
            success=True,
            execution_time_ms=execution_time_ms,
            steps_executed=steps_executed,
            total_steps=total_steps,
            error_message=None,
            exception_type=None
        )

    @classmethod
    def error_result(
        cls,
        ad_response: AdResponse,
        error_message: str,
        exception_type: str,
        execution_time_ms: Optional[float] = None,
        steps_executed: int = 0,
        total_steps: int = 0
    ) -> 'ExecutionResult':
        """
        Create an error execution result.
        
        Args:
            ad_response: The error AdResponse
            error_message: Description of the error
            exception_type: Type of exception
            execution_time_ms: Optional execution time
            steps_executed: Number of steps executed before failure
            total_steps: Total steps in pipeline
            
        Returns:
            ExecutionResult with error status
        """
        return cls(
            ad_response=ad_response,
            success=False,
            execution_time_ms=execution_time_ms,
            steps_executed=steps_executed,
            total_steps=total_steps,
            error_message=error_message,
            exception_type=exception_type
        )


# === Default Implementations ===

class DefaultStepResult(StepResult):
    """
    Default concrete implementation of AdInjectionStepResult.

    Provides factory methods for success and failure scenarios.
    """
    message: str

    @classmethod
    def success(cls, step_class: type['AbstractStep']) -> 'DefaultStepResult':
        """
        Factory method to create a success result.

        Args:
            step_class: The class type of the concrete step

        Returns:
            DefaultStepResult with success message
        """
        return cls(message=f"Step {step_class.__name__} completed successfully")

    @classmethod
    def fail(cls, step_class: type['AbstractStep']) -> 'DefaultStepResult':
        """
        Factory method to create a failure result.

        Args:
            step_class: The class type of the concrete step

        Returns:
            DefaultStepResult with failure message
        """
        return cls(message=f"Step {step_class.__name__} failed")


class FirstStepResult(StepResult):
    """
    Default implementation for the first step in the pipeline.
    """
    message: str = "First executing step"


class DefaultContext(StepContext[FirstStepResult]):
    """
    Default concrete implementation of AdInjectionContext.

    Uses FirstStepResult as the result type.
    """
    pass


# === Initial ===

class InitialResult(StepResult):
    """
    Result class for the first pipeline input.

    Serves as the starting context result for the first step (moderation step).
    """
    pass


# === Conversation & Moderation ===

class ConversationModerationResult(StepResult):
    """
    Result class for the conversation classification and moderation step output.

    Classification and moderation data are persisted directly to the database
    within the step.
    """
    pass


# === Fraud ===

class FraudResult(StepResult):
    """
    Result class for the fraud detection step output.
    """
    pass


# === Context Building ===

class ContextResult(StepResult):
    """
    Result class for context building step output.

    Contains the built context string and the source messages used to build it.
    """
    context: str
    context_messages: List[Message]


# === Embedding ===

class EmbeddingResult(StepResult):
    """
    Result class for the embedding step output.

    Contains only the generated embedding vector representation.
    """
    embedding: List[float]


# === Filtering ===

class FilteringQueries(BaseModel):
    """
    Queries used for filtering vectors based on targeting criteria.

    Each field represents a single targeting criterion value. If a field is None,
    it means no filtering should be applied for that criterion.

    The filtering logic checks if vectors either:
    1. Have targeting for that field and the provided value is in their allowed list, OR
    2. Don't have targeting for that field (meaning they allow all values)

    Example: FilteringQueries(user_country="US", conversation_language="EN")
    """
    user_country: Optional[str] = Field(
        default=None,
        description="Single user country to filter by. If None, no country filtering is applied."
    )
    conversation_language: Optional[str] = Field(
        default=None,
        description="Single conversation language to filter by. If None, no language filtering is applied."
    )
    device: Optional[str] = Field(
        default=None,
        description="Single device type to filter by. If None, no device filtering is applied."
    )
    age_group: Optional[str] = Field(
        default=None,
        description="Single age group to filter by. If None, no age group filtering is applied."
    )
    gender: Optional[str] = Field(
        default=None,
        description="Single gender to filter by. If None, no gender filtering is applied."
    )
    verified: bool = Field(
        default=True,
        description="Only include verified product document vectors."
    )
    eligible: bool = Field(
        default=True,
        description="Only include eligible product document vectors."
    )
    dangling: bool = Field(
        default=False,
        description="Exclude dangling vectors by default."
    )
    blocked: bool = Field(
        default=False,
        description="Exclude blocked vectors by default."
    )


class VectorSearchResult(StepResult):
    """
    Result class for the filtering step output.

    Contains the embedding and the list of filtered vectors that match
    the filtering criteria.
    """
    embedding: list[float]
    searched_points: list[SearchedPoint]  # List of vectors that passed the filtering


# === Auction ===

class AuctionResult(StepResult):
    """
    Result class for the auction step output.

    Contains the completed auction object after the full auction lifecycle
    (creation, execution, and payment processing).
    """
    auction: Auction


# === Finishing ===

class FinishingResult(StepResult):
    """
    Result class for the finishing step output.

    Contains the completed AdResponse object with all related metadata,
    including LinkInstance for click tracking.
    """

    ad_response: AdResponse
