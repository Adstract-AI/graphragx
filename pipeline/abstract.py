"""Abstract building blocks for the graphragX pipeline foundation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, Optional, TypeVar

from pydantic import BaseModel, ConfigDict

from pipeline.exceptions import (
    PipelineException,
    StepNotImplementedException,
)

T_Result = TypeVar("T_Result", bound="StepResult")
T_ContextResult = TypeVar("T_ContextResult", bound="StepResult")


class StepResult(BaseModel, ABC):
    """Base output model for any pipeline step."""

    model_config = ConfigDict(arbitrary_types_allowed=True)


class StepContext(BaseModel, Generic[T_ContextResult], ABC):
    """Input wrapper passed into a pipeline step."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    result: Optional[T_ContextResult] = None
    outcome: bool = False
    exception: Optional[PipelineException] = None


class AbstractStep(ABC, Generic[T_Result, T_ContextResult]):
    """Base class for all graphragX pipeline steps."""

    def __init__(self, force_default: bool = False):
        self.force_default = force_default

    def execute(self, context: StepContext[T_ContextResult]) -> T_Result:
        """Dispatch to implementation/default logic and update context outcome."""
        try:
            if self.force_default:
                result = self.execute_default(context)
            else:
                try:
                    result = self.execute_implementation(context)
                except NotImplementedError:
                    result = self.execute_default(context)

            context.outcome = True
            context.exception = None
            return result
        except Exception as error:
            context.outcome = False
            if isinstance(error, PipelineException):
                context.exception = error
                raise

            wrapped_error = PipelineException(
                f"Step {self.__class__.__name__} failed: {error}"
            )
            context.exception = wrapped_error
            raise wrapped_error from error

    @abstractmethod
    def execute_default(self, context: StepContext[T_ContextResult]) -> T_Result:
        """Fallback/default execution path for the step."""
        raise StepNotImplementedException(
            f"execute_default() must be implemented in {self.__class__.__name__}."
        )

    def execute_implementation(self, context: StepContext[T_ContextResult]) -> T_Result:
        """Primary execution path that subclasses may override."""
        raise NotImplementedError(
            f"execute_implementation() is not implemented in {self.__class__.__name__}."
        )
