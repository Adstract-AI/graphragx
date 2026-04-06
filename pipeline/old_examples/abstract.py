"""
Abstract base classes for Ad Injection Pipeline Steps.

This module defines the abstract base classes that all pipeline steps must inherit from.
The pipeline supports different types of steps with different failure handling strategies.
"""
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Optional, get_args

from pydantic import BaseModel, ConfigDict

from ad_injection.pipeline.exceptions import AdInjectionPipelineException, StepNotImplementedException, \
    CriticalStepException, PassableStepException
from ad_request.models import AdRequest

# Generic type for AdInjectionStepResult
T_Result = TypeVar('T_Result', bound='StepResult')
T_ContextResult = TypeVar('T_ContextResult', bound='StepResult')


class StepResult(BaseModel, ABC):
    """
    Abstract base class for all Ad Injection Step results.

    All concrete step results must inherit from this class.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    pass


class StepContext(BaseModel, Generic[T_ContextResult], ABC):
    """
    Abstract base class for all Ad Injection Pipeline step contexts.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    ad_request: Optional[AdRequest] = None
    result: Optional[T_ContextResult] = None
    outcome: bool = False
    exception: Optional[AdInjectionPipelineException] = None


class AbstractStep(ABC, Generic[T_Result, T_ContextResult]):
    """
    Abstract base class for all Ad Injection Pipeline steps.

    Each step can optionally implement execute_implementation() for custom logic,
    and must implement execute_default() as a fallback.

    T_Result: The type of result this specific step returns
    T_ContextResult: The type of result stored in the context (consumed from the previous step)
    """

    def __init__(self, force_default: bool = False):
        """
        Initialize the step.

        Resolves the concrete T_Result and T_ContextResult types from the class's
        generic parameters and constructs a typed ``PipelineContractHandler`` from
        them.  The handler knows exactly which result types to validate — no MRO
        walking or dynamic ``type()`` inspection is needed at call time.

        Args:
            force_default: If True, always use execute_default() for testing purposes.
        """
        # Deferred import to avoid the circular dependency:
        # abstract.py exports T_Result/T_ContextResult → pipeline_contract_handler.py
        # imports them, so pipeline_contract_handler cannot be imported at module level here.
        from ad_injection.pipeline.pipeline_contract_handler import PipelineContractHandler

        self.force_default = force_default

        result_type, context_result_type = self._resolve_generic_types()
        self.contract_handler = PipelineContractHandler[result_type, context_result_type]()

    @classmethod
    def _resolve_generic_types(cls) -> tuple:
        """
        Walk the MRO to find the first base that supplies two concrete type arguments
        and return them as ``(result_type, context_result_type)``.

        Returns:
            A 2-tuple of (T_Result concrete type, T_ContextResult concrete type).

        Raises:
            TypeError: If the concrete types cannot be resolved — the step class must
                be parameterised as ``SomeStep[ResultType, ContextResultType]``.
        """
        for klass in cls.__mro__:
            for base in getattr(klass, '__orig_bases__', ()):
                args = get_args(base)
                if len(args) >= 2 and isinstance(args[0], type) and isinstance(args[1], type):
                    return args[0], args[1]
        raise TypeError(
            f"{cls.__name__} could not resolve its generic result types. "
            f"Ensure the class is parameterised as AbstractStep[ResultType, ContextResultType]."
        )

    def execute(self, context: StepContext[T_ContextResult]) -> T_Result:
        """
        Main execution method that decides whether to use implementation or default.

        Validates the incoming context against its fulfilment contract before
        dispatching to the concrete step logic.

        Args:
            context: The execution context containing result, outcome, and exception info.

        Returns:
            The result of the step execution (specific to this step's type).
        """
        self.contract_handler.is_fulfilled(context)

        if self.force_default:
            return self.execute_default(context)

        try:
            # Check if execute_implementation is actually implemented
            # by trying to call it and catching NotImplementedError
            return self.execute_implementation(context)
        except NotImplementedError:
            # If not implemented, fall back to default
            return self.execute_default(context)

    @abstractmethod
    def execute_default(self, context: StepContext[T_ContextResult]) -> T_Result:
        """
        Default implementation that must always be provided.

        This method must be implemented by all concrete step classes.
        If not implemented, it should raise StepNotImplementedException.

        Args:
            context: The execution context containing result, outcome, and exception info.

        Returns:
            The result of the default step execution (specific to this step's type).

        Raises:
            StepNotImplementedException: If this method is not properly implemented.
        """
        raise StepNotImplementedException(
            f"execute_default() method must be implemented in {self.__class__.__name__}"
        )

    def execute_implementation(self, context: StepContext[T_ContextResult]) -> T_Result:
        """
        Custom implementation that can be optionally provided.

        By default, this raises NotImplementedError. Concrete classes can override
        this method to provide custom implementation logic.

        Args:
            context: The execution context containing result, outcome, and exception info.

        Returns:
            The result of the custom step implementation (specific to this step's type).

        Raises:
            NotImplementedError: If this method is not implemented (default behavior).
        """
        raise NotImplementedError(
            f"execute_implementation() is not implemented in {self.__class__.__name__}"
        )


class CriticalAbstractStep(AbstractStep[T_Result, T_ContextResult], ABC):
    """
    Abstract step that raises critical exceptions when it fails.

    When a critical step fails, it should raise a CriticalStepException
    which will cause the pipeline to stop execution and fallback to
    a critical situation handling mechanism.
    """

    def execute(self, context: StepContext[T_ContextResult]) -> T_Result:
        """
        Execute the critical step with error handling.

        Delegates fulfilment checking and dispatch to the base class, then
        signs the result before returning it.

        Args:
            context: The execution context containing result, outcome, and exception info.

        Returns:
            The result of the step execution (specific to this step's type).

        Raises:
            AdInjectionPipelineException: If the step execution fails with a known pipeline exception.
            CriticalStepException: If an unexpected non-pipeline exception occurs.
        """
        try:
            result = super().execute(context)
            self.contract_handler.sign(result)
            return result
        except Exception as e:
            # Update context with exception information
            context.outcome = False
            if isinstance(e, AdInjectionPipelineException):
                context.exception = e
                raise

            wrapped_exception = CriticalStepException(
                f"Critical step {self.__class__.__name__} failed: {str(e)}"
            )
            context.exception = wrapped_exception
            raise wrapped_exception from e


class PassableAbstractStep(AbstractStep[T_Result, T_ContextResult], ABC):
    """
    Abstract step that can gracefully handle failures with fallback logic.

    Any exception triggers ``execute_fallback`` rather than stopping the pipeline.
    The fallback result is then signed as well.  If that second signing also fails
    the exception propagates and the pipeline fails completely, because there is no
    valid result to continue with.

    Exception Handling:
    - ``PassableStepException``: raised when a critical
      outcome is detected in a passable step environment — for example when the
      result violates its contract.  Always triggers fallback execution.
    - Other exceptions: also trigger fallback but are wrapped in
      ``AdInjectionPipelineException`` before being stored on the context.

    Contract signing:
    - ``sign()`` is called on the result of the primary execution path.  If it
      raises a ``PassableStepException`` the fallback is attempted.
    - ``sign()`` is called again on the fallback result.  A failure here
      propagates uncaught — the step has exhausted its recovery options.
    - ``context.outcome`` is only set to ``True`` after signing succeeds,
      ensuring it accurately reflects whether a valid result was produced.
    """

    def execute(self, context: StepContext[T_ContextResult]) -> T_Result:
        """
        Execute the passable step with fallback handling.

        Attempts primary execution and signs the result.  On any exception
        (including a contract signing failure) falls back to ``execute_fallback``
        and signs that result.  If the fallback signing also fails the exception
        is allowed to propagate.

        Args:
            context: The execution context containing result, outcome, and exception info.

        Returns:
            The signed result of either the primary or fallback execution.

        Raises:
            Any exception raised by ``execute_fallback`` or by ``sign()`` on the
            fallback result — indicating complete step failure.
        """
        try:
            result = super().execute(context)
            self.contract_handler.sign(result)
            # Mark success only after the contract is verified
            context.outcome = True
            return result
        except PassableStepException as e:
            # PassableStepException specifically indicates that fallback should be used
            # Update context with exception information
            context.outcome = False
            context.exception = e
            # Execute fallback for PassableStepException
            result = self.execute_fallback(context, e)
            self.contract_handler.sign(result)
            return result
        except Exception as e:
            # Other exceptions also trigger fallback but are wrapped differently
            # Update context with exception information
            context.outcome = False
            if isinstance(e, AdInjectionPipelineException):
                context.exception = e
            else:
                context.exception = AdInjectionPipelineException(
                    f"Passable step {self.__class__.__name__} failed: {str(e)}"
                )
            # If execution fails, try the fallback
            result = self.execute_fallback(context, e)
            self.contract_handler.sign(result)
            return result

    def execute_fallback(self, context: StepContext[T_ContextResult], original_exception: Exception) -> T_Result:
        """
        Fallback execution method called when the primary execution or its signing fails.

        This method is called when:
        1. ``sign()`` raises a ``PassableStepException`` because a critical outcome
           was detected in a passable step environment — for example the primary
           result violated its contract.
        2. Any other exception occurs during primary execution (unexpected fallback).

        By default this calls ``execute_default()``.  Subclasses can override to
        provide custom fallback logic (see ``ContextBuildingStep`` for an example).

        The result returned here is signed by the caller before being returned to
        the pipeline.  If signing fails the exception propagates — the step has no
        further recovery path.

        Args:
            context: The execution context; ``context.exception`` holds the reason
                     for the fallback.
            original_exception: The exception that triggered the fallback.  May be
                                 a ``PassableStepException`` (raised on a critical
                                 outcome such as a failed contract signing) or any
                                 other ``Exception``.

        Returns:
            The fallback result, which will be signed by the caller.

        Raises:
            Any exception raised by ``execute_default`` or a custom fallback
            implementation.
        """
        # Default behavior: call execute_default as fallback
        # Subclasses can override this to provide custom fallback logic
        return self.execute_default(context)
