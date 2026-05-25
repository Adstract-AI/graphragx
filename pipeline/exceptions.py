class PipelineException(Exception):
    """Base exception for pipeline-related failures."""


class PipelineExecutionException(PipelineException):
    """Raised when the overall pipeline execution fails."""


class StepNotImplementedException(PipelineException):
    """Raised when a required step method is not implemented."""
