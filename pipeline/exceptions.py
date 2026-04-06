"""Generic exceptions for the graphragX pipeline foundation."""


class PipelineException(Exception):
    """Base exception for pipeline-related failures."""


class PipelineExecutionException(PipelineException):
    """Raised when the overall pipeline execution fails."""


class StepNotImplementedException(PipelineException):
    """Raised when a required step method is not implemented."""


class UnsupportedKnowledgeGraphDatasetException(PipelineException):
    """Raised when a preparation run requests an unsupported KG dataset."""
