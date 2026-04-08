"""Generic exceptions for the graphragX pipeline foundation."""


class PipelineException(Exception):
    """Base exception for pipeline-related failures."""


class PipelineExecutionException(PipelineException):
    """Raised when the overall pipeline execution fails."""


class StepNotImplementedException(PipelineException):
    """Raised when a required step method is not implemented."""


class UnsupportedKnowledgeGraphDatasetException(PipelineException):
    """Raised when a preparation run requests an unsupported KG dataset."""


class InvalidMainLlmSelectionException(PipelineException):
    """Raised when an invalid main LLM option is provided."""


class InvalidAssistantLlmSelectionException(PipelineException):
    """Raised when an invalid assistant LLM option is provided."""


class InvalidSubgraphConstructionSelectionException(PipelineException):
    """Raised when an invalid subgraph construction option is provided."""


class InvalidContextConstructionSelectionException(PipelineException):
    """Raised when an invalid context construction option is provided."""


class InvalidInteractiveConfigurationInputException(PipelineException):
    """Raised when interactive configuration input cannot be read safely."""
