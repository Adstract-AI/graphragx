"""Generic exceptions for the graphragX pipeline foundation."""


class PipelineException(Exception):
    """Base exception for pipeline-related failures."""


class PipelineExecutionException(PipelineException):
    """Raised when the overall pipeline execution fails."""


class StepNotImplementedException(PipelineException):
    """Raised when a required step method is not implemented."""


class UnsupportedDatasetSelectionException(PipelineException):
    """Raised when a preparation run requests an unsupported dataset."""


class InvalidMainLlmSelectionException(PipelineException):
    """Raised when an invalid main LLM option is provided."""


class InvalidAssistantLlmSelectionException(PipelineException):
    """Raised when an invalid assistant LLM option is provided."""


class InvalidSubgraphConstructionSelectionException(PipelineException):
    """Raised when an invalid subgraph construction option is provided."""


class InvalidContextConstructionSelectionException(PipelineException):
    """Raised when an invalid context construction option is provided."""


class InvalidGnnLayerCountSelectionException(PipelineException):
    """Raised when an invalid GNN layer count option is provided."""


class InvalidNodeClassifierSelectionException(PipelineException):
    """Raised when an invalid node classifier option is provided."""


class InvalidQuestionEmbeddingModelSelectionException(PipelineException):
    """Raised when an invalid question embedding model option is provided."""


class InvalidRelationEmbeddingModelSelectionException(PipelineException):
    """Raised when an invalid relation embedding model option is provided."""


class InvalidEntityEmbeddingModelSelectionException(PipelineException):
    """Raised when an invalid entity embedding model option is provided."""


class InvalidInteractiveConfigurationInputException(PipelineException):
    """Raised when interactive configuration input cannot be read safely."""


class MissingTorchDependencyException(PipelineException):
    """Raised when torch is required but not installed."""


class MissingTorchGeometricDependencyException(PipelineException):
    """Raised when torch_geometric is required but not installed."""


class UnsupportedKnowledgeGraphDatasetLoaderException(PipelineException):
    """Raised when no supported dataset loader configuration exists."""


class KnowledgeGraphDatasetLoadingException(PipelineException):
    """Raised when dataset loading fails."""


class MalformedKnowledgeGraphDatasetException(PipelineException):
    """Raised when a loaded dataset does not expose the expected structure."""
