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


class InvalidGnnHiddenDimensionSelectionException(PipelineException):
    """Raised when an invalid GNN hidden dimension option is provided."""


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


class MissingHuggingFaceDatasetsDependencyException(PipelineException):
    """Raised when Hugging Face datasets is required but not installed."""


class UnsupportedDatasetLoaderException(PipelineException):
    """Raised when no supported dataset loader configuration exists."""


class DatasetLoadingException(PipelineException):
    """Raised when dataset loading fails."""


class MalformedDatasetException(PipelineException):
    """Raised when a loaded dataset does not expose the expected structure."""


class UnsupportedDatasetProcessorException(PipelineException):
    """Raised when no supported dataset processor exists for a dataset."""


class MalformedWebQSPExampleException(PipelineException):
    """Raised when a WebQSP example does not expose the expected structure."""


class ProcessedDatasetStorageException(PipelineException):
    """Raised when processed dataset cache storage fails."""


class OpenAiEmbeddingConfigurationException(PipelineException):
    """Raised when OpenAI embedding configuration is incomplete."""


class GnnAnswerRetrieverTrainingException(PipelineException):
    """Raised when GNN answer-retriever training fails."""
