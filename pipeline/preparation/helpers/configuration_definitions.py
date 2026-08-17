"""Typed built-in configuration definitions for pipeline construction."""

from __future__ import annotations

from typing import Final

from pydantic import BaseModel, ConfigDict, Field


class LlmModelDefinition(BaseModel):
    """Typed definition of an available LLM model option."""

    model_config = ConfigDict(frozen=True)

    model_id: str = Field(..., description="Stable model identifier.")
    display_name: str = Field(..., description="Human-readable model name.")
    description: str = Field(..., description="Short description of the model.")


class SubgraphConstructionDefinition(BaseModel):
    """Typed definition of an available subgraph construction algorithm."""

    model_config = ConfigDict(frozen=True)

    algorithm_id: str = Field(..., description="Stable algorithm identifier.")
    display_name: str = Field(..., description="Human-readable algorithm name.")
    description: str = Field(..., description="Short description of the algorithm.")


class ContextConstructionDefinition(BaseModel):
    """Typed definition of an available context construction strategy."""

    model_config = ConfigDict(frozen=True)

    strategy_id: str = Field(..., description="Stable context strategy identifier.")
    display_name: str = Field(..., description="Human-readable context strategy name.")
    description: str = Field(..., description="Short description of the context strategy.")


class GnnLayerCountDefinition(BaseModel):
    """Typed definition of an available GNN layer count option."""

    model_config = ConfigDict(frozen=True)

    layer_count: int = Field(..., description="Number of GNN message-passing layers.")
    display_name: str = Field(..., description="Human-readable layer count label.")
    description: str = Field(..., description="Short description of the layer setting.")


class GnnHiddenDimensionDefinition(BaseModel):
    """Typed definition of an available GNN hidden dimension option."""

    model_config = ConfigDict(frozen=True)

    hidden_dimension: int = Field(
        ...,
        description="Width of projected node states inside the GNN.",
    )
    display_name: str = Field(..., description="Human-readable hidden dimension label.")
    description: str = Field(
        ...,
        description="Short description of the hidden dimension setting.",
    )


class NodeClassifierDefinition(BaseModel):
    """Typed definition of an available node classifier option."""

    model_config = ConfigDict(frozen=True)

    classifier_id: str = Field(..., description="Stable node classifier identifier.")
    display_name: str = Field(..., description="Human-readable classifier name.")
    description: str = Field(..., description="Short description of the classifier.")


class GnnArchitectureDefinition(BaseModel):
    """Typed definition and defaults for one supported GNN architecture."""

    model_config = ConfigDict(frozen=True)

    architecture_id: str
    display_name: str
    description: str
    supported_layer_counts: tuple[int, ...] = (2, 3)
    supported_hidden_dimensions: tuple[int, ...] = (128, 256, 512)
    supported_classifiers: tuple[str, ...] = ("mlp", "linear")
    supported_dropouts: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.5)
    default_layer_count: int = 2
    default_hidden_dimension: int = 256
    default_classifier: str = "mlp"
    default_dropout: float = 0.1
    supports_advanced_options: bool = False
    default_use_edge_mlp: bool = False
    default_use_reverse_edges: bool = False
    default_question_aware_classifier: bool = False
    default_add_layer_normalization: bool = False
    supported_edge_mlp_hidden_dimensions: tuple[int, ...] = ()
    default_edge_mlp_hidden_dimension: int | None = None


class OpenAiEmbeddingModelDefinition(BaseModel):
    """Typed definition of an available OpenAI embedding model option."""

    model_config = ConfigDict(frozen=True)

    model_id: str = Field(..., description="Stable OpenAI embedding model identifier.")
    display_name: str = Field(..., description="Human-readable embedding model name.")
    dimensions: int = Field(..., description="Default embedding vector dimension.")
    description: str = Field(..., description="Short description of the embedding model.")


SHARED_LLM_MODELS: Final[dict[str, LlmModelDefinition]] = {
    "gpt-5.4": LlmModelDefinition(
        model_id="gpt-5.4",
        display_name="GPT-5.4",
        description="Highest-capability shared model option for primary reasoning.",
    ),
    "gpt-5.4-mini": LlmModelDefinition(
        model_id="gpt-5.4-mini",
        display_name="GPT-5.4 Mini",
        description="Balanced shared model option for supporting reasoning tasks.",
    ),
    "gpt-5.4-nano": LlmModelDefinition(
        model_id="gpt-5.4-nano",
        display_name="GPT-5.4 Nano",
        description="Fast lightweight shared model option.",
    ),
    "gpt-5-mini": LlmModelDefinition(
        model_id="gpt-5-mini",
        display_name="GPT-5 Mini",
        description="Faster, cost-efficient GPT-5 model for well-defined tasks.",
    ),
    "gpt-5-nano": LlmModelDefinition(
        model_id="gpt-5-nano",
        display_name="GPT-5 Nano",
        description="Fastest and cheapest GPT-5 model option.",
    ),
    "deepseek-v4-flash": LlmModelDefinition(
        model_id="deepseek-v4-flash",
        display_name="DeepSeek-V4-Flash",
        description="Fast DeepSeek V4 model served through the DeepSeek API.",
    ),
    "deepseek-v4-pro": LlmModelDefinition(
        model_id="deepseek-v4-pro",
        display_name="DeepSeek-V4-Pro",
        description="Higher-capability DeepSeek V4 model served through the DeepSeek API.",
    ),
    "gpt-4.1": LlmModelDefinition(
        model_id="gpt-4.1",
        display_name="GPT-4.1",
        description="Strong shared model option for reasoning and generation.",
    ),
    "gpt-4.1-mini": LlmModelDefinition(
        model_id="gpt-4.1-mini",
        display_name="GPT-4.1 Mini",
        description="Compact shared model option for support tasks.",
    ),
    "gpt-4.1-nano": LlmModelDefinition(
        model_id="gpt-4.1-nano",
        display_name="GPT-4.1 Nano",
        description="Smallest shared model option for lightweight tasks.",
    ),
}

SUBGRAPH_CONSTRUCTION_ALGORITHMS: Final[dict[str, SubgraphConstructionDefinition]] = {
    "shortest_path": SubgraphConstructionDefinition(
        algorithm_id="shortest_path",
        display_name="Shortest Path",
        description="Build a question-specific subgraph using shortest paths between relevant nodes.",
    ),
    # Not Supported Yet
    # "pcst": SubgraphConstructionDefinition(
    #     algorithm_id="pcst",
    #     display_name="Prize-Collecting Steiner Tree (PCST)",
    #     description="Optimize a compact subgraph that balances node value and connection cost.",
    # ),
}

CONTEXT_CONSTRUCTION_STRATEGIES: Final[dict[str, ContextConstructionDefinition]] = {

    "structured_triples": ContextConstructionDefinition(
        strategy_id="structured_triples",
        display_name="Structured Triples",
        description="Represent the subgraph as <Head, Relation, Tail> triples.",
    ),
    # Not Supported Yet
    # "textualized": ContextConstructionDefinition(
    #     strategy_id="textualized",
    #     display_name="Textualized",
    #     description="Represent the subgraph as natural language descriptions for the LLM.",
    # ),
}

GNN_LAYER_COUNT_OPTIONS: Final[dict[str, GnnLayerCountDefinition]] = {
    "2": GnnLayerCountDefinition(
        layer_count=2,
        display_name="2 GNN layers",
        description="A compact message-passing depth for the first retriever baseline.",
    ),
    "3": GnnLayerCountDefinition(
        layer_count=3,
        display_name="3 GNN layers",
        description="A deeper message-passing depth for slightly broader local context.",
    ),
}

GNN_HIDDEN_DIMENSION_OPTIONS: Final[dict[str, GnnHiddenDimensionDefinition]] = {
    "128": GnnHiddenDimensionDefinition(
        hidden_dimension=128,
        display_name="128 hidden dimensions",
        description="A lightweight hidden size for faster early experiments.",
    ),
    "256": GnnHiddenDimensionDefinition(
        hidden_dimension=256,
        display_name="256 hidden dimensions",
        description="A balanced hidden size for the first retriever baseline.",
    ),
    "512": GnnHiddenDimensionDefinition(
        hidden_dimension=512,
        display_name="512 hidden dimensions",
        description="A wider hidden size with more capacity and higher compute cost.",
    ),
}

NODE_CLASSIFIERS: Final[dict[str, NodeClassifierDefinition]] = {
    "mlp": NodeClassifierDefinition(
        classifier_id="mlp",
        display_name="MLP node classifier",
        description="A two-layer classifier over final node hidden states.",
    ),
    "linear": NodeClassifierDefinition(
        classifier_id="linear",
        display_name="Linear node classifier",
        description="A single linear classifier over final node hidden states.",
    ),
}

GRAPH_SAGE_ARCHITECTURE_ID: Final[str] = "graphsage"
AA_GRAPH_SAGE_ARCHITECTURE_ID: Final[str] = "aa-graphsage"

GNN_ARCHITECTURES: Final[dict[str, GnnArchitectureDefinition]] = {
    GRAPH_SAGE_ARCHITECTURE_ID: GnnArchitectureDefinition(
        architecture_id=GRAPH_SAGE_ARCHITECTURE_ID,
        display_name="GraphSAGE",
        description="Baseline GraphSAGE with configurable depth, width, classifier, and dropout.",
    ),
    AA_GRAPH_SAGE_ARCHITECTURE_ID: GnnArchitectureDefinition(
        architecture_id=AA_GRAPH_SAGE_ARCHITECTURE_ID,
        display_name="AA-GraphSAGE",
        description="Advanced answer-aware GraphSAGE with relational and question-aware components.",
        supports_advanced_options=True,
        default_use_edge_mlp=True,
        default_use_reverse_edges=True,
        default_question_aware_classifier=True,
        default_add_layer_normalization=True,
        supported_edge_mlp_hidden_dimensions=(128, 256, 512),
        default_edge_mlp_hidden_dimension=256,
    ),
}

OPENAI_EMBEDDING_MODELS: Final[dict[str, OpenAiEmbeddingModelDefinition]] = {
    "text-embedding-3-small": OpenAiEmbeddingModelDefinition(
        model_id="text-embedding-3-small",
        display_name="text-embedding-3-small",
        dimensions=1536,
        description="Small OpenAI embedding model for cost-efficient text embeddings.",
    ),
    "text-embedding-3-large": OpenAiEmbeddingModelDefinition(
        model_id="text-embedding-3-large",
        display_name="text-embedding-3-large",
        dimensions=3072,
        description="Most capable OpenAI embedding model for text embeddings.",
    )
}

RECOMMENDED_MAIN_LLM_MODEL_ID: Final[str] = "gpt-4.1-nano"
RECOMMENDED_SUBGRAPH_CONSTRUCTION_ALGORITHM_ID: Final[str] = "shortest_path"
RECOMMENDED_CONTEXT_CONSTRUCTION_STRATEGY_ID: Final[str] = "structured_triples"
RECOMMENDED_GNN_LAYER_COUNT: Final[int] = 2
RECOMMENDED_GNN_HIDDEN_DIMENSION: Final[int] = 256
RECOMMENDED_NODE_CLASSIFIER_ID: Final[str] = "mlp"
RECOMMENDED_GNN_ARCHITECTURE_ID: Final[str] = GRAPH_SAGE_ARCHITECTURE_ID
RECOMMENDED_QUESTION_EMBEDDING_MODEL_ID: Final[str] = "text-embedding-3-small"
RECOMMENDED_RELATION_EMBEDDING_MODEL_ID: Final[str] = "text-embedding-3-small"
RECOMMENDED_ENTITY_EMBEDDING_MODEL_ID: Final[str] = "text-embedding-3-small"
