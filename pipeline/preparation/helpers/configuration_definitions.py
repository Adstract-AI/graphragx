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


class NodeClassifierDefinition(BaseModel):
    """Typed definition of an available node classifier option."""

    model_config = ConfigDict(frozen=True)

    classifier_id: str = Field(..., description="Stable node classifier identifier.")
    display_name: str = Field(..., description="Human-readable classifier name.")
    description: str = Field(..., description="Short description of the classifier.")


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
    "pcst": SubgraphConstructionDefinition(
        algorithm_id="pcst",
        display_name="Prize-Collecting Steiner Tree (PCST)",
        description="Optimize a compact subgraph that balances node value and connection cost.",
    ),
}

CONTEXT_CONSTRUCTION_STRATEGIES: Final[dict[str, ContextConstructionDefinition]] = {
    "textualized": ContextConstructionDefinition(
        strategy_id="textualized",
        display_name="Textualized",
        description="Represent the subgraph as natural language descriptions for the LLM.",
    ),
    "structured_triples": ContextConstructionDefinition(
        strategy_id="structured_triples",
        display_name="Structured Triples",
        description="Represent the subgraph as <Head, Relation, Tail> triples.",
    ),
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
    ),
    "text-embedding-ada-002": OpenAiEmbeddingModelDefinition(
        model_id="text-embedding-ada-002",
        display_name="text-embedding-ada-002",
        dimensions=1536,
        description="Older OpenAI embedding model kept as a compatibility option.",
    ),
}

RECOMMENDED_MAIN_LLM_MODEL_ID: Final[str] = "gpt-5.4"
RECOMMENDED_ASSISTANT_LLM_MODEL_ID: Final[str] = "gpt-5.4-mini"
RECOMMENDED_SUBGRAPH_CONSTRUCTION_ALGORITHM_ID: Final[str] = "shortest_path"
RECOMMENDED_CONTEXT_CONSTRUCTION_STRATEGY_ID: Final[str] = "textualized"
RECOMMENDED_GNN_LAYER_COUNT: Final[int] = 2
RECOMMENDED_NODE_CLASSIFIER_ID: Final[str] = "mlp"
RECOMMENDED_QUESTION_EMBEDDING_MODEL_ID: Final[str] = "text-embedding-3-small"
RECOMMENDED_RELATION_EMBEDDING_MODEL_ID: Final[str] = "text-embedding-3-small"
RECOMMENDED_ENTITY_EMBEDDING_MODEL_ID: Final[str] = "text-embedding-3-small"
