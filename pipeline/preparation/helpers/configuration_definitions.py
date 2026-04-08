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

RECOMMENDED_MAIN_LLM_MODEL_ID: Final[str] = "gpt-5.4"
RECOMMENDED_ASSISTANT_LLM_MODEL_ID: Final[str] = "gpt-5.4-mini"
RECOMMENDED_SUBGRAPH_CONSTRUCTION_ALGORITHM_ID: Final[str] = "shortest_path"
RECOMMENDED_CONTEXT_CONSTRUCTION_STRATEGY_ID: Final[str] = "textualized"
