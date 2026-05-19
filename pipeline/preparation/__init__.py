"""Preparation steps and related models for graphragX."""

from typing import Any

from pipeline.preparation.helpers.dataset_definitions import (
    DATASET_CACHE_ROOT,
    DATASET_LOADERS,
    DatasetDefinition,
    DatasetLoaderDefinition,
    PIPELINE_DATASETS,
    WEBQSP_DATASET_ID,
)
from pipeline.preparation.helpers.configuration_definitions import (
    CONTEXT_CONSTRUCTION_STRATEGIES,
    ContextConstructionDefinition,
    GNN_HIDDEN_DIMENSION_OPTIONS,
    GnnHiddenDimensionDefinition,
    GNN_LAYER_COUNT_OPTIONS,
    GnnLayerCountDefinition,
    LlmModelDefinition,
    NODE_CLASSIFIERS,
    NodeClassifierDefinition,
    OPENAI_EMBEDDING_MODELS,
    OpenAiEmbeddingModelDefinition,
    RECOMMENDED_ASSISTANT_LLM_MODEL_ID,
    RECOMMENDED_CONTEXT_CONSTRUCTION_STRATEGY_ID,
    RECOMMENDED_ENTITY_EMBEDDING_MODEL_ID,
    RECOMMENDED_GNN_HIDDEN_DIMENSION,
    RECOMMENDED_GNN_LAYER_COUNT,
    RECOMMENDED_MAIN_LLM_MODEL_ID,
    RECOMMENDED_NODE_CLASSIFIER_ID,
    RECOMMENDED_QUESTION_EMBEDDING_MODEL_ID,
    RECOMMENDED_RELATION_EMBEDDING_MODEL_ID,
    RECOMMENDED_SUBGRAPH_CONSTRUCTION_ALGORITHM_ID,
    SHARED_LLM_MODELS,
    SubgraphConstructionDefinition,
    SUBGRAPH_CONSTRUCTION_ALGORITHMS,
)

_LAZY_EXPORT_MODULES: dict[str, str] = {
    "BuildPipelineConfigurationStep": "pipeline.preparation.steps.configuration_building",
    "BuiltPipelineConfiguration": "pipeline.preparation.steps.configuration_building",
    "PipelineConfigurationInput": "pipeline.preparation.steps.configuration_building",
    "SelectedDataset": "pipeline.preparation.steps.dataset_selection",
    "SelectDatasetStep": "pipeline.preparation.steps.dataset_selection",
    "LoadDatasetStep": "pipeline.preparation.steps.dataset_loading",
    "LoadedDataset": "pipeline.preparation.steps.dataset_loading",
    "PreparedWebQSPGraphDataset": "pipeline.preparation.models.webqsp_local_graph",
    "WebQSPProcessedInstance": "pipeline.preparation.models.webqsp_local_graph",
    "WebQSPVocabularyStore": "pipeline.preparation.models.webqsp_local_graph",
    "BuildGnnAnswerRetrieverContext": "pipeline.preparation.steps.gnn_model_building",
    "BuildGnnAnswerRetrieverStep": "pipeline.preparation.steps.gnn_model_building",
    "BuiltGnnAnswerRetriever": "pipeline.preparation.steps.gnn_model_building",
    "TrainGnnAnswerRetrieverContext": "pipeline.preparation.steps.gnn_answer_retriever_training",
    "TrainGnnAnswerRetrieverStep": "pipeline.preparation.steps.gnn_answer_retriever_training",
    "TrainedGnnAnswerRetriever": "pipeline.preparation.steps.gnn_answer_retriever_training",
    "BuildWebQSPLocalGraphsStep": "pipeline.preparation.steps.webqsp_local_graph_preparation",
}


def __getattr__(name: str) -> Any:
    """Lazy-load preparation exports to avoid service/package import cycles."""
    module_name = _LAZY_EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value

__all__ = [
    "BuildGnnAnswerRetrieverContext",
    "BuildGnnAnswerRetrieverStep",
    "BuildPipelineConfigurationStep",
    "BuildWebQSPLocalGraphsStep",
    "BuiltGnnAnswerRetriever",
    "BuiltPipelineConfiguration",
    "CONTEXT_CONSTRUCTION_STRATEGIES",
    "ContextConstructionDefinition",
    "DATASET_CACHE_ROOT",
    "DATASET_LOADERS",
    "DatasetDefinition",
    "DatasetLoaderDefinition",
    "GNN_HIDDEN_DIMENSION_OPTIONS",
    "GnnHiddenDimensionDefinition",
    "GNN_LAYER_COUNT_OPTIONS",
    "GnnLayerCountDefinition",
    "LoadedDataset",
    "LoadDatasetStep",
    "LlmModelDefinition",
    "NODE_CLASSIFIERS",
    "NodeClassifierDefinition",
    "OPENAI_EMBEDDING_MODELS",
    "OpenAiEmbeddingModelDefinition",
    "PIPELINE_DATASETS",
    "PipelineConfigurationInput",
    "PreparedWebQSPGraphDataset",
    "RECOMMENDED_ASSISTANT_LLM_MODEL_ID",
    "RECOMMENDED_CONTEXT_CONSTRUCTION_STRATEGY_ID",
    "RECOMMENDED_ENTITY_EMBEDDING_MODEL_ID",
    "RECOMMENDED_GNN_HIDDEN_DIMENSION",
    "RECOMMENDED_GNN_LAYER_COUNT",
    "RECOMMENDED_MAIN_LLM_MODEL_ID",
    "RECOMMENDED_NODE_CLASSIFIER_ID",
    "RECOMMENDED_QUESTION_EMBEDDING_MODEL_ID",
    "RECOMMENDED_RELATION_EMBEDDING_MODEL_ID",
    "RECOMMENDED_SUBGRAPH_CONSTRUCTION_ALGORITHM_ID",
    "SelectedDataset",
    "SelectDatasetStep",
    "SHARED_LLM_MODELS",
    "SubgraphConstructionDefinition",
    "SUBGRAPH_CONSTRUCTION_ALGORITHMS",
    "TrainedGnnAnswerRetriever",
    "TrainGnnAnswerRetrieverContext",
    "TrainGnnAnswerRetrieverStep",
    "WEBQSP_DATASET_ID",
    "WebQSPProcessedInstance",
    "WebQSPVocabularyStore",
]
