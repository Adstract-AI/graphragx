"""Knowledge graph dataset loading step for preparation."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from pipeline.abstract import AbstractStep, StepContext, StepResult
from pipeline.exceptions import (
    InvalidInteractiveConfigurationInputException,
    UnsupportedKnowledgeGraphDatasetLoaderException,
)
from pipeline.preparation.helpers.dataset_definitions import (
    KNOWLEDGE_GRAPH_DATASETS,
)
from pipeline.preparation.steps.configuration_building import BuiltPipelineConfiguration
from pipeline.services import (
    AbstractDatasetLoaderService,
    AbstractKnowledgeGraphDatasetProcessingService,
    KnowledgeGraphDatasetProcessingService,
    TorchGeometricKnowledgeGraphLoaderService,
)


class KnowledgeGraphRawTriple(BaseModel):
    """Typed raw triple extracted from a loaded knowledge graph dataset."""

    head_id: int = Field(..., description="Integer id of the head entity.")
    relation_id: int = Field(..., description="Integer id of the relation type.")
    tail_id: int = Field(..., description="Integer id of the tail entity.")


class LoadedKnowledgeGraphDataset(StepResult):
    """Loaded knowledge graph artifact for later standardization and training."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    dataset_id: str = Field(..., description="Selected dataset identifier.")
    dataset_family: str = Field(..., description="Dataset family classification.")
    raw_triples: list[KnowledgeGraphRawTriple] = Field(
        default_factory=list,
        description="Typed raw triple view extracted from the loaded graph.",
    )
    entity_count: int = Field(..., description="Number of entities in the graph.")
    relation_count: int = Field(..., description="Number of relation types in the graph.")
    triple_count: int = Field(..., description="Number of raw triples in the graph.")
    torch_geometric_dataset: object = Field(
        ...,
        description="Loaded Torch Geometric dataset wrapper.",
    )
    torch_geometric_data: object = Field(
        ...,
        description="Concrete Torch Geometric data object for the graph.",
    )


class LoadKnowledgeGraphDatasetStep(
    AbstractStep[LoadedKnowledgeGraphDataset, BuiltPipelineConfiguration]
):
    """Load the selected knowledge graph dataset through Torch Geometric."""

    def __init__(
        self,
        loader_service: AbstractDatasetLoaderService | None = None,
        processing_service: AbstractKnowledgeGraphDatasetProcessingService | None = None,
        force_default: bool = False,
    ):
        super().__init__(force_default=force_default)
        self.loader_service = loader_service or TorchGeometricKnowledgeGraphLoaderService()
        self.processing_service = (
            processing_service or KnowledgeGraphDatasetProcessingService()
        )

    def execute_default(
        self,
        context: StepContext[BuiltPipelineConfiguration],
    ) -> LoadedKnowledgeGraphDataset:
        configuration = context.result
        if configuration is None:
            raise InvalidInteractiveConfigurationInputException(
                "Dataset loading requires a built pipeline configuration in the incoming context."
            )

        dataset_definition = KNOWLEDGE_GRAPH_DATASETS.get(configuration.dataset_id)
        if dataset_definition is None:
            raise UnsupportedKnowledgeGraphDatasetLoaderException(
                f"Unsupported dataset loader configuration for dataset: {configuration.dataset_id}"
            )

        dataset, data = self.loader_service.load_dataset(configuration.dataset_id)
        raw_triples = self.processing_service.extract_raw_triples(
            dataset_id=configuration.dataset_id,
            data=data,
        )

        return LoadedKnowledgeGraphDataset(
            dataset_id=configuration.dataset_id,
            dataset_family=dataset_definition.dataset_family,
            raw_triples=raw_triples,
            entity_count=int(data.num_nodes),
            relation_count=len({triple.relation_id for triple in raw_triples}),
            triple_count=len(raw_triples),
            torch_geometric_dataset=dataset,
            torch_geometric_data=data,
        )
