"""Dataset loading step for preparation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import ConfigDict, Field

from helpers.logging_config import get_logger
from pipeline.abstract import AbstractStep, StepContext, StepResult
from pipeline.preparation.exceptions import (
    InvalidInteractiveConfigurationInputException,
    MalformedDatasetException,
    UnsupportedDatasetLoaderException,
)
from pipeline.preparation.helpers.dataset_definitions import PIPELINE_DATASETS
from pipeline.preparation.steps.configuration_building import BuiltPipelineConfiguration
from pipeline.preparation.services.dataset_loader import (
    AbstractDatasetLoaderService,
    HuggingFaceWebQSPDatasetLoaderService,
)

if TYPE_CHECKING:
    from datasets import DatasetDict as HuggingFaceDatasetDict
else:
    HuggingFaceDatasetDict = Any

logger = get_logger(__name__)


class LoadedDataset(StepResult):
    """Loaded dataset artifact for later local-graph construction and training."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    dataset_id: str = Field(..., description="Selected dataset identifier.")
    dataset_family: str = Field(..., description="Dataset family classification.")
    hugging_face_dataset_name: str = Field(..., description="Hugging Face dataset name.")
    split_names: list[str] = Field(..., description="Available dataset split names.")
    split_sizes: dict[str, int] = Field(..., description="Number of examples per split.")
    hugging_face_dataset: HuggingFaceDatasetDict = Field(
        ...,
        description="Loaded Hugging Face dataset.",
    )


class LoadDatasetStep(AbstractStep[LoadedDataset, BuiltPipelineConfiguration]):
    """Load the selected dataset."""

    def __init__(
        self,
        loader_service: AbstractDatasetLoaderService | None = None,
        force_default: bool = False,
    ):
        super().__init__(force_default=force_default)
        self.loader_service = loader_service or HuggingFaceWebQSPDatasetLoaderService()

    def execute_default(
        self,
        context: StepContext[BuiltPipelineConfiguration],
    ) -> LoadedDataset:
        configuration = context.result
        if configuration is None:
            raise InvalidInteractiveConfigurationInputException(
                "Dataset loading requires a built pipeline configuration in the incoming context."
            )

        dataset_definition = PIPELINE_DATASETS.get(configuration.dataset_id)
        if dataset_definition is None:
            raise UnsupportedDatasetLoaderException(
                f"Unsupported dataset loader configuration for dataset: {configuration.dataset_id}"
            )

        logger.info(f"Loading dataset: {configuration.dataset_id}")
        dataset = self.loader_service.load_dataset(configuration.dataset_id)
        loader_definition = self.loader_service.get_loader_definition(
            configuration.dataset_id
        )
        split_names = list(dataset.keys())
        if not split_names:
            raise MalformedDatasetException(
                f"Loaded dataset {configuration.dataset_id} does not contain any splits."
            )

        split_sizes = {split_name: len(dataset[split_name]) for split_name in split_names}
        logger.info(
            f"Loaded dataset {configuration.dataset_id}: "
            f"splits={split_names} split_sizes={split_sizes}"
        )
        return LoadedDataset(
            dataset_id=configuration.dataset_id,
            dataset_family=dataset_definition.dataset_family,
            hugging_face_dataset_name=loader_definition.hugging_face_dataset_name,
            split_names=split_names,
            split_sizes=split_sizes,
            hugging_face_dataset=dataset,
        )
