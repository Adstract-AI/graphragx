"""Dataset selection for the preparation pipeline."""

from __future__ import annotations

from collections.abc import Callable

from pydantic import Field

from helpers.logging_config import get_logger
from pipeline.abstract import AbstractStep, StepContext, StepResult
from pipeline.preparation.exceptions import UnsupportedDatasetSelectionException
from pipeline.models import InitialStepResult
from pipeline.preparation.helpers.dataset_definitions import (
    PIPELINE_DATASETS,
    WEBQSP_DATASET_ID,
)
from pipeline.preparation.services.selection import SelectionService

logger = get_logger(__name__)


class SelectedDataset(StepResult):
    """Metadata-only result describing the selected dataset."""

    dataset_id: str = Field(..., description="Stable identifier of the selected dataset.")
    display_name: str = Field(..., description="Human-readable dataset name.")
    dataset_family: str = Field(..., description="High-level dataset family.")
    task_domain: str = Field(..., description="Reasoning task or domain classification.")
    description: str = Field(..., description="Short dataset description.")
    supported: bool = Field(..., description="Whether the dataset is currently supported.")


class SelectDatasetStep(
    AbstractStep[SelectedDataset, InitialStepResult]
):
    """Select the dataset for the preparation pipeline."""

    def __init__(
        self,
        requested_dataset: str | None = None,
        input_func: Callable[[str], str] | None = None,
        force_default: bool = False,
    ):
        super().__init__(force_default=force_default)
        self.requested_dataset = requested_dataset
        self.selection_service = SelectionService(input_func=input_func)

    def execute_default(
        self,
        context: StepContext[InitialStepResult],
    ) -> SelectedDataset:
        logger.info(f"Selecting pipeline dataset")
        requested_dataset = self.selection_service.resolve_choice(
            provided_value=self.requested_dataset,
            options=PIPELINE_DATASETS,
            prompt_title="Dataset",
            prompt_help="Select the dataset used for local-graph QA training and evaluation.",
            recommended_id=WEBQSP_DATASET_ID,
            invalid_exception_type=UnsupportedDatasetSelectionException,
            value_getter=lambda item: item.dataset_id,
            label_getter=lambda item: item.display_name,
        )
        dataset_definition = PIPELINE_DATASETS.get(requested_dataset)
        if dataset_definition is None or dataset_definition.dataset_id != WEBQSP_DATASET_ID:
            raise UnsupportedDatasetSelectionException(
                f"Unsupported dataset: {requested_dataset}"
            )

        logger.info(f"Selected dataset: {dataset_definition.dataset_id}")
        return SelectedDataset(
            dataset_id=dataset_definition.dataset_id,
            display_name=dataset_definition.display_name,
            dataset_family=dataset_definition.dataset_family,
            task_domain=dataset_definition.task_domain,
            description=dataset_definition.description,
            supported=dataset_definition.supported,
        )
