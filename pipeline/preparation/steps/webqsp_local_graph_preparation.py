"""WebQSP processed graph dataset preparation step."""

from __future__ import annotations

from pipeline.abstract import AbstractStep, StepContext
from pipeline.exceptions import InvalidInteractiveConfigurationInputException
from pipeline.preparation.models.webqsp_local_graph import PreparedWebQSPGraphDataset
from pipeline.preparation.steps.dataset_loading import LoadedDataset
from pipeline.services.webqsp_local_graph_processing import (
    WebQSPLocalGraphProcessorService,
)
from pipeline.services.webqsp_local_graph_storage import (
    WebQSPLocalGraphStorageService,
)

WEBQSP_LOCAL_GRAPH_PROCESSING_VERSION = "2"


class BuildWebQSPLocalGraphsStep(
    AbstractStep[PreparedWebQSPGraphDataset, LoadedDataset]
):
    """Build or load cached WebQSP processed graph instances."""

    def __init__(
        self,
        processor_service: WebQSPLocalGraphProcessorService | None = None,
        storage_service: WebQSPLocalGraphStorageService | None = None,
        force_default: bool = False,
    ):
        super().__init__(force_default=force_default)
        self.processor_service = processor_service or WebQSPLocalGraphProcessorService()
        self.storage_service = storage_service or WebQSPLocalGraphStorageService()

    def execute_default(
        self,
        context: StepContext[LoadedDataset],
    ) -> PreparedWebQSPGraphDataset:
        loaded_dataset = context.result
        if loaded_dataset is None:
            raise InvalidInteractiveConfigurationInputException(
                "WebQSP local graph preparation requires a loaded dataset."
            )

        cached_dataset = self.storage_service.load_if_available(
            dataset_id=loaded_dataset.dataset_id,
            processing_version=WEBQSP_LOCAL_GRAPH_PROCESSING_VERSION,
        )
        if cached_dataset is not None:
            return cached_dataset

        prepared_dataset = self.processor_service.process_loaded_dataset(
            loaded_dataset=loaded_dataset,
            processing_version=WEBQSP_LOCAL_GRAPH_PROCESSING_VERSION,
            cache_directory=self.storage_service.get_cache_directory(
                loaded_dataset.dataset_id
            ),
        )
        self.storage_service.save(prepared_dataset)
        return prepared_dataset
