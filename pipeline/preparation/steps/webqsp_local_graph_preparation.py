"""WebQSP processed graph dataset preparation step."""

from __future__ import annotations

from helpers.logging_config import get_logger
from pipeline.abstract import AbstractStep, StepContext
from pipeline.preparation.exceptions import InvalidInteractiveConfigurationInputException
from pipeline.preparation.models.webqsp_local_graph import PreparedWebQSPGraphDataset
from pipeline.preparation.steps.dataset_loading import LoadedDataset
from pipeline.preparation.services.webqsp_local_graph_processing import (
    WebQSPLocalGraphProcessorService,
)
from pipeline.preparation.services.webqsp_local_graph_storage import (
    WebQSPLocalGraphStorageService,
)

WEBQSP_LOCAL_GRAPH_PROCESSING_VERSION = "5"
logger = get_logger(__name__)


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

        logger.info(
            f"Preparing WebQSP graph instances for dataset={loaded_dataset.dataset_id} "
            f"processing_version={WEBQSP_LOCAL_GRAPH_PROCESSING_VERSION}"
        )
        cached_dataset = self.storage_service.load_if_available(
            dataset_id=loaded_dataset.dataset_id,
            processing_version=WEBQSP_LOCAL_GRAPH_PROCESSING_VERSION,
        )
        if cached_dataset is not None:
            logger.info(
                f"Loaded prepared WebQSP graph dataset from cache: "
                f"train_size={cached_dataset.train_size} test_size={cached_dataset.test_size}"
            )
            return cached_dataset

        logger.info(f"Prepared WebQSP graph cache miss; processing loaded dataset")
        prepared_dataset = self.processor_service.process_loaded_dataset(
            loaded_dataset=loaded_dataset,
            processing_version=WEBQSP_LOCAL_GRAPH_PROCESSING_VERSION,
            cache_directory=self.storage_service.get_cache_directory(
                loaded_dataset.dataset_id
            ),
        )
        self.storage_service.save(prepared_dataset)
        logger.info(
            f"Saved prepared WebQSP graph dataset: "
            f"train_size={prepared_dataset.train_size} test_size={prepared_dataset.test_size}"
        )
        return prepared_dataset
