"""Prepare compact frozen embedding matrices for GNN training."""

from __future__ import annotations

from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from helpers.constants import (
    DEFAULT_TRAINING_EMBEDDING_CACHE_DEVICE,
    DEFAULT_TRAINING_EMBEDDING_CACHE_DTYPE,
    DEFAULT_TRAINING_GPU_CACHE_RESERVE_GB,
)
from helpers.logging_config import get_logger
from pipeline.preparation.exceptions import GnnAnswerRetrieverTrainingException
from pipeline.preparation.helpers.configuration_definitions import GNN_ARCHITECTURES
from pipeline.preparation.models.gnn_training_data import (
    PreparedGnnTrainingData,
    PreparedGnnTrainingInstance,
)
from pipeline.preparation.models.webqsp_local_graph import (
    PreparedWebQSPGraphDataset,
    WebQSPProcessedInstance,
)
from pipeline.preparation.services.embedding_cache import (
    TextEmbeddingCache,
    WebQSPEmbeddingCacheService,
)
from pipeline.preparation.services.gnn_answer_retriever_model_runs import (
    GnnAnswerRetrieverModelRunService,
)
from pipeline.preparation.services.gnn_embedding_tensor_cache import (
    GnnEmbeddingTensorCacheService,
)
from pipeline.preparation.services.gnn_relation_vocabulary import (
    build_relation_aggregation_metadata,
    build_sorted_typed_edges,
)
from pipeline.preparation.steps.configuration_building import BuiltPipelineConfiguration
from pipeline.preparation.steps.gnn_model_building import BuiltGnnAnswerRetriever
from pipeline.services import AbstractService

logger = get_logger(__name__)

if TYPE_CHECKING:
    from torch import Tensor, dtype as TorchDtype


class GnnTrainingDataPreparationConfig(BaseModel):
    """Settings controlling training-slice embedding preparation and placement."""

    start_instance: int = 0
    max_instances: int | None = None
    training_device: str = "auto"
    embedding_cache_device: Literal["auto", "gpu", "cpu"] = Field(
        default=DEFAULT_TRAINING_EMBEDDING_CACHE_DEVICE
    )
    embedding_cache_dtype: Literal["auto", "float32", "bfloat16"] = Field(
        default=DEFAULT_TRAINING_EMBEDDING_CACHE_DTYPE
    )
    gpu_cache_reserve_gb: float = DEFAULT_TRAINING_GPU_CACHE_RESERVE_GB
    continue_from_model_run_name: str | None = None
    continue_from_model_run_number: int | None = None


class GnnTrainingDataPreparationService(AbstractService):
    """Build compact embedding matrices and integer-indexed training graphs."""

    bytes_per_gibibyte = 1024**3

    def __init__(
        self,
        embedding_cache_service: WebQSPEmbeddingCacheService | None = None,
        embedding_tensor_cache_service: GnnEmbeddingTensorCacheService | None = None,
        model_run_service: GnnAnswerRetrieverModelRunService | None = None,
    ) -> None:
        self.embedding_cache_service = (
            embedding_cache_service or WebQSPEmbeddingCacheService()
        )
        self.embedding_tensor_cache_service = (
            embedding_tensor_cache_service
            or GnnEmbeddingTensorCacheService(self.embedding_cache_service)
        )
        self.model_run_service = model_run_service or GnnAnswerRetrieverModelRunService()

    def prepare(
        self,
        built_retriever: BuiltGnnAnswerRetriever,
        prepared_dataset: PreparedWebQSPGraphDataset,
        configuration: BuiltPipelineConfiguration,
        preparation_config: GnnTrainingDataPreparationConfig,
    ) -> PreparedGnnTrainingData:
        """Prepare selected graph embeddings once and place them for repeated epochs."""
        import torch

        if preparation_config.gpu_cache_reserve_gb < 0:
            raise GnnAnswerRetrieverTrainingException(
                "training_gpu_cache_reserve_gb must be greater than or equal to zero."
            )

        selected_instances, start_index, end_index = self._select_instances(
            prepared_dataset=prepared_dataset,
            start_instance=preparation_config.start_instance,
            max_instances=preparation_config.max_instances,
        )
        selected_device = self._resolve_training_device(
            torch=torch,
            requested_device=preparation_config.training_device,
        )
        (
            entity_embedding_model,
            question_embedding_model,
            relation_embedding_model,
        ) = self._resolve_embedding_models(
            cache_root=prepared_dataset.cache_directory.parent,
            built_retriever=built_retriever,
            configuration=configuration,
            preparation_config=preparation_config,
        )
        architecture = GNN_ARCHITECTURES[built_retriever.gnn_architecture]
        requirements = architecture.data_requirements
        relation_vocabulary = self._resolve_relation_vocabulary(
            cache_root=prepared_dataset.cache_directory.parent,
            built_retriever=built_retriever,
            preparation_config=preparation_config,
        )

        node_texts, node_index_tensors = self._compact_text_index_tensors(
            instance_texts=[instance.nodes for instance in selected_instances],
            torch=torch,
        )
        relation_texts: list[str] = []
        relation_index_tensors: list[Tensor | None] = [None] * len(selected_instances)
        if requirements.uses_relation_embeddings:
            relation_texts, relation_index_tensors = self._compact_text_index_tensors(
                instance_texts=[
                    instance.edge_relations for instance in selected_instances
                ],
                torch=torch,
            )
        question_texts: list[str] = []
        question_indices: list[list[int]] = [[] for _ in selected_instances]
        if requirements.uses_question_embeddings:
            question_texts, question_indices = self._compact_text_indices(
                [[instance.question] for instance in selected_instances]
            )

        cache_root = prepared_dataset.cache_directory.parent
        node_cache = self.embedding_cache_service.load_node_cache(
            cache_root=cache_root,
            model_id=entity_embedding_model,
            vocabulary={text: index for index, text in enumerate(node_texts)},
            dataset_id=prepared_dataset.dataset_id,
            ensure_collection=False,
        )
        relation_cache = (
            self.embedding_cache_service.load_relation_cache(
                cache_root=cache_root,
                model_id=relation_embedding_model,
                vocabulary={text: index for index, text in enumerate(relation_texts)},
                dataset_id=prepared_dataset.dataset_id,
                ensure_collection=False,
            )
            if requirements.uses_relation_embeddings
            else None
        )
        question_cache = (
            self.embedding_cache_service.load_question_cache(
                cache_root=cache_root,
                model_id=question_embedding_model,
                vocabulary={text: index for index, text in enumerate(question_texts)},
                dataset_id=prepared_dataset.dataset_id,
                ensure_collection=False,
            )
            if requirements.uses_question_embeddings
            else None
        )

        embedding_dtype = self._resolve_embedding_dtype(
            torch=torch,
            requested_dtype=preparation_config.embedding_cache_dtype,
            selected_device=selected_device,
        )
        cache_specs = [(node_cache, node_texts)]
        if relation_cache is not None:
            cache_specs.append((relation_cache, relation_texts))
        if question_cache is not None:
            cache_specs.append((question_cache, question_texts))
        total_embedding_bytes = sum(
            len(texts) * cache.vector_size * (2 if embedding_dtype == "bfloat16" else 4)
            for cache, texts in cache_specs
        )
        embedding_device = self._resolve_embedding_device(
            torch=torch,
            requested_device=preparation_config.embedding_cache_device,
            selected_device=selected_device,
            required_bytes=total_embedding_bytes,
            reserve_gb=preparation_config.gpu_cache_reserve_gb,
        )
        logger.info(
            f"Preparing compact GNN training embeddings: instances={len(selected_instances)} "
            f"nodes={len(node_texts)} relations={len(relation_texts)} "
            f"questions={len(question_texts)} storage_gib="
            f"{total_embedding_bytes / self.bytes_per_gibibyte:.2f} "
            f"device={embedding_device} dtype={embedding_dtype}"
        )

        torch_dtype = torch.bfloat16 if embedding_dtype == "bfloat16" else torch.float32
        cuda_allocation_failed = False
        try:
            matrices = tuple(
                self._load_embedding_matrix(
                    torch=torch,
                    cache_root=cache_root,
                    cache=cache,
                    texts=texts,
                    dtype=torch_dtype,
                    dtype_name=embedding_dtype,
                    device=embedding_device,
                    preprocess=cache.cache_kind == "relations",
                )
                for cache, texts in cache_specs
            )
        except torch.OutOfMemoryError as error:
            if not embedding_device.startswith("cuda"):
                raise GnnAnswerRetrieverTrainingException(
                    "Host memory was exhausted while loading compact embeddings."
                ) from error
            if preparation_config.embedding_cache_device == "gpu":
                raise GnnAnswerRetrieverTrainingException(
                    "CUDA ran out of memory while loading the forced GPU embedding cache."
                ) from error
            logger.warning(
                "CUDA allocation failed while preparing compact embeddings; "
                "retrying with CPU storage."
            )
            embedding_device = "cpu"
            matrices = tuple(
                self._load_embedding_matrix(
                    torch=torch,
                    cache_root=cache_root,
                    cache=cache,
                    texts=texts,
                    dtype=torch_dtype,
                    dtype_name=embedding_dtype,
                    device=embedding_device,
                    preprocess=cache.cache_kind == "relations",
                )
                for cache, texts in cache_specs
            )
            cuda_allocation_failed = True
        if cuda_allocation_failed:
            torch.cuda.empty_cache()
        node_embeddings = matrices[0]
        matrix_offset = 1
        relation_embeddings = None
        if relation_cache is not None:
            relation_embeddings = matrices[matrix_offset]
            matrix_offset += 1
        question_embeddings = (
            matrices[matrix_offset] if question_cache is not None else None
        )
        prepared_instances: list[PreparedGnnTrainingInstance] = []
        for offset, instance in enumerate(selected_instances):
            edge_index = instance.edge_index
            edge_type = None
            edge_norm = None
            active_relation_ids = None
            edge_relation_index = None
            if requirements.uses_relation_types:
                if relation_vocabulary is None:
                    raise GnnAnswerRetrieverTrainingException(
                        "R-GCN training requires an authoritative relation vocabulary."
                    )
                try:
                    edge_index, edge_type = build_sorted_typed_edges(
                        edge_index=edge_index,
                        edge_relations=instance.edge_relations,
                        vocabulary=relation_vocabulary,
                        torch=torch,
                    )
                    (
                        edge_norm,
                        active_relation_ids,
                        edge_relation_index,
                    ) = build_relation_aggregation_metadata(
                        edge_index=edge_index,
                        edge_type=edge_type,
                        node_count=len(instance.nodes),
                        torch=torch,
                    )
                except ValueError as error:
                    raise GnnAnswerRetrieverTrainingException(
                        f"Could not prepare R-GCN edge types for training instance "
                        f"{start_index + offset}: {error}"
                    ) from error
            prepared_instances.append(
                PreparedGnnTrainingInstance(
                    source_instance_index=start_index + offset,
                    node_embedding_indices=node_index_tensors[offset],
                    relation_embedding_indices=relation_index_tensors[offset],
                    question_embedding_index=(
                        question_indices[offset][0]
                        if question_indices[offset]
                        else None
                    ),
                    edge_index=edge_index,
                    edge_type=edge_type,
                    edge_norm=edge_norm,
                    active_relation_ids=active_relation_ids,
                    edge_relation_index=edge_relation_index,
                    node_labels=instance.node_labels,
                )
            )
        logger.info(
            f"Prepared GNN training data once for epochs: "
            f"instances={len(prepared_instances)} device={embedding_device} "
            f"dtype={embedding_dtype}"
        )
        return PreparedGnnTrainingData(
            built_retriever=built_retriever,
            instances=prepared_instances,
            node_embeddings=node_embeddings,
            relation_embeddings=relation_embeddings,
            question_embeddings=question_embeddings,
            training_start_instance=start_index,
            training_end_instance=end_index,
            selected_device=selected_device,
            embedding_cache_device=embedding_device,
            embedding_cache_dtype=embedding_dtype,
            entity_embedding_model=entity_embedding_model,
            question_embedding_model=question_embedding_model,
            relation_embedding_model=relation_embedding_model,
            cache_root=cache_root,
        )

    def _resolve_relation_vocabulary(
        self,
        *,
        cache_root: Path,
        built_retriever: BuiltGnnAnswerRetriever,
        preparation_config: GnnTrainingDataPreparationConfig,
    ) -> dict[str, int] | None:
        """Use a continued R-GCN run's saved relation IDs as authoritative."""
        if (
            preparation_config.continue_from_model_run_name is None
            and preparation_config.continue_from_model_run_number is None
        ):
            return built_retriever.relation_vocabulary
        saved_run = self.model_run_service.resolve_run(
            model_root=cache_root / "models",
            run_name=preparation_config.continue_from_model_run_name,
            run_number=preparation_config.continue_from_model_run_number,
        )
        if saved_run.config.resolved_gnn_architecture != built_retriever.gnn_architecture:
            raise GnnAnswerRetrieverTrainingException(
                "Continued training cannot change GNN architecture from "
                f"{saved_run.config.resolved_gnn_architecture} to "
                f"{built_retriever.gnn_architecture}."
            )
        return saved_run.relation_vocabulary

    @staticmethod
    def _select_instances(
        prepared_dataset: PreparedWebQSPGraphDataset,
        start_instance: int,
        max_instances: int | None,
    ) -> tuple[list[WebQSPProcessedInstance], int, int]:
        """Select the configured contiguous training split slice."""
        if start_instance < 0:
            raise GnnAnswerRetrieverTrainingException(
                "training_start_instance must be greater than or equal to 0."
            )
        end_instance = (
            len(prepared_dataset.train_instances)
            if max_instances is None
            else start_instance + max_instances
        )
        selected_instances = prepared_dataset.train_instances[start_instance:end_instance]
        if not selected_instances:
            raise GnnAnswerRetrieverTrainingException(
                f"GNN answer retriever training selected no instances: "
                f"start={start_instance} end={end_instance} "
                f"available={len(prepared_dataset.train_instances)}."
            )
        return selected_instances, start_instance, start_instance + len(selected_instances)

    @staticmethod
    def _compact_text_indices(
        instance_texts: list[list[str]],
    ) -> tuple[list[str], list[list[int]]]:
        """Create one compact vocabulary and per-instance integer indices."""
        text_to_index: dict[str, int] = {}
        compact_texts: list[str] = []
        instance_indices: list[list[int]] = []
        for texts in instance_texts:
            indices: list[int] = []
            for text in texts:
                compact_index = text_to_index.get(text)
                if compact_index is None:
                    compact_index = len(compact_texts)
                    text_to_index[text] = compact_index
                    compact_texts.append(text)
                indices.append(compact_index)
            instance_indices.append(indices)
        return compact_texts, instance_indices

    @staticmethod
    def _compact_text_index_tensors(
        instance_texts: list[list[str]],
        torch: ModuleType,
    ) -> tuple[list[str], list[Tensor]]:
        """Build compact IDs as tensors without retaining all Python integers."""
        text_to_index: dict[str, int] = {}
        compact_texts: list[str] = []
        instance_index_tensors: list[Tensor] = []
        for texts in instance_texts:
            indices: list[int] = []
            for text in texts:
                compact_index = text_to_index.get(text)
                if compact_index is None:
                    compact_index = len(compact_texts)
                    text_to_index[text] = compact_index
                    compact_texts.append(text)
                indices.append(compact_index)
            instance_index_tensors.append(torch.tensor(indices, dtype=torch.long))
        return compact_texts, instance_index_tensors

    def _resolve_embedding_models(
        self,
        cache_root: Path,
        built_retriever: BuiltGnnAnswerRetriever,
        configuration: BuiltPipelineConfiguration,
        preparation_config: GnnTrainingDataPreparationConfig,
    ) -> tuple[str, str, str]:
        """Resolve embedding model IDs for fresh or continued training."""
        if (
            preparation_config.continue_from_model_run_name is None
            and preparation_config.continue_from_model_run_number is None
        ):
            embedding_model = (
                configuration.embedding_model
                or built_retriever.entity_embedding_model
            )
            return (
                embedding_model,
                embedding_model,
                embedding_model,
            )
        saved_run = self.model_run_service.resolve_run(
            model_root=cache_root / "models",
            run_name=preparation_config.continue_from_model_run_name,
            run_number=preparation_config.continue_from_model_run_number,
        )
        embedding_model = saved_run.config.resolved_embedding_model
        return (embedding_model, embedding_model, embedding_model)

    @staticmethod
    def _resolve_training_device(torch: ModuleType, requested_device: str) -> str:
        """Resolve the accelerator used by the training service."""
        if requested_device != "auto":
            if requested_device.startswith("cuda") and not torch.cuda.is_available():
                raise GnnAnswerRetrieverTrainingException(
                    "CUDA training was requested, but CUDA is not available."
                )
            if requested_device == "mps" and not torch.backends.mps.is_available():
                raise GnnAnswerRetrieverTrainingException(
                    "MPS training was requested, but MPS is not available."
                )
            return requested_device
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @staticmethod
    def _resolve_embedding_dtype(
        torch: ModuleType,
        requested_dtype: str,
        selected_device: str,
    ) -> str:
        """Resolve compact embedding precision for the training device."""
        if not selected_device.startswith("cuda"):
            if requested_dtype == "bfloat16":
                raise GnnAnswerRetrieverTrainingException(
                    "bfloat16 training embeddings require a CUDA training device."
                )
            return "float32"
        if requested_dtype == "float32":
            return "float32"
        if torch.cuda.is_bf16_supported():
            return "bfloat16"
        if requested_dtype == "bfloat16":
            raise GnnAnswerRetrieverTrainingException(
                "The selected CUDA device does not support bfloat16 embeddings."
            )
        return "float32"

    def _resolve_embedding_device(
        self,
        torch: ModuleType,
        requested_device: str,
        selected_device: str,
        required_bytes: int,
        reserve_gb: float,
    ) -> str:
        """Place embeddings on CUDA when requested and safely within free VRAM."""
        if requested_device == "cpu":
            return "cpu"
        if not selected_device.startswith("cuda"):
            if requested_device == "gpu":
                raise GnnAnswerRetrieverTrainingException(
                    "GPU embedding cache requested without a CUDA training device."
                )
            return "cpu"
        free_bytes, _ = torch.cuda.mem_get_info()
        reserve_bytes = int(reserve_gb * self.bytes_per_gibibyte)
        fits_safely = required_bytes <= max(0, free_bytes - reserve_bytes)
        if fits_safely:
            return selected_device
        if requested_device == "gpu":
            raise GnnAnswerRetrieverTrainingException(
                f"GPU embedding cache requires {required_bytes / self.bytes_per_gibibyte:.2f} "
                f"GiB with a {reserve_gb:.2f} GiB reserve, but only "
                f"{free_bytes / self.bytes_per_gibibyte:.2f} GiB is free."
            )
        logger.warning(
            f"Compact embeddings do not fit the safe CUDA budget; using CPU storage: "
            f"required_gib={required_bytes / self.bytes_per_gibibyte:.2f} "
            f"free_gib={free_bytes / self.bytes_per_gibibyte:.2f} "
            f"reserve_gib={reserve_gb:.2f}"
        )
        return "cpu"

    @staticmethod
    def _embedding_storage_bytes(
        text_counts: tuple[int, int, int],
        vector_sizes: tuple[int, int, int],
        element_size: int,
    ) -> int:
        """Calculate storage required by all compact embedding matrices."""
        return sum(
            text_count * vector_size * element_size
            for text_count, vector_size in zip(text_counts, vector_sizes, strict=True)
        )

    def _load_embedding_matrix(
        self,
        torch: ModuleType,
        cache_root: Path,
        cache: TextEmbeddingCache,
        texts: list[str],
        dtype: TorchDtype,
        dtype_name: str,
        device: str,
        preprocess: bool,
    ) -> Tensor:
        """Load one compact matrix through the incremental local tensor cache."""
        return self.embedding_tensor_cache_service.load_matrix(
            torch=torch,
            cache_root=cache_root,
            cache=cache,
            texts=texts,
            dtype=dtype,
            dtype_name=dtype_name,
            device=device,
            preprocess=preprocess,
        )

    def _load_embedding_matrices(
        self,
        torch: ModuleType,
        cache_root: Path,
        caches: tuple[TextEmbeddingCache, TextEmbeddingCache, TextEmbeddingCache],
        text_groups: tuple[list[str], list[str], list[str]],
        dtype: TorchDtype,
        dtype_name: str,
        device: str,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Load node, relation, and question matrices on one storage device."""
        return tuple(
            self._load_embedding_matrix(
                torch=torch,
                cache_root=cache_root,
                cache=cache,
                texts=texts,
                dtype=dtype,
                dtype_name=dtype_name,
                device=device,
                preprocess=cache.cache_kind == "relations",
            )
            for cache, texts in zip(caches, text_groups, strict=True)
        )
