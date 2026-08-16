"""Training service for the GNN answer retriever."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Protocol
from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field

from helpers.path_serialization import make_project_paths_relative
from helpers.constants import (
    DEFAULT_TRAINING_DEVICE,
    DEFAULT_TRAINING_EPOCHS,
    DEFAULT_TRAINING_LEARNING_RATE,
    DEFAULT_TRAINING_LOG_EVERY,
    DEFAULT_TRAINING_PROFILE,
    DEFAULT_TRAINING_WEIGHT_DECAY,
    GNN_ANSWER_RETRIEVER_CONFIG_FILENAME,
    GNN_ANSWER_RETRIEVER_WEIGHTS_FILENAME,
)
from helpers.logging_config import get_logger
from pipeline.preparation.exceptions import GnnAnswerRetrieverTrainingException
from pipeline.preparation.models.gnn_training_data import PreparedGnnTrainingData
from pipeline.preparation.models.webqsp_local_graph import PreparedWebQSPGraphDataset
from pipeline.preparation.steps.gnn_model_building import BuiltGnnAnswerRetriever
from pipeline.services import AbstractService
from pipeline.preparation.services.gnn_answer_retriever_model_runs import (
    GnnAnswerRetrieverModelRunService,
    LoadedGnnAnswerRetrieverRun,
)

if TYPE_CHECKING:
    from torch import Tensor
    from pipeline.preparation.steps.configuration_building import BuiltPipelineConfiguration

logger = get_logger(__name__)


class SynchronizableDeviceRuntime(Protocol):
    """Accelerator runtime capable of synchronizing queued operations."""

    def synchronize(self) -> None:
        """Wait for queued accelerator operations to finish."""


class TorchRuntime(Protocol):
    """Subset of the torch runtime needed by training profiling."""

    cuda: SynchronizableDeviceRuntime
    mps: SynchronizableDeviceRuntime


class GnnAnswerRetrieverTrainingConfig(BaseModel):
    """Runtime training settings for the answer retriever."""

    epochs: int = Field(default=DEFAULT_TRAINING_EPOCHS)
    learning_rate: float = Field(default=DEFAULT_TRAINING_LEARNING_RATE)
    weight_decay: float = Field(default=DEFAULT_TRAINING_WEIGHT_DECAY)
    max_instances: int | None = Field(default=None)
    start_instance: int = Field(default=0)
    log_every: int = Field(default=DEFAULT_TRAINING_LOG_EVERY)
    device: str = Field(default=DEFAULT_TRAINING_DEVICE)
    profile: bool = Field(default=DEFAULT_TRAINING_PROFILE)
    run_name: str | None = Field(default=None)
    continue_from_model_run_name: str | None = Field(default=None)
    continue_from_model_run_number: int | None = Field(default=None)


@dataclass
class GnnTrainingPhaseTimings:
    """Accumulate synchronized wall-clock timings for training phases."""

    input_preparation_seconds: float = 0.0
    forward_seconds: float = 0.0
    loss_seconds: float = 0.0
    backward_seconds: float = 0.0
    optimizer_seconds: float = 0.0
    instance_count: int = 0

    @property
    def total_seconds(self) -> float:
        """Return total measured time across all training phases."""
        return (
            self.input_preparation_seconds
            + self.forward_seconds
            + self.loss_seconds
            + self.backward_seconds
            + self.optimizer_seconds
        )


class GnnAnswerRetrieverTrainingOutcome(BaseModel):
    """Result produced by the training service."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    final_loss: float = Field(..., description="Final average epoch loss.")
    loss_history: list[dict[str, float | int]] = Field(
        default_factory=list,
        description="Average loss per training epoch.",
    )
    trained_instances: int = Field(..., description="Number of training instances used.")
    training_start_instance: int = Field(..., description="Inclusive training slice start.")
    training_end_instance: int = Field(..., description="Exclusive training slice end.")
    model_artifact_path: Path = Field(..., description="Saved model weights path.")
    model_config_path: Path = Field(..., description="Saved model config path.")
    model_run_directory: Path = Field(..., description="Versioned training run directory.")
    model_run_name: str = Field(..., description="Resolved training run folder name.")
    model_run_number: int = Field(..., description="Incremental training run number.")
    embedding_cache_directory: Path = Field(..., description="Embedding cache root.")
    selected_device: str = Field(..., description="Resolved PyTorch training device.")
    embedding_cache_device: str = Field(..., description="Resolved embedding location.")
    embedding_cache_dtype: str = Field(..., description="Frozen embedding precision.")
    dataset_id: str = Field(..., description="Dataset id used by the trained model.")
    entity_embedding_model: str = Field(..., description="Entity embedding model id.")
    question_embedding_model: str = Field(..., description="Question embedding model id.")
    relation_embedding_model: str = Field(..., description="Relation embedding model id.")
    entity_embedding_dimension: int = Field(..., description="Entity embedding dimension.")
    question_embedding_dimension: int = Field(..., description="Question embedding dimension.")
    relation_embedding_dimension: int = Field(..., description="Relation embedding dimension.")
    hidden_dimension: int = Field(..., description="GNN hidden dimension.")
    gnn_layer_count: int = Field(..., description="GNN layer count.")
    node_classifier: str = Field(..., description="Node classifier id.")
    use_edge_mlp: bool = Field(default=False)
    question_aware_classifier: bool = Field(default=False)
    use_reverse_edges: bool = Field(default=False)
    add_layer_normalization: bool = Field(default=False)
    edge_mlp_hidden_dim: int = Field(..., description="Edge MLP hidden dimension.")
    dropout: float = Field(default=0.1)
    model: object = Field(..., description="Trained model instance.")
    is_fine_tuned_model: bool = Field(..., description="Whether training continued a saved run.")
    continued_from_model_run_name: str | None = Field(
        default=None,
        description="Source model run name when continuation was used.",
    )
    continued_from_model_run_number: int | None = Field(
        default=None,
        description="Source model run number when continuation was used.",
    )


class GnnAnswerRetrieverTrainingService(AbstractService):
    """Train and persist the GNN answer retriever."""

    model_weights_filename = GNN_ANSWER_RETRIEVER_WEIGHTS_FILENAME
    model_config_filename = GNN_ANSWER_RETRIEVER_CONFIG_FILENAME

    def __init__(
        self,
        model_run_service: GnnAnswerRetrieverModelRunService | None = None,
    ) -> None:
        self.model_run_service = model_run_service or GnnAnswerRetrieverModelRunService()

    def train(
        self,
        prepared_training_data: PreparedGnnTrainingData,
        prepared_dataset: PreparedWebQSPGraphDataset,
        configuration: BuiltPipelineConfiguration,
        training_config: GnnAnswerRetrieverTrainingConfig,
    ) -> GnnAnswerRetrieverTrainingOutcome:
        """Train the answer retriever from compact indexed embedding matrices."""
        import torch
        import torch.nn.functional as torch_functional

        device = self._resolve_device(torch, training_config.device)
        if device != prepared_training_data.selected_device:
            raise GnnAnswerRetrieverTrainingException(
                f"Prepared training device {prepared_training_data.selected_device} "
                f"does not match resolved training device {device}."
            )
        logger.info(f"Starting GNN answer retriever training on device={device}")

        cache_root = prepared_training_data.cache_root
        continued_run = self._load_continued_run(
            cache_root=cache_root,
            configuration=configuration,
            training_config=training_config,
            device=device,
        )
        effective_retriever = self._build_effective_retriever(
            built_retriever=prepared_training_data.built_retriever,
            continued_run=continued_run,
        )
        question_embedding_model = (
            continued_run.question_embedding_model
            if continued_run is not None
            else configuration.question_embedding_model
        )
        relation_embedding_model = (
            continued_run.relation_embedding_model
            if continued_run is not None
            else configuration.relation_embedding_model
        )
        if (
            effective_retriever.entity_embedding_model
            != prepared_training_data.entity_embedding_model
            or question_embedding_model
            != prepared_training_data.question_embedding_model
            or relation_embedding_model
            != prepared_training_data.relation_embedding_model
        ):
            raise GnnAnswerRetrieverTrainingException(
                "Prepared training embeddings do not match the effective model run."
            )
        if effective_retriever.dataset_id != prepared_dataset.dataset_id:
            raise GnnAnswerRetrieverTrainingException(
                f"Cannot continue training model run from dataset "
                f"{effective_retriever.dataset_id} on dataset {prepared_dataset.dataset_id}."
            )

        logger.info(
            f"Selected GNN training slice: "
            f"start={prepared_training_data.training_start_instance} "
            f"end={prepared_training_data.training_end_instance} "
            f"instances={len(prepared_training_data.instances)} "
            f"embedding_device={prepared_training_data.embedding_cache_device} "
            f"embedding_dtype={prepared_training_data.embedding_cache_dtype}"
        )
        if continued_run is not None:
            logger.info(
                f"Continuing GNN answer retriever from run={continued_run.run_name} "
                f"run_number={continued_run.run_number} "
                f"additional_epochs={training_config.epochs}"
            )

        model = effective_retriever.model
        model.to(device)
        model.train()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
        )

        final_loss = 0.0
        loss_history: list[dict[str, float | int]] = []
        for epoch in range(1, training_config.epochs + 1):
            logger.info(
                f"Training epoch {epoch}/{training_config.epochs} "
                f"over {len(prepared_training_data.instances)} WebQSP instances"
            )
            epoch_loss = torch.zeros((), dtype=torch.float, device=device)
            phase_timings = GnnTrainingPhaseTimings()
            for instance_index, instance in enumerate(
                prepared_training_data.instances,
                start=1,
            ):
                optimizer.zero_grad(set_to_none=True)
                phase_started_at = self._start_profiled_phase(
                    torch=torch,
                    device=device,
                    enabled=training_config.profile,
                )

                entity_features = self._gather_embedding_features(
                    embedding_matrix=prepared_training_data.node_embeddings,
                    indices=instance.node_embedding_indices,
                    torch=torch,
                    device=device,
                )
                question_features = prepared_training_data.question_embeddings[
                    instance.question_embedding_index
                ].to(device=device, non_blocking=True)
                relation_features = self._gather_embedding_features(
                    embedding_matrix=prepared_training_data.relation_embeddings,
                    indices=instance.relation_embedding_indices,
                    torch=torch,
                    device=device,
                )
                edge_weight = (
                    None
                    if effective_retriever.use_edge_mlp
                    else self._build_edge_weight_tensor(
                        relation_features=relation_features,
                        question_features=question_features,
                        torch=torch,
                        torch_functional=torch_functional,
                        device=device,
                    )
                )
                edge_index = instance.edge_index.to(device=device, non_blocking=True)
                node_labels = instance.node_labels.to(device=device, non_blocking=True)
                phase_started_at, elapsed_seconds = self._finish_profiled_phase(
                    torch=torch,
                    device=device,
                    enabled=training_config.profile,
                    phase_started_at=phase_started_at,
                )
                phase_timings.input_preparation_seconds += elapsed_seconds

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                    enabled=(
                        device.startswith("cuda")
                        and prepared_training_data.uses_bfloat16
                    ),
                ):
                    logits = model(
                        entity_features=entity_features,
                        edge_index=edge_index,
                        edge_weight=edge_weight,
                        question_features=question_features,
                        relation_features=relation_features,
                    )
                phase_started_at, elapsed_seconds = self._finish_profiled_phase(
                    torch=torch,
                    device=device,
                    enabled=training_config.profile,
                    phase_started_at=phase_started_at,
                )
                phase_timings.forward_seconds += elapsed_seconds
                loss = self._compute_loss(
                    logits=logits.float(),
                    node_labels=node_labels,
                    torch=torch,
                    torch_functional=torch_functional,
                )
                phase_started_at, elapsed_seconds = self._finish_profiled_phase(
                    torch=torch,
                    device=device,
                    enabled=training_config.profile,
                    phase_started_at=phase_started_at,
                )
                phase_timings.loss_seconds += elapsed_seconds
                loss.backward()
                phase_started_at, elapsed_seconds = self._finish_profiled_phase(
                    torch=torch,
                    device=device,
                    enabled=training_config.profile,
                    phase_started_at=phase_started_at,
                )
                phase_timings.backward_seconds += elapsed_seconds
                optimizer.step()
                _, elapsed_seconds = self._finish_profiled_phase(
                    torch=torch,
                    device=device,
                    enabled=training_config.profile,
                    phase_started_at=phase_started_at,
                )
                phase_timings.optimizer_seconds += elapsed_seconds
                phase_timings.instance_count += 1

                detached_loss = loss.detach()
                epoch_loss += detached_loss
                if (
                    training_config.log_every > 0
                    and instance_index % training_config.log_every == 0
                ):
                    logger.info(
                        f"Epoch {epoch}/{training_config.epochs} progress: "
                        f"{instance_index}/{len(prepared_training_data.instances)} "
                        f"instances, "
                        f"latest_loss={detached_loss.item():.6f}"
                    )
                    if training_config.profile:
                        self._log_phase_timings(
                            epoch=epoch,
                            processed_instances=instance_index,
                            timings=phase_timings,
                        )

            final_loss = (epoch_loss / len(prepared_training_data.instances)).item()
            loss_history.append(
                {
                    "epoch": epoch,
                    "average_loss": final_loss,
                }
            )
            logger.info(
                f"Finished epoch {epoch}/{training_config.epochs} "
                f"with average_loss={final_loss:.6f}"
            )
            if training_config.profile and (
                training_config.log_every <= 0
                or len(prepared_training_data.instances) % training_config.log_every != 0
            ):
                self._log_phase_timings(
                    epoch=epoch,
                    processed_instances=len(prepared_training_data.instances),
                    timings=phase_timings,
                )

        model_artifact_path = self._save_model_artifacts(
            model=model,
            built_retriever=effective_retriever,
            question_embedding_model=question_embedding_model,
            relation_embedding_model=relation_embedding_model,
            training_config=training_config,
            selected_device=device,
            final_loss=final_loss,
            loss_history=loss_history,
            trained_instances=len(prepared_training_data.instances),
            training_start_instance=prepared_training_data.training_start_instance,
            training_end_instance=prepared_training_data.training_end_instance,
            continued_run=continued_run,
            model_run_directory=self._create_model_run_directory(
                model_root=cache_root / "models",
                run_name=training_config.run_name,
            ),
            torch=torch,
            embedding_cache_device=prepared_training_data.embedding_cache_device,
            embedding_cache_dtype=prepared_training_data.embedding_cache_dtype,
        )
        logger.info(f"Saved trained GNN answer retriever to {model_artifact_path}")
        model_run_directory = model_artifact_path.parent
        model_run_number = self._extract_run_number(model_run_directory.name)

        return GnnAnswerRetrieverTrainingOutcome(
            final_loss=final_loss,
            loss_history=loss_history,
            trained_instances=len(prepared_training_data.instances),
            training_start_instance=prepared_training_data.training_start_instance,
            training_end_instance=prepared_training_data.training_end_instance,
            model_artifact_path=model_artifact_path,
            model_config_path=model_artifact_path.with_name(self.model_config_filename),
            model_run_directory=model_run_directory,
            model_run_name=model_run_directory.name,
            model_run_number=model_run_number,
            embedding_cache_directory=cache_root / "embeddings",
            selected_device=device,
            embedding_cache_device=prepared_training_data.embedding_cache_device,
            embedding_cache_dtype=prepared_training_data.embedding_cache_dtype,
            dataset_id=effective_retriever.dataset_id,
            entity_embedding_model=effective_retriever.entity_embedding_model,
            question_embedding_model=question_embedding_model,
            relation_embedding_model=relation_embedding_model,
            entity_embedding_dimension=effective_retriever.entity_embedding_dimension,
            question_embedding_dimension=effective_retriever.question_embedding_dimension,
            relation_embedding_dimension=effective_retriever.relation_embedding_dimension,
            hidden_dimension=effective_retriever.hidden_dimension,
            gnn_layer_count=effective_retriever.gnn_layer_count,
            node_classifier=effective_retriever.node_classifier,
            use_edge_mlp=effective_retriever.use_edge_mlp,
            question_aware_classifier=effective_retriever.question_aware_classifier,
            use_reverse_edges=effective_retriever.use_reverse_edges,
            add_layer_normalization=effective_retriever.add_layer_normalization,
            edge_mlp_hidden_dim=effective_retriever.edge_mlp_hidden_dim,
            dropout=effective_retriever.dropout,
            model=model,
            is_fine_tuned_model=continued_run is not None,
            continued_from_model_run_name=(
                continued_run.run_name if continued_run is not None else None
            ),
            continued_from_model_run_number=(
                continued_run.run_number if continued_run is not None else None
            ),
        )

    @staticmethod
    def release_prepared_embeddings(
        prepared_training_data: PreparedGnnTrainingData,
    ) -> None:
        """Release compact embedding matrices after the training step finishes."""
        import torch

        used_cuda = prepared_training_data.embedding_cache_device.startswith("cuda")
        prepared_training_data.node_embeddings = torch.empty(0)
        prepared_training_data.relation_embeddings = torch.empty(0)
        prepared_training_data.question_embeddings = torch.empty(0)
        if used_cuda:
            torch.cuda.empty_cache()

    def _load_continued_run(
        self,
        cache_root: Path,
        configuration: BuiltPipelineConfiguration,
        training_config: GnnAnswerRetrieverTrainingConfig,
        device: str,
    ) -> LoadedGnnAnswerRetrieverRun | None:
        if (
            training_config.continue_from_model_run_name is None
            and training_config.continue_from_model_run_number is None
        ):
            return None

        return self.model_run_service.load_run(
            model_root=cache_root / "models",
            run_name=training_config.continue_from_model_run_name,
            run_number=training_config.continue_from_model_run_number,
            pipeline_configuration=configuration,
            device=device,
        )

    @staticmethod
    def _build_effective_retriever(
        built_retriever: BuiltGnnAnswerRetriever,
        continued_run: LoadedGnnAnswerRetrieverRun | None,
    ) -> BuiltGnnAnswerRetriever:
        if continued_run is None:
            return built_retriever

        return BuiltGnnAnswerRetriever(
            dataset_id=continued_run.config.dataset_id,
            entity_embedding_model=continued_run.config.entity_embedding_model,
            entity_embedding_dimension=continued_run.config.entity_embedding_dimension,
            question_embedding_dimension=(
                continued_run.config.question_embedding_dimension
                or built_retriever.question_embedding_dimension
            ),
            relation_embedding_dimension=(
                continued_run.config.relation_embedding_dimension
                or built_retriever.relation_embedding_dimension
            ),
            hidden_dimension=continued_run.config.resolved_hidden_dimension,
            gnn_layer_count=continued_run.config.resolved_gnn_layer_count,
            node_classifier=continued_run.config.node_classifier,
            use_edge_mlp=continued_run.config.use_edge_mlp,
            question_aware_classifier=continued_run.config.question_aware_classifier,
            use_reverse_edges=continued_run.config.use_reverse_edges,
            add_layer_normalization=continued_run.config.add_layer_normalization,
            edge_mlp_hidden_dim=continued_run.config.resolved_edge_mlp_hidden_dim,
            dropout=continued_run.config.dropout,
            model=continued_run.model,
        )

    @staticmethod
    def _resolve_device(torch, requested_device: str) -> str:
        if requested_device != "auto":
            return requested_device

        if torch.cuda.is_available():
            return "cuda"

        if torch.backends.mps.is_available():
            return "mps"

        return "cpu"

    @staticmethod
    def _gather_embedding_features(
        embedding_matrix: Tensor,
        indices: Tensor,
        torch: ModuleType,
        device: str,
    ) -> Tensor:
        """Gather compact frozen embeddings and move the result to training device."""
        matrix_device = embedding_matrix.device
        resolved_indices = indices.to(device=matrix_device, non_blocking=True)
        features = torch.index_select(embedding_matrix, 0, resolved_indices)
        return features.to(device=device, non_blocking=True)

    def _build_edge_weight_tensor(
        self,
        relation_features,
        question_features,
        torch,
        torch_functional,
        device: str,
    ):
        if relation_features.shape[0] == 0:
            return torch.empty(0, dtype=relation_features.dtype, device=device)
        return torch_functional.cosine_similarity(
            question_features.reshape(1, -1),
            relation_features,
            dim=1,
        )

    @staticmethod
    def _compute_loss(
        logits,
        node_labels,
        torch,
        torch_functional,
    ):
        """Compute class-balanced binary loss without synchronizing the device."""
        positive_count = node_labels.sum()
        negative_count = node_labels.numel() - positive_count
        positive_weight = torch.where(
            positive_count > 0,
            negative_count / positive_count.clamp_min(1),
            torch.ones_like(positive_count),
        )
        return torch_functional.binary_cross_entropy_with_logits(
            logits,
            node_labels,
            pos_weight=positive_weight,
        )

    @staticmethod
    def _start_profiled_phase(
        torch: TorchRuntime,
        device: str,
        enabled: bool,
    ) -> float:
        """Synchronize the selected device and start one measured phase."""
        if not enabled:
            return 0.0
        GnnAnswerRetrieverTrainingService._synchronize_device(torch, device)
        return time.perf_counter()

    @staticmethod
    def _finish_profiled_phase(
        torch: TorchRuntime,
        device: str,
        enabled: bool,
        phase_started_at: float,
    ) -> tuple[float, float]:
        """Finish one measured phase and return its next start and duration."""
        if not enabled:
            return 0.0, 0.0
        GnnAnswerRetrieverTrainingService._synchronize_device(torch, device)
        elapsed_seconds = time.perf_counter() - phase_started_at
        return time.perf_counter(), elapsed_seconds

    @staticmethod
    def _synchronize_device(torch: TorchRuntime, device: str) -> None:
        """Wait for asynchronous accelerator work before collecting timings."""
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()

    @staticmethod
    def _log_phase_timings(
        epoch: int,
        processed_instances: int,
        timings: GnnTrainingPhaseTimings,
    ) -> None:
        """Log cumulative and per-instance training phase timings."""
        if timings.instance_count == 0:
            return
        milliseconds_per_instance = 1000.0 / timings.instance_count
        logger.info(
            f"Training profile epoch={epoch} instances={processed_instances}: "
            f"input_ms={timings.input_preparation_seconds * milliseconds_per_instance:.2f} "
            f"forward_ms={timings.forward_seconds * milliseconds_per_instance:.2f} "
            f"loss_ms={timings.loss_seconds * milliseconds_per_instance:.2f} "
            f"backward_ms={timings.backward_seconds * milliseconds_per_instance:.2f} "
            f"optimizer_ms={timings.optimizer_seconds * milliseconds_per_instance:.2f} "
            f"total_ms={timings.total_seconds * milliseconds_per_instance:.2f}"
        )

    def _save_model_artifacts(
        self,
        model,
        built_retriever: BuiltGnnAnswerRetriever,
        question_embedding_model: str,
        relation_embedding_model: str,
        training_config: GnnAnswerRetrieverTrainingConfig,
        selected_device: str,
        final_loss: float,
        loss_history: list[dict[str, float | int]],
        trained_instances: int,
        training_start_instance: int,
        training_end_instance: int,
        continued_run: LoadedGnnAnswerRetrieverRun | None,
        model_run_directory: Path,
        torch,
        embedding_cache_device: str = "qdrant",
        embedding_cache_dtype: str = "float32",
    ) -> Path:
        model_run_directory.mkdir(parents=True, exist_ok=False)
        model_artifact_path = model_run_directory / self.model_weights_filename
        model_config_path = model_run_directory / self.model_config_filename
        model_run_name = model_run_directory.name
        model_run_number = self._extract_run_number(model_run_name)
        training_payload = training_config.model_dump(
            exclude={
                "run_name",
                "continue_from_model_run_name",
                "continue_from_model_run_number",
            }
        )
        training_payload["device"] = selected_device
        training_payload["gnn_layer_count"] = built_retriever.gnn_layer_count
        training_payload["hidden_dimension"] = built_retriever.hidden_dimension
        training_payload["loss_function"] = "BCEWithLogitsLoss"
        training_payload["use_edge_mlp"] = built_retriever.use_edge_mlp
        training_payload["question_aware_classifier"] = (
            built_retriever.question_aware_classifier
        )
        training_payload["use_reverse_edges"] = built_retriever.use_reverse_edges
        training_payload["add_layer_normalization"] = (
            built_retriever.add_layer_normalization
        )
        training_payload["edge_mlp_hidden_dim"] = built_retriever.edge_mlp_hidden_dim
        training_payload["dropout"] = built_retriever.dropout
        training_payload["embedding_cache_device"] = embedding_cache_device
        training_payload["embedding_cache_dtype"] = embedding_cache_dtype
        is_fine_tuned_model = continued_run is not None
        config_payload = {
            "dataset_id": built_retriever.dataset_id,
            "run_name": model_run_name,
            "run_number": model_run_number,
            "entity_embedding_model": built_retriever.entity_embedding_model,
            "question_embedding_model": question_embedding_model,
            "relation_embedding_model": relation_embedding_model,
            "entity_embedding_dimension": built_retriever.entity_embedding_dimension,
            "question_embedding_dimension": built_retriever.question_embedding_dimension,
            "relation_embedding_dimension": built_retriever.relation_embedding_dimension,
            "hidden_dimension": built_retriever.hidden_dimension,
            "gnn_layer_count": built_retriever.gnn_layer_count,
            "node_classifier": built_retriever.node_classifier,
            "use_edge_mlp": built_retriever.use_edge_mlp,
            "question_aware_classifier": built_retriever.question_aware_classifier,
            "use_reverse_edges": built_retriever.use_reverse_edges,
            "add_layer_normalization": built_retriever.add_layer_normalization,
            "edge_mlp_hidden_dim": built_retriever.edge_mlp_hidden_dim,
            "dropout": built_retriever.dropout,
            "is_fine_tuned_model": is_fine_tuned_model,
            "continued_from_model_run_name": (
                continued_run.run_name if continued_run is not None else None
            ),
            "continued_from_model_run_number": (
                continued_run.run_number if continued_run is not None else None
            ),
            "training_start_instance": training_start_instance,
            "training_end_instance": training_end_instance,
            "trained_instance_range": {
                "start": training_start_instance,
                "end": training_end_instance,
            },
            "training": training_payload,
            "loss_history": loss_history,
            "final_loss": final_loss,
            "trained_instances": trained_instances,
        }
        torch.save(model.state_dict(), model_artifact_path)
        model_config_path.write_text(
            json.dumps(
                make_project_paths_relative(config_payload),
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return model_artifact_path

    def _create_model_run_directory(
        self,
        model_root: Path,
        run_name: str | None,
    ) -> Path:
        model_root.mkdir(parents=True, exist_ok=True)
        run_number = self._next_run_number(model_root)
        run_label = self._resolve_run_label(run_name)
        return model_root / f"{run_number}_{run_label}"

    @classmethod
    def _next_run_number(cls, model_root: Path) -> int:
        existing_run_numbers = [
            cls._extract_run_number(path.name)
            for path in model_root.iterdir()
            if path.is_dir() and cls._extract_run_number(path.name) > 0
        ]
        return max(existing_run_numbers, default=0) + 1

    @staticmethod
    def _extract_run_number(run_directory_name: str) -> int:
        run_number_match = re.match(r"^(\d+)_", run_directory_name)
        if run_number_match is None:
            return 0

        return int(run_number_match.group(1))

    @classmethod
    def _resolve_run_label(cls, run_name: str | None) -> str:
        if run_name is None or not run_name.strip():
            return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        return cls._sanitize_run_name(run_name)

    @staticmethod
    def _sanitize_run_name(run_name: str) -> str:
        sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", run_name.strip())
        sanitized = sanitized.strip("._-")
        return sanitized or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
