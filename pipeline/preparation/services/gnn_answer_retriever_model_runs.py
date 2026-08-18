"""Services for resolving and loading saved GNN answer-retriever runs."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from helpers.constants import (
    GNN_ANSWER_RETRIEVER_CONFIG_FILENAME,
    GNN_ANSWER_RETRIEVER_WEIGHTS_FILENAME,
)
from helpers.logging_config import get_logger
from pipeline.preparation.helpers.configuration_definitions import (
    GNN_ARCHITECTURES,
    OPENAI_EMBEDDING_MODELS,
)
from pipeline.preparation.helpers.gnn_architecture import (
    architecture_defaults,
    infer_gnn_architecture,
)
from pipeline.preparation.exceptions import GnnAnswerRetrieverModelRunException
from pipeline.preparation.models.interfaces import AnswerRetrieverModel
from pipeline.preparation.steps.configuration_building import BuiltPipelineConfiguration
from pipeline.services import AbstractService

logger = get_logger(__name__)


class SavedGnnAnswerRetrieverTrainingConfig(BaseModel):
    """Training config loaded from a saved model run."""

    epochs: int = 0
    learning_rate: float = 0.0
    weight_decay: float = 0.0
    max_instances: int | None = None
    start_instance: int = 0
    log_every: int = 0
    device: str = "cpu"
    profile: bool = False
    embedding_cache_device: str | None = None
    embedding_cache_dtype: str | None = None
    loss_history: list[dict[str, float | int]] = Field(default_factory=list)
    final_loss: float | None = None
    trained_instances: dict[str, int] | int | None = None
    training_start_instance: int | None = None
    training_end_instance: int | None = None
    trained_instance_range: dict[str, int] | None = None
    gnn_architecture: str | None = None
    gnn_architecture_options: dict[str, Any] = Field(default_factory=dict)
    gnn_layer_count: int | None = None
    hidden_dimension: int | None = None
    loss_function: str | None = None
    use_edge_mlp: bool = False
    question_aware_classifier: bool = False
    use_reverse_edges: bool = False
    add_layer_normalization: bool = False
    edge_mlp_hidden_dim: int | None = None
    dropout: float = 0.0


class SavedGnnAnswerRetrieverConfig(BaseModel):
    """Architecture and training metadata loaded from a saved model run."""

    dataset_id: str
    gnn_architecture: str | None = None
    gnn_architecture_options: dict[str, Any] = Field(default_factory=dict)
    embedding_model: str | None = None
    embedding_dimension: int | None = None
    # Legacy per-resource fields are accepted only when loading old artifacts.
    entity_embedding_model: str | None = None
    question_embedding_model: str | None = None
    relation_embedding_model: str | None = None
    entity_embedding_dimension: int | None = None
    question_embedding_dimension: int | None = None
    relation_embedding_dimension: int | None = None
    hidden_dimension: int | None = None
    gnn_layer_count: int | None = None
    node_classifier: str | None = None
    use_edge_mlp: bool = False
    question_aware_classifier: bool = False
    use_reverse_edges: bool = False
    add_layer_normalization: bool = False
    edge_mlp_hidden_dim: int | None = None
    dropout: float = 0.0
    training: SavedGnnAnswerRetrieverTrainingConfig = Field(
        default_factory=SavedGnnAnswerRetrieverTrainingConfig
    )
    run_name: str | None = None
    run_number: int | None = None
    is_fine_tuned_model: bool = False
    continued_from_model_run_name: str | None = None
    continued_from_model_run_number: int | None = None
    training_start_instance: int = 0
    training_end_instance: int | None = None
    trained_instance_range: dict[str, int] | None = None
    loss_history: list[dict[str, float | int]] = Field(default_factory=list)
    final_loss: float = 0.0
    trained_instances: int = 0

    @property
    def resolved_embedding_model(self) -> str:
        """Return the unified embedding model, with legacy fallback."""
        return (
            self.embedding_model
            or self.entity_embedding_model
            or self.question_embedding_model
            or self.relation_embedding_model
            or "text-embedding-3-small"
        )

    @property
    def resolved_embedding_dimension(self) -> int:
        """Return the unified embedding dimension, with legacy fallback."""
        if self.embedding_dimension is not None:
            return self.embedding_dimension
        for dimension in (
            self.entity_embedding_dimension,
            self.question_embedding_dimension,
            self.relation_embedding_dimension,
        ):
            if dimension is not None:
                return dimension
        return OPENAI_EMBEDDING_MODELS[self.resolved_embedding_model].dimensions

    @property
    def resolved_loss_history(self) -> list[dict[str, float | int]]:
        return self.loss_history or self.training.loss_history

    @property
    def resolved_training_instance_range(self) -> dict[str, int]:
        """Return the unified training range/count, including legacy fallback."""
        training_instances = self.training.trained_instances
        if isinstance(training_instances, dict):
            start = int(training_instances.get("start", 0))
            count = int(training_instances.get("count", 0))
            end = int(training_instances.get("end", 0))
            return {"start": start, "end": end or start + count, "count": count}
        if isinstance(training_instances, int):
            return {"start": 0, "end": training_instances, "count": training_instances}
        if self.trained_instance_range is not None:
            start = int(self.trained_instance_range.get("start", 0))
            end = int(self.trained_instance_range.get("end", 0))
            return {
                "start": start,
                "end": end,
                "count": int(self.trained_instances or max(end - start, 0)),
            }
        start = int(self.training_start_instance or 0)
        end = int(self.training_end_instance or 0)
        count = int(self.trained_instances or max(end - start, 0))
        return {"start": start, "end": end or start + count, "count": count}

    @property
    def resolved_final_loss(self) -> float:
        return self.final_loss or self.training.final_loss or 0.0

    @property
    def resolved_trained_instances(self) -> int:
        training_instances = self.training.trained_instances
        if isinstance(training_instances, dict):
            return int(training_instances.get("count", 0))
        if isinstance(training_instances, int):
            return training_instances
        return self.trained_instances or 0

    @property
    def resolved_hidden_dimension(self) -> int:
        return (
            self.gnn_architecture_options.get("gnn_hidden_dimension")
            or self.hidden_dimension
            or self.training.hidden_dimension
            or 0
        )

    @property
    def resolved_gnn_layer_count(self) -> int:
        return (
            self.gnn_architecture_options.get("gnn_layer_count")
            or self.gnn_layer_count
            or self.training.gnn_layer_count
            or 0
        )

    @property
    def resolved_gnn_architecture(self) -> str:
        return infer_gnn_architecture(self.model_dump())

    @property
    def resolved_gnn_architecture_options(self) -> dict[str, Any]:
        defaults = architecture_defaults(self.resolved_gnn_architecture)
        defaults.pop("gnn_architecture", None)
        persisted = {
            **self.training.gnn_architecture_options,
            **self.gnn_architecture_options,
        }
        legacy_values = {
            "gnn_layer_count": self.resolved_gnn_layer_count,
            "gnn_hidden_dimension": self.resolved_hidden_dimension,
            "node_classifier": self.node_classifier,
            "dropout": self.dropout,
            "use_edge_mlp": self.use_edge_mlp,
            "question_aware_classifier": self.question_aware_classifier,
            "use_reverse_edges": self.use_reverse_edges,
            "add_layer_normalization": self.add_layer_normalization,
            "edge_mlp_hidden_dim": self.resolved_edge_mlp_hidden_dim,
        }
        supported = GNN_ARCHITECTURES[self.resolved_gnn_architecture].option_map
        resolved = {
            option_id: persisted.get(option_id, legacy_values.get(option_id, default))
            for option_id, default in defaults.items()
            if option_id in supported
        }
        if resolved.get("use_edge_mlp") is False:
            resolved["edge_mlp_hidden_dim"] = None
        return resolved

    @property
    def resolved_edge_mlp_hidden_dim(self) -> int:
        return (
            self.edge_mlp_hidden_dim
            or self.training.edge_mlp_hidden_dim
            or self.resolved_hidden_dimension
        )


class SavedGnnAnswerRetrieverRun(BaseModel):
    """Resolved saved model run on disk."""

    run_directory: Path = Field(..., description="Saved model run directory.")
    run_name: str = Field(..., description="Saved run folder name.")
    run_number: int = Field(..., description="Saved run numeric prefix.")
    weights_path: Path = Field(..., description="Saved state-dict path.")
    config_path: Path = Field(..., description="Saved model config path.")
    config: SavedGnnAnswerRetrieverConfig = Field(..., description="Saved config.")


class LoadedGnnAnswerRetrieverRun(BaseModel):
    """Loaded saved model run ready for evaluation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    run_directory: Path = Field(..., description="Saved model run directory.")
    run_name: str = Field(..., description="Saved run folder name.")
    run_number: int = Field(..., description="Saved run numeric prefix.")
    weights_path: Path = Field(..., description="Saved state-dict path.")
    config_path: Path = Field(..., description="Saved model config path.")
    config: SavedGnnAnswerRetrieverConfig = Field(..., description="Saved config.")
    model: AnswerRetrieverModel = Field(..., description="Loaded PyTorch model.")
    question_embedding_model: str = Field(..., description="Question embedding model id.")
    relation_embedding_model: str = Field(..., description="Relation embedding model id.")


class GnnAnswerRetrieverModelRunService(AbstractService):
    """Resolve and load saved GNN answer-retriever model runs."""

    weights_filename = GNN_ANSWER_RETRIEVER_WEIGHTS_FILENAME
    config_filename = GNN_ANSWER_RETRIEVER_CONFIG_FILENAME
    legacy_config_filename = "gnn_answer_retriever_config.json"

    def resolve_run(
        self,
        model_root: Path,
        run_name: str | None,
        run_number: int | None,
    ) -> SavedGnnAnswerRetrieverRun:
        """Resolve the requested saved run, falling back to the latest run."""
        candidate_directories = self._list_numbered_run_directories(model_root)
        if not candidate_directories:
            raise GnnAnswerRetrieverModelRunException(
                f"No saved GNN answer-retriever runs found under {model_root}."
            )

        if run_name is not None and run_name.strip():
            selected_directory = self._resolve_by_run_name(
                candidate_directories,
                run_name.strip(),
            )
        elif run_number is not None:
            selected_directory = self._resolve_by_run_number(
                candidate_directories,
                run_number,
            )
        else:
            selected_directory = max(
                candidate_directories,
                key=lambda path: self._extract_run_number(path.name),
            )

        return self._build_saved_run(selected_directory)

    def load_run(
        self,
        model_root: Path,
        run_name: str | None,
        run_number: int | None,
        pipeline_configuration: BuiltPipelineConfiguration,
        device: str,
    ) -> LoadedGnnAnswerRetrieverRun:
        """Resolve a saved run, rebuild its model, and load its weights."""
        import torch

        saved_run = self.resolve_run(
            model_root=model_root,
            run_name=run_name,
            run_number=run_number,
        )
        from pipeline.preparation.models.gnn_answer_retriever import build_gnn_answer_retriever

        model = build_gnn_answer_retriever(
            gnn_architecture=saved_run.config.resolved_gnn_architecture,
            architecture_options=saved_run.config.resolved_gnn_architecture_options,
            entity_embedding_dimension=saved_run.config.resolved_embedding_dimension,
            question_embedding_dimension=self._embedding_dimension(
                saved_run.config.question_embedding_model,
                saved_run.config.question_embedding_dimension,
                pipeline_configuration.embedding_model,
            ),
            relation_embedding_dimension=self._embedding_dimension(
                saved_run.config.relation_embedding_model,
                saved_run.config.relation_embedding_dimension,
                pipeline_configuration.embedding_model,
            ),
        )
        try:
            state_dict = torch.load(saved_run.weights_path, map_location=device)
            model.load_state_dict(state_dict)
        except RuntimeError as error:
            raise GnnAnswerRetrieverModelRunException(
                f"Could not load saved GNN answer-retriever weights from "
                f"{saved_run.weights_path}: {error}"
            ) from error
        model.to(device)
        model.eval()

        question_embedding_model = saved_run.config.resolved_embedding_model
        relation_embedding_model = saved_run.config.resolved_embedding_model
        if (
            saved_run.config.question_embedding_model is None
            and saved_run.config.embedding_model is None
        ):
            logger.warning(
                f"Model run {saved_run.run_name} does not store a question embedding "
                f"model id. Falling back to current configuration value "
                f"{question_embedding_model}."
            )
        if (
            saved_run.config.relation_embedding_model is None
            and saved_run.config.embedding_model is None
        ):
            logger.warning(
                f"Model run {saved_run.run_name} does not store a relation embedding "
                f"model id. Falling back to current configuration value "
                f"{relation_embedding_model}."
            )

        logger.info(
            f"Loaded GNN answer-retriever run {saved_run.run_name} "
            f"from {saved_run.run_directory}"
        )
        return LoadedGnnAnswerRetrieverRun(
            run_directory=saved_run.run_directory,
            run_name=saved_run.run_name,
            run_number=saved_run.run_number,
            weights_path=saved_run.weights_path,
            config_path=saved_run.config_path,
            config=saved_run.config,
            model=model,
            question_embedding_model=question_embedding_model,
            relation_embedding_model=relation_embedding_model,
        )

    @staticmethod
    def _embedding_dimension(
        saved_model_id: str | None,
        saved_dimension: int | None,
        fallback_model_id: str | None,
    ) -> int:
        if saved_dimension is not None:
            return saved_dimension
        model_id = saved_model_id or fallback_model_id or "text-embedding-3-small"
        return OPENAI_EMBEDDING_MODELS[model_id].dimensions

    @classmethod
    def _list_numbered_run_directories(cls, model_root: Path) -> list[Path]:
        if not model_root.exists():
            return []

        return [
            path
            for path in model_root.iterdir()
            if path.is_dir() and cls._extract_run_number(path.name) > 0
        ]

    @classmethod
    def _resolve_by_run_name(
        cls,
        candidate_directories: list[Path],
        run_name: str,
    ) -> Path:
        matching_directories = [
            path
            for path in candidate_directories
            if path.name == run_name or cls._extract_run_label(path.name) == run_name
        ]
        if not matching_directories:
            raise GnnAnswerRetrieverModelRunException(
                f"No saved GNN answer-retriever run matched name '{run_name}'."
            )

        return max(
            matching_directories,
            key=lambda path: cls._extract_run_number(path.name),
        )

    @classmethod
    def _resolve_by_run_number(
        cls,
        candidate_directories: list[Path],
        run_number: int,
    ) -> Path:
        matching_directories = [
            path
            for path in candidate_directories
            if cls._extract_run_number(path.name) == run_number
        ]
        if not matching_directories:
            raise GnnAnswerRetrieverModelRunException(
                f"No saved GNN answer-retriever run matched number {run_number}."
            )

        return matching_directories[0]

    def _build_saved_run(self, run_directory: Path) -> SavedGnnAnswerRetrieverRun:
        weights_path = run_directory / self.weights_filename
        config_path = run_directory / self.config_filename
        if not config_path.exists():
            legacy_config_path = run_directory / self.legacy_config_filename
            if legacy_config_path.exists():
                config_path = legacy_config_path
        if not weights_path.exists():
            raise GnnAnswerRetrieverModelRunException(
                f"Selected model run is missing weights: {weights_path}"
            )
        if not config_path.exists():
            raise GnnAnswerRetrieverModelRunException(
                f"Selected model run is missing config: {config_path}"
            )

        try:
            config = SavedGnnAnswerRetrieverConfig.model_validate_json(
                config_path.read_text(encoding="utf-8")
            )
        except (json.JSONDecodeError, ValueError) as error:
            raise GnnAnswerRetrieverModelRunException(
                f"Saved model config is invalid: {config_path}"
            ) from error

        return SavedGnnAnswerRetrieverRun(
            run_directory=run_directory,
            run_name=run_directory.name,
            run_number=self._extract_run_number(run_directory.name),
            weights_path=weights_path,
            config_path=config_path,
            config=config,
        )

    @staticmethod
    def _extract_run_number(run_directory_name: str) -> int:
        run_number_match = re.match(r"^(\d+)_", run_directory_name)
        if run_number_match is None:
            return 0

        return int(run_number_match.group(1))

    @staticmethod
    def _extract_run_label(run_directory_name: str) -> str:
        run_label_match = re.match(r"^\d+_(.+)$", run_directory_name)
        if run_label_match is None:
            return run_directory_name

        return run_label_match.group(1)
