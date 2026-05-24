"""Services for resolving and loading saved GNN answer-retriever runs."""

from __future__ import annotations

import json
import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from helpers.constants import (
    GNN_ANSWER_RETRIEVER_CONFIG_FILENAME,
    GNN_ANSWER_RETRIEVER_WEIGHTS_FILENAME,
)
from helpers.logging_config import get_logger
from pipeline.exceptions import GnnAnswerRetrieverModelRunException
from pipeline.preparation.models.interfaces import AnswerRetrieverModel
from pipeline.preparation.steps.configuration_building import BuiltPipelineConfiguration
from pipeline.services.abstract import AbstractService

logger = get_logger(__name__)


class SavedGnnAnswerRetrieverTrainingConfig(BaseModel):
    """Training config loaded from a saved model run."""

    epochs: int
    learning_rate: float
    weight_decay: float
    max_instances: int | None = None
    log_every: int
    device: str
    gnn_layer_count: int | None = None
    hidden_dimension: int | None = None
    loss_function: str | None = None


class SavedGnnAnswerRetrieverConfig(BaseModel):
    """Architecture and training metadata loaded from a saved model run."""

    dataset_id: str
    entity_embedding_model: str
    question_embedding_model: str | None = None
    relation_embedding_model: str | None = None
    entity_embedding_dimension: int
    hidden_dimension: int | None = None
    gnn_layer_count: int | None = None
    node_classifier: str
    training: SavedGnnAnswerRetrieverTrainingConfig
    run_name: str | None = None
    run_number: int | None = None
    loss_history: list[dict[str, float | int]] = Field(default_factory=list)
    final_loss: float
    trained_instances: int

    @property
    def resolved_hidden_dimension(self) -> int:
        return self.hidden_dimension or self.training.hidden_dimension or 0

    @property
    def resolved_gnn_layer_count(self) -> int:
        return self.gnn_layer_count or self.training.gnn_layer_count or 0


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
        from pipeline.preparation.models.gnn_answer_retriever import GnnAnswerRetriever

        model = GnnAnswerRetriever(
            entity_embedding_dimension=saved_run.config.entity_embedding_dimension,
            hidden_dimension=saved_run.config.resolved_hidden_dimension,
            gnn_layer_count=saved_run.config.resolved_gnn_layer_count,
            node_classifier=saved_run.config.node_classifier,
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

        question_embedding_model = (
            saved_run.config.question_embedding_model
            or pipeline_configuration.question_embedding_model
        )
        relation_embedding_model = (
            saved_run.config.relation_embedding_model
            or pipeline_configuration.relation_embedding_model
        )
        if saved_run.config.question_embedding_model is None:
            logger.warning(
                f"Model run {saved_run.run_name} does not store a question embedding "
                f"model id. Falling back to current configuration value "
                f"{question_embedding_model}."
            )
        if saved_run.config.relation_embedding_model is None:
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
