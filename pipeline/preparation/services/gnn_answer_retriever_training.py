"""Training service for the GNN answer retriever."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING
from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field

from helpers.path_serialization import make_project_paths_relative
from helpers.constants import (
    DEFAULT_TRAINING_DEVICE,
    DEFAULT_TRAINING_EPOCHS,
    DEFAULT_TRAINING_LEARNING_RATE,
    DEFAULT_TRAINING_LOG_EVERY,
    DEFAULT_TRAINING_WEIGHT_DECAY,
    GNN_ANSWER_RETRIEVER_CONFIG_FILENAME,
    GNN_ANSWER_RETRIEVER_WEIGHTS_FILENAME,
)
from helpers.logging_config import get_logger
from pipeline.preparation.exceptions import GnnAnswerRetrieverTrainingException
from pipeline.preparation.models.webqsp_local_graph import (
    PreparedWebQSPGraphDataset,
    WebQSPProcessedInstance,
)
from pipeline.preparation.steps.gnn_model_building import BuiltGnnAnswerRetriever
from pipeline.services import AbstractService
from pipeline.preparation.services.embedding_cache import (
    TextEmbeddingCache,
    WebQSPEmbeddingCacheService,
)
from pipeline.preparation.services.gnn_answer_retriever_model_runs import (
    GnnAnswerRetrieverModelRunService,
    LoadedGnnAnswerRetrieverRun,
)

if TYPE_CHECKING:
    from pipeline.preparation.steps.configuration_building import BuiltPipelineConfiguration

logger = get_logger(__name__)


class GnnAnswerRetrieverTrainingConfig(BaseModel):
    """Runtime training settings for the answer retriever."""

    epochs: int = Field(default=DEFAULT_TRAINING_EPOCHS)
    learning_rate: float = Field(default=DEFAULT_TRAINING_LEARNING_RATE)
    weight_decay: float = Field(default=DEFAULT_TRAINING_WEIGHT_DECAY)
    max_instances: int | None = Field(default=None)
    start_instance: int = Field(default=0)
    log_every: int = Field(default=DEFAULT_TRAINING_LOG_EVERY)
    device: str = Field(default=DEFAULT_TRAINING_DEVICE)
    run_name: str | None = Field(default=None)
    continue_from_model_run_name: str | None = Field(default=None)
    continue_from_model_run_number: int | None = Field(default=None)


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
        embedding_cache_service: WebQSPEmbeddingCacheService | None = None,
        model_run_service: GnnAnswerRetrieverModelRunService | None = None,
    ):
        self.embedding_cache_service = (
            embedding_cache_service or WebQSPEmbeddingCacheService()
        )
        self.model_run_service = model_run_service or GnnAnswerRetrieverModelRunService()

    def train(
        self,
        built_retriever: BuiltGnnAnswerRetriever,
        prepared_dataset: PreparedWebQSPGraphDataset,
        configuration: BuiltPipelineConfiguration,
        training_config: GnnAnswerRetrieverTrainingConfig,
    ) -> GnnAnswerRetrieverTrainingOutcome:
        """Train the built answer retriever on prepared WebQSP instances."""
        import torch
        import torch.nn.functional as torch_functional
        from torch import nn

        (
            train_instances,
            training_start_instance,
            training_end_instance,
        ) = self._select_train_instances(
            prepared_dataset=prepared_dataset,
            start_instance=training_config.start_instance,
            max_instances=training_config.max_instances,
        )

        device = self._resolve_device(torch, training_config.device)
        logger.info(f"Starting GNN answer retriever training on device={device}")

        cache_root = prepared_dataset.cache_directory.parent
        continued_run = self._load_continued_run(
            cache_root=cache_root,
            configuration=configuration,
            training_config=training_config,
            device=device,
        )
        effective_retriever = self._build_effective_retriever(
            built_retriever=built_retriever,
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
        if effective_retriever.dataset_id != prepared_dataset.dataset_id:
            raise GnnAnswerRetrieverTrainingException(
                f"Cannot continue training model run from dataset "
                f"{effective_retriever.dataset_id} on dataset {prepared_dataset.dataset_id}."
            )

        logger.info(
            f"Selected GNN training slice: start={training_start_instance} "
            f"end={training_end_instance} instances={len(train_instances)}"
        )
        if continued_run is not None:
            logger.info(
                f"Continuing GNN answer retriever from run={continued_run.run_name} "
                f"run_number={continued_run.run_number} "
                f"additional_epochs={training_config.epochs}"
            )

        node_cache = self.embedding_cache_service.load_node_cache(
            cache_root=cache_root,
            model_id=effective_retriever.entity_embedding_model,
            vocabulary=prepared_dataset.vocabulary_store.nodes,
            dataset_id=prepared_dataset.dataset_id,
        )
        relation_cache = self.embedding_cache_service.load_relation_cache(
            cache_root=cache_root,
            model_id=relation_embedding_model,
            vocabulary=prepared_dataset.vocabulary_store.relations,
            dataset_id=prepared_dataset.dataset_id,
        )
        question_cache = self.embedding_cache_service.load_question_cache(
            cache_root=cache_root,
            model_id=question_embedding_model,
            vocabulary=prepared_dataset.vocabulary_store.questions,
            dataset_id=prepared_dataset.dataset_id,
        )
        logger.info(
            f"Loaded Qdrant embedding cache handles: "
            f"nodes_collection={node_cache.collection_name} "
            f"relations_collection={relation_cache.collection_name} "
            f"questions_collection={question_cache.collection_name}"
        )

        self._populate_embedding_caches(
            train_instances=train_instances,
            node_cache=node_cache,
            relation_cache=relation_cache,
            question_cache=question_cache,
        )
        logger.info(
            f"Qdrant embedding caches ready: "
            f"node_vocab={len(node_cache.vocabulary)} "
            f"relation_vocab={len(relation_cache.vocabulary)} "
            f"question_vocab={len(question_cache.vocabulary)}"
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
                f"over {len(train_instances)} WebQSP instances"
            )
            epoch_loss = 0.0
            for instance_index, instance in enumerate(train_instances, start=1):
                optimizer.zero_grad()

                entity_features = self._build_entity_feature_tensor(
                    instance=instance,
                    node_cache=node_cache,
                    torch=torch,
                    device=device,
                )
                question_features = self._build_question_feature_tensor(
                    instance=instance,
                    question_cache=question_cache,
                    torch=torch,
                    device=device,
                )
                relation_features = self._build_relation_feature_tensor(
                    instance=instance,
                    relation_cache=relation_cache,
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
                edge_index = instance.edge_index.to(device)
                node_labels = instance.node_labels.to(device)

                logits = model(
                    entity_features=entity_features,
                    edge_index=edge_index,
                    edge_weight=edge_weight,
                    question_features=question_features,
                    relation_features=relation_features,
                )
                loss = self._compute_loss(
                    logits=logits,
                    node_labels=node_labels,
                    torch=torch,
                    loss_module=nn,
                    device=device,
                )
                loss.backward()
                optimizer.step()

                epoch_loss += float(loss.detach().cpu().item())
                if (
                    training_config.log_every > 0
                    and instance_index % training_config.log_every == 0
                ):
                    logger.info(
                        f"Epoch {epoch}/{training_config.epochs} progress: "
                        f"{instance_index}/{len(train_instances)} instances, "
                        f"latest_loss={float(loss.detach().cpu().item()):.6f}"
                    )

            final_loss = epoch_loss / len(train_instances)
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

        model_artifact_path = self._save_model_artifacts(
            model=model,
            built_retriever=effective_retriever,
            question_embedding_model=question_embedding_model,
            relation_embedding_model=relation_embedding_model,
            training_config=training_config,
            selected_device=device,
            final_loss=final_loss,
            loss_history=loss_history,
            trained_instances=len(train_instances),
            training_start_instance=training_start_instance,
            training_end_instance=training_end_instance,
            continued_run=continued_run,
            model_run_directory=self._create_model_run_directory(
                model_root=cache_root / "models",
                run_name=training_config.run_name,
            ),
            torch=torch,
        )
        logger.info(f"Saved trained GNN answer retriever to {model_artifact_path}")
        model_run_directory = model_artifact_path.parent
        model_run_number = self._extract_run_number(model_run_directory.name)

        return GnnAnswerRetrieverTrainingOutcome(
            final_loss=final_loss,
            loss_history=loss_history,
            trained_instances=len(train_instances),
            training_start_instance=training_start_instance,
            training_end_instance=training_end_instance,
            model_artifact_path=model_artifact_path,
            model_config_path=model_artifact_path.with_name(self.model_config_filename),
            model_run_directory=model_run_directory,
            model_run_name=model_run_directory.name,
            model_run_number=model_run_number,
            embedding_cache_directory=cache_root / "embeddings",
            selected_device=device,
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
    def _select_train_instances(
        prepared_dataset: PreparedWebQSPGraphDataset,
        start_instance: int,
        max_instances: int | None,
    ) -> tuple[list[WebQSPProcessedInstance], int, int]:
        if start_instance < 0:
            raise GnnAnswerRetrieverTrainingException(
                "training_start_instance must be greater than or equal to 0."
            )

        available_instances = prepared_dataset.train_instances
        end_instance = (
            len(available_instances)
            if max_instances is None
            else start_instance + max_instances
        )
        selected_instances = available_instances[start_instance:end_instance]
        if not selected_instances:
            raise GnnAnswerRetrieverTrainingException(
                f"GNN answer retriever training selected no instances: "
                f"start={start_instance} end={end_instance} "
                f"available={len(available_instances)}."
            )

        return (
            selected_instances,
            start_instance,
            start_instance + len(selected_instances),
        )

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

    def _populate_embedding_caches(
        self,
        train_instances: list[WebQSPProcessedInstance],
        node_cache: TextEmbeddingCache,
        relation_cache: TextEmbeddingCache,
        question_cache: TextEmbeddingCache,
    ) -> None:
        node_texts: list[str] = []
        relation_texts: list[str] = []
        question_texts: list[str] = []
        for instance in train_instances:
            node_texts.extend(instance.nodes)
            relation_texts.extend(instance.edge_relations)
            question_texts.append(instance.question)

        self.embedding_cache_service.ensure_embeddings(node_cache, node_texts)
        self.embedding_cache_service.ensure_embeddings(
            relation_cache,
            relation_texts,
            preprocess=True,
        )
        self.embedding_cache_service.ensure_embeddings(question_cache, question_texts)

    def _build_entity_feature_tensor(
        self,
        instance: WebQSPProcessedInstance,
        node_cache: TextEmbeddingCache,
        torch,
        device: str,
    ):
        return torch.tensor(
            self.embedding_cache_service.embeddings_for_texts(
                cache=node_cache,
                texts=instance.nodes,
            ),
            dtype=torch.float,
            device=device,
        )

    def _build_question_feature_tensor(
        self,
        instance: WebQSPProcessedInstance,
        question_cache: TextEmbeddingCache,
        torch,
        device: str,
    ):
        return torch.tensor(
            self.embedding_cache_service.embeddings_for_texts(
                cache=question_cache,
                texts=[instance.question],
            )[0],
            dtype=torch.float,
            device=device,
        )

    def _build_relation_feature_tensor(
        self,
        instance: WebQSPProcessedInstance,
        relation_cache: TextEmbeddingCache,
        torch,
        device: str,
    ):
        if not instance.edge_relations:
            return torch.empty(
                (0, relation_cache.vector_size),
                dtype=torch.float,
                device=device,
            )
        return torch.tensor(
            self.embedding_cache_service.embeddings_for_texts(
                cache=relation_cache,
                texts=instance.edge_relations,
            ),
            dtype=torch.float,
            device=device,
        )

    def _build_edge_weight_tensor(
        self,
        relation_features,
        question_features,
        torch,
        torch_functional,
        device: str,
    ):
        if relation_features.shape[0] == 0:
            return torch.empty(0, dtype=torch.float, device=device)
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
        loss_module,
        device: str,
    ):
        positive_count = node_labels.sum()
        positive_value = float(positive_count.item())
        if positive_value <= 0:
            loss_function = loss_module.BCEWithLogitsLoss()
            return loss_function(logits, node_labels)

        negative_count = float(node_labels.numel()) - positive_value
        positive_weight = torch.tensor(
            [negative_count / positive_value],
            dtype=torch.float,
            device=device,
        )
        loss_function = loss_module.BCEWithLogitsLoss(pos_weight=positive_weight)
        return loss_function(logits, node_labels)

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
