"""Training service for the GNN answer retriever."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING
from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field

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
from pipeline.exceptions import GnnAnswerRetrieverTrainingException
from pipeline.preparation.models.webqsp_local_graph import (
    PreparedWebQSPGraphDataset,
    WebQSPProcessedInstance,
)
from pipeline.preparation.steps.gnn_model_building import BuiltGnnAnswerRetriever
from pipeline.services.abstract import AbstractService
from pipeline.services.embedding_cache import (
    TextEmbeddingCache,
    WebQSPEmbeddingCacheService,
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
    log_every: int = Field(default=DEFAULT_TRAINING_LOG_EVERY)
    device: str = Field(default=DEFAULT_TRAINING_DEVICE)
    run_name: str | None = Field(default=None)


class GnnAnswerRetrieverTrainingOutcome(BaseModel):
    """Result produced by the training service."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    final_loss: float = Field(..., description="Final average epoch loss.")
    loss_history: list[dict[str, float | int]] = Field(
        default_factory=list,
        description="Average loss per training epoch.",
    )
    trained_instances: int = Field(..., description="Number of training instances used.")
    model_artifact_path: Path = Field(..., description="Saved model weights path.")
    model_config_path: Path = Field(..., description="Saved model config path.")
    model_run_directory: Path = Field(..., description="Versioned training run directory.")
    model_run_name: str = Field(..., description="Resolved training run folder name.")
    model_run_number: int = Field(..., description="Incremental training run number.")
    embedding_cache_directory: Path = Field(..., description="Embedding cache root.")
    selected_device: str = Field(..., description="Resolved PyTorch training device.")


class GnnAnswerRetrieverTrainingService(AbstractService):
    """Train and persist the GNN answer retriever."""

    model_weights_filename = GNN_ANSWER_RETRIEVER_WEIGHTS_FILENAME
    model_config_filename = GNN_ANSWER_RETRIEVER_CONFIG_FILENAME

    def __init__(
        self,
        embedding_cache_service: WebQSPEmbeddingCacheService | None = None,
    ):
        self.embedding_cache_service = (
            embedding_cache_service or WebQSPEmbeddingCacheService()
        )

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

        train_instances = self._select_train_instances(
            prepared_dataset=prepared_dataset,
            max_instances=training_config.max_instances,
        )
        if not train_instances:
            raise GnnAnswerRetrieverTrainingException(
                "GNN answer retriever training requires at least one training instance."
            )

        device = self._resolve_device(torch, training_config.device)
        logger.info(f"Starting GNN answer retriever training on device={device}")

        cache_root = prepared_dataset.cache_directory.parent
        node_cache = self.embedding_cache_service.load_node_cache(
            cache_root=cache_root,
            model_id=configuration.entity_embedding_model,
            vocabulary=prepared_dataset.vocabulary_store.nodes,
        )
        relation_cache = self.embedding_cache_service.load_relation_cache(
            cache_root=cache_root,
            model_id=configuration.relation_embedding_model,
            vocabulary=prepared_dataset.vocabulary_store.relations,
        )
        question_cache = self.embedding_cache_service.load_question_cache(
            cache_root=cache_root,
            model_id=configuration.question_embedding_model,
            vocabulary=prepared_dataset.vocabulary_store.questions,
        )
        logger.info(
            f"Loaded embedding caches: nodes={len(node_cache.embeddings)} "
            f"relations={len(relation_cache.embeddings)} "
            f"questions={len(question_cache.embeddings)}"
        )

        self._populate_embedding_caches(
            train_instances=train_instances,
            node_cache=node_cache,
            relation_cache=relation_cache,
            question_cache=question_cache,
        )
        self.embedding_cache_service.save_cache(node_cache)
        self.embedding_cache_service.save_cache(relation_cache)
        self.embedding_cache_service.save_cache(question_cache)
        logger.info(
            f"Saved embedding caches: nodes={len(node_cache.embeddings)} "
            f"relations={len(relation_cache.embeddings)} "
            f"questions={len(question_cache.embeddings)}"
        )

        model = built_retriever.model
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
                edge_weight = self._build_edge_weight_tensor(
                    instance=instance,
                    question_cache=question_cache,
                    relation_cache=relation_cache,
                    torch=torch,
                    torch_functional=torch_functional,
                    device=device,
                )
                edge_index = instance.edge_index.to(device)
                node_labels = instance.node_labels.to(device)

                logits = model(
                    entity_features=entity_features,
                    edge_index=edge_index,
                    edge_weight=edge_weight,
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
            built_retriever=built_retriever,
            configuration=configuration,
            training_config=training_config,
            selected_device=device,
            final_loss=final_loss,
            loss_history=loss_history,
            trained_instances=len(train_instances),
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
            model_artifact_path=model_artifact_path,
            model_config_path=model_artifact_path.with_name(self.model_config_filename),
            model_run_directory=model_run_directory,
            model_run_name=model_run_directory.name,
            model_run_number=model_run_number,
            embedding_cache_directory=cache_root / "embeddings",
            selected_device=device,
        )

    @staticmethod
    def _select_train_instances(
        prepared_dataset: PreparedWebQSPGraphDataset,
        max_instances: int | None,
    ) -> list[WebQSPProcessedInstance]:
        if max_instances is None:
            return prepared_dataset.train_instances

        return prepared_dataset.train_instances[:max_instances]

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
            [
                self.embedding_cache_service.embedding_for_text(node_cache, node)
                for node in instance.nodes
            ],
            dtype=torch.float,
            device=device,
        )

    def _build_edge_weight_tensor(
        self,
        instance: WebQSPProcessedInstance,
        question_cache: TextEmbeddingCache,
        relation_cache: TextEmbeddingCache,
        torch,
        torch_functional,
        device: str,
    ):
        if not instance.edge_relations:
            return torch.empty(0, dtype=torch.float, device=device)

        question_embedding = torch.tensor(
            self.embedding_cache_service.embedding_for_text(
                question_cache,
                instance.question,
            ),
            dtype=torch.float,
            device=device,
        )
        relation_embeddings = torch.tensor(
            [
                self.embedding_cache_service.embedding_for_text(
                    relation_cache,
                    relation,
                )
                for relation in instance.edge_relations
            ],
            dtype=torch.float,
            device=device,
        )
        return torch_functional.cosine_similarity(
            question_embedding.reshape(1, -1),
            relation_embeddings,
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
        configuration: BuiltPipelineConfiguration,
        training_config: GnnAnswerRetrieverTrainingConfig,
        selected_device: str,
        final_loss: float,
        loss_history: list[dict[str, float | int]],
        trained_instances: int,
        model_run_directory: Path,
        torch,
    ) -> Path:
        model_run_directory.mkdir(parents=True, exist_ok=False)
        model_artifact_path = model_run_directory / self.model_weights_filename
        model_config_path = model_run_directory / self.model_config_filename
        model_run_name = model_run_directory.name
        model_run_number = self._extract_run_number(model_run_name)
        training_payload = training_config.model_dump(exclude={"run_name"})
        training_payload["device"] = selected_device
        training_payload["gnn_layer_count"] = built_retriever.gnn_layer_count
        training_payload["hidden_dimension"] = built_retriever.hidden_dimension
        training_payload["loss_function"] = "BCEWithLogitsLoss"
        torch.save(model.state_dict(), model_artifact_path)
        model_config_path.write_text(
            json.dumps(
                {
                    "dataset_id": built_retriever.dataset_id,
                    "run_name": model_run_name,
                    "run_number": model_run_number,
                    "entity_embedding_model": built_retriever.entity_embedding_model,
                    "question_embedding_model": configuration.question_embedding_model,
                    "relation_embedding_model": configuration.relation_embedding_model,
                    "entity_embedding_dimension": built_retriever.entity_embedding_dimension,
                    "node_classifier": built_retriever.node_classifier,
                    "training": training_payload,
                    "loss_history": loss_history,
                    "final_loss": final_loss,
                    "trained_instances": trained_instances,
                },
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
