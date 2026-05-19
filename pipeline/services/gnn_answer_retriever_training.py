"""Training service for the GNN answer retriever."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

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

logger = logging.getLogger(__name__)


class GnnAnswerRetrieverTrainingConfig(BaseModel):
    """Runtime training settings for the answer retriever."""

    epochs: int = Field(default=3)
    learning_rate: float = Field(default=1e-3)
    weight_decay: float = Field(default=0.0)
    max_instances: int | None = Field(default=None)
    log_every: int = Field(default=25)
    device: str = Field(default="auto")


class GnnAnswerRetrieverTrainingOutcome(BaseModel):
    """Result produced by the training service."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    final_loss: float = Field(..., description="Final average epoch loss.")
    trained_instances: int = Field(..., description="Number of training instances used.")
    model_artifact_path: Path = Field(..., description="Saved model weights path.")
    model_config_path: Path = Field(..., description="Saved model config path.")
    embedding_cache_directory: Path = Field(..., description="Embedding cache root.")
    selected_device: str = Field(..., description="Resolved PyTorch training device.")


class GnnAnswerRetrieverTrainingService(AbstractService):
    """Train and persist the GNN answer retriever."""

    model_weights_filename = "gnn_answer_retriever.pt"
    model_config_filename = "gnn_answer_retriever_config.json"

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
        logger.info("Starting GNN answer retriever training on device=%s", device)

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
        )
        logger.info(
            "Loaded embedding caches: nodes=%s relations=%s questions=%s",
            len(node_cache.embeddings),
            len(relation_cache.embeddings),
            len(question_cache.embeddings),
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
            "Saved embedding caches: nodes=%s relations=%s questions=%s",
            len(node_cache.embeddings),
            len(relation_cache.embeddings),
            len(question_cache.embeddings),
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
        for epoch in range(1, training_config.epochs + 1):
            logger.info(
                "Training epoch %s/%s over %s WebQSP instances",
                epoch,
                training_config.epochs,
                len(train_instances),
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
                        "Epoch %s/%s progress: %s/%s instances, latest_loss=%.6f",
                        epoch,
                        training_config.epochs,
                        instance_index,
                        len(train_instances),
                        float(loss.detach().cpu().item()),
                    )

            final_loss = epoch_loss / len(train_instances)
            logger.info(
                "Finished epoch %s/%s with average_loss=%.6f",
                epoch,
                training_config.epochs,
                final_loss,
            )

        model_artifact_path = self._save_model_artifacts(
            model=model,
            built_retriever=built_retriever,
            training_config=training_config,
            final_loss=final_loss,
            trained_instances=len(train_instances),
            model_root=cache_root / "models",
            torch=torch,
        )
        logger.info("Saved trained GNN answer retriever to %s", model_artifact_path)

        return GnnAnswerRetrieverTrainingOutcome(
            final_loss=final_loss,
            trained_instances=len(train_instances),
            model_artifact_path=model_artifact_path,
            model_config_path=model_artifact_path.with_name(self.model_config_filename),
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
        training_config: GnnAnswerRetrieverTrainingConfig,
        final_loss: float,
        trained_instances: int,
        model_root: Path,
        torch,
    ) -> Path:
        model_root.mkdir(parents=True, exist_ok=True)
        model_artifact_path = model_root / self.model_weights_filename
        model_config_path = model_root / self.model_config_filename
        torch.save(model.state_dict(), model_artifact_path)
        model_config_path.write_text(
            json.dumps(
                {
                    "dataset_id": built_retriever.dataset_id,
                    "entity_embedding_model": built_retriever.entity_embedding_model,
                    "entity_embedding_dimension": built_retriever.entity_embedding_dimension,
                    "hidden_dimension": built_retriever.hidden_dimension,
                    "gnn_layer_count": built_retriever.gnn_layer_count,
                    "node_classifier": built_retriever.node_classifier,
                    "training": training_config.model_dump(),
                    "final_loss": final_loss,
                    "trained_instances": trained_instances,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return model_artifact_path
