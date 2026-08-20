"""Architecture-owned preparation, batching, loss, probability, and checkpoint hooks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from helpers.logging_config import get_logger
from pipeline.evaluation.models import PreparedGnnEvaluationData, PreparedGnnEvaluationInstance
from pipeline.preparation.exceptions import (
    GnnAnswerRetrieverEvaluationException,
    GnnAnswerRetrieverTrainingException,
)
from pipeline.preparation.models.gnn_training_data import (
    PreparedGnnTrainingData,
    PreparedGnnTrainingInstance,
)
from pipeline.preparation.helpers.rearev_constants import (
    REAREV_ENCODER_MODEL_ID,
    REAREV_ENCODER_REVISION,
    REAREV_QUESTION_MAX_LENGTH,
    REAREV_RELATION_MAX_LENGTH,
)
from pipeline.preparation.models.webqsp_local_graph import WebQSPProcessedInstance
from pipeline.preparation.services.gnn_relation_vocabulary import build_sorted_typed_edges

logger = get_logger(__name__)


@dataclass
class PreparedReaRevBatch:
    """One disconnected ReaRev batch with token IDs and graph-local metadata."""

    instance_count: int
    question_input_ids: Any
    question_attention_mask: Any
    relation_input_ids: Any
    relation_attention_mask: Any
    edge_index: Any
    edge_relation_index: Any
    initialization_edge_index: Any
    initialization_relation_index: Any
    node_graph_index: Any
    seed_distribution: Any
    node_labels: Any
    graph_count: int
    valid_target_graphs: Any


class DefaultGnnRuntimeStrategy:
    """Default runtime contract used by embedding-based retrievers."""

    strategy_id = "default"
    handles_data_preparation = False
    handles_training_batches = False

    @staticmethod
    def checkpoint_state_dict(model):
        return model.state_dict()

    @staticmethod
    def load_checkpoint_state_dict(model, state_dict) -> None:
        model.load_state_dict(state_dict)

    @staticmethod
    def probabilities(scores, *, graph_index=None, graph_count=None):
        import torch

        return torch.sigmoid(scores)


class ReaRevRuntimeStrategy(DefaultGnnRuntimeStrategy):
    """Runtime behavior for token-level ReaRev reasoning."""

    strategy_id = "rearev"
    handles_data_preparation = True
    handles_training_batches = True

    def __init__(self, tokenizer=None) -> None:
        self._tokenizer = tokenizer

    def _get_tokenizer(self):
        if self._tokenizer is not None:
            return self._tokenizer
        try:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                REAREV_ENCODER_MODEL_ID,
                revision=REAREV_ENCODER_REVISION,
            )
        except Exception as error:
            raise RuntimeError(
                "ReaRev requires the pinned MiniLM tokenizer "
                f"{REAREV_ENCODER_MODEL_ID}@{REAREV_ENCODER_REVISION}. The tokenizer "
                "is not available locally and could not be downloaded."
            ) from error
        return self._tokenizer

    @staticmethod
    def normalize_relation_text(relation: str) -> str:
        """Apply the reference-compatible Freebase relation token schema."""
        reverse = relation.startswith("reverse__")
        relation_name = relation[len("reverse__") :] if reverse else relation
        components = [part for part in relation_name.replace("/", ".").split(".") if part]
        selected = components[-2:] if len(components) >= 2 else components
        tokens: list[str] = []
        for component in selected:
            tokens.extend(part for part in component.split("_") if part)
        if reverse:
            tokens.reverse()
        return " ".join(tokens) or relation_name

    def _tokenize(
        self,
        texts: list[str],
        *,
        max_length: int,
        torch: ModuleType,
    ) -> tuple[Any, Any]:
        if not texts:
            return (
                torch.empty((0, max_length), dtype=torch.long),
                torch.empty((0, max_length), dtype=torch.long),
            )
        encoded = self._get_tokenizer()(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return encoded["input_ids"].long(), encoded["attention_mask"].long()

    @staticmethod
    def _ordered_relations(relation_vocabulary: dict[str, int]) -> list[str]:
        relations: list[str | None] = [None] * len(relation_vocabulary)
        for relation, relation_id in relation_vocabulary.items():
            if relation_id < 0 or relation_id >= len(relations):
                raise ValueError("Relation vocabulary IDs must be contiguous from zero.")
            relations[relation_id] = relation
        if any(relation is None for relation in relations):
            raise ValueError("Relation vocabulary IDs must be contiguous from zero.")
        return [str(relation) for relation in relations]

    @staticmethod
    def _seed_distribution(instance: WebQSPProcessedInstance, torch: ModuleType):
        node_count = len(instance.nodes)
        if node_count <= 0:
            raise ValueError("ReaRev graphs must contain at least one node.")
        seed_ids = sorted(
            {
                instance.node2id[entity]
                for entity in instance.q_entity
                if entity in instance.node2id
            }
        )
        distribution = torch.zeros(node_count, dtype=torch.float32)
        if seed_ids:
            distribution[torch.tensor(seed_ids, dtype=torch.long)] = 1.0 / len(seed_ids)
            return distribution
        logger.warning(
            "ReaRev graph has no question entity in the local graph; using a uniform "
            "seed distribution over all nodes."
        )
        return torch.full((node_count,), 1.0 / node_count, dtype=torch.float32)

    @staticmethod
    def _initialization_edges(
        instance: WebQSPProcessedInstance,
        relation_vocabulary: dict[str, int],
        torch: ModuleType,
    ) -> tuple[Any, Any]:
        original_indices = [
            index
            for index, relation in enumerate(instance.edge_relations)
            if not relation.startswith("reverse__")
        ]
        if not original_indices:
            return (
                torch.empty((2, 0), dtype=torch.long),
                torch.empty(0, dtype=torch.long),
            )
        index_tensor = torch.tensor(original_indices, dtype=torch.long)
        edge_index = instance.edge_index.index_select(1, index_tensor).long()
        edge_type = torch.tensor(
            [relation_vocabulary[instance.edge_relations[index]] for index in original_indices],
            dtype=torch.long,
        )
        return edge_index, edge_type

    def _prepare_instances(
        self,
        *,
        instances: list[WebQSPProcessedInstance],
        relation_vocabulary: dict[str, int],
        source_start: int,
        torch: ModuleType,
        evaluation: bool,
    ) -> tuple[list[Any], Any, Any]:
        ordered_relations = self._ordered_relations(relation_vocabulary)
        relation_input_ids, relation_attention_mask = self._tokenize(
            [self.normalize_relation_text(relation) for relation in ordered_relations],
            max_length=REAREV_RELATION_MAX_LENGTH,
            torch=torch,
        )
        question_input_ids, question_attention_mask = self._tokenize(
            [instance.question for instance in instances],
            max_length=REAREV_QUESTION_MAX_LENGTH,
            torch=torch,
        )
        prepared: list[Any] = []
        for offset, instance in enumerate(instances):
            source_instance_index = source_start + offset
            if not instance.nodes:
                logger.warning(
                    "Skipping ReaRev graph with no nodes: "
                    f"instance_index={source_instance_index} "
                    f"phase={'evaluation' if evaluation else 'training'}"
                )
                continue
            try:
                edge_index, edge_type = build_sorted_typed_edges(
                    edge_index=instance.edge_index,
                    edge_relations=instance.edge_relations,
                    vocabulary=relation_vocabulary,
                    torch=torch,
                )
                initialization_edge_index, initialization_edge_type = (
                    self._initialization_edges(instance, relation_vocabulary, torch)
                )
                common = dict(
                    source_instance_index=source_instance_index,
                    edge_index=edge_index,
                    edge_type=edge_type,
                    question_input_ids=question_input_ids[offset],
                    question_attention_mask=question_attention_mask[offset],
                    seed_distribution=self._seed_distribution(instance, torch),
                    initialization_edge_index=initialization_edge_index,
                    initialization_edge_type=initialization_edge_type,
                )
                if evaluation:
                    prepared.append(PreparedGnnEvaluationInstance(instance=instance, **common))
                else:
                    prepared.append(
                        PreparedGnnTrainingInstance(node_labels=instance.node_labels, **common)
                    )
            except (KeyError, ValueError) as error:
                exception_type = (
                    GnnAnswerRetrieverEvaluationException
                    if evaluation
                    else GnnAnswerRetrieverTrainingException
                )
                raise exception_type(
                    f"Could not prepare ReaRev graph {source_instance_index}: {error}"
                ) from error
        if not prepared:
            exception_type = (
                GnnAnswerRetrieverEvaluationException
                if evaluation
                else GnnAnswerRetrieverTrainingException
            )
            raise exception_type(
                "ReaRev requires at least one non-empty graph in the selected "
                f"{'evaluation' if evaluation else 'training'} slice."
            )
        return prepared, relation_input_ids, relation_attention_mask

    def prepare_training_data(
        self,
        *,
        built_retriever,
        instances: list[WebQSPProcessedInstance],
        relation_vocabulary: dict[str, int] | None,
        start_index: int,
        end_index: int,
        selected_device: str,
        cache_root: Path,
        torch: ModuleType,
        autocast_dtype: str = "float32",
    ) -> PreparedGnnTrainingData:
        if relation_vocabulary is None:
            raise GnnAnswerRetrieverTrainingException(
                "ReaRev training requires an authoritative relation vocabulary."
            )
        prepared, relation_ids, relation_mask = self._prepare_instances(
            instances=instances,
            relation_vocabulary=relation_vocabulary,
            source_start=start_index,
            torch=torch,
            evaluation=False,
        )
        logger.info(
            "Prepared ReaRev token IDs without OpenAI/Qdrant embeddings: "
            f"instances={len(prepared)} relations={len(relation_vocabulary)}"
        )
        return PreparedGnnTrainingData(
            built_retriever=built_retriever,
            instances=prepared,
            training_start_instance=start_index,
            training_end_instance=end_index,
            selected_device=selected_device,
            embedding_cache_device="not-applicable",
            embedding_cache_dtype="not-applicable",
            relation_input_ids=relation_ids,
            relation_attention_mask=relation_mask,
            runtime_strategy=self.strategy_id,
            autocast_dtype=autocast_dtype,
            cache_root=cache_root,
        )

    def prepare_evaluation_data(
        self,
        *,
        instances: list[WebQSPProcessedInstance],
        relation_vocabulary: dict[str, int] | None,
        selected_device: str,
        torch: ModuleType,
        autocast_dtype: str = "float32",
    ) -> PreparedGnnEvaluationData:
        if relation_vocabulary is None:
            raise GnnAnswerRetrieverEvaluationException(
                "ReaRev evaluation requires the saved relation vocabulary."
            )
        prepared, relation_ids, relation_mask = self._prepare_instances(
            instances=instances,
            relation_vocabulary=relation_vocabulary,
            source_start=0,
            torch=torch,
            evaluation=True,
        )
        return PreparedGnnEvaluationData(
            instances=prepared,
            selected_device=selected_device,
            embedding_cache_device="not-applicable",
            embedding_cache_dtype="not-applicable",
            relation_input_ids=relation_ids,
            relation_attention_mask=relation_mask,
            runtime_strategy=self.strategy_id,
            autocast_dtype=autocast_dtype,
        )

    def build_training_batches(
        self,
        *,
        prepared_data: PreparedGnnTrainingData,
        batch_size: int,
        torch: ModuleType,
        device: str,
    ) -> list[PreparedReaRevBatch]:
        return [
            self._build_batch(
                instances=prepared_data.instances[start : start + batch_size],
                relation_input_ids=prepared_data.relation_input_ids,
                relation_attention_mask=prepared_data.relation_attention_mask,
                torch=torch,
                device=device,
            )
            for start in range(0, len(prepared_data.instances), batch_size)
        ]

    def build_evaluation_batch(
        self,
        *,
        prepared_data: PreparedGnnEvaluationData,
        prepared_instance: PreparedGnnEvaluationInstance,
        torch: ModuleType,
        device: str,
    ) -> PreparedReaRevBatch:
        return self._build_batch(
            instances=[prepared_instance],
            relation_input_ids=prepared_data.relation_input_ids,
            relation_attention_mask=prepared_data.relation_attention_mask,
            torch=torch,
            device=device,
        )

    @staticmethod
    def _build_batch(
        *,
        instances: list[Any],
        relation_input_ids,
        relation_attention_mask,
        torch: ModuleType,
        device: str,
    ) -> PreparedReaRevBatch:
        if not instances:
            raise ValueError("ReaRev batches cannot be empty.")
        edge_parts, edge_type_parts = [], []
        init_edge_parts, init_type_parts = [], []
        graph_parts, seed_parts, label_parts = [], [], []
        node_offset = 0
        for graph_id, instance in enumerate(instances):
            node_count = int(instance.seed_distribution.shape[0])
            edge_parts.append(instance.edge_index + node_offset)
            edge_type_parts.append(instance.edge_type)
            init_edge_parts.append(instance.initialization_edge_index + node_offset)
            init_type_parts.append(instance.initialization_edge_type)
            graph_parts.append(torch.full((node_count,), graph_id, dtype=torch.long))
            seed_parts.append(instance.seed_distribution)
            labels = getattr(instance, "node_labels", None)
            label_parts.append(
                labels.float() if labels is not None else torch.zeros(node_count)
            )
            node_offset += node_count
        edge_index = torch.cat(edge_parts, dim=1)
        edge_types = torch.cat(edge_type_parts)
        init_edge_index = torch.cat(init_edge_parts, dim=1)
        init_types = torch.cat(init_type_parts)
        all_types = torch.cat((edge_types, init_types))
        active_types = torch.unique(all_types, sorted=True)
        edge_relation_index = torch.searchsorted(active_types, edge_types)
        init_relation_index = torch.searchsorted(active_types, init_types)
        if relation_input_ids is None or relation_attention_mask is None:
            raise ValueError("Prepared ReaRev relation token IDs are missing.")
        selected_relation_ids = relation_input_ids.index_select(0, active_types)
        selected_relation_mask = relation_attention_mask.index_select(0, active_types)
        labels = torch.cat(label_parts).float()
        valid = torch.tensor(
            [float(part.sum().item()) > 0 for part in label_parts],
            dtype=torch.bool,
        )
        return PreparedReaRevBatch(
            instance_count=len(instances),
            question_input_ids=torch.stack(
                [instance.question_input_ids for instance in instances]
            ).to(device),
            question_attention_mask=torch.stack(
                [instance.question_attention_mask for instance in instances]
            ).to(device),
            relation_input_ids=selected_relation_ids.to(device),
            relation_attention_mask=selected_relation_mask.to(device),
            edge_index=edge_index.to(device),
            edge_relation_index=edge_relation_index.to(device),
            initialization_edge_index=init_edge_index.to(device),
            initialization_relation_index=init_relation_index.to(device),
            node_graph_index=torch.cat(graph_parts).to(device),
            seed_distribution=torch.cat(seed_parts).to(device),
            node_labels=labels.to(device),
            graph_count=len(instances),
            valid_target_graphs=valid.to(device),
        )

    @staticmethod
    def model_inputs(batch: PreparedReaRevBatch) -> dict[str, Any]:
        return {
            "question_input_ids": batch.question_input_ids,
            "question_attention_mask": batch.question_attention_mask,
            "relation_input_ids": batch.relation_input_ids,
            "relation_attention_mask": batch.relation_attention_mask,
            "edge_index": batch.edge_index,
            "edge_relation_index": batch.edge_relation_index,
            "initialization_edge_index": batch.initialization_edge_index,
            "initialization_relation_index": batch.initialization_relation_index,
            "node_graph_index": batch.node_graph_index,
            "seed_distribution": batch.seed_distribution,
            "graph_count": batch.graph_count,
        }

    @staticmethod
    def compute_loss(scores, batch: PreparedReaRevBatch):
        """Graph-balanced KL divergence over normalized answer-node targets."""
        import torch
        from pipeline.preparation.models.rearev_answer_retriever import graph_softmax

        valid_graph_ids = torch.nonzero(batch.valid_target_graphs, as_tuple=False).flatten()
        if valid_graph_ids.numel() == 0:
            return None
        probabilities = graph_softmax(scores, batch.node_graph_index, batch.graph_count)
        losses = []
        for graph_id in valid_graph_ids.tolist():
            mask = batch.node_graph_index == graph_id
            labels = batch.node_labels[mask].float()
            target = labels / labels.sum().clamp_min(1.0)
            positive = target > 0
            losses.append(
                torch.sum(
                    target[positive]
                    * (
                        target[positive].log()
                        - probabilities[mask][positive].clamp_min(1e-12).log()
                    )
                )
            )
        return torch.stack(losses).mean()

    @staticmethod
    def probabilities(scores, *, graph_index=None, graph_count=None):
        if graph_index is None or graph_count is None:
            raise ValueError("ReaRev probability conversion requires graph metadata.")
        from pipeline.preparation.models.rearev_answer_retriever import graph_softmax

        return graph_softmax(scores, graph_index, graph_count)

    @staticmethod
    def checkpoint_state_dict(model):
        return model.trainable_state_dict()

    @staticmethod
    def load_checkpoint_state_dict(model, state_dict) -> None:
        model.load_trainable_state_dict(state_dict)
