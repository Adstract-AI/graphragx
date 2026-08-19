"""Pure-PyTorch Heterogeneous Graph Transformer answer retriever."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as torch_functional
from torch import Tensor, nn

from pipeline.preparation.helpers.configuration_definitions import HGT_ARCHITECTURE_ID
from pipeline.preparation.models.gnn_answer_retriever import build_node_classifier
from pipeline.preparation.models.interfaces import AnswerRetrieverModel


class HeterogeneousGraphTransformerLayer(nn.Module):
    """One-node-type HGT layer with categorical relation-aware attention."""

    def __init__(
        self,
        hidden_dimension: int,
        num_relations: int,
        attention_heads: int,
        dropout: float,
        *,
        use_layer_normalization: bool = True,
    ) -> None:
        super().__init__()
        if hidden_dimension <= 0:
            raise ValueError("hidden_dimension must be greater than zero.")
        if num_relations <= 0:
            raise ValueError("num_relations must be greater than zero.")
        if attention_heads <= 0:
            raise ValueError("attention_heads must be greater than zero.")
        if hidden_dimension % attention_heads != 0:
            raise ValueError(
                "hidden_dimension must be divisible by attention_heads."
            )

        self.hidden_dimension = hidden_dimension
        self.num_relations = num_relations
        self.attention_heads = attention_heads
        self.head_dimension = hidden_dimension // attention_heads
        self.use_layer_normalization = use_layer_normalization

        self.query_projection = nn.Linear(hidden_dimension, hidden_dimension)
        self.key_projection = nn.Linear(hidden_dimension, hidden_dimension)
        self.value_projection = nn.Linear(hidden_dimension, hidden_dimension)
        self.output_projection = nn.Linear(hidden_dimension, hidden_dimension)
        self.relation_attention = nn.Parameter(
            torch.empty(
                num_relations,
                attention_heads,
                self.head_dimension,
                self.head_dimension,
            )
        )
        self.relation_message = nn.Parameter(
            torch.empty(
                num_relations,
                attention_heads,
                self.head_dimension,
                self.head_dimension,
            )
        )
        self.relation_prior = nn.Parameter(
            torch.ones(num_relations, attention_heads)
        )
        self.skip = nn.Parameter(torch.ones(1))
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = (
            nn.LayerNorm(hidden_dimension)
            if use_layer_normalization
            else nn.Identity()
        )
        self.last_attention: Tensor | None = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.query_projection.reset_parameters()
        self.key_projection.reset_parameters()
        self.value_projection.reset_parameters()
        self.output_projection.reset_parameters()
        nn.init.xavier_uniform_(self.relation_attention)
        nn.init.xavier_uniform_(self.relation_message)
        nn.init.ones_(self.relation_prior)
        nn.init.ones_(self.skip)
        if isinstance(self.layer_norm, nn.LayerNorm):
            self.layer_norm.reset_parameters()

    def forward(
        self,
        node_features: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
        active_relation_ids: Tensor | None = None,
        active_relation_offsets: Tensor | None = None,
    ) -> Tensor:
        self._validate_inputs(node_features, edge_index, edge_type)
        active_relation_ids, active_relation_offsets = self._resolve_relation_groups(
            edge_type=edge_type,
            active_relation_ids=active_relation_ids,
            active_relation_offsets=active_relation_offsets,
        )

        node_count = node_features.shape[0]
        query = self.query_projection(node_features).view(
            node_count, self.attention_heads, self.head_dimension
        )
        key = self.key_projection(node_features).view(
            node_count, self.attention_heads, self.head_dimension
        )
        value = self.value_projection(node_features).view(
            node_count, self.attention_heads, self.head_dimension
        )

        source_nodes = edge_index[0].long()
        target_nodes = edge_index[1].long()
        attention_chunks: list[Tensor] = []
        message_chunks: list[Tensor] = []
        relation_ids = active_relation_ids.detach().cpu().tolist()
        relation_offsets = active_relation_offsets.detach().cpu().tolist()
        scale = math.sqrt(self.head_dimension)
        for relation_index, relation_id in enumerate(relation_ids):
            start = relation_offsets[relation_index]
            end = relation_offsets[relation_index + 1]
            relation_sources = source_nodes[start:end]
            relation_targets = target_nodes[start:end]
            relation_keys = torch.einsum(
                "ehd,hdf->ehf",
                key.index_select(0, relation_sources),
                self.relation_attention[relation_id],
            )
            relation_values = torch.einsum(
                "ehd,hdf->ehf",
                value.index_select(0, relation_sources),
                self.relation_message[relation_id],
            )
            relation_queries = query.index_select(0, relation_targets)
            attention_chunks.append(
                (
                    (relation_queries * relation_keys).sum(dim=-1)
                    * self.relation_prior[relation_id]
                    / scale
                ).float()
            )
            message_chunks.append(relation_values)

        if attention_chunks:
            attention_logits = torch.cat(attention_chunks, dim=0)
            transformed_messages = torch.cat(message_chunks, dim=0)
            attention = self._target_softmax(
                attention_logits,
                target_nodes=target_nodes,
                node_count=node_count,
            )
            aggregated = torch.zeros(
                node_count,
                self.attention_heads,
                self.head_dimension,
                dtype=torch.float32,
                device=node_features.device,
            )
            aggregated.index_add_(
                0,
                target_nodes,
                transformed_messages.float() * attention.unsqueeze(-1),
            )
            self.last_attention = attention.detach()
        else:
            aggregated = torch.zeros(
                node_count,
                self.attention_heads,
                self.head_dimension,
                dtype=torch.float32,
                device=node_features.device,
            )
            self.last_attention = node_features.new_empty(
                (0, self.attention_heads),
                dtype=torch.float32,
            )

        transformed = self.output_projection(
            torch_functional.gelu(
                aggregated.reshape(node_count, self.hidden_dimension).to(
                    dtype=node_features.dtype
                )
            )
        )
        transformed = self.dropout(transformed)
        alpha = torch.sigmoid(self.skip).to(dtype=transformed.dtype)
        output = alpha * transformed + (1.0 - alpha) * node_features
        return self.layer_norm(output)

    @staticmethod
    def _target_softmax(
        logits: Tensor,
        *,
        target_nodes: Tensor,
        node_count: int,
    ) -> Tensor:
        """Compute a stable FP32 softmax per target node and attention head."""
        expanded_targets = target_nodes.reshape(-1, 1).expand_as(logits)
        maxima = torch.full(
            (node_count, logits.shape[1]),
            -torch.inf,
            dtype=torch.float32,
            device=logits.device,
        )
        maxima.scatter_reduce_(
            0,
            expanded_targets,
            logits,
            reduce="amax",
            include_self=True,
        )
        exponentials = torch.exp(logits - maxima.index_select(0, target_nodes))
        denominators = torch.zeros_like(maxima)
        denominators.index_add_(0, target_nodes, exponentials)
        return exponentials / denominators.index_select(0, target_nodes).clamp_min(
            torch.finfo(torch.float32).tiny
        )

    @staticmethod
    def _resolve_relation_groups(
        *,
        edge_type: Tensor,
        active_relation_ids: Tensor | None,
        active_relation_offsets: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        if active_relation_ids is not None and active_relation_offsets is not None:
            if active_relation_offsets.ndim != 1:
                raise ValueError("active_relation_offsets must be one-dimensional.")
            if active_relation_offsets.numel() != active_relation_ids.numel() + 1:
                raise ValueError(
                    "active_relation_offsets must contain one boundary per active "
                    "relation plus the final edge boundary."
                )
            return active_relation_ids, active_relation_offsets
        if edge_type.numel() == 0:
            return (
                edge_type.new_empty(0),
                edge_type.new_zeros(1),
            )
        active_relation_ids, counts = torch.unique_consecutive(
            edge_type,
            return_counts=True,
        )
        offsets = torch.cat(
            [counts.new_zeros(1), counts.cumsum(dim=0)],
            dim=0,
        )
        return active_relation_ids, offsets

    def _validate_inputs(
        self,
        node_features: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
    ) -> None:
        if node_features.ndim != 2 or node_features.shape[1] != self.hidden_dimension:
            raise ValueError(
                "node_features must have shape [num_nodes, hidden_dimension]."
            )
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, num_edges].")
        if edge_type.ndim != 1 or edge_type.shape[0] != edge_index.shape[1]:
            raise ValueError("edge_type must contain one relation id per graph edge.")
        if edge_type.dtype != torch.long:
            raise ValueError("edge_type must use torch.long dtype.")
        if edge_type.numel() > 1 and torch.any(edge_type[1:] < edge_type[:-1]):
            raise ValueError("HGT edge_type values must be sorted by relation id.")
        if edge_type.numel() > 0 and (
            torch.any(edge_type < 0) or torch.any(edge_type >= self.num_relations)
        ):
            raise ValueError("edge_type contains a relation id outside the vocabulary.")


class HGTAnswerRetriever(nn.Module, AnswerRetrieverModel):
    """Entity projection, stacked HGT layers, and shared answer-node MLP."""

    def __init__(
        self,
        entity_embedding_dimension: int,
        hidden_dimension: int,
        gnn_layer_count: int,
        num_relations: int,
        attention_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.gnn_architecture = HGT_ARCHITECTURE_ID
        self.entity_embedding_dimension = entity_embedding_dimension
        self.hidden_dimension = hidden_dimension
        self.gnn_layer_count = gnn_layer_count
        self.num_relations = num_relations
        self.attention_heads = attention_heads
        self.dropout_value = dropout
        self.node_classifier = "mlp"
        self.use_edge_mlp = False
        self.question_aware_classifier = False
        self.add_layer_normalization = True
        self.edge_mlp_hidden_dim = None

        self.entity_projection = nn.Linear(
            entity_embedding_dimension,
            hidden_dimension,
        )
        self.gnn_layers = nn.ModuleList(
            HeterogeneousGraphTransformerLayer(
                hidden_dimension=hidden_dimension,
                num_relations=num_relations,
                attention_heads=attention_heads,
                dropout=dropout,
            )
            for _ in range(gnn_layer_count)
        )
        self.classifier = build_node_classifier(
            node_classifier="mlp",
            hidden_dimension=hidden_dimension,
            question_aware_classifier=False,
            dropout=dropout,
        )

    def forward(
        self,
        entity_features: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor | None = None,
        question_features: Tensor | None = None,
        relation_features: Tensor | None = None,
        edge_type: Tensor | None = None,
        edge_norm: Tensor | None = None,
        active_relation_ids: Tensor | None = None,
        edge_relation_index: Tensor | None = None,
        active_relation_offsets: Tensor | None = None,
    ) -> Tensor:
        if edge_type is None:
            raise ValueError("edge_type is required for HGT message passing.")
        node_features = self.entity_projection(entity_features)
        for layer in self.gnn_layers:
            node_features = layer(
                node_features,
                edge_index,
                edge_type,
                active_relation_ids=active_relation_ids,
                active_relation_offsets=active_relation_offsets,
            )
        return self.classifier(node_features).squeeze(-1)


def build_hgt_model(
    *,
    architecture_options: dict[str, Any],
    architecture_context: dict[str, Any] | None = None,
    entity_embedding_dimension: int,
    **_: Any,
) -> HGTAnswerRetriever:
    """Registry callback for the manual HGT architecture."""
    context = architecture_context or {}
    relation_type_count = context.get("relation_type_count")
    if not isinstance(relation_type_count, int) or relation_type_count <= 0:
        raise ValueError("HGT construction requires a positive relation_type_count.")
    return HGTAnswerRetriever(
        entity_embedding_dimension=entity_embedding_dimension,
        hidden_dimension=int(architecture_options["gnn_hidden_dimension"]),
        gnn_layer_count=int(architecture_options["gnn_layer_count"]),
        num_relations=relation_type_count,
        attention_heads=int(architecture_options["attention_heads"]),
        dropout=float(architecture_options["dropout"]),
    )
