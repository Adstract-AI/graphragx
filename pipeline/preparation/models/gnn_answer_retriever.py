"""PyTorch modules for answer-node retrieval."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from pipeline.preparation.models.interfaces import AnswerRetrieverModel


class WeightedMessagePassingLayer(nn.Module):
    """Message-passing layer that weights source-node messages per edge."""

    def __init__(self, hidden_dimension: int):
        super().__init__()
        self.message_projection = nn.Linear(hidden_dimension, hidden_dimension)
        self.update_projection = nn.Linear(hidden_dimension * 2, hidden_dimension)
        self.activation = nn.ReLU()

    def forward(
        self,
        node_features: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor,
    ) -> Tensor:
        source_nodes = edge_index[0].long()
        target_nodes = edge_index[1].long()
        source_messages = self.message_projection(node_features[source_nodes])
        weighted_messages = source_messages * edge_weight.reshape(-1, 1)

        aggregated_messages = torch.zeros_like(node_features)
        aggregated_messages.index_add_(0, target_nodes, weighted_messages)

        updated_features = torch.cat([node_features, aggregated_messages], dim=1)
        return self.activation(self.update_projection(updated_features))


class GnnAnswerRetriever(nn.Module, AnswerRetrieverModel):
    """Weighted GNN plus node-classifier head for answer retrieval."""

    def __init__(
        self,
        entity_embedding_dimension: int,
        hidden_dimension: int,
        gnn_layer_count: int,
        node_classifier: str,
    ):
        super().__init__()
        self.entity_embedding_dimension = entity_embedding_dimension
        self.hidden_dimension = hidden_dimension
        self.gnn_layer_count = gnn_layer_count
        self.node_classifier = node_classifier

        self.entity_projection = nn.Linear(
            entity_embedding_dimension,
            hidden_dimension,
        )
        self.gnn_layers = nn.ModuleList(
            WeightedMessagePassingLayer(hidden_dimension)
            for _ in range(gnn_layer_count)
        )
        self.classifier = self._build_classifier(node_classifier, hidden_dimension)

    def forward(
        self,
        entity_features: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor,
    ) -> Tensor:
        node_features = self.entity_projection(entity_features)
        for layer in self.gnn_layers:
            node_features = layer(node_features, edge_index, edge_weight)

        return self.classifier(node_features).squeeze(-1)

    @staticmethod
    def _build_classifier(node_classifier: str, hidden_dimension: int) -> nn.Module:
        if node_classifier == "linear":
            return nn.Linear(hidden_dimension, 1)

        if node_classifier == "mlp":
            return nn.Sequential(
                nn.Linear(hidden_dimension, hidden_dimension),
                nn.ReLU(),
                nn.Linear(hidden_dimension, 1),
            )

        raise ValueError(f"Unsupported node classifier: {node_classifier}")
