## Knowledge Graph Dataset Scope

At this stage of the project, we focus on **knowledge graph (KG) datasets** as the primary structured data source for retrieval and reasoning.

Knowledge graphs represent information as a set of entities (nodes) connected through typed relationships (edges), typically in the form of triples (head, relation, tail). This structure makes them particularly suitable for graph-based learning and multi-hop reasoning tasks.

The KG dataset serves as the **core “world” of the system**, providing the relational structure over which:
- Graph Neural Networks (GNNs) learn representations
- Subgraphs are retrieved during inference
- Context is constructed for LLM-based reasoning

This dataset is independent of the question-answering process and exists as a standalone structured representation of knowledge. All reasoning and retrieval operations are grounded in this graph.