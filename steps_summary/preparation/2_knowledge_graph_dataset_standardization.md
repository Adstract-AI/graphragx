## Knowledge Graph Dataset Standardization

The goal of this step is to transform raw knowledge graph data into a structured and consistent format suitable for GNN training and downstream retrieval.

Raw knowledge graphs are typically provided as triples of the form (head, relation, tail). These must be converted into a unified graph representation.

This involves:
- Assigning unique IDs to all entities (nodes)
- Constructing directed edges while preserving relation types
- Creating mappings for relation types (and optionally node types)
- Defining node representations (e.g., entity names, types, or textual descriptors)
- Optionally adding structural enhancements such as reverse edges

The output is a graph object consisting of:
- Nodes (entities)
- Typed edges (relations)
- Supporting mappings and metadata

This standardized graph becomes the direct input for GNN training and enables consistent handling of structure across the entire pipeline.