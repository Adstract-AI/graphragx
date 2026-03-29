## Query Transformation (Natural Language → Graph Representation)

The goal of this step is to convert a natural language question into a graph-compatible representation that can be used to query the knowledge graph.

Given a question from the QA dataset, the system must extract **graph-aligned query signals**, which act as entry points into the graph. This process, commonly referred to as **query grounding**, involves identifying entities (nodes) and optionally relations mentioned in the question.

This transformation can be performed using different strategies, such as:
- **Embedding-based similarity**, where the question is embedded and matched against node representations to identify relevant entities
- **LLM-based extraction**, where a language model extracts entities and relations in a structured format

The output of this step is a set of **query nodes**, represented as node IDs in the standardized knowledge graph, and optionally associated relations.

This step serves as the bridge between unstructured language input and structured graph data, enabling the system to interact with the graph in a meaningful way.