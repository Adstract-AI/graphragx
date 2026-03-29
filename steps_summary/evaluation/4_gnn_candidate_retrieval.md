## GNN-Based Candidate Retrieval

The goal of this step is to retrieve a set of relevant node candidates from the knowledge graph using the query nodes obtained from the transformation step.

Given the query nodes, the trained GNN is used to explore the graph and identify neighboring nodes that are likely to be relevant for answering the question. This is achieved by leveraging the learned node embeddings and the structural information encoded in the graph.

The retrieval process typically involves:
- Expanding from the query nodes to their local neighborhoods
- Using node embeddings to rank or filter candidate nodes based on similarity or relevance
- Selecting a subset of nodes that form a focused candidate set

The output of this step is a set of **candidate nodes**, which represent a narrowed-down portion of the graph that is likely to contain the answer or relevant supporting information.

This step constitutes the **retrieval component of the RAG pipeline**, ensuring that only relevant graph information is passed to the next stage for subgraph construction and LLM-based reasoning.