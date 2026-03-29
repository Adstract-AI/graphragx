# Evaluation

## Objective

The goal of this step is to evaluate the performance of the full pipeline across three key dimensions:
- Retrieval quality (GNN)
- Answer quality (LLM)
- Explanation quality (LLM reasoning / grounding)

The evaluation measures not only whether the system produces correct answers, but also whether those answers are supported by the retrieved graph context and how efficiently the system operates.

---

## Evaluation Components

### 1. Retrieval Quality (GNN)

Evaluate how well the system retrieves relevant nodes and subgraphs from the knowledge graph.

- **Recall@k (e.g., Recall@5)**  
  - Measures whether the correct answer node is included in the top-k retrieved nodes.
  - Critical for ensuring the answer is present in the retrieved subgraph.

- **nDCG@k (e.g., nDCG@5)**  
  - Measures the quality of ranking among retrieved nodes.
  - Rewards placing relevant nodes higher in the ranking.

These metrics evaluate the effectiveness of the GNN as a retrieval mechanism.

---

### 2. Answer Quality (LLM)

Evaluate the correctness of the final generated answer.

- **Correctness (Exact Match / LLM-based judgment)**  
  - Determines whether the predicted answer is factually correct.

- **F1 Score**  
  - Measures overlap between predicted and ground truth answers at the token level.

- **Semantic Similarity (optional)**  
  - Uses embedding-based similarity to account for paraphrased answers.

These metrics measure how well the LLM produces accurate answers given the retrieved context.

---

### 3. Faithfulness (Grounding)

Evaluate whether the generated answer is supported by the retrieved subgraph.

- **Faithfulness**  
  - Measures whether the answer is grounded in the provided context.
  - Detects hallucinations (answers not supported by the graph).

Evaluation can be performed using:
- rule-based checks (answer present in subgraph)
- LLM-based verification

This is critical for ensuring trustworthiness of the system.

---

### 4. Explanation Quality (LLM Reasoning)

Evaluate the quality of the explanation provided by the LLM as justification for the answer.

The explanation serves as a “proof” of reasoning over the graph.

- **Explanation Faithfulness**  
  - Checks whether the explanation is consistent with the retrieved subgraph.

- **Explanation Correctness**  
  - Verifies whether the reasoning steps logically support the final answer.

- **Explanation Completeness**  
  - Measures whether the explanation includes the key entities and relations needed for reasoning.

Evaluation can be performed using:
- LLM-based judgment (e.g., scoring explanation quality)
- comparison with expected reasoning paths (if available)

---

### 5. System Efficiency

Evaluate the performance and scalability of the system.

- **Avg Subgraph Triples**  
  - Average number of triples included in the constructed subgraph.
  - Reflects context size and potential noise.

- **Avg Latency (seconds)**  
  - Average time taken per query (including transformation, retrieval, and generation).

These metrics capture the trade-off between performance and efficiency.

---

## Evaluation Procedure

1. Use a QA dataset with known ground truth answers.
2. For each question:
   - Transform the query
   - Retrieve candidate nodes and construct subgraph
   - Generate answer and explanation using the LLM
3. Compare:
   - Predicted answer vs ground truth
   - Retrieved nodes vs expected nodes
   - Explanation vs retrieved context
4. Compute all metrics across the dataset.

---

## Summary

This evaluation framework measures the full pipeline by jointly analyzing:
- Retrieval effectiveness (Recall@k, nDCG@k)
- Answer accuracy (Correctness, F1)
- Grounding and reliability (Faithfulness)
- Explanation quality (reasoning validity and completeness)
- System efficiency (subgraph size and latency)

Together, these metrics provide a comprehensive view of how well the system performs both as a retrieval mechanism and as a reasoning engine.