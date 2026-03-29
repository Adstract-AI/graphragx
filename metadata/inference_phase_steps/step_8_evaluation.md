# Step 8: Evaluation

Evaluate the performance of the pipeline in terms of answer quality, retrieval quality, and overall system behavior.

---

## Objective

Measure how well the system retrieves relevant graph information and generates correct and grounded answers.

---

## Evaluation Components

### 1. Answer Quality

Evaluate the final LLM output.

- **Accuracy / Exact Match (EM)**  
  - Checks if the predicted answer matches the ground truth exactly.

- **F1 Score**  
  - Measures token-level overlap between predicted and true answers.

- **Semantic Similarity**  
  - Use embedding-based similarity (e.g. cosine similarity) to handle paraphrased answers.

---

### 2. Retrieval Quality

Evaluate how well the system retrieves relevant nodes and subgraphs.

- **Hit@k**  
  - Checks if the correct answer node is within the top-k retrieved nodes.

- **Recall@k**  
  - Measures how many relevant nodes are retrieved among the top-k.

- **Mean Reciprocal Rank (MRR)**  
  - Evaluates how high the correct node appears in the ranking.

These retrieval metrics require a graph-grounded target. This means the dataset must either contain gold answer nodes, gold relevant subgraphs, or a reliable way to map the textual ground-truth answer to graph nodes.

---

### 3. Subgraph Quality

Evaluate whether the constructed subgraph contains useful reasoning paths.

- Check if:
  - query nodes and answer nodes are connected
  - relevant paths are present
- Optional:
  - measure path length (shorter paths are usually better)

---

### 4. LLM Grounding

Evaluate whether the LLM uses the provided context.

- Check if:
  - the answer is supported by the textualized subgraph
  - hallucinations are minimized
- Optional:
  - manual inspection or LLM-based evaluation

---

## Evaluation Procedure

1. Use a dataset with known question–answer pairs.
2. Run the full pipeline for each question.
3. Compare:
   - predicted answer vs ground truth
   - retrieved nodes vs expected nodes
4. Compute metrics across all samples.

---

## Notes

- Start with simple metrics (Accuracy, Hit@k, MRR).
- Use small evaluation sets first for debugging.
- Combine automatic metrics with manual inspection for deeper analysis.
