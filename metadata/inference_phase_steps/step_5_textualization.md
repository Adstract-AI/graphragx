# Step 5: Context Building / Textualization

Convert the constructed subgraph into a structured textual format that can be used as context for the LLM.

---

## Objective

Transform the graph-based representation (`Gq`) into a clear and informative text context that preserves relevant relationships for reasoning.

---

## Input

- Subgraph `Gq = (Vq, Eq)` from Step 3  

---

## Processing Steps

1. **Edge Traversal**
   - Iterate over edges in the subgraph.
   - Each edge represents a relation between two nodes.

2. **Text Conversion**
   - Convert each edge into a textual statement.

   Recommended format:

   ```text
   Node1 relation Node2.
   ```

   Example:
   ```text
   Aspirin treats headache.
   Headache is a symptom of flu.
   ```

3. **Context Aggregation**
   - Combine all statements into a single context block.
   - Optionally:
     - group related statements
     - remove duplicates
     - maintain logical ordering (e.g. path order)

---

## Output

- Textual context:

```text
Context:
- Node1 relation Node2.
- Node2 relation Node3.
...
```

---

## Alternative Formats

- **Triple format**
  ```text
  (Node1, relation, Node2)
  ```

- **Hybrid format**
  - mix of natural language and structured triples

---

## Notes

- Natural language format is recommended for better LLM performance.
- The context should be concise and relevant; avoid including unnecessary edges.
- The ordering of statements can influence reasoning quality.

---

## Implementation Considerations

- Ensure consistent formatting across all queries.
- Limit context size to avoid exceeding LLM input limits.
- Optionally prioritize edges from shortest paths for better relevance.
