# Subgraph Construction

Construct a question-specific subgraph that connects the query nodes and candidate nodes, capturing relevant relationships for reasoning.

---

## Objective

Build a compact and meaningful subgraph (`Gq`) that preserves the most relevant paths and relationships needed to answer the question.

---

## Input

- `query_nodes`: nodes obtained from Step 1  
- `candidate_nodes`: nodes selected in Step 2  
- Graph `G = (V, E)`

---

## Processing Steps

1. **Path Discovery**
   - For each candidate node, find the shortest path(s) from any query node.

   ```text
   for each candidate node c:
       find shortest path from any q ∈ query_nodes to c
   ```

2. **Subgraph Assembly**
   - Combine all discovered paths into a single subgraph.

   ```text
   Gq = union of all shortest paths
   ```

3. **Optional Expansion**
   - Optionally include:
     - immediate neighbors of nodes in the paths
     - additional edges between selected nodes
   - This can improve context but should be controlled to avoid large subgraphs.

---

## Output

- Subgraph `Gq = (Vq, Eq)`

```text
Vq = nodes in selected paths  
Eq = edges connecting those nodes
```

---

## Notes

- Shortest paths provide a simple and effective approximation of reasoning chains.
- This step reduces the large graph into a focused, query-specific structure.
- The quality of the subgraph directly impacts the final answer.
- If no path exists, consider using direct neighbors or fallback connections.

---

## Implementation Considerations

- Use standard graph algorithms (e.g. BFS for unweighted graphs).
- Limit path length if needed to avoid overly large subgraphs.
- Ensure the final subgraph remains small enough for efficient processing and textualization.
