# GNN Retrieval

There are two main approaches for retrieving relevant nodes using GNN embeddings. Both rely on the same embedding space but differ in how similarity is computed.

---

### Approach 1: Vector Database (Recommended for Scalability)

Use a vector database (e.g. Qdrant, FAISS) to perform efficient nearest-neighbor search.

#### Process

For each query node `q`:
1. Retrieve its embedding `h_q`
2. Perform vector search using cosine similarity
3. Retrieve top-k most similar nodes

```text
neighbors_q = vector_search(h_q, top_k)
```

After processing all query nodes:

```text
candidate_nodes = union of neighbors_q for all q ∈ query_nodes
```

The value of `k` should be chosen carefully. A small `k` keeps the next graph steps efficient, while a larger `k` improves the chance of keeping relevant nodes. A practical starting range is around 5-20 nodes, then tune it during evaluation.

#### Advantages

- Efficient for large graphs
- Avoids full similarity computation over all nodes
- Scales well with increasing graph size

#### Notes

- Embeddings must be precomputed and stored in the vector database
- Cosine similarity is typically used as the distance metric

---

### Approach 2: Direct Similarity Computation (For Small Graphs)

Compute similarity scores between query nodes and all nodes in the graph.

#### Process

For each node `v` in the graph:

```text
score(v) = max(sim(h_v, h_q)) for all q ∈ query_nodes
```

Where:
- `h_v` = embedding of node `v`
- `h_q` = embedding of query node `q`
- `sim` = similarity function

Then:

```text
rank nodes by score(v) in descending order
select top-k nodes
```

After retrieval, the final node set can be lightly cleaned before subgraph construction. Possible actions:
- remove duplicate nodes
- remove query nodes if they create trivial matches
- apply a score threshold if needed
- keep the final set small enough for efficient subgraph construction

#### Similarity Functions

- Cosine similarity (recommended)
- Euclidean distance (converted to similarity)

#### Advantages

- Simple to implement
- No additional infrastructure required

#### Limitations

- Not scalable for large graphs (O(N) per query)
- Slower as graph size increases

---

## Output

- ranked relevant nodes
- final top-k `candidate_nodes` for subgraph construction

## Notes

- This is the main retrieval step of the pipeline.
- The final output of this step should already be ready for the subgraph construction stage.
