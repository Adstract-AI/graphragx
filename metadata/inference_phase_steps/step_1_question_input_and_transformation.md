# Step 1: Question Input and Transformation

Receive the user's question, preserve it for the final LLM step, and transform it into graph-usable query nodes through entity linking.

---

## Objective

Convert a natural language question into a set of graph node identifiers (`query_nodes`) that anchor the query in the graph.

---

## Input

- Natural language question (string)

---

## Processing Steps

1. **Validate Input**
   - Ensure the input is a non-empty string.

2. **Store Original Question**
   - Keep the question unchanged for later use in the LLM step.

3. **Entity Extraction**
   - Identify relevant entities or concepts from the question.
   - Possible methods:
     - rule-based keyword extraction
     - named entity recognition (NER)
     - noun phrase extraction
     - lightweight LLM-based extraction
   - For this project, a simple and consistent method is sufficient.

4. **Entity Linking (Core Step)**
   - Map extracted entities to graph nodes.
   - Possible techniques:
     - exact string matching
     - alias or dictionary matching
     - fuzzy matching (e.g. Levenshtein distance)
     - synonym resources
     - lightweight semantic similarity

   The goal is to **anchor the question to existing graph nodes**, not to perform full semantic retrieval.

---

## Example

**Input:**
```text
What drugs treat headache?
```

**Output:**
```json
{
  "question_original": "What drugs treat headache?",
  "entities": ["headache"],
  "query_nodes": ["node_id_123"]
}
```

---

## Output

- `question_original`: unchanged input question  
- `entities`: extracted entities from the question  
- `query_nodes`: mapped graph node identifiers  

The key output of this step is `query_nodes`, which may contain one or multiple nodes depending on the question.

Multiple query nodes are expected and encouraged, as many questions naturally involve more than one entity or concept. These nodes collectively define the starting points for retrieval in the graph.

---

## Fallback Strategy (Important)

Entity linking may fail if:
- the entity is not present in the graph  
- wording differs from graph labels  
- the query is abstract or ambiguous  

The system should apply fallback strategies instead of failing.

### Recommended fallback order

1. **Fuzzy / Approximate Matching**
   - Use string similarity or synonym matching to find close candidates.

2. **Embedding-Based Fallback (Hybrid)**
   - Use lightweight semantic similarity to retrieve relevant nodes if entity linking fails completely.
   - This is a fallback only, not the primary method.

---

## Notes

- This step does **not perform retrieval** over the graph.
- Its only purpose is to produce a reliable set of starting nodes (`query_nodes`) for the next stage.
- The quality of this step directly affects all downstream components.