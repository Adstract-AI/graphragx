# Context Building / Textualization

## Objective

The goal of this step is to transform the retrieved subgraph into a coherent natural language context that can be effectively used by the LLM for reasoning.

Instead of presenting the graph as a set of disconnected triples, the subgraph is converted into a connected textual narrative that preserves relationships between entities and expresses them as a unified, readable description.

---

## Input

- Subgraph `Gq = (Vq, Eq)` obtained from the retrieval step

---

## Core Idea

The subgraph consists of structured relations of the form:

    Node1 → relation → Node2

However, LLMs operate best on natural language. Therefore, the goal is to convert the subgraph into a clear and logically connected piece of text, similar to a short “story” that describes how the entities are related.

---

## Processing Steps

1. **Relation Interpretation**
   - Treat each edge as a semantic relationship between two entities.

2. **Textualization**
   - Convert each relation into a natural language sentence.

   Example:

       Aspirin treats headache.
       Headache is a symptom of flu.

3. **Narrative Construction**
   - Combine sentences into a coherent and logically flowing paragraph.
   - Group related information and maintain meaningful ordering.

   Example:

       Aspirin is commonly used to treat headaches.
       Headaches can be a symptom of flu.

4. **Context Refinement**
   - Remove redundant or irrelevant information
   - Keep the context concise and focused
   - Optionally prioritize important paths or relations

---

## Output

A natural language context representing the subgraph, for example:

    Context:
    Aspirin is commonly used to treat headaches.
    Headaches can be a symptom of flu.

---

## Notes

- Natural language representation is preferred over raw triples for better LLM performance.
- The context should be coherent, concise, and relevant.
- The ordering of information can significantly influence reasoning quality.
- Context size should be controlled to avoid exceeding LLM input limits.

---

## Summary

This step bridges structured graph data and language-based reasoning by converting the retrieved subgraph into a coherent textual narrative, enabling the LLM to interpret and reason over relational information effectively.