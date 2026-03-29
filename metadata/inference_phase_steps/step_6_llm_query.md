# Step 6: LLM Query for Final Answer

Use the textualized subgraph as context and query the LLM to generate the final answer.

---

## Objective

Leverage the LLM to perform reasoning over the structured context and produce a final answer to the original question.

---

## Input

- `question_original`: unchanged question from Step 1  
- `context`: textualized subgraph from Step 5  

---

## Processing Steps

1. **Prompt Construction**
   - Combine the question and context into a structured prompt.

   Recommended format:

   ```text
   Question:
   {question_original}

   Context:
   {textualized_subgraph}
   ```

2. **LLM Invocation**
   - Send the constructed prompt to the LLM.
   - The LLM processes both the question and the provided context.

3. **Answer Generation**
   - The LLM generates an answer based only on the given context and question.

---

## Output

- `answer`: generated response from the LLM  

---

## Notes

- The original question should always be used, not the transformed version.
- Instruct the LLM to answer only using the provided context.
- The quality of the answer depends heavily on the quality of the subgraph and its textualization.

---

## Implementation Considerations

- Use a consistent prompt template across all queries.
- Optionally instruct the LLM to:
  - base answers only on the provided context
  - be concise and precise
- Ensure the total input (question + context) stays within model limits.
