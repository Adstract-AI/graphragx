# Pipeline Overview

This file connects the step summaries across the two main phases of the project and shows how outputs from one step become inputs to the next.

For the actual implementation-oriented description of each step, use the linked summary files below.

## Preparation Phase

### 1. Knowledge Graph Dataset Scope

This step defines the graph dataset that acts as the structured world of the system. It establishes the base knowledge source used by all later retrieval and reasoning steps.

See: [1_knowledge_graph_dataset_scope.md](/Users/itonkdong/Work/Fax/NLP/project/graphragx/steps_summary/preparation/1_knowledge_graph_dataset_scope.md)

### 2. Knowledge Graph Dataset Standardization

This step converts the raw graph data into a consistent internal representation. The resulting standardized graph becomes the main structural artifact used throughout the pipeline.

See: [2_knowledge_graph_dataset_standardization.md](/Users/itonkdong/Work/Fax/NLP/project/graphragx/steps_summary/preparation/2_knowledge_graph_dataset_standardization.md)

### 3. Train GNN and Graph Embeddings

This step learns graph-based representations from the standardized graph. Its outputs power the retrieval stage during evaluation.

See: [3_train_gnn_and_graph_embeddings.md](/Users/itonkdong/Work/Fax/NLP/project/graphragx/steps_summary/preparation/3_train_gnn_and_graph_embeddings.md)

## Bridge Between Phases

The preparation phase produces the main reusable assets of the system: the standardized graph, the trained GNN, and the learned node embeddings.

The evaluation phase consumes those assets to process questions, retrieve relevant graph context, generate answers, and measure system performance.

## Evaluation Phase

### 1. Question-Answer Dataset Scope

This step defines the QA dataset used to probe the graph with questions. It provides the evaluation side of the pipeline.

See: [1_question_answer_dataset_scope.md](/Users/itonkdong/Work/Fax/NLP/project/graphragx/steps_summary/evaluation/1_question_answer_dataset_scope.md)

### 2. Question-Answer Dataset Standardization

This step normalizes the QA dataset into a form that can be passed through the inference pipeline consistently. It aligns the question-side data with the graph-side representation.

See: [2_question_answer_dataset_standardization.md](/Users/itonkdong/Work/Fax/NLP/project/graphragx/steps_summary/evaluation/2_question_answer_dataset_standardization.md)

### 3. Query Transformation

This step converts natural language questions into graph-aligned query signals. It is the first direct bridge from language input into graph retrieval.

See: [3_query_transformation.md](/Users/itonkdong/Work/Fax/NLP/project/graphragx/steps_summary/evaluation/3_query_transformation.md)

### 4. GNN Retrieval

This step uses the learned graph representations to retrieve relevant candidate nodes. It narrows the large graph down to the most promising region for the current question.

See: [4_gnn_retrieval.md](/Users/itonkdong/Work/Fax/NLP/project/graphragx/steps_summary/evaluation/4_gnn_retrieval.md)

### 5. Subgraph Construction

This step turns the retrieved candidates into a compact question-specific subgraph. That subgraph becomes the structured reasoning context for the rest of the flow.

See: [5_subgraph_construction.md](/Users/itonkdong/Work/Fax/NLP/project/graphragx/steps_summary/evaluation/5_subgraph_construction.md)

### 6. Context Building / Textualization

This step translates the subgraph into a language form that the LLM can reason over. It connects structured graph evidence with language-based inference.

See: [6_textualization.md](/Users/itonkdong/Work/Fax/NLP/project/graphragx/steps_summary/evaluation/6_textualization.md)

### 7. LLM Query

This step combines the original question with the textualized context and asks the LLM for an answer. It is the main answer-generation step of the pipeline.

See: [7_llm_query.md](/Users/itonkdong/Work/Fax/NLP/project/graphragx/steps_summary/evaluation/7_llm_query.md)

### 8. Answer and Explanation Output

This step formats the final response and optionally provides supporting explanation. It makes the output easier to inspect and analyze.

See: [8_answer_and_explanation.md](/Users/itonkdong/Work/Fax/NLP/project/graphragx/steps_summary/evaluation/8_answer_and_explanation.md)

### 9. Evaluation

This step measures retrieval quality, answer quality, grounding, explanation quality, and efficiency. It closes the loop by assessing how well the full pipeline performs.

See: [9_evaluation.md](/Users/itonkdong/Work/Fax/NLP/project/graphragx/steps_summary/evaluation/9_evaluation.md)

## End-to-End Flow

`KG definition`
→ `KG standardization`
→ `GNN training and embeddings`
→ `QA definition`
→ `QA standardization`
→ `query transformation`
→ `retrieval`
→ `subgraph construction`
→ `textualization`
→ `LLM answering`
→ `answer/explanation output`
→ `evaluation`
