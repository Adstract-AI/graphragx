## Question-Answer Dataset Scope

In addition to the knowledge graph, the system requires a **question-answer (QA) dataset** to drive evaluation and inference.

The QA dataset consists of:
- Natural language questions
- Ground truth answers
- (Optionally) linked entities or query annotations

Unlike the KG dataset, the QA dataset does not define the structure of the world. Instead, it provides **queries over that world**, allowing the system to test its ability to:
- Retrieve relevant subgraphs
- Construct meaningful context
- Generate correct answers

It is important to note that QA datasets (e.g., WebQSP) operate on top of an underlying knowledge graph and do not replace it.

In this setup:
- The KG defines the knowledge
- The QA dataset probes that knowledge