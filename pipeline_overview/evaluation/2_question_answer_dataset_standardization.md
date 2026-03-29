## Question-Answer Dataset Standardization

The goal of this step is to transform raw QA data into a consistent format suitable for inference and evaluation.

QA datasets may vary in structure, but typically include questions, answers, and optionally linked entities or query structures. These must be normalized into a unified representation.

This involves:
- Cleaning and standardizing question text
- Normalizing answer formats (e.g., entity IDs, strings)
- Aligning question entities with the standardized KG (entity linking or ID mapping)
- Defining a consistent schema for each sample (question, target entities, expected answers)

The output is a structured QA dataset where each query can be directly fed into the inference pipeline.

This standardized format ensures that:
- Questions can be transformed into graph queries
- Retrieved subgraphs can be evaluated against ground truth
- Metrics and comparisons remain consistent across experiments