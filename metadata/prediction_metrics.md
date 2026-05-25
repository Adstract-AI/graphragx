# Prediction Metrics

These metrics describe the final generated answers and reasoning outputs. They are computed in the final results step from `answers.jsonl`, `reasoning.jsonl`, and retrieval `predictions.jsonl`.

Aggregate metrics are saved as `data/webqsp/results/<run>/reasoning_metrics.json`. Per-question rows are saved as `data/webqsp/results/<run>/per_instance_results.jsonl`.

## Metadata Fields

### `dataset_id`
Dataset evaluated in the run.

### `evaluation_run_name`
GNN evaluation run used as the retrieval source.

### `inference_run_name`
LLM inference run used as the answer source.

### `model_run_name`
Saved GNN model run used for retrieval.

### `model_id`
LLM model used to generate final answers.

### `evaluated_instances`
Number of questions evaluated.

### `successful_answers`
Number of answer rows without an LLM generation error.

### `failed_answers`
Number of answer rows with an LLM generation error.

Failed rows are treated as having no predicted answer.

## Answer Matching Metrics

Before matching, answers are normalized by lowercasing, trimming, removing simple punctuation/articles, collapsing whitespace, and splitting comma-separated answer strings. `Unknown`, empty answers, and failed rows count as no prediction.

### `accuracy`
Exact-match accuracy over normalized answer sets.

Per question:

```text
exact_match = 1 if normalized_predicted_answers == normalized_gold_answers
exact_match = 0 otherwise
```

Aggregate:

```text
accuracy = exact_match_count / evaluated_instances
```

### `exact_match_count`
Number of questions where the full normalized predicted answer set exactly matches the gold answer set.

### `hit_rate`
Share of questions where any predicted/generated answer is correct.

Per question:

```text
hit = 1 if any predicted answer appears in gold_answers
hit = 0 otherwise
```

Aggregate:

```text
hit_rate = hit_count / evaluated_instances
```

### `hit_count`
Number of questions where at least one generated answer matches a gold answer.

### `hits_at_1`
Share of questions where the first generated answer is correct.

Per question:

```text
Hits@1 = 1 if the first predicted answer appears in gold_answers
Hits@1 = 0 otherwise
```

Aggregate:

```text
hits_at_1 = hits_at_1_count / evaluated_instances
```

For comma-separated LLM answers, the first comma-separated answer is treated as the top prediction.

### `hits_at_1_count`
Number of questions where the first generated answer is correct.

### `precision`
Micro precision over normalized answer sets.

```text
precision = true_positive_count / (true_positive_count + false_positive_count)
```

### `recall`
Micro recall over normalized answer sets.

```text
recall = true_positive_count / (true_positive_count + false_negative_count)
```

### `f1`
Micro F1 score computed from precision and recall.

```text
f1 = 2 * precision * recall / (precision + recall)
```

If the denominator is zero, F1 is `0.0`.

### `true_positive_count`
Total number of normalized predicted answers that appear in the normalized gold answer sets.

### `false_positive_count`
Total number of normalized predicted answers that do not appear in the normalized gold answer sets.

### `false_negative_count`
Total number of normalized gold answers that were not predicted.

## Explanation Grounding Metrics

These metrics check whether triples mentioned in the generated explanation are actually present in the reasoning subgraph from `reasoning.jsonl`.

### `grounded_explanation_rate`
Share of questions where at least one explanation-mentioned triple is found in the reasoning subgraph.

```text
grounded_explanation_rate = grounded_explanation_count / evaluated_instances
```

### `grounded_explanation_count`
Number of questions with at least one grounded explanation triple.

### `fully_grounded_explanation_rate`
Share of questions where every explanation-mentioned triple is found in the reasoning subgraph.

```text
fully_grounded_explanation_rate = fully_grounded_explanation_count / evaluated_instances
```

### `fully_grounded_explanation_count`
Number of questions where all mentioned explanation triples are grounded.

### `mentioned_triple_count`
Total number of triples mentioned across generated explanations.

### `grounded_mentioned_triple_count`
Total number of explanation-mentioned triples that are present in the reasoning subgraphs.

## Ranking Metrics

These are retrieval-ranking metrics included with final prediction results for end-to-end reporting. They are computed from GNN candidate probabilities and binary relevance from `is_gold_answer`.

### `ndcg_at_1`
Mean normalized discounted cumulative gain at rank 1.

### `ndcg_at_5`
Mean normalized discounted cumulative gain at rank 5.

### `ndcg_at_10`
Mean normalized discounted cumulative gain at rank 10.

### `ndcg_at_candidate_limit`
Mean normalized discounted cumulative gain at the configured candidate limit.

### `candidate_limit`
Candidate limit from the GNN evaluation configuration.

For each question:

```text
DCG@K = sum(relevance_i / log2(i + 1))
IDCG@K = best possible DCG@K for the same relevance labels
nDCG@K = DCG@K / IDCG@K
```

where `i` is the 1-based rank and relevance is `1` when the candidate is a gold answer, otherwise `0`. If there are no gold candidates, nDCG is `0.0`.
