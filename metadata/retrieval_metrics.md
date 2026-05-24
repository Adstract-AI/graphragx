# Retrieval Metrics

These metrics describe the GNN answer-retriever step. They are computed in the final results step from `predictions.jsonl` and saved as `data/webqsp/results/<run>/retrieval_metrics.json`.

## Metadata Fields

### `dataset_id`
Dataset evaluated in the run.

### `model_run_name`
Saved GNN model run used for retrieval.

### `model_run_number`
Numeric run id of the saved GNN model.

### `evaluation_run_name`
Saved GNN evaluation run that produced `predictions.jsonl`.

### `evaluation_run_number`
Numeric run id of the GNN evaluation run.

### `evaluated_instances`
Number of questions evaluated.

## Retrieval Quality Metrics

### `hits_at_1`
Share of questions where the top-ranked retrieved candidate is a gold answer.

Per question:

```text
Hits@1 = 1 if the first/highest-probability candidate is in gold_answers
Hits@1 = 0 otherwise
```

Aggregate:

```text
hits_at_1 = hits_at_1_count / evaluated_instances
```

### `hits_at_1_count`
Number of questions where `Hits@1 = 1`.

### `hit_at_k`
Share of questions where at least one selected retrieved candidate is a gold answer.

Per question:

```text
Hit@K = 1 if any retrieved answer candidate is in gold_answers
Hit@K = 0 otherwise
```

Aggregate:

```text
hit_at_k = hit_at_k_count / evaluated_instances
```

Here `K` is the number of selected candidates for that question, after threshold/top-k fallback and candidate limit filtering.

### `hit_at_k_count`
Number of questions where `Hit@K = 1`.

### `average_candidate_count`
Average number of retrieved answer candidates per question.

```text
average_candidate_count = total retrieved candidates / evaluated_instances
```

### `missing_gold_in_graph_count`
Number of questions where none of the gold answers are present in the local graph. These questions cannot be retrieved correctly by the GNN because the correct answer is absent from the candidate graph.
