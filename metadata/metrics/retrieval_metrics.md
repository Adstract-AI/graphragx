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

### `hits_at_5`
Share of questions where at least one of the top 5 retrieved candidates is a gold answer.

Per question:

```text
Hits@5 = 1 if any of the first 5 retrieved answer candidates is in gold_answers
Hits@5 = 0 otherwise
```

Aggregate:

```text
hits_at_5 = hits_at_5_count / evaluated_instances
```

### `hits_at_5_count`
Number of questions where `Hits@5 = 1`.

### `hits_at_10`
Share of questions where at least one of the top 10 retrieved candidates is a gold answer.

Per question:

```text
Hits@10 = 1 if any of the first 10 retrieved answer candidates is in gold_answers
Hits@10 = 0 otherwise
```

Aggregate:

```text
hits_at_10 = hits_at_10_count / evaluated_instances
```

### `hits_at_10_count`
Number of questions where `Hits@10 = 1`.

### `hits_at_candidate_limit`
Share of questions where at least one retrieved candidate within the configured
`candidate_limit` is a gold answer.

Per question:

```text
Hits@candidate_limit = 1 if any selected answer candidate is in gold_answers
Hits@candidate_limit = 0 otherwise
```

Aggregate:

```text
hits_at_candidate_limit = hits_at_candidate_limit_count / evaluated_instances
```

### `hits_at_candidate_limit_count`
Number of questions where `Hits@candidate_limit = 1`.

### `candidate_limit`
The configured maximum number of answer candidates retained by the retriever.

### `average_candidate_count`
Average number of retrieved answer candidates per question.

```text
average_candidate_count = total retrieved candidates / evaluated_instances
```

### `missing_gold_in_graph_count`
Number of questions where none of the gold answers are present in the local graph. These questions cannot be retrieved correctly by the GNN because the correct answer is absent from the candidate graph.
