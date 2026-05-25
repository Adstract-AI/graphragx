# graphragX

`graphragX` is an experimental GraphRAG pipeline for knowledge-graph question answering. It combines graph-based candidate retrieval with GNN scoring, reasoning-subgraph extraction, LLM answer generation, final evaluation metrics, and optional Weights & Biases visualization.

The current implementation is focused on WebQSP. The pipeline prepares local graph data, trains a GNN answer retriever, evaluates candidate-answer retrieval, builds reasoning subgraphs, calls an LLM for final answers, computes final metrics, and saves all artifacts under `data/webqsp`.

This project was highly inspired by **GNN-RAG: Graph Neural Retrieval for Large Language Model Reasoning**: https://arxiv.org/pdf/2405.20139

For the full project explanation, the project paper, architecture graphs, and metric explanations, see the `metadata/` folder.

## Setup

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create your local environment file:

```bash
cp .env.example .env
```

Fill in at least:

```bash
OPENAI_API_KEY=your_openai_key
```

Start Qdrant before running any training or evaluation step that uses embeddings:

```bash
docker compose up -d qdrant
```

By default the pipeline connects to `http://localhost:6333` and stores embeddings in Qdrant collections prefixed with `graphragx_embeddings`.

W&B logging is enabled by default for full runs. For first-time W&B usage, run:

```bash
wandb login
```

or set `WANDB_API_KEY` in your shell environment. If you do not want W&B for a run, use `--no-wandb`.

## Running The Pipeline

Run the main entrypoint:

```bash
python main.py
```

With no arguments, the pipeline runs in full mode, but it still asks you to select configurable project options interactively. Full mode includes training, evaluation, LLM inference, final result computation, and W&B logging.

To use recommended default selections without interactive prompts:

```bash
python main.py --default
```

To run the full pipeline without W&B:

```bash
python main.py --default --no-wandb
```

To run only GNN training:

```bash
python main.py --train-only --default
```

To evaluate a saved model run:

```bash
python main.py --evaluation-only --evaluation-model-run-number 12 --default
```

## CLI Flags

### Run Mode

| Flag | Description |
| --- | --- |
| `--full` | Runs setup, GNN training, GNN evaluation, LLM inference, final results, and W&B logging. This is the default run mode. |
| `--train-only` | Runs dataset selection/configuration, dataset loading, local WebQSP graph preparation, GNN model construction, and GNN training. It stops after saving the trained model run. |
| `--evaluation-only` | Runs setup and evaluates a previously saved GNN model run. Use this with `--evaluation-model-run-name` or `--evaluation-model-run-number`. |

### Dataset And Pipeline Configuration

| Flag | Description |
| --- | --- |
| `--dataset DATASET` | Dataset id to use. The current supported dataset is `WebQSP`. |
| `--main-llm-model MAIN_LLM_MODEL` | LLM model id used for final answer generation. |
| `--subgraph-algorithm SUBGRAPH_ALGORITHM` | Subgraph construction algorithm. The current supported option is `shortest_path`. |
| `--context-strategy CONTEXT_STRATEGY` | How the reasoning subgraph is represented for the LLM. The current supported option is `structured_triples`. |
| `--gnn-layers GNN_LAYERS` | Number of GNN message-passing layers. |
| `--gnn-hidden-dim GNN_HIDDEN_DIM` | Hidden dimension used inside the GNN. |
| `--node-classifier NODE_CLASSIFIER` | Node classifier head used after the GNN. Supported options include `mlp` and `linear`. |
| `--question-embedding-model QUESTION_EMBEDDING_MODEL` | OpenAI embedding model used for question text. |
| `--relation-embedding-model RELATION_EMBEDDING_MODEL` | OpenAI embedding model used for relation text. |
| `--entity-embedding-model ENTITY_EMBEDDING_MODEL` | OpenAI embedding model used for entity text. |

### Training

| Flag | Description |
| --- | --- |
| `--training-epochs TRAINING_EPOCHS` | Number of GNN training epochs. |
| `--training-learning-rate TRAINING_LEARNING_RATE` | Learning rate for GNN training. |
| `--training-weight-decay TRAINING_WEIGHT_DECAY` | Weight decay for GNN training. |
| `--training-max-instances TRAINING_MAX_INSTANCES` | Optional limit for how many WebQSP training instances to use. If omitted, the full train split is used. |
| `--training-log-every TRAINING_LOG_EVERY` | How often training progress is logged, measured in processed instances. |
| `--training-device {auto,cpu,cuda,mps}` | Device used for GNN training. `auto` selects the best available supported device. |
| `--training-run-name TRAINING_RUN_NAME` | Optional label for the saved training run folder. |

### GNN Evaluation

| Flag | Description |
| --- | --- |
| `--evaluation-model-run-name EVALUATION_MODEL_RUN_NAME` | Saved model run folder name or suffix to evaluate. |
| `--evaluation-model-run-number EVALUATION_MODEL_RUN_NUMBER` | Saved model run numeric prefix to evaluate. |
| `--answer-threshold ANSWER_THRESHOLD` | Minimum answer-node probability for threshold candidate selection. |
| `--candidate-top-k CANDIDATE_TOP_K` | Minimum number of selected candidates when threshold selection returns too few. |
| `--candidate-limit CANDIDATE_LIMIT` | Maximum number of selected answer candidates after threshold and top-k selection. `--limit` is an alias. |
| `--evaluation-run-name EVALUATION_RUN_NAME` | Optional label for the saved evaluation run folder. |
| `--evaluation-max-instances EVALUATION_MAX_INSTANCES` | Optional limit for how many WebQSP test instances to evaluate. If omitted, the full test split is used. |
| `--evaluation-log-every EVALUATION_LOG_EVERY` | How often GNN evaluation progress is logged, measured in evaluated instances. |

### LLM Inference And Results

| Flag | Description |
| --- | --- |
| `--no-llm-inference` | Stops after GNN candidate retrieval and skips reasoning-subgraph extraction, LLM answer generation, final results, and W&B logging. If this is used, also pass `--no-wandb`. |
| `--inference-run-name INFERENCE_RUN_NAME` | Optional label for the saved LLM inference run folder. |
| `--llm-inference-batch-size LLM_INFERENCE_BATCH_SIZE` | Number of samples to process per persistence batch during LLM inference. The LLM calls remain one-by-one. |

### W&B

| Flag | Description |
| --- | --- |
| `--no-wandb` | Skips W&B upload. Local result files are still saved. |
| `--wandb-project WANDB_PROJECT` | W&B project name. Defaults to `WANDB_PROJECT` from the environment, then `graphragx`. |
| `--wandb-entity WANDB_ENTITY` | Optional W&B entity/team. Defaults to `WANDB_ENTITY` from the environment. |
| `--wandb-mode {online,offline,disabled}` | W&B mode. Defaults to `WANDB_MODE` from the environment, then `online`. |

### Execution Helpers

| Flag | Description |
| --- | --- |
| `--default` | Uses recommended default values for configurable selections instead of prompting interactively. |
| `--force-default` | Forces every pipeline step to use its default execution path. This is mostly useful for tests and controlled runs. |

## Outputs

Pipeline outputs are saved under `data/webqsp`:

`data/webqsp/processed`

Processed WebQSP graph cache and vocabulary artifacts.

`data/webqsp/models/<run>`

GNN training outputs, including `model_config.json`, model weights, and loss history.

`data/webqsp/evaluations/<run>`

GNN retrieval evaluation outputs, including `evaluation_config.json` and `predictions.jsonl`.

`data/webqsp/inference/<run>`

LLM inference outputs, including `inference_config.json`, `answers.jsonl`, and `reasoning.jsonl`.

`data/webqsp/results/<run>`

Final result outputs, including `results_config.json`, retrieval metrics, reasoning/answer metrics, and per-instance metrics.

## Project Structure

`main.py`

CLI entrypoint and pipeline composition.

`pipeline/`

Core pipeline implementation. It contains preparation steps, evaluation steps, services, models, exceptions, and the pipeline runner.

`helpers/`

Shared constants, environment-variable handling, logging, and serialization utilities.

`metadata/`

Human-facing project materials: the project report, architecture diagrams, and metric explanations.

`agents_metadata/`

Agent-facing project context. Contributors using AI coding agents should start here. The folder contains project guidelines, service guidelines, error-handling rules, general conventions, and a pipeline overview. Use it to understand how agents should explore the codebase, preserve conventions, and continue work safely.

`tests/`

Unit and integration-style tests for the pipeline, services, metrics, W&B logging, and storage behavior.

`mappings/`

Static mapping files used by the WebQSP processing flow.

`data/`

Local generated data, model runs, evaluations, inference runs, and final results.

## Contributing With Agents

If you want to contribute with coding agents, first read `agents_metadata/guidlines/PROJECT_GUIDELINES.MD` and `agents_metadata/pipeline_overview/PIPELINE_OVERVIEW.md`.

The `agents_metadata/` folder explains the expected project conventions, how services and steps should be structured, how errors should be handled, and how the pipeline is intended to flow. Give those files to the agent as context before asking it to change the project.

## Credits

This project was made by **Andrea Stevanoska** and **Viktor Kostadinoski**.

It was supervised by the TA **M.Sc. Martina Toshevska** and the Professor: **PhD Sonja Gievska**.

All contributors and supervisors are part of **FINKI, the Faculty of Computer Science and Engineering in Skopje**.

## License

This project is released under the MIT License. See `LICENSE`.
