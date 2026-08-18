# graphragX

`graphragX` is an experimental GraphRAG pipeline for knowledge-graph question answering. It combines graph-based candidate retrieval with GNN scoring, reasoning-subgraph extraction, LLM answer generation, final evaluation metrics, and optional Weights & Biases visualization.

The current implementation is focused on WebQSP. The pipeline prepares local graph data, trains a GNN answer retriever, evaluates candidate-answer retrieval, builds reasoning subgraphs, calls an LLM for final answers, computes final metrics, and saves all artifacts under `data/webqsp`.

This project was highly inspired by **GNN-RAG: Graph Neural Retrieval for Large Language Model Reasoning**: https://arxiv.org/pdf/2405.20139

For the full project explanation, read the final paper at [`metadata/GraphRagX.pdf`](metadata/GraphRagX.pdf). Architecture graphs, supporting reports, and metric explanations are also available in the [`metadata/`](metadata/) folder.

## Setup

Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/) if it is not already available, then create the project environment and install the locked dependencies:

```bash
uv sync --frozen
```

The repository pins Python 3.11 in `.python-version`. If Python 3.11 is not already installed, `uv` will install and manage it automatically. Run project commands through the managed environment:

```bash
uv run python main.py
uv run pytest
```

To activate the environment in the current shell instead, run `source .venv/bin/activate` on macOS/Linux or `.venv\Scripts\activate` on Windows.

The generated `uv.lock` file is the reproducible dependency source for remote machines. `--frozen` makes setup fail clearly if the project definition and lockfile ever drift instead of silently resolving different versions. After intentionally changing dependencies in `pyproject.toml`, update the lock and environment with `uv lock` followed by `uv sync`. The existing `requirements.txt` remains available for compatibility with pip-based environments.

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

W&B logging is enabled by default for every run mode. For first-time W&B usage, run:

```bash
wandb login
```

or set `WANDB_API_KEY` in your shell environment. If you do not want W&B for a run, use `--no-wandb`.

## Running The Pipeline

Run the main entrypoint:

```bash
uv run python main.py
```

With no arguments, the pipeline runs in full mode, but it still asks you to select configurable project options interactively. Full mode includes training, evaluation, LLM inference, final result computation, and W&B logging.

To use recommended default selections without interactive prompts:

```bash
uv run python main.py --default
```

To run the full pipeline without W&B:

```bash
uv run python main.py --default --no-wandb
```

To run only GNN training:

```bash
uv run python main.py --train-only --default
```

To evaluate a saved model run:

```bash
uv run python main.py --evaluation-only --evaluation-model-run-number 12 --default
```

To train and evaluate only the retriever:

```bash
uv run python main.py --retriever-only --default
```

To run the advanced architecture with its recommended feature set:

```bash
uv run python main.py --retriever-only --gnn-architecture aa-graphsage --default
```

To run LLM inference from an existing retriever evaluation:

```bash
uv run python main.py --inference-only --retriever-run-number 7 --default
```

## CLI Flags

### Run Mode

| Flag | Description |
| --- | --- |
| `--full` | Runs setup, GNN training, GNN evaluation, LLM inference, final results, and W&B logging. This is the default run mode. |
| `--train-only` | Runs dataset selection/configuration, dataset loading, local WebQSP graph preparation, GNN model construction, and GNN training. It stops after saving the trained model run. |
| `--retriever-only` | Trains the GNN, evaluates it on the test split, saves retriever predictions and metrics, and stops before LLM inference. |
| `--evaluation-only` | Evaluates a previously saved GNN model and then runs LLM inference/final results. Use this with `--evaluation-model-run-name` or `--evaluation-model-run-number`. |
| `--inference-only` | Loads a saved retriever evaluation and runs LLM inference/final results without loading or training the GNN. Use a retriever run selector. |

### Dataset And Pipeline Configuration

| Flag | Description |
| --- | --- |
| `--dataset DATASET` | Dataset id to use. The current supported dataset is `WebQSP`. |
| `--main-llm-model MAIN_LLM_MODEL` | LLM model id used for final answer generation. |
| `--subgraph-algorithm SUBGRAPH_ALGORITHM` | Subgraph construction algorithm. The current supported option is `shortest_path`. |
| `--context-strategy CONTEXT_STRATEGY` | How the reasoning subgraph is represented for the LLM. The current supported option is `structured_triples`. |
| `--gnn-architecture {graphsage,aa-graphsage,rgcn}` | Select GraphSAGE, Advance GraphSAGE, or R-GCN. GraphSAGE is the default; `aa-graphsage` and `rgcn` are the stable CLI/configuration ids. |
| `--gnn-layers {2,3}` | Number of GNN message-passing layers. Default: `2`. |
| `--gnn-hidden-dim {128,256,512}` | Hidden dimension used inside the GNN. Default: `256`. |
| `--node-classifier NODE_CLASSIFIER` | Node classifier head used after the GNN. Supported options include `mlp` and `linear`. |
| `--dropout {0.0,0.1,0.2,0.3,0.5}` | Shared architecture dropout. Default: `0.1`. |
| `--embedding-model EMBEDDING_MODEL` | OpenAI embedding model used consistently for question, relation, and entity text. |

### Training

| Flag | Description |
| --- | --- |
| `--training-epochs TRAINING_EPOCHS` | Number of GNN training epochs. |
| `--training-learning-rate TRAINING_LEARNING_RATE` | Learning rate for GNN training. |
| `--training-weight-decay TRAINING_WEIGHT_DECAY` | Weight decay for GNN training. |
| `--training-max-instances TRAINING_MAX_INSTANCES` | Optional limit for how many WebQSP training instances to use. If omitted, the full train split is used. |
| `--training-start-instance TRAINING_START_INSTANCE` | Zero-based train split index where training starts. With `--training-max-instances 100 --training-start-instance 101`, the slice is `[101:201]`. |
| `--training-log-every TRAINING_LOG_EVERY` | How often training progress is written to the console, measured in processed instances. Use `0` to disable progress messages. |
| `--training-batch-size TRAINING_BATCH_SIZE` | Number of WebQSP graphs combined into each disconnected R-GCN batch and optimizer step. Default: `16`. GraphSAGE retains its existing single-graph optimizer steps. |
| `--training-device {auto,cpu,cuda,mps}` | Device used for GNN training. `auto` selects the best available supported device. |
| `--training-profile` | Reports synchronized input, forward, loss, backward, and optimizer timings. Use only for short diagnostics because synchronization reduces throughput. |
| `--training-embedding-cache-device {auto,gpu,cpu}` | Placement for compact frozen embeddings prepared before training. `auto` uses CUDA when the matrices fit after the configured reserve. |
| `--training-embedding-cache-dtype {auto,float32,bfloat16}` | Storage precision for compact embeddings. `auto` uses BF16 on supported CUDA devices and float32 otherwise. |
| `--training-gpu-cache-reserve-gb TRAINING_GPU_CACHE_RESERVE_GB` | VRAM kept free for model parameters, graph activations, gradients, and CUDA overhead. Default: `6.0`. |
| `--training-run-name TRAINING_RUN_NAME` | Optional label for the saved training run folder. |
| `--continue-training-model-run-name CONTINUE_TRAINING_MODEL_RUN_NAME` | Continue training from a saved GNN model run folder name or suffix. Valid in full and train-only runs. |
| `--continue-training-model-run-number CONTINUE_TRAINING_MODEL_RUN_NUMBER` | Continue training from a saved GNN model run numeric prefix. Valid in full and train-only runs. |
| `--use-edge-mlp` / `--no-use-edge-mlp` | Enable or disable Advance GraphSAGE's trainable question-relation edge scorer. |
| `--question-aware-classifier` / `--no-question-aware-classifier` | Enable or disable Advance GraphSAGE's question-aware node head. A linear classifier requires the negative form. |
| `--use-reverse-edges` / `--no-use-reverse-edges` | Enable or disable reverse-edge graph preparation for Advance GraphSAGE. |
| `--add-layer-normalization` / `--no-add-layer-normalization` | Enable or disable Advance GraphSAGE residual LayerNorm blocks. |
| `--edge-mlp-hidden-dim {128,256,512}` | Advance GraphSAGE edge-MLP width. Valid only when edge MLP is enabled. Default: `256`. |
| `--num-bases {8,16,30,64}` | Number of shared relation-weight bases for R-GCN. Default: `30`. |

R-GCN always prepares distinct inverse relation types and uses categorical relation-specific transformations with a trainable root transform. Its configurable options are layers, hidden dimension, dropout, and basis count; GraphSAGE classifier, edge-MLP, question-conditioning, normalization, edge-width, and reverse-edge flags are rejected. The existing GraphSAGE variants retain their semantic question–relation scalar weighting behavior.

#### Adding another GNN architecture

GNN configuration is registry-driven. Each `GnnArchitectureDefinition` owns:

- Its complete `GnnArchitectureOptionDefinition` list, including CLI flags, types, choices, defaults, interactive labels, and conditional visibility.
- A lazy `model_builder_path` callback that constructs the architecture without importing PyTorch during CLI setup.
- An optional `validator_path` callback for relationships between its options.

After registering a new definition in `GNN_ARCHITECTURES`, the CLI union and interactive prompts are generated automatically. Architecture-specific values are carried in `gnn_architecture_options`, persisted in model and training configurations, restored for evaluation or continuation, and exposed to the registered model builder. The central argument parser and configuration step do not need architecture-specific branches.

Before the epoch loop, training deduplicates embeddings used by the selected instance slice and builds compact integer-indexed matrices. Retrieved vectors are also persisted under `data/webqsp/training_embedding_tensors` as append-only local tensor shards. A full local hit bypasses Qdrant; a partial hit retrieves and appends only vectors that have not been persisted yet. For example, training first on 100 instances and then on 300 reuses the vectors from the first run and fills only embeddings introduced by the additional 200 instances. Separate local caches are maintained for each dataset, embedding model, text category, vector dimension, and storage dtype.

R-GCN additionally precomputes relation-mean normalization and compact active-relation indices. Multiple question graphs are combined as disconnected components, and the vectorized layer constructs active relation transforms once per batch before processing edge messages in bounded chunks. Static batch graph tensors remain on the training device across epochs. Use `--training-batch-size 1` for the original per-graph optimizer-step behavior or to compare throughput.

The compact matrices are still copied into VRAM at the start of every process because GPU memory is not persistent across runs. GPU-resident matrices remain frozen, are excluded from the optimizer and model checkpoint, and are released when training finishes. If the safe CUDA memory budget is exceeded in `auto` mode, the matrices remain on CPU.

### GNN Evaluation

| Flag | Description |
| --- | --- |
| `--evaluation-model-run-name EVALUATION_MODEL_RUN_NAME` | Saved model run folder name or suffix to evaluate. |
| `--evaluation-model-run-number EVALUATION_MODEL_RUN_NUMBER` | Saved model run numeric prefix to evaluate. |
| `--answer-threshold ANSWER_THRESHOLD` | Minimum answer-node probability for threshold candidate selection. |
| `--candidate-top-k CANDIDATE_TOP_K` | Minimum number of selected candidates when threshold selection returns too few. |
| `--candidate-limit CANDIDATE_LIMIT` | Maximum number of selected answer candidates after threshold and top-k selection. `--limit` is an alias. |
| `--evaluation-run-name EVALUATION_RUN_NAME` | Optional label for the saved evaluation run folder. |
| `--retriever-run-name RETRIEVER_RUN_NAME` | Saved retriever evaluation folder name or suffix required by inference-only mode. |
| `--retriever-run-number RETRIEVER_RUN_NUMBER` | Saved retriever evaluation numeric prefix required by inference-only mode. |
| `--evaluation-max-instances EVALUATION_MAX_INSTANCES` | Optional limit for how many WebQSP test instances to evaluate. If omitted, the full test split is used. |
| `--evaluation-log-every EVALUATION_LOG_EVERY` | How often GNN evaluation progress is logged, measured in evaluated instances. |
| `--evaluation-profile` | Reports synchronized model loading, embedding preparation, input, forward, prediction, and persistence timings. Use for short diagnostics because synchronization reduces throughput. |
| `--evaluation-embedding-cache-device {auto,gpu,cpu}` | Placement for compact frozen evaluation embeddings. `auto` uses CUDA when the matrices fit after the configured reserve. |
| `--evaluation-embedding-cache-dtype {auto,float32,bfloat16}` | Storage precision for compact evaluation embeddings. `auto` uses BF16 on supported CUDA devices and float32 otherwise. |
| `--evaluation-gpu-cache-reserve-gb EVALUATION_GPU_CACHE_RESERVE_GB` | VRAM kept free outside the evaluation embedding matrices. Default: `6.0`. |

Evaluation compacts the selected test instances into reusable embedding matrices before model inference. GraphSAGE loads node, relation, and question embeddings; R-GCN loads only node embeddings and uses the saved categorical relation vocabulary. It uses the same incremental tensor shards as training, so matching dataset/model/type/dtype vectors are reused immediately. Only missing vectors are fetched from Qdrant and appended; no Qdrant requests occur inside the per-instance inference loop. Evaluation uses `torch.inference_mode()` and BF16 autocast when BF16 cache storage is selected on CUDA.

### LLM Inference And Results

| Flag | Description |
| --- | --- |
| `--no-llm-inference` | Stops full or evaluation-only mode after GNN candidate retrieval. Training and retriever W&B logging still run unless `--no-wandb` is supplied. |
| `--inference-run-name INFERENCE_RUN_NAME` | Optional label for the saved LLM inference run folder. |
| `--llm-inference-batch-size LLM_INFERENCE_BATCH_SIZE` | Number of samples to process per persistence batch during LLM inference. The LLM calls remain one-by-one. |

### W&B

| Flag | Description |
| --- | --- |
| `--no-wandb` | Skips W&B upload. Local result files are still saved. |
| `--wandb-project WANDB_PROJECT` | W&B project name. Defaults to `WANDB_PROJECT` from the environment, then `graphragx`. |
| `--wandb-entity WANDB_ENTITY` | Optional W&B entity/team. Defaults to `WANDB_ENTITY` from the environment. |
| `--wandb-mode {online,offline,disabled}` | W&B mode. Defaults to `WANDB_MODE` from the environment, then `online`. |
| `--wandb-training-log-every WANDB_TRAINING_LOG_EVERY` | How often live training loss is sent to W&B, measured in processed instances. Use `0` to disable live loss events. |

New W&B runs use a dataset-wide sequential identifier in the form `run_number_YYYYMMDD_HHMMSS`, independent of which pipeline mode creates them. Full, training, and retriever stages reuse their logical experiment within the command. Every evaluation-only command creates a new W&B run and copies the selected model's training metrics, configuration, tags, and artifact before adding retrieval and optional inference results. Every inference-only command creates a new W&B run and copies the selected retriever metrics and configuration into it. This keeps repeated evaluations and LLM inference runs independently comparable without modifying their upstream W&B runs. If an older artifact has no W&B lineage, the pipeline creates a run and backfills the available upstream metrics and artifacts.

W&B tags are populated incrementally from the stages available in each mode. Depending on the completed stages, tags include the dataset, GNN architecture (`graphsage`, `aa-graphsage`, or `rgcn`), LLM id, embedding models, trained/evaluated instance counts, and model, evaluation, and inference run numbers. Resumed runs preserve their existing tags, and duplicate values are removed.

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

GNN training outputs, including `model_config.json`, model weights, and loss history. R-GCN runs also contain the authoritative `relation_vocabulary.json` used to construct categorical edge types.

`data/webqsp/training_embedding_tensors`

Incremental append-only tensor shards used to bypass Qdrant for embeddings already loaded by previous training runs.

`data/webqsp/evaluations/<run>`

GNN retrieval evaluation outputs, including `evaluation_config.json` and `predictions.jsonl`.

`data/webqsp/inference/<run>`

LLM inference outputs, including `inference_config.json`, `answers.jsonl`, and `reasoning.jsonl`.

`data/webqsp/results/<run>`

Final result outputs, including `results_config.json`, retrieval metrics, reasoning/answer metrics, and per-instance metrics.

## Project Structure

```text
graphragx/
├── main.py
├── pyproject.toml
├── uv.lock
├── .python-version
├── requirements.txt
├── docker-compose.yml
├── README.md
├── LICENSE
├── CONTRIBUTING.md
├── helpers/
│   ├── constants.py
│   ├── env_variables.py
│   ├── logging_config.py
│   ├── openai_rate_limit_logging.py
│   └── path_serialization.py
├── pipeline/
│   ├── abstract.py
│   ├── context_builder.py
│   ├── exceptions.py
│   ├── pipeline.py
│   ├── services.py
│   ├── preparation/
│   │   ├── exceptions/
│   │   ├── helpers/
│   │   ├── models/
│   │   ├── services/
│   │   └── steps/
│   └── evaluation/
│       ├── exceptions/
│       ├── models/
│       ├── services/
│       └── steps/
├── mappings/
│   └── webqsp/
├── metadata/
│   ├── GraphRagX.pdf
│   ├── metrics/
│   │   ├── prediction_metrics.md
│   │   └── retrieval_metrics.md
│   └── other/
│       ├── REPORT.pdf
│       ├── graphragx-final.drawio.png
│       └── graphragx.drawio
├── agents_metadata/
│   ├── guidlines/
│   │   ├── ERROR_HANDLING_GUIDELINES.MD
│   │   ├── GENERAL_GUIDELINES.MD
│   │   ├── PROJECT_GUIDELINES.MD
│   │   ├── PROJECT_OVERVIEW.md
│   │   └── SERVICE_GUIDELINES.MD
│   └── pipeline_overview/
│       ├── PIPELINE_OVERVIEW.md
│       ├── preparation/
│       └── evaluation/
├── tests/
│   ├── preparation/
│   └── evaluation/
└── data/
    └── webqsp/
        ├── processed/
        ├── processed_reverse_edges/
        ├── models/
        ├── evaluations/
        ├── inference/
        └── results/
```

`main.py` is the CLI entrypoint and composes the full pipeline. `pipeline/` contains the actual preparation and evaluation steps, services, models, exceptions, and pipeline runner. `helpers/` contains shared constants, environment handling, logging, rate-limit visibility, and path serialization utilities.

`metadata/` is the human-facing project documentation folder. It contains the final paper, project report, architecture diagrams, and metric explanations. `agents_metadata/` is the agent-facing context folder; contributors using AI coding agents should start there to understand conventions, expected service structure, error handling, and the intended pipeline flow.

`data/webqsp/` is generated locally during runs. It contains processed graph caches, trained model runs, evaluation outputs, LLM inference outputs, and final result folders.

## Contributing With Agents

If you want to contribute with coding agents, first read `agents_metadata/guidlines/PROJECT_GUIDELINES.MD` and `agents_metadata/pipeline_overview/PIPELINE_OVERVIEW.md`.

The `agents_metadata/` folder explains the expected project conventions, how services and steps should be structured, how errors should be handled, and how the pipeline is intended to flow. Give those files to the agent as context before asking it to change the project.

## Credits

This project was made by **Andrea Stevanoska** and **Viktor Kostadinoski**.

It was supervised by the TA **M.Sc. Martina Toshevska** and the Professor: **PhD Sonja Gievska**.

All contributors and supervisors are part of **FINKI, the Faculty of Computer Science and Engineering in Skopje**.

## License

This project is released under the MIT License. See `LICENSE`.
