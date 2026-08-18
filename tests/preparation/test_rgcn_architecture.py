"""R-GCN registry, layer, vocabulary, and preparation tests."""

from __future__ import annotations

from types import SimpleNamespace
import json

import pytest
import main

torch = pytest.importorskip("torch")

from pipeline.abstract import StepContext
from pipeline.preparation.exceptions import (
    InvalidGnnArchitectureConfigurationException,
)
from pipeline.preparation.helpers.configuration_definitions import GNN_ARCHITECTURES
from pipeline.preparation.helpers.gnn_architecture import architecture_defaults
from pipeline.preparation.models.rgcn_answer_retriever import (
    ActiveRelationBasisRGCNLayer,
    RGCNAnswerRetriever,
)
from pipeline.preparation.models.webqsp_local_graph import (
    PreparedWebQSPGraphDataset,
    WebQSPProcessedInstance,
    WebQSPVocabularyStore,
)
from pipeline.preparation.services.gnn_relation_vocabulary import (
    build_relation_architecture_context,
    build_sorted_typed_edges,
    validate_relation_architecture_context,
)
from pipeline.preparation.services.gnn_training_data_preparation import (
    GnnTrainingDataPreparationConfig,
    GnnTrainingDataPreparationService,
)
from pipeline.preparation.services.gnn_evaluation_data_preparation import (
    GnnEvaluationDataPreparationService,
)
from pipeline.evaluation.models import GnnAnswerRetrieverEvaluationConfig
from pipeline.evaluation.services.model_config_normalization import (
    normalize_model_config,
)
from pipeline.evaluation.services.wandb_experiment import WandbExperimentCoordinator
from pipeline.preparation.services.gnn_answer_retriever_model_runs import (
    GnnAnswerRetrieverModelRunService,
)
from pipeline.preparation.exceptions import GnnAnswerRetrieverModelRunException
from pipeline.preparation.steps.configuration_building import (
    BuildPipelineConfigurationStep,
    BuiltPipelineConfiguration,
)
from pipeline.preparation.steps.dataset_selection import SelectedDataset
from pipeline.preparation.steps.gnn_model_building import BuiltGnnAnswerRetriever


def _dataset_context() -> StepContext[SelectedDataset]:
    return StepContext(
        result=SelectedDataset(
            dataset_id="WebQSP",
            display_name="WebQSP",
            dataset_family="question_answering",
            task_domain="knowledge_graph_question_answering",
            description="dataset",
            supported=True,
        )
    )


def _pipeline_config(options: dict) -> BuiltPipelineConfiguration:
    return BuiltPipelineConfiguration(
        dataset_id="WebQSP",
        gnn_architecture="rgcn",
        gnn_architecture_options=options,
        main_llm_model="gpt-5.4",
        subgraph_construction_algorithm="shortest_path",
        context_construction_strategy="structured_triples",
        embedding_model="text-embedding-3-small",
        question_embedding_model="text-embedding-3-small",
        relation_embedding_model="text-embedding-3-small",
        entity_embedding_model="text-embedding-3-small",
        use_reverse_edges=True,
    )


def test_rgcn_registry_defaults_and_requirements() -> None:
    defaults = architecture_defaults("rgcn")
    definition = GNN_ARCHITECTURES["rgcn"]

    assert defaults == {
        "gnn_layer_count": 2,
        "gnn_hidden_dimension": 256,
        "dropout": 0.1,
        "num_bases": 30,
        "gnn_architecture": "rgcn",
    }
    assert definition.display_name == "R-GCN"
    assert definition.data_requirements.requires_reverse_edges
    assert definition.data_requirements.uses_relation_types
    assert not definition.data_requirements.uses_question_embeddings
    assert not definition.data_requirements.uses_relation_embeddings


def test_rgcn_cli_exposes_only_its_registered_numeric_option() -> None:
    args = main.build_parser().parse_args(
        ["--gnn-architecture", "rgcn", "--num-bases", "16"]
    )

    assert args.gnn_architecture == "rgcn"
    assert args.num_bases == 16


def test_rgcn_interactive_prompts_follow_registry_order(capsys) -> None:
    answers = iter(["3", "1", "2", "2", "3"])
    prompts: list[str] = []

    def choose(prompt: str) -> str:
        prompts.append(prompt)
        return next(answers)

    result = BuildPipelineConfigurationStep(
        main_llm_model="gpt-5.4",
        subgraph_algorithm="shortest_path",
        context_strategy="structured_triples",
        embedding_model="text-embedding-3-small",
        input_func=choose,
    ).execute(_dataset_context())

    assert result.gnn_architecture == "rgcn"
    assert result.gnn_architecture_options == {
        "gnn_layer_count": 2,
        "gnn_hidden_dimension": 256,
        "dropout": 0.1,
        "num_bases": 30,
    }
    prompt_text = capsys.readouterr().out
    assert prompt_text.index("GNN Architecture") < prompt_text.index("GNN Layer Count")
    assert prompt_text.index("GNN Layer Count") < prompt_text.index(
        "GNN Hidden Dimension"
    )
    assert prompt_text.index("GNN Hidden Dimension") < prompt_text.index("GNN Dropout")
    assert prompt_text.index("GNN Dropout") < prompt_text.index("R-GCN Basis Count")
    assert "Node Classifier" not in prompt_text


@pytest.mark.parametrize(
    ("option_id", "value"),
    [
        ("node_classifier", "mlp"),
        ("use_reverse_edges", True),
        ("use_reverse_edges", False),
        ("use_edge_mlp", False),
        ("question_aware_classifier", False),
        ("add_layer_normalization", False),
        ("edge_mlp_hidden_dim", 256),
    ],
)
def test_rgcn_rejects_graphsage_specific_options(option_id: str, value) -> None:
    step = BuildPipelineConfigurationStep(
        gnn_architecture="rgcn",
        gnn_options={
            "gnn_layer_count": 2,
            "gnn_hidden_dimension": 256,
            "dropout": 0.1,
            "num_bases": 30,
            option_id: value,
        },
    )

    with pytest.raises(
        InvalidGnnArchitectureConfigurationException,
        match=f"does not support: {option_id}",
    ):
        step.execute(_dataset_context())


def test_rgcn_configuration_makes_reverse_edges_mandatory() -> None:
    result = BuildPipelineConfigurationStep(
        gnn_architecture="rgcn",
        gnn_options={
            "gnn_layer_count": 2,
            "gnn_hidden_dimension": 256,
            "dropout": 0.1,
            "num_bases": 30,
        },
        main_llm_model="gpt-5.4",
        subgraph_algorithm="shortest_path",
        context_strategy="structured_triples",
        embedding_model="text-embedding-3-small",
    ).execute(_dataset_context())

    assert result.use_reverse_edges
    assert result.node_classifier is None


def test_typed_edges_are_sorted_and_keep_inverse_relations_distinct() -> None:
    vocabulary = {"reverse__parent": 0, "parent": 1}
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])

    sorted_edges, edge_type = build_sorted_typed_edges(
        edge_index=edge_index,
        edge_relations=["parent", "reverse__parent", "parent"],
        vocabulary=vocabulary,
        torch=torch,
    )

    assert edge_type.tolist() == [0, 1, 1]
    assert sorted_edges.tolist() == [[1, 0, 2], [2, 1, 0]]
    context = build_relation_architecture_context(vocabulary)
    validate_relation_architecture_context(context, vocabulary)
    with pytest.raises(ValueError, match="relation_vocabulary_sha256"):
        validate_relation_architecture_context(
            context,
            {"reverse__parent": 1, "parent": 0},
        )


def test_rgcn_layer_uses_per_relation_target_mean_and_root_transform() -> None:
    layer = ActiveRelationBasisRGCNLayer(
        hidden_dimension=1,
        num_relations=2,
        num_bases=2,
    )
    with torch.no_grad():
        layer.basis_weights.copy_(torch.tensor([[[2.0]], [[5.0]]]))
        layer.relation_coefficients.copy_(torch.eye(2))
        layer.root_weight.fill_(3.0)
        layer.bias.fill_(1.0)
    node_features = torch.tensor([[1.0], [3.0], [10.0]])
    edge_index = torch.tensor([[0, 1, 2], [2, 2, 2]])
    edge_type = torch.tensor([0, 0, 1])

    output = layer(node_features, edge_index, edge_type)

    # Relation 0 contributes mean(1*2, 3*2)=4, relation 1 contributes
    # 10*5=50, and the target's root/bias contributes 10*3+1.
    assert output[2].item() == pytest.approx(85.0)
    assert output[0].item() == pytest.approx(4.0)


def test_rgcn_layer_supports_empty_edges_gradients_and_bfloat16_autocast() -> None:
    model = RGCNAnswerRetriever(
        entity_embedding_dimension=4,
        hidden_dimension=8,
        gnn_layer_count=2,
        num_relations=3,
        num_bases=2,
        dropout=0.1,
    )
    entity_features = torch.randn(4, 4)
    edge_index = torch.empty((2, 0), dtype=torch.long)
    edge_type = torch.empty(0, dtype=torch.long)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        logits = model(entity_features, edge_index, edge_type=edge_type)
        loss = logits.float().sum()
    loss.backward()

    assert logits.shape == (4,)
    assert model.entity_projection.weight.grad is not None
    assert model.gnn_layers[0].root_weight.grad is not None


def test_custom_layer_matches_pyg_rgcn_conv() -> None:
    geometric_nn = pytest.importorskip("torch_geometric.nn")
    torch.manual_seed(4)
    custom = ActiveRelationBasisRGCNLayer(4, num_relations=3, num_bases=2)
    reference = geometric_nn.RGCNConv(
        4,
        4,
        num_relations=3,
        num_bases=2,
        aggr="mean",
        root_weight=True,
        bias=True,
    )
    with torch.no_grad():
        reference.weight.copy_(custom.basis_weights)
        reference.comp.copy_(custom.relation_coefficients)
        reference.root.copy_(custom.root_weight)
        reference.bias.copy_(custom.bias)
    node_features = torch.randn(5, 4)
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [2, 2, 3, 4, 2]])
    edge_type = torch.tensor([0, 0, 2, 1, 2])

    assert torch.allclose(
        custom(node_features, edge_index, edge_type),
        reference(node_features, edge_index, edge_type),
        atol=1e-6,
    )


class _EmbeddingService:
    def __init__(self) -> None:
        self.node_calls = 0
        self.relation_calls = 0
        self.question_calls = 0

    def load_node_cache(self, **_):
        self.node_calls += 1
        return SimpleNamespace(vector_size=4, cache_kind="nodes")

    def load_relation_cache(self, **_):
        self.relation_calls += 1
        raise AssertionError("R-GCN must not load relation embeddings")

    def load_question_cache(self, **_):
        self.question_calls += 1
        raise AssertionError("R-GCN must not load question embeddings")


class _TensorCacheService:
    def load_matrix(self, *, texts, dtype, device, **_):
        return torch.ones((len(texts), 4), dtype=dtype, device=device)


def test_rgcn_training_preparation_loads_only_entity_embeddings(tmp_path) -> None:
    vocabulary = {"parent": 0, "reverse__parent": 1}
    context = build_relation_architecture_context(vocabulary)
    model = RGCNAnswerRetriever(
        entity_embedding_dimension=4,
        hidden_dimension=8,
        gnn_layer_count=2,
        num_relations=2,
        num_bases=2,
        dropout=0.1,
    )
    built = BuiltGnnAnswerRetriever(
        dataset_id="WebQSP",
        gnn_architecture="rgcn",
        gnn_architecture_options={
            "gnn_layer_count": 2,
            "gnn_hidden_dimension": 8,
            "dropout": 0.1,
            "num_bases": 2,
        },
        gnn_architecture_context=context,
        relation_vocabulary=vocabulary,
        entity_embedding_model="text-embedding-3-small",
        entity_embedding_dimension=4,
        question_embedding_dimension=4,
        relation_embedding_dimension=4,
        hidden_dimension=8,
        gnn_layer_count=2,
        node_classifier="mlp",
        use_reverse_edges=True,
        model=model,
    )
    instance = WebQSPProcessedInstance(
        question="Who is the parent?",
        q_entity=["A"],
        a_entity=["B"],
        nodes=["A", "B"],
        node2id={"A": 0, "B": 1},
        edge_index=torch.tensor([[1, 0], [0, 1]]),
        edge_relations=["reverse__parent", "parent"],
        node_labels=torch.tensor([0.0, 1.0]),
    )
    dataset = PreparedWebQSPGraphDataset(
        dataset_id="WebQSP",
        processing_version="test",
        use_reverse_edges=True,
        train_instances=[instance],
        test_instances=[],
        vocabulary_store=WebQSPVocabularyStore(
            nodes={"A": 0, "B": 1},
            relations=vocabulary,
            questions={instance.question: 0},
        ),
        cache_directory=tmp_path / "processed_reverse_edges",
    )
    embedding_service = _EmbeddingService()
    service = GnnTrainingDataPreparationService(
        embedding_cache_service=embedding_service,
        embedding_tensor_cache_service=_TensorCacheService(),
    )
    configuration = _pipeline_config(built.gnn_architecture_options)

    prepared = service.prepare(
        built_retriever=built,
        prepared_dataset=dataset,
        configuration=configuration,
        preparation_config=GnnTrainingDataPreparationConfig(
            training_device="cpu",
            embedding_cache_device="cpu",
            embedding_cache_dtype="float32",
        ),
    )

    assert embedding_service.node_calls == 1
    assert embedding_service.relation_calls == 0
    assert embedding_service.question_calls == 0
    assert prepared.relation_embeddings is None
    assert prepared.question_embeddings is None
    assert prepared.instances[0].edge_type.tolist() == [0, 1]
    assert prepared.instances[0].edge_index.tolist() == [[0, 1], [1, 0]]


def test_rgcn_evaluation_preparation_loads_only_entity_embeddings(tmp_path) -> None:
    vocabulary = {"parent": 0, "reverse__parent": 1}
    instance = WebQSPProcessedInstance(
        question="Who is the parent?",
        q_entity=["A"],
        a_entity=["B"],
        nodes=["A", "B"],
        node2id={"A": 0, "B": 1},
        edge_index=torch.tensor([[1, 0], [0, 1]]),
        edge_relations=["reverse__parent", "parent"],
        node_labels=torch.tensor([0.0, 1.0]),
    )
    embedding_service = _EmbeddingService()
    service = GnnEvaluationDataPreparationService(
        embedding_cache_service=embedding_service,
        embedding_tensor_cache_service=_TensorCacheService(),
    )

    prepared = service.prepare(
        torch=torch,
        test_instances=[instance],
        cache_root=tmp_path,
        dataset_id="WebQSP",
        entity_embedding_model="text-embedding-3-small",
        relation_embedding_model="text-embedding-3-small",
        question_embedding_model="text-embedding-3-small",
        selected_device="cpu",
        evaluation_config=GnnAnswerRetrieverEvaluationConfig(
            embedding_cache_device="cpu",
            embedding_cache_dtype="float32",
        ),
        gnn_architecture="rgcn",
        relation_vocabulary=vocabulary,
    )

    assert embedding_service.node_calls == 1
    assert embedding_service.relation_calls == 0
    assert embedding_service.question_calls == 0
    assert prepared.relation_embeddings is None
    assert prepared.question_embeddings is None
    assert prepared.instances[0].edge_type.tolist() == [0, 1]


def test_rgcn_model_artifact_round_trip_uses_saved_relation_context(tmp_path) -> None:
    vocabulary = {"parent": 0, "reverse__parent": 1}
    context = build_relation_architecture_context(vocabulary)
    options = {
        "gnn_layer_count": 2,
        "gnn_hidden_dimension": 128,
        "dropout": 0.1,
        "num_bases": 8,
    }
    model = RGCNAnswerRetriever(
        entity_embedding_dimension=4,
        hidden_dimension=128,
        gnn_layer_count=2,
        num_relations=2,
        num_bases=8,
        dropout=0.1,
    )
    run_directory = tmp_path / "models" / "1_rgcn"
    run_directory.mkdir(parents=True)
    torch.save(model.state_dict(), run_directory / "gnn_answer_retriever.pt")
    (run_directory / "model_config.json").write_text(
        json.dumps(
            {
                "dataset_id": "WebQSP",
                "gnn_architecture": "rgcn",
                "gnn_architecture_options": options,
                "gnn_architecture_context": context,
                "embedding_model": "text-embedding-3-small",
                "embedding_dimension": 4,
                "training": {},
            }
        ),
        encoding="utf-8",
    )
    (run_directory / "relation_vocabulary.json").write_text(
        json.dumps(vocabulary),
        encoding="utf-8",
    )

    loaded = GnnAnswerRetrieverModelRunService().load_run(
        model_root=tmp_path / "models",
        run_name=None,
        run_number=1,
        pipeline_configuration=_pipeline_config(options),
        device="cpu",
    )

    assert loaded.relation_vocabulary == vocabulary
    assert loaded.config.resolved_use_reverse_edges
    assert loaded.model.num_relations == 2


def test_rgcn_model_artifact_rejects_tampered_relation_vocabulary(tmp_path) -> None:
    vocabulary = {"parent": 0, "reverse__parent": 1}
    run_directory = tmp_path / "models" / "1_rgcn"
    run_directory.mkdir(parents=True)
    (run_directory / "gnn_answer_retriever.pt").touch()
    (run_directory / "model_config.json").write_text(
        json.dumps(
            {
                "dataset_id": "WebQSP",
                "gnn_architecture": "rgcn",
                "gnn_architecture_options": {
                    "gnn_layer_count": 2,
                    "gnn_hidden_dimension": 128,
                    "dropout": 0.1,
                    "num_bases": 8,
                },
                "gnn_architecture_context": build_relation_architecture_context(
                    vocabulary
                ),
                "embedding_model": "text-embedding-3-small",
                "embedding_dimension": 4,
            }
        ),
        encoding="utf-8",
    )
    (run_directory / "relation_vocabulary.json").write_text(
        json.dumps({"parent": 1, "reverse__parent": 0}),
        encoding="utf-8",
    )

    with pytest.raises(
        GnnAnswerRetrieverModelRunException,
        match="relation vocabulary is invalid",
    ):
        GnnAnswerRetrieverModelRunService().resolve_run(
            model_root=tmp_path / "models",
            run_name=None,
            run_number=1,
        )


def test_rgcn_wandb_model_config_and_tag_are_canonical() -> None:
    vocabulary = {"parent": 0, "reverse__parent": 1}
    normalized = normalize_model_config(
        {
            "dataset_id": "WebQSP",
            "gnn_architecture": "rgcn",
            "gnn_architecture_options": {
                "gnn_layer_count": 2,
                "gnn_hidden_dimension": 256,
                "dropout": 0.1,
                "num_bases": 30,
            },
            "gnn_architecture_context": build_relation_architecture_context(
                vocabulary
            ),
            "embedding_model": "text-embedding-3-small",
            "training": {"epochs": 2, "loss_history": []},
        }
    )

    assert normalized["gnn_architecture"] == "rgcn"
    assert normalized["gnn_architecture_options"]["num_bases"] == 30
    assert normalized["gnn_architecture_context"]["relation_type_count"] == 2
    assert "gnn_architecture" not in normalized["training"]
    tags = WandbExperimentCoordinator._build_tags_from_config(
        {"configs": {"model": normalized}}
    )
    assert "rgcn" in tags
