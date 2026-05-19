"""Evaluation service for saved GNN answer-retriever runs."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from logging_config import get_logger
from pipeline.evaluation.models import (
    AnswerCandidateScore,
    EvaluatedAnswerRetrievalInstance,
    GnnAnswerRetrieverEvaluationConfig,
    GoldAnswerScore,
)
from pipeline.exceptions import GnnAnswerRetrieverEvaluationException
from pipeline.preparation.models.webqsp_local_graph import (
    PreparedWebQSPGraphDataset,
    WebQSPProcessedInstance,
)
from pipeline.preparation.steps.configuration_building import BuiltPipelineConfiguration
from pipeline.services.abstract import AbstractService
from pipeline.services.embedding_cache import (
    TextEmbeddingCache,
    WebQSPEmbeddingCacheService,
)
from pipeline.services.gnn_answer_retriever_evaluation_storage import (
    GnnAnswerRetrieverEvaluationStoragePayload,
    GnnAnswerRetrieverEvaluationStorageResult,
    GnnAnswerRetrieverEvaluationStorageService,
    JsonObjectLevel1,
    JsonObjectLevel3,
)
from pipeline.services.gnn_answer_retriever_model_runs import (
    GnnAnswerRetrieverModelRunService,
    LoadedGnnAnswerRetrieverRun,
)

logger = get_logger(__name__)


class GnnAnswerRetrieverEvaluationOutcome(BaseModel):
    """Evaluation output and persisted run metadata."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    loaded_model_run: LoadedGnnAnswerRetrieverRun = Field(
        ...,
        description="Loaded model run used for evaluation.",
    )
    storage_result: GnnAnswerRetrieverEvaluationStorageResult = Field(
        ...,
        description="Saved evaluation run paths.",
    )
    evaluated_instances: int = Field(..., description="Number of evaluated instances.")
    hits_at_1: float = Field(..., description="Hits@1 rate.")
    hits_at_1_count: int = Field(..., description="Hits@1 count.")
    hit_at_k: float = Field(..., description="Answer coverage rate.")
    hit_at_k_count: int = Field(..., description="Answer coverage count.")
    average_candidate_count: float = Field(
        ...,
        description="Average number of selected candidate nodes.",
    )
    missing_gold_in_graph_count: int = Field(
        ...,
        description="Instances where no gold answer appears in the local graph.",
    )


class GnnAnswerRetrieverEvaluationService(AbstractService):
    """Evaluate a saved GNN answer-retriever model over prepared WebQSP test graphs."""

    def __init__(
        self,
        model_run_service: GnnAnswerRetrieverModelRunService | None = None,
        embedding_cache_service: WebQSPEmbeddingCacheService | None = None,
        storage_service: GnnAnswerRetrieverEvaluationStorageService | None = None,
    ):
        self.model_run_service = model_run_service or GnnAnswerRetrieverModelRunService()
        self.embedding_cache_service = (
            embedding_cache_service or WebQSPEmbeddingCacheService()
        )
        self.storage_service = (
            storage_service or GnnAnswerRetrieverEvaluationStorageService()
        )

    def evaluate(
        self,
        prepared_dataset: PreparedWebQSPGraphDataset,
        pipeline_configuration: BuiltPipelineConfiguration,
        evaluation_config: GnnAnswerRetrieverEvaluationConfig,
    ) -> GnnAnswerRetrieverEvaluationOutcome:
        """Run saved-model retrieval evaluation and persist all outputs."""
        import torch
        import torch.nn.functional as torch_functional

        test_instances = self._select_test_instances(
            prepared_dataset=prepared_dataset,
            max_instances=evaluation_config.max_instances,
        )
        if not test_instances:
            raise GnnAnswerRetrieverEvaluationException(
                "GNN answer-retriever evaluation requires at least one test instance."
            )

        device = self._resolve_device(torch)
        cache_root = prepared_dataset.cache_directory.parent
        loaded_model_run = self.model_run_service.load_run(
            model_root=cache_root / "models",
            run_name=evaluation_config.model_run_name,
            run_number=evaluation_config.model_run_number,
            pipeline_configuration=pipeline_configuration,
            device=device,
        )
        if loaded_model_run.config.dataset_id != prepared_dataset.dataset_id:
            raise GnnAnswerRetrieverEvaluationException(
                f"Model run dataset {loaded_model_run.config.dataset_id} does not "
                f"match prepared dataset {prepared_dataset.dataset_id}."
            )

        node_cache = self.embedding_cache_service.load_node_cache(
            cache_root=cache_root,
            model_id=loaded_model_run.config.entity_embedding_model,
            vocabulary=prepared_dataset.vocabulary_store.nodes,
        )
        relation_cache = self.embedding_cache_service.load_relation_cache(
            cache_root=cache_root,
            model_id=loaded_model_run.relation_embedding_model,
            vocabulary=prepared_dataset.vocabulary_store.relations,
        )
        question_cache = self.embedding_cache_service.load_question_cache(
            cache_root=cache_root,
            model_id=loaded_model_run.question_embedding_model,
            vocabulary=prepared_dataset.vocabulary_store.questions,
        )

        logger.info(
            f"Starting GNN answer-retriever evaluation: "
            f"model_run={loaded_model_run.run_name} "
            f"instances={len(test_instances)} device={device} "
            f"threshold={evaluation_config.answer_threshold} "
            f"candidate_top_k={evaluation_config.candidate_top_k}"
        )
        self._populate_embedding_caches(
            test_instances=test_instances,
            node_cache=node_cache,
            relation_cache=relation_cache,
            question_cache=question_cache,
        )

        predictions: list[EvaluatedAnswerRetrievalInstance] = []
        hits_at_1_count = 0
        hit_at_k_count = 0
        missing_gold_in_graph_count = 0
        total_candidate_count = 0

        model = loaded_model_run.model
        with torch.no_grad():
            for instance_index, instance in enumerate(test_instances):
                entity_features = self._build_entity_feature_tensor(
                    instance=instance,
                    node_cache=node_cache,
                    torch=torch,
                    device=device,
                )
                edge_weight = self._build_edge_weight_tensor(
                    instance=instance,
                    question_cache=question_cache,
                    relation_cache=relation_cache,
                    torch=torch,
                    torch_functional=torch_functional,
                    device=device,
                )
                logits = model(
                    entity_features=entity_features,
                    edge_index=instance.edge_index.to(device),
                    edge_weight=edge_weight,
                )
                probabilities = torch.sigmoid(logits)
                prediction = self._build_prediction(
                    instance_index=instance_index,
                    instance=instance,
                    logits=logits.detach().cpu(),
                    probabilities=probabilities.detach().cpu(),
                    answer_threshold=evaluation_config.answer_threshold,
                    candidate_top_k=evaluation_config.candidate_top_k,
                    global_node_vocabulary=prepared_dataset.vocabulary_store.nodes,
                    torch=torch,
                )
                predictions.append(prediction)
                hits_at_1_count += int(prediction.hit_at_1)
                hit_at_k_count += int(prediction.hit_at_k)
                missing_gold_in_graph_count += int(prediction.missing_gold_in_graph)
                total_candidate_count += len(prediction.answer_candidates)

        evaluated_instances = len(predictions)
        hits_at_1 = hits_at_1_count / evaluated_instances
        hit_at_k = hit_at_k_count / evaluated_instances
        average_candidate_count = total_candidate_count / evaluated_instances
        summary_metrics: JsonObjectLevel1 = {
            "dataset_id": prepared_dataset.dataset_id,
            "model_run_name": loaded_model_run.run_name,
            "model_run_number": loaded_model_run.run_number,
            "evaluated_instances": evaluated_instances,
            "hits_at_1": hits_at_1,
            "hits_at_1_count": hits_at_1_count,
            "hit_at_k": hit_at_k,
            "hit_at_k_count": hit_at_k_count,
            "average_candidate_count": average_candidate_count,
            "missing_gold_in_graph_count": missing_gold_in_graph_count,
        }
        storage_result = self.storage_service.save_evaluation_run(
            evaluation_root=cache_root / "evaluations",
            run_name=evaluation_config.run_name,
            payload=GnnAnswerRetrieverEvaluationStoragePayload(
                evaluation_config=self._build_evaluation_config_payload(
                    evaluation_config=evaluation_config,
                    loaded_model_run=loaded_model_run,
                    pipeline_configuration=pipeline_configuration,
                    device=device,
                ),
                summary_metrics=summary_metrics,
                predictions=predictions,
            ),
        )
        logger.info(
            f"Finished GNN answer-retriever evaluation: "
            f"run={storage_result.evaluation_run_name} "
            f"hits_at_1={hits_at_1:.4f} hit_at_k={hit_at_k:.4f} "
            f"average_candidate_count={average_candidate_count:.2f}"
        )
        return GnnAnswerRetrieverEvaluationOutcome(
            loaded_model_run=loaded_model_run,
            storage_result=storage_result,
            evaluated_instances=evaluated_instances,
            hits_at_1=hits_at_1,
            hits_at_1_count=hits_at_1_count,
            hit_at_k=hit_at_k,
            hit_at_k_count=hit_at_k_count,
            average_candidate_count=average_candidate_count,
            missing_gold_in_graph_count=missing_gold_in_graph_count,
        )

    @staticmethod
    def _select_test_instances(
        prepared_dataset: PreparedWebQSPGraphDataset,
        max_instances: int | None,
    ) -> list[WebQSPProcessedInstance]:
        if max_instances is None:
            return prepared_dataset.test_instances

        return prepared_dataset.test_instances[:max_instances]

    @staticmethod
    def _resolve_device(torch) -> str:
        if torch.cuda.is_available():
            return "cuda"

        if torch.backends.mps.is_available():
            return "mps"

        return "cpu"

    def _populate_embedding_caches(
        self,
        test_instances: list[WebQSPProcessedInstance],
        node_cache: TextEmbeddingCache,
        relation_cache: TextEmbeddingCache,
        question_cache: TextEmbeddingCache,
    ) -> None:
        node_texts: list[str] = []
        relation_texts: list[str] = []
        question_texts: list[str] = []
        for instance in test_instances:
            node_texts.extend(instance.nodes)
            relation_texts.extend(instance.edge_relations)
            question_texts.append(instance.question)

        self.embedding_cache_service.ensure_embeddings(node_cache, node_texts)
        self.embedding_cache_service.ensure_embeddings(
            relation_cache,
            relation_texts,
            preprocess=True,
        )
        self.embedding_cache_service.ensure_embeddings(question_cache, question_texts)

    def _build_entity_feature_tensor(
        self,
        instance: WebQSPProcessedInstance,
        node_cache: TextEmbeddingCache,
        torch,
        device: str,
    ):
        return torch.tensor(
            [
                self.embedding_cache_service.embedding_for_text(node_cache, node)
                for node in instance.nodes
            ],
            dtype=torch.float,
            device=device,
        )

    def _build_edge_weight_tensor(
        self,
        instance: WebQSPProcessedInstance,
        question_cache: TextEmbeddingCache,
        relation_cache: TextEmbeddingCache,
        torch,
        torch_functional,
        device: str,
    ):
        if not instance.edge_relations:
            return torch.empty(0, dtype=torch.float, device=device)

        question_embedding = torch.tensor(
            self.embedding_cache_service.embedding_for_text(
                question_cache,
                instance.question,
            ),
            dtype=torch.float,
            device=device,
        )
        relation_embeddings = torch.tensor(
            [
                self.embedding_cache_service.embedding_for_text(
                    relation_cache,
                    relation,
                )
                for relation in instance.edge_relations
            ],
            dtype=torch.float,
            device=device,
        )
        return torch_functional.cosine_similarity(
            question_embedding.reshape(1, -1),
            relation_embeddings,
            dim=1,
        )

    def _build_prediction(
        self,
        instance_index: int,
        instance: WebQSPProcessedInstance,
        logits,
        probabilities,
        answer_threshold: float,
        candidate_top_k: int,
        global_node_vocabulary: dict[str, int],
        torch,
    ) -> EvaluatedAnswerRetrievalInstance:
        gold_answers = set(instance.a_entity)
        selected_ids = [
            node_id
            for node_id, probability in enumerate(probabilities.tolist())
            if probability >= answer_threshold
        ]
        selection_reason = "threshold"
        if not selected_ids:
            selected_ids = torch.topk(
                probabilities,
                k=min(candidate_top_k, len(instance.nodes)),
            ).indices.tolist()
            selection_reason = "fallback_top_k"

        selected_ids = sorted(
            selected_ids,
            key=lambda node_id: float(probabilities[node_id].item()),
            reverse=True,
        )
        answer_candidates = [
            AnswerCandidateScore(
                node=instance.nodes[node_id],
                local_node_id=node_id,
                global_node_id=global_node_vocabulary[instance.nodes[node_id]],
                logit=float(logits[node_id].item()),
                probability=float(probabilities[node_id].item()),
                is_gold_answer=instance.nodes[node_id] in gold_answers,
                selection_reason=selection_reason,
            )
            for node_id in selected_ids
        ]
        gold_answer_scores = [
            self._build_gold_answer_score(
                gold_answer=gold_answer,
                instance=instance,
                logits=logits,
                probabilities=probabilities,
                global_node_vocabulary=global_node_vocabulary,
            )
            for gold_answer in instance.a_entity
        ]

        top_node_id = int(torch.argmax(probabilities).item())
        hit_at_1 = instance.nodes[top_node_id] in gold_answers
        hit_at_k = any(candidate.is_gold_answer for candidate in answer_candidates)
        missing_gold_in_graph = not any(score.present_in_graph for score in gold_answer_scores)

        return EvaluatedAnswerRetrievalInstance(
            instance_index=instance_index,
            question=instance.question,
            q_entity=instance.q_entity,
            a_entity=instance.a_entity,
            answer_candidates=answer_candidates,
            gold_answer_scores=gold_answer_scores,
            hit_at_1=hit_at_1,
            hit_at_k=hit_at_k,
            missing_gold_in_graph=missing_gold_in_graph,
        )

    @staticmethod
    def _build_gold_answer_score(
        gold_answer: str,
        instance: WebQSPProcessedInstance,
        logits,
        probabilities,
        global_node_vocabulary: dict[str, int],
    ) -> GoldAnswerScore:
        if gold_answer not in instance.node2id:
            return GoldAnswerScore(
                node=gold_answer,
                local_node_id=None,
                global_node_id=global_node_vocabulary.get(gold_answer),
                logit=None,
                probability=None,
                present_in_graph=False,
            )

        node_id = instance.node2id[gold_answer]
        return GoldAnswerScore(
            node=gold_answer,
            local_node_id=node_id,
            global_node_id=global_node_vocabulary[gold_answer],
            logit=float(logits[node_id].item()),
            probability=float(probabilities[node_id].item()),
            present_in_graph=True,
        )

    @staticmethod
    def _build_evaluation_config_payload(
        evaluation_config: GnnAnswerRetrieverEvaluationConfig,
        loaded_model_run: LoadedGnnAnswerRetrieverRun,
        pipeline_configuration: BuiltPipelineConfiguration,
        device: str,
    ) -> JsonObjectLevel3:
        return {
            "dataset_id": pipeline_configuration.dataset_id,
            "selected_device": device,
            "model_run": {
                "name": loaded_model_run.run_name,
                "number": loaded_model_run.run_number,
                "directory": str(loaded_model_run.run_directory),
                "config_path": str(loaded_model_run.config_path),
                "weights_path": str(loaded_model_run.weights_path),
            },
            "model_configuration": loaded_model_run.config.model_dump(mode="json"),
            "evaluation": evaluation_config.model_dump(mode="json"),
        }
