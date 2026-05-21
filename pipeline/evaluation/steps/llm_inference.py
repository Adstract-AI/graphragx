"""Post-retrieval batch steps for reasoning paths and LLM inference."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import Field

from pipeline.abstract import AbstractStep, StepContext
from pipeline.evaluation.exceptions import (
    LlmAnswerGenerationException,
    ShortestPathExtractionException,
)
from pipeline.evaluation.models import (
    BuiltReasoningSamples,
    CandidateNodeScore,
    CandidateNodeScores,
    EvaluatedAnswerRetrievalInstance,
    EvaluationSample,
    ExtractedReasoningPathsBatch,
    GeneratedAnswerForPrediction,
    GeneratedFinalAnswersBatch,
    GnnAnswerRetrieverEvaluationResult,
    GraphTriple,
    ReasoningPathsForPrediction,
    ReasoningSampleForPrediction,
    SavedLlmInferenceRun,
)
from pipeline.evaluation.services import (
    LangChainOpenAiAnswerGenerationService,
    LlmInferenceStoragePayload,
    LlmInferenceStorageService,
    ShortestPathExtractionService,
)
from pipeline.preparation.models.webqsp_local_graph import (
    PreparedWebQSPGraphDataset,
    WebQSPProcessedInstance,
)


class BuildReasoningSamplesFromGnnEvaluationContext(
    StepContext[GnnAnswerRetrieverEvaluationResult]
):
    """Specialized context for adapting GNN predictions to reasoning samples."""

    prepared_dataset: PreparedWebQSPGraphDataset = Field(
        ...,
        description="Prepared WebQSP graph dataset used by GNN evaluation.",
    )


class BuildReasoningSamplesFromGnnEvaluationStep(
    AbstractStep[BuiltReasoningSamples, GnnAnswerRetrieverEvaluationResult]
):
    """Build shortest-path inputs from the previous GNN evaluation result."""

    def execute_default(
        self,
        context: BuildReasoningSamplesFromGnnEvaluationContext,
    ) -> BuiltReasoningSamples:
        evaluation_result = context.result
        if evaluation_result is None:
            raise ShortestPathExtractionException(
                "Reasoning sample construction requires a GNN evaluation result."
            )

        predictions = self._load_predictions(evaluation_result.predictions_path)
        samples = [
            self._build_sample_for_prediction(
                prediction=prediction,
                instance=context.prepared_dataset.test_instances[
                    prediction.instance_index
                ],
            )
            for prediction in predictions
        ]
        return BuiltReasoningSamples(
            dataset_id=evaluation_result.dataset_id,
            evaluation_run_name=evaluation_result.evaluation_run_name,
            evaluation_run_directory=Path(evaluation_result.evaluation_run_directory),
            samples=samples,
        )

    @staticmethod
    def _load_predictions(
        predictions_path: Path,
    ) -> list[EvaluatedAnswerRetrievalInstance]:
        try:
            raw_lines = predictions_path.read_text(encoding="utf-8").splitlines()
        except OSError as error:
            raise ShortestPathExtractionException(
                f"Could not read GNN predictions file {predictions_path}: {error}"
            ) from error

        predictions: list[EvaluatedAnswerRetrievalInstance] = []
        for line_number, raw_line in enumerate(raw_lines, start=1):
            stripped_line = raw_line.strip()
            if not stripped_line:
                continue
            try:
                predictions.append(
                    EvaluatedAnswerRetrievalInstance.model_validate_json(
                        stripped_line
                    )
                )
            except (ValueError, json.JSONDecodeError) as error:
                raise ShortestPathExtractionException(
                    f"Invalid prediction JSON on line {line_number} of "
                    f"{predictions_path}: {error}"
                ) from error

        return predictions

    @classmethod
    def _build_sample_for_prediction(
        cls,
        prediction: EvaluatedAnswerRetrievalInstance,
        instance: WebQSPProcessedInstance,
    ) -> ReasoningSampleForPrediction:
        graph_triples = cls._build_graph_triples(instance)
        candidate_scores = CandidateNodeScores(
            sample=cls._build_evaluation_sample(prediction, graph_triples),
            candidates=[
                CandidateNodeScore(
                    node_id=candidate.node,
                    score=candidate.probability,
                    local_node_id=candidate.local_node_id,
                    global_node_id=candidate.global_node_id,
                    logit=candidate.logit,
                    probability=candidate.probability,
                    is_gold_answer=candidate.is_gold_answer,
                    selection_reason=candidate.selection_reason,
                )
                for candidate in prediction.answer_candidates
            ],
            top_k=len(prediction.answer_candidates),
        )
        return ReasoningSampleForPrediction(
            instance_index=prediction.instance_index,
            prediction=prediction,
            candidate_scores=candidate_scores,
        )

    @staticmethod
    def _build_evaluation_sample(
        prediction: EvaluatedAnswerRetrievalInstance,
        graph_triples: list[GraphTriple],
    ) -> EvaluationSample:
        return EvaluationSample(
            sample_id=str(prediction.instance_index),
            question=prediction.question,
            q_entities=prediction.q_entity,
            a_entities=prediction.a_entity,
            graph_triples=graph_triples,
        )

    @staticmethod
    def _build_graph_triples(
        instance: WebQSPProcessedInstance,
    ) -> list[GraphTriple]:
        edge_index = instance.edge_index
        edge_count = len(instance.edge_relations)
        triples: list[GraphTriple] = []
        for edge_offset in range(edge_count):
            source_index = int(edge_index[0][edge_offset].item())
            target_index = int(edge_index[1][edge_offset].item())
            triples.append(
                GraphTriple(
                    source=instance.nodes[source_index],
                    relation=instance.edge_relations[edge_offset],
                    target=instance.nodes[target_index],
                )
            )
        return triples


class ExtractShortestPathsBatchStep(
    AbstractStep[ExtractedReasoningPathsBatch, BuiltReasoningSamples]
):
    """Extract shortest reasoning paths for each GNN prediction."""

    def __init__(
        self,
        shortest_path_service: ShortestPathExtractionService | None = None,
        force_default: bool = False,
    ):
        super().__init__(force_default=force_default)
        self.shortest_path_service = shortest_path_service or ShortestPathExtractionService()

    def execute_default(
        self,
        context: StepContext[BuiltReasoningSamples],
    ) -> ExtractedReasoningPathsBatch:
        built_samples = context.result
        if built_samples is None:
            raise ShortestPathExtractionException(
                "Batch shortest path extraction requires built reasoning samples."
            )

        return ExtractedReasoningPathsBatch(
            dataset_id=built_samples.dataset_id,
            evaluation_run_name=built_samples.evaluation_run_name,
            items=[
                ReasoningPathsForPrediction(
                    instance_index=item.instance_index,
                    prediction=item.prediction,
                    extracted_paths=self.shortest_path_service.extract_paths(
                        sample=item.candidate_scores.sample,
                        candidates=item.candidate_scores.candidates,
                    ),
                )
                for item in built_samples.samples
            ],
        )


class GenerateFinalAnswersBatchStep(
    AbstractStep[GeneratedFinalAnswersBatch, ExtractedReasoningPathsBatch]
):
    """Generate final answers for a batch of extracted reasoning paths."""

    def __init__(
        self,
        model_id: str = "gpt-4.1-mini",
        answer_generation_service: LangChainOpenAiAnswerGenerationService | None = None,
        force_default: bool = False,
    ):
        super().__init__(force_default=force_default)
        self.model_id = model_id
        self.answer_generation_service = (
            answer_generation_service or LangChainOpenAiAnswerGenerationService()
        )

    def execute_default(
        self,
        context: StepContext[ExtractedReasoningPathsBatch],
    ) -> GeneratedFinalAnswersBatch:
        paths_batch = context.result
        if paths_batch is None:
            raise LlmAnswerGenerationException(
                "Batch answer generation requires extracted reasoning paths."
            )

        items = [
            self._generate_answer(item)
            for item in paths_batch.items
        ]
        return GeneratedFinalAnswersBatch(
            dataset_id=paths_batch.dataset_id,
            evaluation_run_name=paths_batch.evaluation_run_name,
            model_id=self.model_id,
            items=items,
        )

    def _generate_answer(
        self,
        item: ReasoningPathsForPrediction,
    ) -> GeneratedAnswerForPrediction:
        extracted_paths = item.extracted_paths
        prompt = ""
        answer = ""
        error_message = None
        try:
            answer, prompt = self.answer_generation_service.generate_answer(
                question=extracted_paths.sample.question,
                reasoning_paths_text=extracted_paths.reasoning_paths_text,
                model_id=self.model_id,
            )
        except Exception as error:  # keep batch inference usable for later review
            error_message = str(error)

        return GeneratedAnswerForPrediction(
            instance_index=item.instance_index,
            question=extracted_paths.sample.question,
            q_entity=item.prediction.q_entity,
            a_entity=item.prediction.a_entity,
            answer_candidates=[
                candidate.node for candidate in item.prediction.answer_candidates
            ],
            reasoning_subgraph_triples=extracted_paths.reasoning_subgraph_triples,
            reasoning_paths_text=extracted_paths.reasoning_paths_text,
            model_id=self.model_id,
            prompt=prompt,
            answer=answer,
            error_message=error_message,
        )


class SaveInferenceRunStep(
    AbstractStep[SavedLlmInferenceRun, GeneratedFinalAnswersBatch]
):
    """Persist a complete LLM inference run."""

    def __init__(
        self,
        inference_root: str | Path = "data/inference",
        inference_run_name: str | None = None,
        storage_service: LlmInferenceStorageService | None = None,
        force_default: bool = False,
    ):
        super().__init__(force_default=force_default)
        self.inference_root = Path(inference_root)
        self.inference_run_name = inference_run_name
        self.storage_service = storage_service or LlmInferenceStorageService()

    def execute_default(
        self,
        context: StepContext[GeneratedFinalAnswersBatch],
    ) -> SavedLlmInferenceRun:
        answers = context.result
        if answers is None:
            raise LlmAnswerGenerationException(
                "Inference storage requires generated final answers."
            )

        storage_result = self.storage_service.save_inference_run(
            inference_root=self.inference_root,
            run_name=self.inference_run_name,
            payload=LlmInferenceStoragePayload(
                answers=answers,
            ),
        )
        return SavedLlmInferenceRun(
            dataset_id=answers.dataset_id,
            evaluation_run_name=answers.evaluation_run_name,
            inference_run_directory=storage_result.inference_run_directory,
            inference_run_name=storage_result.inference_run_name,
            inference_run_number=storage_result.inference_run_number,
            model_id=answers.model_id,
            total_instances=len(answers.items),
            successful_answers=answers.successful_answers,
            failed_answers=answers.failed_answers,
            reasoning_paths_path=storage_result.reasoning_paths_path,
            reasoning_subgraphs_path=storage_result.reasoning_subgraphs_path,
            prompts_path=storage_result.prompts_path,
            answers_path=storage_result.answers_path,
            summary_path=storage_result.summary_path,
        )
