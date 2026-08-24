"""Post-retrieval batch steps for reasoning paths and LLM inference."""

from __future__ import annotations

import json
import time
from pathlib import Path

from pydantic import Field

from helpers.logging_config import get_logger
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
from pipeline.preparation.steps.configuration_building import BuiltPipelineConfiguration
from pipeline.preparation.models.webqsp_local_graph import (
    PreparedWebQSPGraphDataset,
    WebQSPProcessedInstance,
)

logger = get_logger(__name__)


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

        logger.info(
            f"Building reasoning samples from GNN predictions: "
            f"evaluation_run={evaluation_result.evaluation_run_name} "
            f"predictions_path={evaluation_result.predictions_path}"
        )
        predictions = self._load_predictions(evaluation_result.predictions_path)
        samples = []
        test_instances = context.prepared_dataset.test_instances
        for prediction in predictions:
            if (
                prediction.instance_index < 0
                or prediction.instance_index >= len(test_instances)
            ):
                raise ShortestPathExtractionException(
                    f"Prediction instance index {prediction.instance_index} is outside "
                    f"the prepared test split of size {len(test_instances)}."
                )
            samples.append(
                self._build_sample_for_prediction(
                    prediction=prediction,
                    instance=test_instances[prediction.instance_index],
                )
            )
        logger.info(
            f"Built reasoning samples: evaluation_run={evaluation_result.evaluation_run_name} "
            f"samples={len(samples)}"
        )
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
            predictions_file = predictions_path.open("r", encoding="utf-8")
        except OSError as error:
            raise ShortestPathExtractionException(
                f"Could not read GNN predictions file {predictions_path}: {error}"
            ) from error

        with predictions_file:
            predictions: list[EvaluatedAnswerRetrievalInstance] = []
            for line_number, raw_line in enumerate(predictions_file, start=1):
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
        candidate_scores = CandidateNodeScores(
            sample=cls._build_evaluation_sample(prediction, []),
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
            graph_instance=instance,
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
        source_indices, target_indices = instance.edge_index.tolist()
        return [
            GraphTriple(
                source=instance.nodes[source_index],
                relation=relation,
                target=instance.nodes[target_index],
            )
            for source_index, target_index, relation in zip(
                source_indices,
                target_indices,
                instance.edge_relations,
                strict=True,
            )
        ]


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

        logger.info(
            f"Extracting shortest reasoning paths: "
            f"evaluation_run={built_samples.evaluation_run_name} "
            f"samples={len(built_samples.samples)}"
        )
        result = ExtractedReasoningPathsBatch(
            dataset_id=built_samples.dataset_id,
            evaluation_run_name=built_samples.evaluation_run_name,
            items=[
                ReasoningPathsForPrediction(
                    instance_index=item.instance_index,
                    prediction=item.prediction,
                    extracted_paths=(
                        self.shortest_path_service.extract_paths_from_processed_graph(
                            instance=item.graph_instance,
                            sample=item.candidate_scores.sample,
                            candidates=item.candidate_scores.candidates,
                        )
                        if item.graph_instance is not None
                        else self.shortest_path_service.extract_paths(
                            sample=item.candidate_scores.sample,
                            candidates=item.candidate_scores.candidates,
                        )
                    ),
                )
                for item in built_samples.samples
            ],
        )
        found_paths = sum(item.extracted_paths.found_paths for item in result.items)
        missing_paths = sum(item.extracted_paths.missing_paths for item in result.items)
        logger.info(
            f"Finished shortest reasoning paths: "
            f"evaluation_run={built_samples.evaluation_run_name} "
            f"samples={len(result.items)} found_paths={found_paths} "
            f"missing_paths={missing_paths}"
        )
        return result


class GenerateFinalAnswersBatchStep(
    AbstractStep[GeneratedFinalAnswersBatch, ExtractedReasoningPathsBatch]
):
    """Generate final answers for a batch of extracted reasoning paths."""

    def __init__(
        self,
        model_id: str = "gpt-4.1-mini",
        llm_provider: str = "openai",
        reasoning_effort: str | None = None,
        answer_generation_service: LangChainOpenAiAnswerGenerationService | None = None,
        force_default: bool = False,
    ):
        super().__init__(force_default=force_default)
        self.model_id = model_id
        self.llm_provider = llm_provider
        self.reasoning_effort = reasoning_effort
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

        logger.info(
            f"Generating LLM answers: evaluation_run={paths_batch.evaluation_run_name} "
            f"model={self.model_id} samples={len(paths_batch.items)}"
        )
        items = [
            self._generate_answer(item)
            for item in paths_batch.items
        ]
        result = GeneratedFinalAnswersBatch(
            dataset_id=paths_batch.dataset_id,
            evaluation_run_name=paths_batch.evaluation_run_name,
            model_id=self.model_id,
            llm_provider=self.llm_provider,
            reasoning_effort=self.reasoning_effort,
            items=items,
        )
        logger.info(
            f"Finished LLM answers: evaluation_run={paths_batch.evaluation_run_name} "
            f"model={self.model_id} successful={result.successful_answers} "
            f"failed={result.failed_answers}"
        )
        return result

    def _generate_answer(
        self,
        item: ReasoningPathsForPrediction,
    ) -> GeneratedAnswerForPrediction:
        extracted_paths = item.extracted_paths
        answer = ""
        explanation = ""
        raw_response = ""
        error_message = None
        started_at = time.monotonic()
        try:
            generation_kwargs = {
                "question": extracted_paths.sample.question,
                "reasoning_paths_text": extracted_paths.reasoning_paths_text,
                "model_id": self.model_id,
            }
            try:
                result = self.answer_generation_service.generate_answer_with_explanation(
                    **generation_kwargs,
                    provider_id=self.llm_provider,
                    reasoning_effort=self.reasoning_effort,
                )
            except TypeError as error:
                if not any(
                    name in str(error)
                    for name in ("provider_id", "reasoning_effort")
                ):
                    raise
                # Keep existing injected/custom services compatible with the
                # pre-provider method contract.
                result = self.answer_generation_service.generate_answer_with_explanation(
                    **generation_kwargs,
                )
            answer = result["answer"]
            explanation = result["explanation"]
            raw_response = result["raw_response"]
            prompt_tokens = int(result.get("prompt_tokens", 0))
            completion_tokens = int(result.get("completion_tokens", 0))
            total_tokens = int(result.get("total_tokens", 0))
            estimated_cost_usd = float(result.get("estimated_cost_usd", 0.0))
        except Exception as error:  # keep batch inference usable for later review
            error_message = str(error)
            prompt_tokens = 0
            completion_tokens = 0
            total_tokens = 0
            estimated_cost_usd = 0.0
            elapsed_seconds = time.monotonic() - started_at
            logger.warning(
                f"LLM answer generation failed: instance_index={item.instance_index} "
                f"elapsed_seconds={elapsed_seconds:.2f} error={error_message}"
            )
        else:
            elapsed_seconds = time.monotonic() - started_at
            slow_warning_seconds = getattr(
                self.answer_generation_service,
                "slow_request_warning_seconds",
                30.0,
            )
            if elapsed_seconds >= slow_warning_seconds:
                logger.warning(
                    f"Slow LLM answer generation instance: "
                    f"instance_index={item.instance_index} "
                    f"elapsed_seconds={elapsed_seconds:.2f}"
                )

        return GeneratedAnswerForPrediction(
            instance_index=item.instance_index,
            question=extracted_paths.sample.question,
            q_entity=item.prediction.q_entity,
            a_entity=item.prediction.a_entity,
            answer_candidates=[
                candidate.node for candidate in item.prediction.answer_candidates
            ],
            reasoning_subgraph_triples=extracted_paths.reasoning_subgraph_triples,
            reasoning_path_lengths=[
                len(path.triples)
                for path in extracted_paths.paths
                if path.path_found
            ],
            found_reasoning_paths=extracted_paths.found_paths,
            missing_reasoning_paths=extracted_paths.missing_paths,
            reasoning_paths_text=extracted_paths.reasoning_paths_text,
            model_id=self.model_id,
            llm_provider=self.llm_provider,
            answer=answer,
            explanation=explanation,
            raw_response=raw_response,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=estimated_cost_usd,
            error_message=error_message,
        )


class GenerateAndSaveFinalAnswersBatchesContext(
    StepContext[ExtractedReasoningPathsBatch]
):
    """Context for batched LLM inference with resolved pipeline configuration."""

    pipeline_configuration: BuiltPipelineConfiguration = Field(
        ...,
        description="Pipeline configuration containing the selected main LLM model.",
    )


class GenerateAndSaveFinalAnswersBatchesStep(
    AbstractStep[SavedLlmInferenceRun, ExtractedReasoningPathsBatch]
):
    """Generate LLM answers in small batches and persist each batch immediately."""

    def __init__(
        self,
        llm_provider: str | None = None,
        reasoning_effort: str | None = None,
        model_id: str | None = None,
        inference_root: str | Path = "data/webqsp/inference",
        inference_run_name: str | None = None,
        inference_batch_size: int = 10,
        answer_generation_service: LangChainOpenAiAnswerGenerationService | None = None,
        storage_service: LlmInferenceStorageService | None = None,
        force_default: bool = False,
    ):
        super().__init__(force_default=force_default)
        self.model_id = model_id
        self.llm_provider = llm_provider
        self.reasoning_effort = reasoning_effort
        self.inference_root = Path(inference_root)
        self.inference_run_name = inference_run_name
        self.inference_batch_size = max(1, inference_batch_size)
        self.answer_generation_service = (
            answer_generation_service or LangChainOpenAiAnswerGenerationService()
        )
        self.storage_service = storage_service or LlmInferenceStorageService()

    def execute_default(
        self,
        context: StepContext[ExtractedReasoningPathsBatch],
    ) -> SavedLlmInferenceRun:
        paths_batch = context.result
        if paths_batch is None:
            raise LlmAnswerGenerationException(
                "Batched inference requires extracted reasoning paths."
            )
        model_id = self._resolve_model_id(context)
        llm_provider = self._resolve_llm_provider(context)

        total_items = len(paths_batch.items)
        logger.info(
            f"Starting batched LLM inference: "
            f"evaluation_run={paths_batch.evaluation_run_name} "
            f"provider={llm_provider} model={model_id} samples={total_items} "
            f"batch_size={self.inference_batch_size} root={self.inference_root}"
        )
        run = self.storage_service.create_inference_run(
            inference_root=self.inference_root,
            run_name=self.inference_run_name,
        )

        all_items: list[GeneratedAnswerForPrediction] = []
        for batch_number, batch_items in enumerate(
            self._chunk_items(paths_batch.items, self.inference_batch_size),
            start=1,
        ):
            logger.info(
                f"Generating LLM inference batch: "
                f"evaluation_run={paths_batch.evaluation_run_name} "
                f"batch={batch_number} batch_size={len(batch_items)}"
            )
            generated_batch = GeneratedFinalAnswersBatch(
                dataset_id=paths_batch.dataset_id,
                evaluation_run_name=paths_batch.evaluation_run_name,
                model_id=model_id,
                llm_provider=llm_provider,
                reasoning_effort=self.reasoning_effort,
                items=[
                    self._generate_answer(item, model_id, llm_provider)
                    for item in batch_items
                ],
            )
            all_items.extend(generated_batch.items)
            cumulative_batch = GeneratedFinalAnswersBatch(
                dataset_id=paths_batch.dataset_id,
                evaluation_run_name=paths_batch.evaluation_run_name,
                model_id=model_id,
                llm_provider=llm_provider,
                reasoning_effort=self.reasoning_effort,
                items=all_items,
            )
            self.storage_service.append_inference_batch(
                run=run,
                answers=generated_batch,
            )
            self.storage_service.write_inference_config(
                run=run,
                answers=cumulative_batch,
            )
            logger.info(
                f"Saved LLM inference batch: "
                f"evaluation_run={paths_batch.evaluation_run_name} "
                f"batch={batch_number} total_saved={len(all_items)} "
                f"successful={cumulative_batch.successful_answers} "
                f"failed={cumulative_batch.failed_answers}"
            )

        final_answers = GeneratedFinalAnswersBatch(
            dataset_id=paths_batch.dataset_id,
            evaluation_run_name=paths_batch.evaluation_run_name,
            model_id=model_id,
            llm_provider=llm_provider,
            reasoning_effort=self.reasoning_effort,
            items=all_items,
        )
        self.storage_service.write_inference_config(run=run, answers=final_answers)
        logger.info(
            f"Finished batched LLM inference: "
            f"evaluation_run={paths_batch.evaluation_run_name} "
            f"run={run.inference_run_name} total={len(all_items)} "
            f"successful={final_answers.successful_answers} "
            f"failed={final_answers.failed_answers}"
        )
        return SavedLlmInferenceRun(
            dataset_id=final_answers.dataset_id,
            evaluation_run_name=final_answers.evaluation_run_name,
            inference_run_directory=run.inference_run_directory,
            inference_run_name=run.inference_run_name,
            inference_run_number=run.inference_run_number,
            model_id=final_answers.model_id,
            llm_provider=final_answers.llm_provider,
            reasoning_effort=final_answers.reasoning_effort,
            total_instances=len(final_answers.items),
            successful_answers=final_answers.successful_answers,
            failed_answers=final_answers.failed_answers,
            reasoning_path=run.reasoning_path,
            answers_path=run.answers_path,
            inference_config_path=run.inference_config_path,
        )

    @staticmethod
    def _chunk_items(
        items: list[ReasoningPathsForPrediction],
        batch_size: int,
    ) -> list[list[ReasoningPathsForPrediction]]:
        return [
            items[start_index : start_index + batch_size]
            for start_index in range(0, len(items), batch_size)
        ]

    def _generate_answer(
        self,
        item: ReasoningPathsForPrediction,
        model_id: str,
        llm_provider: str,
    ) -> GeneratedAnswerForPrediction:
        return GenerateFinalAnswersBatchStep(
            model_id=model_id,
            llm_provider=llm_provider,
            reasoning_effort=self.reasoning_effort,
            answer_generation_service=self.answer_generation_service,
        )._generate_answer(item)

    def _resolve_llm_provider(
        self,
        context: StepContext[ExtractedReasoningPathsBatch],
    ) -> str:
        if self.llm_provider is not None:
            return self.llm_provider
        pipeline_configuration = getattr(context, "pipeline_configuration", None)
        if isinstance(pipeline_configuration, BuiltPipelineConfiguration):
            return pipeline_configuration.llm_provider
        return "openai"

    def _resolve_model_id(
        self,
        context: StepContext[ExtractedReasoningPathsBatch],
    ) -> str:
        if self.model_id is not None:
            return self.model_id

        pipeline_configuration = getattr(context, "pipeline_configuration", None)
        if isinstance(pipeline_configuration, BuiltPipelineConfiguration):
            return pipeline_configuration.main_llm_model

        raise LlmAnswerGenerationException(
            "LLM inference requires a main LLM model from pipeline configuration."
        )


class SaveInferenceRunStep(
    AbstractStep[SavedLlmInferenceRun, GeneratedFinalAnswersBatch]
):
    """Persist a complete LLM inference run."""

    def __init__(
        self,
        inference_root: str | Path = "data/webqsp/inference",
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

        logger.info(
            f"Saving LLM inference run: evaluation_run={answers.evaluation_run_name} "
            f"model={answers.model_id} root={self.inference_root}"
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
            llm_provider=answers.llm_provider,
            reasoning_effort=answers.reasoning_effort,
            total_instances=len(answers.items),
            successful_answers=answers.successful_answers,
            failed_answers=answers.failed_answers,
            reasoning_path=storage_result.reasoning_path,
            answers_path=storage_result.answers_path,
            inference_config_path=storage_result.inference_config_path,
        )
