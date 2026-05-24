"""Tests for the graphragX entry point."""

import json
import unittest
from io import StringIO
from unittest.mock import patch

import main
from pipeline import (
    BuildReasoningSamplesFromGnnEvaluationStep,
    BuildGnnAnswerRetrieverStep,
    BuildWebQSPLocalGraphsStep,
    BuiltGnnAnswerRetriever,
    ComputeFinalResultsStep,
    EvaluateGnnAnswerRetrieverStep,
    ExtractShortestPathsBatchStep,
    GenerateAndSaveFinalAnswersBatchesStep,
    GnnAnswerRetrieverEvaluationResult,
    LogFinalResultsToWandbStep,
    LoadDatasetStep,
    PreparedWebQSPGraphDataset,
    TrainGnnAnswerRetrieverStep,
    TrainedGnnAnswerRetriever,
    WebQSPVocabularyStore,
)
from pipeline.preparation.models.interfaces import AnswerRetrieverModel
from pipeline.services import AbstractDatasetLoaderService


class FakeSplit:
    def __init__(self, size: int):
        self.size = size

    def __len__(self) -> int:
        return self.size


class FakeLoaderService(AbstractDatasetLoaderService):
    def load_dataset(self, dataset_id: str) -> dict[str, FakeSplit]:
        return {
            "train": FakeSplit(3),
            "validation": FakeSplit(2),
            "test": FakeSplit(1),
        }


class FakeAnswerRetrieverModel(AnswerRetrieverModel):
    pass


class FakeGnnAnswerRetrieverStep(BuildGnnAnswerRetrieverStep):
    def execute_default(self, context):
        return BuiltGnnAnswerRetriever(
            dataset_id=context.result.dataset_id,
            entity_embedding_model="text-embedding-3-small",
            entity_embedding_dimension=1536,
            hidden_dimension=256,
            gnn_layer_count=2,
            node_classifier="mlp",
            model=FakeAnswerRetrieverModel(),
        )


class FakeWebQSPLocalGraphsStep(BuildWebQSPLocalGraphsStep):
    def execute_default(self, context):
        return PreparedWebQSPGraphDataset(
            dataset_id=context.result.dataset_id,
            processing_version="test",
            train_instances=[],
            test_instances=[],
            vocabulary_store=WebQSPVocabularyStore(),
            cache_directory="/tmp/graphragx-test",
        )


class FakeTrainGnnAnswerRetrieverStep(TrainGnnAnswerRetrieverStep):
    def execute_default(self, context):
        return TrainedGnnAnswerRetriever(
            dataset_id=context.result.dataset_id,
            hidden_dimension=context.result.hidden_dimension,
            gnn_layer_count=context.result.gnn_layer_count,
            node_classifier=context.result.node_classifier,
            training_epochs=1,
            training_learning_rate=1e-3,
            training_weight_decay=0.0,
            training_max_instances=None,
            training_log_every=25,
            training_device="cpu",
            training_run_name=None,
            selected_device="cpu",
            final_loss=0.0,
            trained_instances=0,
            model=context.result.model,
            model_artifact_path="/tmp/graphragx-test/gnn_answer_retriever.pt",
            model_config_path="/tmp/graphragx-test/model_config.json",
            model_run_directory="/tmp/graphragx-test/1_test",
            model_run_name="1_test",
            model_run_number=1,
            embedding_cache_directory="/tmp/graphragx-test/embeddings",
        )


class FakeEvaluateGnnAnswerRetrieverStep(EvaluateGnnAnswerRetrieverStep):
    def execute_default(self, context):
        return GnnAnswerRetrieverEvaluationResult(
            dataset_id=context.result.dataset_id,
            model_run_directory=context.result.model_run_directory,
            model_run_name=context.result.model_run_name,
            model_run_number=context.result.model_run_number,
            evaluation_run_directory="/tmp/graphragx-test/evaluations/1_test",
            evaluation_run_name="1_test",
            evaluation_run_number=1,
            evaluated_instances=0,
            hits_at_1=0.0,
            hits_at_1_count=0,
            hits_at_5=0.0,
            hits_at_5_count=0,
            hits_at_10=0.0,
            hits_at_10_count=0,
            average_candidate_count=0.0,
            missing_gold_in_graph_count=0,
            predictions_path="/tmp/graphragx-test/evaluations/1_test/predictions.jsonl",
            evaluation_config_path="/tmp/graphragx-test/evaluations/1_test/evaluation_config.json",
        )


class MainEntrypointTests(unittest.TestCase):
    @staticmethod
    def _extract_json_payload(raw_output: str) -> dict:
        json_start = raw_output.rfind("{")
        while json_start != -1:
            try:
                return json.loads(raw_output[json_start:])
            except json.JSONDecodeError:
                json_start = raw_output.rfind("{", 0, json_start)

        raise AssertionError("No JSON payload found in output.")

    @staticmethod
    def _patch_dataset_loading_step():
        return patch(
            "main.LoadDatasetStep",
            return_value=LoadDatasetStep(
                loader_service=FakeLoaderService(),
            ),
        )

    @staticmethod
    def _patch_gnn_builder_step():
        return patch(
            "main.BuildGnnAnswerRetrieverStep",
            return_value=FakeGnnAnswerRetrieverStep(),
        )

    @staticmethod
    def _patch_webqsp_local_graph_step():
        return patch(
            "main.BuildWebQSPLocalGraphsStep",
            return_value=FakeWebQSPLocalGraphsStep(),
        )

    @staticmethod
    def _patch_training_step():
        return patch(
            "main.TrainGnnAnswerRetrieverStep",
            return_value=FakeTrainGnnAnswerRetrieverStep(),
        )

    @staticmethod
    def _patch_evaluation_step():
        return patch(
            "main.EvaluateGnnAnswerRetrieverStep",
            return_value=FakeEvaluateGnnAnswerRetrieverStep(),
        )

    def test_run_pipeline_returns_success_for_webqsp(self) -> None:
        with self._patch_dataset_loading_step(), self._patch_webqsp_local_graph_step(), self._patch_gnn_builder_step(), self._patch_training_step(), self._patch_evaluation_step():
            result = main.run_pipeline(
                config=main.PipelineRuntimeConfig(
                    dataset="WebQSP",
                    main_llm_model="gpt-5.4",
                    assistant_llm_model="gpt-5.4-mini",
                    subgraph_algorithm="shortest_path",
                    context_strategy="structured_triples",
                    gnn_layer_count=2,
                    gnn_hidden_dimension=256,
                    node_classifier="mlp",
                    question_embedding_model="text-embedding-3-small",
                    relation_embedding_model="text-embedding-3-small",
                    entity_embedding_model="text-embedding-3-small",
                ),
            )

        self.assertTrue(result.success)
        self.assertEqual(result.final_result.dataset_id, "WebQSP")
        self.assertEqual(result.final_result.model_run_number, 1)

    def test_main_prints_success_payload_for_full_run(self) -> None:
        with self._patch_dataset_loading_step(), self._patch_webqsp_local_graph_step(), self._patch_gnn_builder_step(), self._patch_training_step(), self._patch_evaluation_step(), patch(
            "sys.stdout",
            new_callable=StringIO,
        ) as stdout:
            exit_code = main.main(
                [
                    "--dataset",
                    "WebQSP",
                    "--main-llm-model",
                    "gpt-5.4",
                    "--assistant-llm-model",
                    "gpt-5.4-mini",
                    "--subgraph-algorithm",
                    "shortest_path",
                    "--context-strategy",
                    "structured_triples",
                    "--gnn-layers",
                    "2",
                    "--gnn-hidden-dim",
                    "256",
                    "--node-classifier",
                    "mlp",
                    "--question-embedding-model",
                    "text-embedding-3-small",
                    "--relation-embedding-model",
                    "text-embedding-3-small",
                    "--entity-embedding-model",
                    "text-embedding-3-small",
                ]
            )

        payload = self._extract_json_payload(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertTrue(payload["success"])
        self.assertEqual(payload["final_result"]["dataset_id"], "WebQSP")
        self.assertEqual(payload["final_result"]["model_run_number"], 1)

    def test_main_returns_error_for_unsupported_dataset(self) -> None:
        with self._patch_dataset_loading_step(), self._patch_webqsp_local_graph_step(), self._patch_gnn_builder_step(), self._patch_training_step(), self._patch_evaluation_step(), patch(
            "sys.stdout",
            new_callable=StringIO,
        ) as stdout:
            exit_code = main.main(
                [
                    "--dataset",
                    "WN18RR",
                    "--main-llm-model",
                    "gpt-5.4",
                    "--assistant-llm-model",
                    "gpt-5.4-mini",
                    "--subgraph-algorithm",
                    "shortest_path",
                    "--context-strategy",
                    "structured_triples",
                    "--gnn-layers",
                    "2",
                    "--gnn-hidden-dim",
                    "256",
                    "--node-classifier",
                    "mlp",
                    "--question-embedding-model",
                    "text-embedding-3-small",
                    "--relation-embedding-model",
                    "text-embedding-3-small",
                    "--entity-embedding-model",
                    "text-embedding-3-small",
                ]
            )

        payload = self._extract_json_payload(stdout.getvalue())
        self.assertEqual(exit_code, 1)
        self.assertFalse(payload["success"])
        self.assertEqual(
            payload["exception_type"],
            "UnsupportedDatasetSelectionException",
        )

    def test_main_interactively_prompts_missing_configuration_flags(self) -> None:
        with self._patch_dataset_loading_step(), self._patch_webqsp_local_graph_step(), self._patch_gnn_builder_step(), self._patch_training_step(), self._patch_evaluation_step(), patch(
            "builtins.input",
            side_effect=["1", "2", "1", "1", "2", "1", "1", "1", "1", "1"],
        ), patch(
            "sys.stdout",
            new_callable=StringIO,
        ) as stdout:
            exit_code = main.main(["--dataset", "WebQSP"])

        payload = self._extract_json_payload(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertTrue(payload["success"])
        self.assertEqual(payload["final_result"]["model_run_number"], 1)

    def test_main_default_flag_runs_without_prompting(self) -> None:
        with self._patch_dataset_loading_step(), self._patch_webqsp_local_graph_step(), self._patch_gnn_builder_step(), self._patch_training_step(), self._patch_evaluation_step(), patch(
            "builtins.input",
            side_effect=AssertionError("input should not be called"),
        ), patch(
            "sys.stdout",
            new_callable=StringIO,
        ) as stdout:
            exit_code = main.main(["--default"])

        payload = self._extract_json_payload(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertTrue(payload["success"])
        self.assertEqual(payload["final_result"]["dataset_id"], "WebQSP")
        self.assertEqual(payload["final_result"]["model_run_number"], 1)

    def test_run_pipeline_default_config_succeeds_from_neutral_initial_result(self) -> None:
        with self._patch_dataset_loading_step(), self._patch_webqsp_local_graph_step(), self._patch_gnn_builder_step(), self._patch_training_step(), self._patch_evaluation_step(), patch(
            "builtins.input",
            side_effect=AssertionError("input should not be called"),
        ):
            result = main.run_pipeline(
                config=main.PipelineRuntimeConfig(use_default_config_values=True),
            )

        self.assertTrue(result.success)
        self.assertEqual(result.final_result.dataset_id, "WebQSP")
        self.assertEqual(result.final_result.model_run_number, 1)

    def test_force_default_flag_sets_step_execution_mode_only(self) -> None:
        pipeline = main.build_pipeline(
            config=main.PipelineRuntimeConfig(force_all_default=True),
        )

        self.assertTrue(pipeline.force_all_default)
        self.assertTrue(all(step.force_default for step in pipeline.preparation_steps))
        self.assertIsNone(
            pipeline.preparation_steps[0].requested_dataset,
        )

    def test_llm_inference_flag_appends_post_retrieval_steps_only(self) -> None:
        pipeline = main.build_pipeline(
            config=main.PipelineRuntimeConfig(with_llm_inference=True),
        )

        self.assertIsInstance(pipeline.evaluation_steps[0], EvaluateGnnAnswerRetrieverStep)
        self.assertIsInstance(
            pipeline.evaluation_steps[1],
            BuildReasoningSamplesFromGnnEvaluationStep,
        )
        self.assertIsInstance(pipeline.evaluation_steps[2], ExtractShortestPathsBatchStep)
        self.assertIsInstance(
            pipeline.evaluation_steps[3],
            GenerateAndSaveFinalAnswersBatchesStep,
        )
        self.assertIsInstance(pipeline.evaluation_steps[4], ComputeFinalResultsStep)
        self.assertEqual(len(pipeline.evaluation_steps), 5)

    def test_wandb_flag_appends_after_final_results_only_when_requested(self) -> None:
        pipeline = main.build_pipeline(
            config=main.PipelineRuntimeConfig(
                with_llm_inference=True,
                with_wandb=True,
                wandb_project="project",
                wandb_mode="disabled",
            ),
        )

        self.assertIsInstance(pipeline.evaluation_steps[4], ComputeFinalResultsStep)
        self.assertIsInstance(
            pipeline.evaluation_steps[5],
            LogFinalResultsToWandbStep,
        )
        self.assertEqual(len(pipeline.evaluation_steps), 6)

    def test_wandb_requires_llm_inference(self) -> None:
        with self.assertRaisesRegex(
            main.PipelineException,
            "requires --with-llm-inference",
        ):
            main.build_pipeline(
                config=main.PipelineRuntimeConfig(with_wandb=True),
            )


if __name__ == "__main__":
    unittest.main()
