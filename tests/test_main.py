"""Tests for the graphragX entry point."""

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
from pipeline.preparation.services import AbstractDatasetLoaderService


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
                    subgraph_algorithm="shortest_path",
                    context_strategy="structured_triples",
                    gnn_layer_count=2,
                    gnn_hidden_dimension=256,
                    node_classifier="mlp",
                    question_embedding_model="text-embedding-3-small",
                    relation_embedding_model="text-embedding-3-small",
                    entity_embedding_model="text-embedding-3-small",
                    no_llm_inference=True,
                    no_wandb=True,
                ),
            )

        self.assertTrue(result.success)
        self.assertEqual(result.final_result.dataset_id, "WebQSP")
        self.assertEqual(result.final_result.model_run_number, 1)

    def test_main_logs_success_summary_without_printing_full_payload(self) -> None:
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
                    "--no-llm-inference",
                    "--no-wandb",
                ]
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(stdout.getvalue(), "")

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
                    "--no-llm-inference",
                    "--no-wandb",
                ]
            )

        self.assertEqual(exit_code, 1)
        self.assertEqual(stdout.getvalue(), "")

    def test_main_interactively_prompts_missing_configuration_flags(self) -> None:
        with self._patch_dataset_loading_step(), self._patch_webqsp_local_graph_step(), self._patch_gnn_builder_step(), self._patch_training_step(), self._patch_evaluation_step(), patch(
            "builtins.input",
            side_effect=["1", "2", "1", "1", "1", "1", "1", "1", "1"],
        ), patch(
            "sys.stdout",
            new_callable=StringIO,
        ) as stdout:
            exit_code = main.main(["--dataset", "WebQSP", "--no-llm-inference", "--no-wandb"])

        self.assertEqual(exit_code, 0)
        self.assertNotIn('"success"', stdout.getvalue())

    def test_main_default_flag_runs_without_prompting(self) -> None:
        with self._patch_dataset_loading_step(), self._patch_webqsp_local_graph_step(), self._patch_gnn_builder_step(), self._patch_training_step(), self._patch_evaluation_step(), patch(
            "builtins.input",
            side_effect=AssertionError("input should not be called"),
        ), patch(
            "sys.stdout",
            new_callable=StringIO,
        ) as stdout:
            exit_code = main.main(["--default", "--no-llm-inference", "--no-wandb"])

        self.assertEqual(exit_code, 0)
        self.assertEqual(stdout.getvalue(), "")

    def test_main_no_arguments_does_not_force_default_configuration_values(self) -> None:
        captured_configs: list[main.PipelineRuntimeConfig] = []

        def fake_run_pipeline(config: main.PipelineRuntimeConfig):
            captured_configs.append(config)
            return main.PipelineExecutionResult.success_result(
                final_result=main.InitialStepResult(),
                execution_time_ms=0.0,
                steps_executed=0,
                total_steps=0,
            )

        with patch("main.run_pipeline", side_effect=fake_run_pipeline), patch(
            "sys.stdout",
            new_callable=StringIO,
        ):
            exit_code = main.main([])

        self.assertEqual(exit_code, 0)
        self.assertEqual(len(captured_configs), 1)
        self.assertFalse(captured_configs[0].use_default_config_values)
        self.assertFalse(captured_configs[0].no_llm_inference)
        self.assertFalse(captured_configs[0].no_wandb)
        self.assertIsNone(captured_configs[0].training_max_instances)
        self.assertIsNone(captured_configs[0].evaluation_max_instances)

    def test_run_pipeline_default_config_succeeds_from_neutral_initial_result(self) -> None:
        with self._patch_dataset_loading_step(), self._patch_webqsp_local_graph_step(), self._patch_gnn_builder_step(), self._patch_training_step(), self._patch_evaluation_step(), patch(
            "builtins.input",
            side_effect=AssertionError("input should not be called"),
        ):
            result = main.run_pipeline(
                config=main.PipelineRuntimeConfig(
                    use_default_config_values=True,
                    no_llm_inference=True,
                    no_wandb=True,
                ),
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

    def test_default_pipeline_appends_final_results_and_wandb_steps(self) -> None:
        pipeline = main.build_pipeline(
            config=main.PipelineRuntimeConfig(),
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
        self.assertIsInstance(
            pipeline.evaluation_steps[5],
            LogFinalResultsToWandbStep,
        )
        self.assertEqual(len(pipeline.evaluation_steps), 6)

    def test_no_wandb_keeps_final_results_without_wandb_step(self) -> None:
        pipeline = main.build_pipeline(
            config=main.PipelineRuntimeConfig(
                no_wandb=True,
                wandb_project="project",
                wandb_mode="disabled",
            ),
        )

        self.assertIsInstance(pipeline.evaluation_steps[4], ComputeFinalResultsStep)
        self.assertEqual(len(pipeline.evaluation_steps), 5)

    def test_no_llm_inference_skips_post_retrieval_steps_when_wandb_is_also_disabled(self) -> None:
        pipeline = main.build_pipeline(
            config=main.PipelineRuntimeConfig(
                no_llm_inference=True,
                no_wandb=True,
            ),
        )

        self.assertIsInstance(pipeline.evaluation_steps[0], EvaluateGnnAnswerRetrieverStep)
        self.assertEqual(len(pipeline.evaluation_steps), 1)

    def test_wandb_requires_llm_inference(self) -> None:
        with self.assertRaisesRegex(
            main.PipelineException,
            "requires LLM inference",
        ):
            main.build_pipeline(
                config=main.PipelineRuntimeConfig(no_llm_inference=True),
            )


if __name__ == "__main__":
    unittest.main()
