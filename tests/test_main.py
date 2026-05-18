"""Tests for the graphragX entry point."""

import json
import unittest
from io import StringIO
from unittest.mock import patch

import main
from pipeline import LoadDatasetStep
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

    def test_run_pipeline_returns_success_for_webqsp(self) -> None:
        with self._patch_dataset_loading_step():
            result = main.run_pipeline(
                config=main.PipelineRuntimeConfig(
                    dataset="WebQSP",
                    main_llm_model="gpt-5.4",
                    assistant_llm_model="gpt-5.4-mini",
                    subgraph_algorithm="shortest_path",
                    context_strategy="textualized",
                    gnn_layer_count=2,
                    node_classifier="mlp",
                    question_embedding_model="text-embedding-3-small",
                    relation_embedding_model="text-embedding-3-small",
                    entity_embedding_model="text-embedding-3-small",
                ),
            )

        self.assertTrue(result.success)
        self.assertEqual(result.final_result.dataset_id, "WebQSP")
        self.assertEqual(result.final_result.split_sizes["train"], 3)

    def test_main_prints_success_payload_for_full_run(self) -> None:
        with self._patch_dataset_loading_step(), patch("sys.stdout", new_callable=StringIO) as stdout:
            exit_code = main.main([
                "--dataset", "WebQSP",
                "--main-llm-model", "gpt-5.4",
                "--assistant-llm-model", "gpt-5.4-mini",
                "--subgraph-algorithm", "shortest_path",
                "--context-strategy", "textualized",
                "--gnn-layers", "2",
                "--node-classifier", "mlp",
                "--question-embedding-model", "text-embedding-3-small",
                "--relation-embedding-model", "text-embedding-3-small",
                "--entity-embedding-model", "text-embedding-3-small",
            ])

        payload = self._extract_json_payload(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertTrue(payload["success"])
        self.assertEqual(payload["final_result"]["dataset_id"], "WebQSP")
        self.assertEqual(payload["final_result"]["split_sizes"]["train"], 3)

    def test_main_returns_error_for_unsupported_dataset(self) -> None:
        with self._patch_dataset_loading_step(), patch("sys.stdout", new_callable=StringIO) as stdout:
            exit_code = main.main([
                "--dataset", "WN18RR",
                "--main-llm-model", "gpt-5.4",
                "--assistant-llm-model", "gpt-5.4-mini",
                "--subgraph-algorithm", "shortest_path",
                "--context-strategy", "textualized",
                "--gnn-layers", "2",
                "--node-classifier", "mlp",
                "--question-embedding-model", "text-embedding-3-small",
                "--relation-embedding-model", "text-embedding-3-small",
                "--entity-embedding-model", "text-embedding-3-small",
            ])

        payload = self._extract_json_payload(stdout.getvalue())
        self.assertEqual(exit_code, 1)
        self.assertFalse(payload["success"])
        self.assertEqual(
            payload["exception_type"],
            "UnsupportedDatasetSelectionException",
        )

    def test_main_interactively_prompts_missing_configuration_flags(self) -> None:
        with self._patch_dataset_loading_step(), patch("builtins.input", side_effect=["1", "1", "1", "2", "1", "1", "1", "1", "1"]), patch(
            "sys.stdout", new_callable=StringIO
        ) as stdout:
            exit_code = main.main(["--dataset", "WebQSP"])

        payload = self._extract_json_payload(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertTrue(payload["success"])
        self.assertEqual(payload["final_result"]["split_sizes"]["train"], 3)

    def test_main_default_flag_runs_without_prompting(self) -> None:
        with self._patch_dataset_loading_step(), patch("builtins.input", side_effect=AssertionError("input should not be called")), patch(
            "sys.stdout", new_callable=StringIO
        ) as stdout:
            exit_code = main.main(["--default"])

        payload = self._extract_json_payload(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertTrue(payload["success"])
        self.assertEqual(payload["final_result"]["dataset_id"], "WebQSP")
        self.assertEqual(payload["final_result"]["split_sizes"]["train"], 3)

    def test_run_pipeline_default_config_succeeds_from_neutral_initial_result(self) -> None:
        with self._patch_dataset_loading_step(), patch("builtins.input", side_effect=AssertionError("input should not be called")):
            result = main.run_pipeline(
                config=main.PipelineRuntimeConfig(use_default_config_values=True),
            )

        self.assertTrue(result.success)
        self.assertEqual(result.final_result.dataset_id, "WebQSP")
        self.assertEqual(result.final_result.split_sizes["train"], 3)

    def test_force_default_flag_sets_step_execution_mode_only(self) -> None:
        pipeline = main.build_pipeline(
            config=main.PipelineRuntimeConfig(force_all_default=True),
        )

        self.assertTrue(pipeline.force_all_default)
        self.assertTrue(all(step.force_default for step in pipeline.preparation_steps))
        self.assertIsNone(
            pipeline.preparation_steps[0].requested_dataset,
        )


if __name__ == "__main__":
    unittest.main()
