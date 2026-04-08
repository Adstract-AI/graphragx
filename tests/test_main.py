"""Tests for the graphragX entry point."""

import json
import unittest
from io import StringIO
from unittest.mock import patch

import main
from pipeline import LoadKnowledgeGraphDatasetStep
from pipeline.services import AbstractDatasetLoaderService


class FakeTensor:
    def __init__(self, values):
        self._values = values

    def tolist(self):
        return list(self._values)

    def __getitem__(self, index):
        return FakeTensor(self._values[index])


class FakeData:
    edge_index = FakeTensor([[0, 1], [1, 0]])
    edge_type = FakeTensor([0, 0])
    num_nodes = 2


class FakeDataset:
    num_relations = 1

    def __getitem__(self, index):
        if index != 0:
            raise IndexError(index)
        return FakeData()


class FakeLoaderService(AbstractDatasetLoaderService):
    def load_dataset(self, dataset_id: str) -> tuple[FakeDataset, FakeData]:
        dataset = FakeDataset()
        return dataset, dataset[0]


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
            "main.LoadKnowledgeGraphDatasetStep",
            return_value=LoadKnowledgeGraphDatasetStep(
                loader_service=FakeLoaderService(),
            ),
        )

    def test_run_pipeline_returns_success_for_fb15k_237(self) -> None:
        with self._patch_dataset_loading_step():
            result = main.run_pipeline(
                config=main.PipelineRuntimeConfig(
                    dataset="FB15K-237",
                    main_llm_model="gpt-5.4",
                    assistant_llm_model="gpt-5.4-mini",
                    subgraph_algorithm="shortest_path",
                    context_strategy="textualized",
                    gnn_architecture="rgcn",
                ),
            )

        self.assertTrue(result.success)
        self.assertEqual(result.final_result.dataset_id, "FB15K-237")
        self.assertEqual(result.final_result.triple_count, 2)

    def test_main_prints_success_payload_for_full_run(self) -> None:
        with self._patch_dataset_loading_step(), patch("sys.stdout", new_callable=StringIO) as stdout:
            exit_code = main.main([
                "--dataset", "FB15K-237",
                "--main-llm-model", "gpt-5.4",
                "--assistant-llm-model", "gpt-5.4-mini",
                "--subgraph-algorithm", "shortest_path",
                "--context-strategy", "textualized",
                "--gnn-architecture", "rgcn",
            ])

        payload = self._extract_json_payload(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertTrue(payload["success"])
        self.assertEqual(payload["final_result"]["dataset_id"], "FB15K-237")
        self.assertEqual(payload["final_result"]["triple_count"], 2)

    def test_main_returns_error_for_unsupported_dataset(self) -> None:
        with self._patch_dataset_loading_step(), patch("sys.stdout", new_callable=StringIO) as stdout:
            exit_code = main.main([
                "--dataset", "WN18RR",
                "--main-llm-model", "gpt-5.4",
                "--assistant-llm-model", "gpt-5.4-mini",
                "--subgraph-algorithm", "shortest_path",
                "--context-strategy", "textualized",
                "--gnn-architecture", "rgcn",
            ])

        payload = self._extract_json_payload(stdout.getvalue())
        self.assertEqual(exit_code, 1)
        self.assertFalse(payload["success"])
        self.assertEqual(
            payload["exception_type"],
            "UnsupportedKnowledgeGraphDatasetException",
        )

    def test_main_interactively_prompts_missing_configuration_flags(self) -> None:
        with self._patch_dataset_loading_step(), patch("builtins.input", side_effect=["1", "2", "1", "1", "1"]), patch(
            "sys.stdout", new_callable=StringIO
        ) as stdout:
            exit_code = main.main(["--dataset", "FB15K-237"])

        payload = self._extract_json_payload(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertTrue(payload["success"])
        self.assertEqual(payload["final_result"]["triple_count"], 2)

    def test_main_default_flag_runs_without_prompting(self) -> None:
        with self._patch_dataset_loading_step(), patch("builtins.input", side_effect=AssertionError("input should not be called")), patch(
            "sys.stdout", new_callable=StringIO
        ) as stdout:
            exit_code = main.main(["--default"])

        payload = self._extract_json_payload(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertTrue(payload["success"])
        self.assertEqual(payload["final_result"]["dataset_id"], "FB15K-237")
        self.assertEqual(payload["final_result"]["triple_count"], 2)
        self.assertEqual(payload["final_result"]["entity_count"], 2)

    def test_run_pipeline_default_config_succeeds_from_neutral_initial_result(self) -> None:
        with self._patch_dataset_loading_step(), patch("builtins.input", side_effect=AssertionError("input should not be called")):
            result = main.run_pipeline(
                config=main.PipelineRuntimeConfig(use_default_config_values=True),
            )

        self.assertTrue(result.success)
        self.assertEqual(result.final_result.dataset_id, "FB15K-237")
        self.assertEqual(result.final_result.triple_count, 2)

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
