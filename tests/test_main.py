"""Tests for the graphragX entry point."""

import json
import unittest
from io import StringIO
from unittest.mock import patch

import main


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

    def test_run_pipeline_returns_success_for_fb15k_237(self) -> None:
        result = main.run_pipeline(
            config=main.PipelineRuntimeConfig(
                dataset="FB15K-237",
                main_llm_model="gpt-5.4",
                assistant_llm_model="gpt-5.4-mini",
                subgraph_algorithm="shortest_path",
                context_strategy="textualized",
            ),
        )

        self.assertTrue(result.success)
        self.assertEqual(result.final_result.dataset_id, "FB15K-237")
        self.assertEqual(result.final_result.main_llm_model, "gpt-5.4")

    def test_main_prints_success_payload_for_full_run(self) -> None:
        with patch("sys.stdout", new_callable=StringIO) as stdout:
            exit_code = main.main([
                "--dataset", "FB15K-237",
                "--main-llm-model", "gpt-5.4",
                "--assistant-llm-model", "gpt-5.4-mini",
                "--subgraph-algorithm", "shortest_path",
                "--context-strategy", "textualized",
            ])

        payload = self._extract_json_payload(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertTrue(payload["success"])
        self.assertEqual(payload["final_result"]["dataset_id"], "FB15K-237")
        self.assertEqual(payload["final_result"]["main_llm_model"], "gpt-5.4")

    def test_main_returns_error_for_unsupported_dataset(self) -> None:
        with patch("sys.stdout", new_callable=StringIO) as stdout:
            exit_code = main.main([
                "--dataset", "WN18RR",
                "--main-llm-model", "gpt-5.4",
                "--assistant-llm-model", "gpt-5.4-mini",
                "--subgraph-algorithm", "shortest_path",
                "--context-strategy", "textualized",
            ])

        payload = self._extract_json_payload(stdout.getvalue())
        self.assertEqual(exit_code, 1)
        self.assertFalse(payload["success"])
        self.assertEqual(
            payload["exception_type"],
            "UnsupportedKnowledgeGraphDatasetException",
        )

    def test_main_interactively_prompts_missing_configuration_flags(self) -> None:
        with patch("builtins.input", side_effect=["1", "2", "1", "1"]), patch(
            "sys.stdout", new_callable=StringIO
        ) as stdout:
            exit_code = main.main(["--dataset", "FB15K-237"])

        payload = self._extract_json_payload(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertTrue(payload["success"])
        self.assertEqual(payload["final_result"]["main_llm_model"], "gpt-5.4")

    def test_main_default_flag_runs_without_prompting(self) -> None:
        with patch("builtins.input", side_effect=AssertionError("input should not be called")), patch(
            "sys.stdout", new_callable=StringIO
        ) as stdout:
            exit_code = main.main(["--default"])

        payload = self._extract_json_payload(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertTrue(payload["success"])
        self.assertEqual(payload["final_result"]["dataset_id"], "FB15K-237")
        self.assertEqual(payload["final_result"]["main_llm_model"], "gpt-5.4")
        self.assertEqual(payload["final_result"]["assistant_llm_model"], "gpt-5.4-mini")
        self.assertEqual(
            payload["final_result"]["subgraph_construction_algorithm"],
            "shortest_path",
        )
        self.assertEqual(
            payload["final_result"]["context_construction_strategy"],
            "textualized",
        )

    def test_run_pipeline_default_config_succeeds_from_neutral_initial_result(self) -> None:
        with patch("builtins.input", side_effect=AssertionError("input should not be called")):
            result = main.run_pipeline(
                config=main.PipelineRuntimeConfig(force_all_default=True),
            )

        self.assertTrue(result.success)
        self.assertEqual(result.final_result.dataset_id, "FB15K-237")
        self.assertEqual(result.final_result.main_llm_model, "gpt-5.4")


if __name__ == "__main__":
    unittest.main()
