"""Tests for the graphragX entry point."""

import json
import unittest
from io import StringIO
from unittest.mock import patch

import main


class MainEntrypointTests(unittest.TestCase):
    def test_run_pipeline_returns_success_for_fb15k_237(self) -> None:
        result = main.run_pipeline(dataset="FB15K-237")

        self.assertTrue(result.success)
        self.assertEqual(result.final_result.dataset_id, "FB15K-237")

    def test_main_prints_success_payload_for_full_run(self) -> None:
        with patch("sys.stdout", new_callable=StringIO) as stdout:
            exit_code = main.main(["--dataset", "FB15K-237"])

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertTrue(payload["success"])
        self.assertEqual(payload["final_result"]["dataset_id"], "FB15K-237")

    def test_main_returns_error_for_unsupported_dataset(self) -> None:
        with patch("sys.stdout", new_callable=StringIO) as stdout:
            exit_code = main.main(["--dataset", "WN18RR"])

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 1)
        self.assertFalse(payload["success"])
        self.assertEqual(
            payload["exception_type"],
            "UnsupportedKnowledgeGraphDatasetException",
        )


if __name__ == "__main__":
    unittest.main()
