"""Tests for the first preparation step: dataset selection."""

import unittest

from pipeline import (
    InitialStepResult,
    Pipeline,
    SelectDatasetStep,
    StepContext,
    UnsupportedDatasetSelectionException,
)


class SelectDatasetStepTests(unittest.TestCase):
    def test_step_succeeds_for_webqsp(self) -> None:
        step = SelectDatasetStep(requested_dataset="WebQSP")
        context = StepContext(result=InitialStepResult())

        result = step.execute(context)

        self.assertEqual(result.dataset_id, "WebQSP")
        self.assertEqual(result.display_name, "WebQSP")
        self.assertEqual(result.dataset_family, "question_answering")
        self.assertEqual(result.task_domain, "knowledge_graph_question_answering")
        self.assertTrue(result.supported)

    def test_step_uses_default_dataset_choice(self) -> None:
        step = SelectDatasetStep(requested_dataset="WebQSP")
        context = StepContext(result=InitialStepResult())

        result = step.execute(context)

        self.assertEqual(result.dataset_id, "WebQSP")

    def test_step_fails_for_unsupported_dataset(self) -> None:
        step = SelectDatasetStep(requested_dataset="WN18RR")
        context = StepContext(result=InitialStepResult())

        with self.assertRaises(UnsupportedDatasetSelectionException):
            step.execute(context)

    def test_pipeline_prepare_runs_dataset_selection_as_first_step(self) -> None:
        pipeline = Pipeline(preparation_steps=[SelectDatasetStep(requested_dataset="WebQSP")])
        initial_context = StepContext(result=InitialStepResult())

        execution_result = pipeline.prepare(initial_context)

        self.assertTrue(execution_result.success)
        self.assertEqual(execution_result.steps_executed, 1)
        self.assertEqual(execution_result.final_result.dataset_id, "WebQSP")

    def test_pipeline_surfaces_failure_metadata_for_unsupported_dataset(self) -> None:
        pipeline = Pipeline(preparation_steps=[SelectDatasetStep(requested_dataset="CustomKG")])
        initial_context = StepContext(result=InitialStepResult())

        execution_result = pipeline.prepare(initial_context)

        self.assertFalse(execution_result.success)
        self.assertEqual(execution_result.steps_executed, 0)
        self.assertEqual(
            execution_result.exception_type,
            UnsupportedDatasetSelectionException.__name__,
        )
        self.assertIn("Invalid selection", execution_result.error_message)

    def test_step_interactively_selects_dataset_when_missing(self) -> None:
        step = SelectDatasetStep(input_func=lambda _: "1")
        context = StepContext(result=InitialStepResult())

        result = step.execute(context)

        self.assertEqual(result.dataset_id, "WebQSP")


if __name__ == "__main__":
    unittest.main()
