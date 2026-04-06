"""Tests for the first preparation step: KG dataset selection."""

import unittest

from pipeline import (
    KnowledgeGraphDatasetSelection,
    Pipeline,
    SelectKnowledgeGraphDatasetStep,
    StepContext,
    UnsupportedKnowledgeGraphDatasetException,
)


class SelectKnowledgeGraphDatasetStepTests(unittest.TestCase):
    def test_step_succeeds_for_fb15k_237(self) -> None:
        step = SelectKnowledgeGraphDatasetStep()
        context = StepContext(result=KnowledgeGraphDatasetSelection(requested_dataset="FB15K-237"))

        result = step.execute(context)

        self.assertEqual(result.dataset_id, "FB15K-237")
        self.assertEqual(result.display_name, "FB15K-237")
        self.assertEqual(result.dataset_family, "knowledge_graph")
        self.assertEqual(result.task_domain, "multi_hop_reasoning")
        self.assertTrue(result.supported)

    def test_step_uses_default_dataset_choice(self) -> None:
        step = SelectKnowledgeGraphDatasetStep()
        context = StepContext(result=KnowledgeGraphDatasetSelection())

        result = step.execute(context)

        self.assertEqual(result.dataset_id, "FB15K-237")

    def test_step_fails_for_unsupported_dataset(self) -> None:
        step = SelectKnowledgeGraphDatasetStep()
        context = StepContext(result=KnowledgeGraphDatasetSelection(requested_dataset="WN18RR"))

        with self.assertRaises(UnsupportedKnowledgeGraphDatasetException):
            step.execute(context)

    def test_pipeline_prepare_runs_dataset_selection_as_first_step(self) -> None:
        pipeline = Pipeline(preparation_steps=[SelectKnowledgeGraphDatasetStep()])
        initial_context = StepContext(result=KnowledgeGraphDatasetSelection())

        execution_result = pipeline.prepare(initial_context)

        self.assertTrue(execution_result.success)
        self.assertEqual(execution_result.steps_executed, 1)
        self.assertEqual(execution_result.final_result.dataset_id, "FB15K-237")

    def test_pipeline_surfaces_failure_metadata_for_unsupported_dataset(self) -> None:
        pipeline = Pipeline(preparation_steps=[SelectKnowledgeGraphDatasetStep()])
        initial_context = StepContext(
            result=KnowledgeGraphDatasetSelection(requested_dataset="CustomKG"),
        )

        execution_result = pipeline.prepare(initial_context)

        self.assertFalse(execution_result.success)
        self.assertEqual(execution_result.steps_executed, 0)
        self.assertEqual(
            execution_result.exception_type,
            UnsupportedKnowledgeGraphDatasetException.__name__,
        )
        self.assertIn("Unsupported knowledge graph dataset", execution_result.error_message)


if __name__ == "__main__":
    unittest.main()
