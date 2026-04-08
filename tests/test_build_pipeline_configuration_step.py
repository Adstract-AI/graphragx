"""Tests for the preparation configuration-building step."""

import unittest

from pipeline import (
    BuildPipelineConfigurationStep,
    InvalidAssistantLlmSelectionException,
    InvalidContextConstructionSelectionException,
    InvalidGnnArchitectureSelectionException,
    InvalidMainLlmSelectionException,
    InvalidSubgraphConstructionSelectionException,
    Pipeline,
    SelectedKnowledgeGraphDataset,
    StepContext,
)


class BuildPipelineConfigurationStepTests(unittest.TestCase):
    def make_dataset_context(self) -> StepContext[SelectedKnowledgeGraphDataset]:
        return StepContext(
            result=SelectedKnowledgeGraphDataset(
                dataset_id="FB15K-237",
                display_name="FB15K-237",
                dataset_family="knowledge_graph",
                task_domain="multi_hop_reasoning",
                description="dataset",
                supported=True,
            )
        )

    def test_constructor_only_path_succeeds(self) -> None:
        step = BuildPipelineConfigurationStep(
            main_llm_model="gpt-5.4",
            assistant_llm_model="gpt-5.4-mini",
            subgraph_algorithm="shortest_path",
            context_strategy="textualized",
            gnn_architecture="rgcn",
        )

        result = step.execute(self.make_dataset_context())

        self.assertEqual(result.dataset_id, "FB15K-237")
        self.assertEqual(result.main_llm_model, "gpt-5.4")
        self.assertEqual(result.assistant_llm_model, "gpt-5.4-mini")
        self.assertEqual(result.subgraph_construction_algorithm, "shortest_path")
        self.assertEqual(result.context_construction_strategy, "textualized")
        self.assertEqual(result.gnn_architecture, "rgcn")

    def test_mixed_path_prompts_only_missing_fields(self) -> None:
        answers = iter(["2", "2", "1"])
        prompts: list[str] = []

        def fake_input(prompt: str) -> str:
            prompts.append(prompt)
            return next(answers)

        step = BuildPipelineConfigurationStep(
            main_llm_model="gpt-5.4",
            subgraph_algorithm="shortest_path",
            input_func=fake_input,
        )

        result = step.execute(self.make_dataset_context())

        self.assertEqual(result.main_llm_model, "gpt-5.4")
        self.assertEqual(result.assistant_llm_model, "gpt-5.4-mini")
        self.assertEqual(result.subgraph_construction_algorithm, "shortest_path")
        self.assertEqual(result.context_construction_strategy, "structured_triples")
        self.assertEqual(result.gnn_architecture, "rgcn")
        self.assertEqual(len(prompts), 3)

    def test_fully_interactive_path_works(self) -> None:
        answers = iter(["1", "2", "1", "2", "1"])

        step = BuildPipelineConfigurationStep(input_func=lambda _: next(answers))
        result = step.execute(self.make_dataset_context())

        self.assertEqual(result.main_llm_model, "gpt-5.4")
        self.assertEqual(result.assistant_llm_model, "gpt-5.4-mini")
        self.assertEqual(result.subgraph_construction_algorithm, "shortest_path")
        self.assertEqual(result.context_construction_strategy, "structured_triples")
        self.assertEqual(result.gnn_architecture, "rgcn")

    def test_invalid_main_llm_constructor_value_raises(self) -> None:
        step = BuildPipelineConfigurationStep(
            main_llm_model="invalid-model",
            assistant_llm_model="gpt-5.4-mini",
            subgraph_algorithm="shortest_path",
            context_strategy="textualized",
            gnn_architecture="rgcn",
        )

        with self.assertRaises(InvalidMainLlmSelectionException):
            step.execute(self.make_dataset_context())

    def test_invalid_assistant_llm_constructor_value_raises(self) -> None:
        step = BuildPipelineConfigurationStep(
            main_llm_model="gpt-5.4",
            assistant_llm_model="invalid-model",
            subgraph_algorithm="shortest_path",
            context_strategy="textualized",
            gnn_architecture="rgcn",
        )

        with self.assertRaises(InvalidAssistantLlmSelectionException):
            step.execute(self.make_dataset_context())

    def test_invalid_subgraph_constructor_value_raises(self) -> None:
        step = BuildPipelineConfigurationStep(
            main_llm_model="gpt-5.4",
            assistant_llm_model="gpt-5.4-mini",
            subgraph_algorithm="invalid",
            context_strategy="textualized",
            gnn_architecture="rgcn",
        )

        with self.assertRaises(InvalidSubgraphConstructionSelectionException):
            step.execute(self.make_dataset_context())

    def test_invalid_context_strategy_constructor_value_raises(self) -> None:
        step = BuildPipelineConfigurationStep(
            main_llm_model="gpt-5.4",
            assistant_llm_model="gpt-5.4-mini",
            subgraph_algorithm="shortest_path",
            context_strategy="invalid",
            gnn_architecture="rgcn",
        )

        with self.assertRaises(InvalidContextConstructionSelectionException):
            step.execute(self.make_dataset_context())

    def test_invalid_gnn_architecture_constructor_value_raises(self) -> None:
        step = BuildPipelineConfigurationStep(
            main_llm_model="gpt-5.4",
            assistant_llm_model="gpt-5.4-mini",
            subgraph_algorithm="shortest_path",
            context_strategy="textualized",
            gnn_architecture="invalid",
        )

        with self.assertRaises(InvalidGnnArchitectureSelectionException):
            step.execute(self.make_dataset_context())

    def test_interactive_invalid_numeric_input_reprompts(self) -> None:
        answers = iter(["abc", "1", "2", "1", "1", "1"])
        step = BuildPipelineConfigurationStep(input_func=lambda _: next(answers))

        result = step.execute(self.make_dataset_context())

        self.assertEqual(result.main_llm_model, "gpt-5.4")

    def test_interactive_out_of_range_input_reprompts(self) -> None:
        answers = iter(["99", "1", "2", "1", "1", "1"])
        step = BuildPipelineConfigurationStep(input_func=lambda _: next(answers))

        result = step.execute(self.make_dataset_context())

        self.assertEqual(result.main_llm_model, "gpt-5.4")

    def test_configuration_result_is_stored_in_result_bank(self) -> None:
        pipeline = Pipeline(
            preparation_steps=[
                BuildPipelineConfigurationStep(
                    main_llm_model="gpt-5.4",
                    assistant_llm_model="gpt-5.4-mini",
                    subgraph_algorithm="shortest_path",
                    context_strategy="textualized",
                    gnn_architecture="rgcn",
                )
            ]
        )

        result = pipeline.prepare(self.make_dataset_context())

        self.assertTrue(result.success)
        self.assertTrue(
            pipeline.context_builder.has_stored_result(type(result.final_result))
        )
