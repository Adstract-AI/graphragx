"""Tests for the preparation configuration-building step."""

import unittest

from pipeline import (
    BuildPipelineConfigurationStep,
    InvalidAssistantLlmSelectionException,
    InvalidContextConstructionSelectionException,
    InvalidEntityEmbeddingModelSelectionException,
    InvalidGnnLayerCountSelectionException,
    InvalidMainLlmSelectionException,
    InvalidNodeClassifierSelectionException,
    InvalidSubgraphConstructionSelectionException,
    Pipeline,
    SelectedDataset,
    StepContext,
)


class BuildPipelineConfigurationStepTests(unittest.TestCase):
    def make_dataset_context(self) -> StepContext[SelectedDataset]:
        return StepContext(
            result=SelectedDataset(
                dataset_id="WebQSP",
                display_name="WebQSP",
                dataset_family="question_answering",
                task_domain="knowledge_graph_question_answering",
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
            gnn_layer_count=2,
            node_classifier="mlp",
            question_embedding_model="text-embedding-3-small",
            relation_embedding_model="text-embedding-3-small",
            entity_embedding_model="text-embedding-3-small",
        )

        result = step.execute(self.make_dataset_context())

        self.assertEqual(result.dataset_id, "WebQSP")
        self.assertEqual(result.main_llm_model, "gpt-5.4")
        self.assertEqual(result.assistant_llm_model, "gpt-5.4-mini")
        self.assertEqual(result.subgraph_construction_algorithm, "shortest_path")
        self.assertEqual(result.context_construction_strategy, "textualized")
        self.assertEqual(result.gnn_layer_count, 2)
        self.assertEqual(result.node_classifier, "mlp")
        self.assertEqual(result.question_embedding_model, "text-embedding-3-small")
        self.assertEqual(result.relation_embedding_model, "text-embedding-3-small")
        self.assertEqual(result.entity_embedding_model, "text-embedding-3-small")

    def test_mixed_path_prompts_only_missing_fields(self) -> None:
        answers = iter(["2", "2"])
        prompts: list[str] = []

        def fake_input(prompt: str) -> str:
            prompts.append(prompt)
            return next(answers)

        step = BuildPipelineConfigurationStep(
            main_llm_model="gpt-5.4",
            subgraph_algorithm="shortest_path",
            gnn_layer_count=2,
            node_classifier="mlp",
            question_embedding_model="text-embedding-3-small",
            relation_embedding_model="text-embedding-3-small",
            entity_embedding_model="text-embedding-3-small",
            input_func=fake_input,
        )

        result = step.execute(self.make_dataset_context())

        self.assertEqual(result.main_llm_model, "gpt-5.4")
        self.assertEqual(result.assistant_llm_model, "gpt-5.4-mini")
        self.assertEqual(result.subgraph_construction_algorithm, "shortest_path")
        self.assertEqual(result.context_construction_strategy, "structured_triples")
        self.assertEqual(result.gnn_layer_count, 2)
        self.assertEqual(result.node_classifier, "mlp")
        self.assertEqual(len(prompts), 2)

    def test_fully_interactive_path_works(self) -> None:
        answers = iter(["1", "1", "1", "2", "1", "2", "1", "1", "1"])

        step = BuildPipelineConfigurationStep(input_func=lambda _: next(answers))
        result = step.execute(self.make_dataset_context())

        self.assertEqual(result.gnn_layer_count, 2)
        self.assertEqual(result.node_classifier, "mlp")
        self.assertEqual(result.main_llm_model, "gpt-5.4")
        self.assertEqual(result.assistant_llm_model, "gpt-5.4-mini")
        self.assertEqual(result.subgraph_construction_algorithm, "shortest_path")
        self.assertEqual(result.context_construction_strategy, "structured_triples")
        self.assertEqual(result.question_embedding_model, "text-embedding-3-small")
        self.assertEqual(result.relation_embedding_model, "text-embedding-3-small")
        self.assertEqual(result.entity_embedding_model, "text-embedding-3-small")

    def test_invalid_main_llm_constructor_value_raises(self) -> None:
        step = BuildPipelineConfigurationStep(
            main_llm_model="invalid-model",
            assistant_llm_model="gpt-5.4-mini",
            subgraph_algorithm="shortest_path",
            context_strategy="textualized",
            gnn_layer_count=2,
            node_classifier="mlp",
            question_embedding_model="text-embedding-3-small",
            relation_embedding_model="text-embedding-3-small",
            entity_embedding_model="text-embedding-3-small",
        )

        with self.assertRaises(InvalidMainLlmSelectionException):
            step.execute(self.make_dataset_context())

    def test_invalid_assistant_llm_constructor_value_raises(self) -> None:
        step = BuildPipelineConfigurationStep(
            main_llm_model="gpt-5.4",
            assistant_llm_model="invalid-model",
            subgraph_algorithm="shortest_path",
            context_strategy="textualized",
            gnn_layer_count=2,
            node_classifier="mlp",
            question_embedding_model="text-embedding-3-small",
            relation_embedding_model="text-embedding-3-small",
            entity_embedding_model="text-embedding-3-small",
        )

        with self.assertRaises(InvalidAssistantLlmSelectionException):
            step.execute(self.make_dataset_context())

    def test_invalid_subgraph_constructor_value_raises(self) -> None:
        step = BuildPipelineConfigurationStep(
            main_llm_model="gpt-5.4",
            assistant_llm_model="gpt-5.4-mini",
            subgraph_algorithm="invalid",
            context_strategy="textualized",
            gnn_layer_count=2,
            node_classifier="mlp",
            question_embedding_model="text-embedding-3-small",
            relation_embedding_model="text-embedding-3-small",
            entity_embedding_model="text-embedding-3-small",
        )

        with self.assertRaises(InvalidSubgraphConstructionSelectionException):
            step.execute(self.make_dataset_context())

    def test_invalid_context_strategy_constructor_value_raises(self) -> None:
        step = BuildPipelineConfigurationStep(
            main_llm_model="gpt-5.4",
            assistant_llm_model="gpt-5.4-mini",
            subgraph_algorithm="shortest_path",
            context_strategy="invalid",
            gnn_layer_count=2,
            node_classifier="mlp",
            question_embedding_model="text-embedding-3-small",
            relation_embedding_model="text-embedding-3-small",
            entity_embedding_model="text-embedding-3-small",
        )

        with self.assertRaises(InvalidContextConstructionSelectionException):
            step.execute(self.make_dataset_context())

    def test_invalid_gnn_layer_count_constructor_value_raises(self) -> None:
        step = BuildPipelineConfigurationStep(
            main_llm_model="gpt-5.4",
            assistant_llm_model="gpt-5.4-mini",
            subgraph_algorithm="shortest_path",
            context_strategy="textualized",
            gnn_layer_count=9,
            node_classifier="mlp",
            question_embedding_model="text-embedding-3-small",
            relation_embedding_model="text-embedding-3-small",
            entity_embedding_model="text-embedding-3-small",
        )

        with self.assertRaises(InvalidGnnLayerCountSelectionException):
            step.execute(self.make_dataset_context())

    def test_invalid_node_classifier_constructor_value_raises(self) -> None:
        step = BuildPipelineConfigurationStep(
            main_llm_model="gpt-5.4",
            assistant_llm_model="gpt-5.4-mini",
            subgraph_algorithm="shortest_path",
            context_strategy="textualized",
            gnn_layer_count=2,
            node_classifier="invalid",
            question_embedding_model="text-embedding-3-small",
            relation_embedding_model="text-embedding-3-small",
            entity_embedding_model="text-embedding-3-small",
        )

        with self.assertRaises(InvalidNodeClassifierSelectionException):
            step.execute(self.make_dataset_context())

    def test_invalid_embedding_model_constructor_value_raises(self) -> None:
        step = BuildPipelineConfigurationStep(
            main_llm_model="gpt-5.4",
            assistant_llm_model="gpt-5.4-mini",
            subgraph_algorithm="shortest_path",
            context_strategy="textualized",
            gnn_layer_count=2,
            node_classifier="mlp",
            question_embedding_model="text-embedding-3-small",
            relation_embedding_model="text-embedding-3-small",
            entity_embedding_model="invalid",
        )

        with self.assertRaises(InvalidEntityEmbeddingModelSelectionException):
            step.execute(self.make_dataset_context())

    def test_interactive_invalid_numeric_input_reprompts(self) -> None:
        answers = iter(["abc", "1", "1", "1", "2", "1", "1", "1", "1", "1"])
        step = BuildPipelineConfigurationStep(input_func=lambda _: next(answers))

        result = step.execute(self.make_dataset_context())

        self.assertEqual(result.gnn_layer_count, 2)

    def test_interactive_out_of_range_input_reprompts(self) -> None:
        answers = iter(["99", "1", "1", "1", "2", "1", "1", "1", "1", "1"])
        step = BuildPipelineConfigurationStep(input_func=lambda _: next(answers))

        result = step.execute(self.make_dataset_context())

        self.assertEqual(result.gnn_layer_count, 2)

    def test_configuration_result_is_stored_in_result_bank(self) -> None:
        pipeline = Pipeline(
            preparation_steps=[
                BuildPipelineConfigurationStep(
                    main_llm_model="gpt-5.4",
                    assistant_llm_model="gpt-5.4-mini",
                    subgraph_algorithm="shortest_path",
                    context_strategy="textualized",
                    gnn_layer_count=2,
                    node_classifier="mlp",
                    question_embedding_model="text-embedding-3-small",
                    relation_embedding_model="text-embedding-3-small",
                    entity_embedding_model="text-embedding-3-small",
                )
            ]
        )

        result = pipeline.prepare(self.make_dataset_context())

        self.assertTrue(result.success)
        self.assertTrue(
            pipeline.context_builder.has_stored_result(type(result.final_result))
        )


if __name__ == "__main__":
    unittest.main()
