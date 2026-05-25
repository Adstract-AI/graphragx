"""Tests for preparation-time GNN answer-retriever construction."""

import builtins
import sys
import unittest
from unittest.mock import patch

from pipeline import (
    AbstractStep,
    BuildGnnAnswerRetrieverStep,
    BuiltGnnAnswerRetriever,
    BuiltPipelineConfiguration,
    LoadedDataset,
    Pipeline,
    PipelineException,
    StepContext,
    StepContextBuilder,
)


try:
    import torch
    from torch import nn
except ModuleNotFoundError:
    torch = None
    nn = None


class FakeDatasetDict(dict):
    pass


class FakeConfigurationStep(AbstractStep[BuiltPipelineConfiguration, LoadedDataset]):
    def execute_default(
        self,
        context: StepContext[LoadedDataset],
    ) -> BuiltPipelineConfiguration:
        return BuildGnnAnswerRetrieverStepTests.make_configuration()


class FakeLoadedDatasetStep(AbstractStep[LoadedDataset, BuiltPipelineConfiguration]):
    def execute_default(
        self,
        context: StepContext[BuiltPipelineConfiguration],
    ) -> LoadedDataset:
        return LoadedDataset(
            dataset_id="WebQSP",
            dataset_family="question_answering",
            hugging_face_dataset_name="ml1996/webqsp",
            split_names=["train", "validation", "test"],
            split_sizes={"train": 3, "validation": 2, "test": 1},
            hugging_face_dataset=FakeDatasetDict(),
        )


class BuildGnnAnswerRetrieverStepTests(unittest.TestCase):
    @staticmethod
    def make_configuration(
        node_classifier: str = "mlp",
    ) -> BuiltPipelineConfiguration:
        return BuiltPipelineConfiguration(
            dataset_id="WebQSP",
            main_llm_model="gpt-5.4",
            subgraph_construction_algorithm="shortest_path",
            context_construction_strategy="structured_triples",
            gnn_layer_count=2,
            gnn_hidden_dimension=256,
            node_classifier=node_classifier,
            question_embedding_model="text-embedding-3-small",
            relation_embedding_model="text-embedding-3-small",
            entity_embedding_model="text-embedding-3-small",
        )

    @staticmethod
    def make_loaded_dataset_context(
        builder: StepContextBuilder,
    ) -> StepContext[LoadedDataset]:
        loaded_dataset = LoadedDataset(
            dataset_id="WebQSP",
            dataset_family="question_answering",
            hugging_face_dataset_name="ml1996/webqsp",
            split_names=["train", "validation", "test"],
            split_sizes={"train": 3, "validation": 2, "test": 1},
            hugging_face_dataset=FakeDatasetDict(),
        )
        return builder.create_context(result=loaded_dataset)

    @unittest.skipIf(torch is None, "PyTorch is not installed.")
    def test_builds_gnn_answer_retriever_from_loaded_dataset_and_stored_configuration(self) -> None:
        builder = StepContextBuilder()
        builder.store_result(self.make_configuration())
        step = BuildGnnAnswerRetrieverStep()

        result = step.execute(self.make_loaded_dataset_context(builder))

        self.assertIsInstance(result, BuiltGnnAnswerRetriever)
        self.assertEqual(result.dataset_id, "WebQSP")
        self.assertEqual(result.entity_embedding_model, "text-embedding-3-small")
        self.assertEqual(result.entity_embedding_dimension, 1536)
        self.assertEqual(result.hidden_dimension, 256)
        self.assertEqual(result.gnn_layer_count, 2)
        self.assertEqual(result.node_classifier, "mlp")
        self.assertEqual(len(result.model.gnn_layers), 2)

    @unittest.skipIf(torch is None, "PyTorch is not installed.")
    def test_linear_classifier_builds_linear_head(self) -> None:
        builder = StepContextBuilder()
        builder.store_result(self.make_configuration(node_classifier="linear"))
        step = BuildGnnAnswerRetrieverStep()

        result = step.execute(self.make_loaded_dataset_context(builder))

        self.assertIsInstance(result.model.classifier, nn.Linear)

    @unittest.skipIf(torch is None, "PyTorch is not installed.")
    def test_mlp_classifier_builds_sequential_head(self) -> None:
        builder = StepContextBuilder()
        builder.store_result(self.make_configuration(node_classifier="mlp"))
        step = BuildGnnAnswerRetrieverStep()

        result = step.execute(self.make_loaded_dataset_context(builder))

        self.assertIsInstance(result.model.classifier, nn.Sequential)

    @unittest.skipIf(torch is None, "PyTorch is not installed.")
    def test_model_forward_returns_one_logit_per_node(self) -> None:
        builder = StepContextBuilder()
        builder.store_result(self.make_configuration())
        step = BuildGnnAnswerRetrieverStep()
        result = step.execute(self.make_loaded_dataset_context(builder))

        entity_features = torch.randn(4, result.entity_embedding_dimension)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
        edge_weight = torch.tensor([0.9, 0.5, 0.2])

        logits = result.model(entity_features, edge_index, edge_weight)

        self.assertEqual(tuple(logits.shape), (4,))

    def test_missing_configuration_in_result_bank_raises(self) -> None:
        builder = StepContextBuilder()
        step = BuildGnnAnswerRetrieverStep()

        with self.assertRaises(PipelineException):
            step.execute(self.make_loaded_dataset_context(builder))

    @unittest.skipIf(torch is None, "PyTorch is not installed.")
    def test_pipeline_stores_built_gnn_answer_retriever(self) -> None:
        pipeline = Pipeline(
            preparation_steps=[
                FakeConfigurationStep(),
                FakeLoadedDatasetStep(),
                BuildGnnAnswerRetrieverStep(),
            ],
        )

        result = pipeline.prepare(StepContext(result=None))

        self.assertTrue(result.success)
        self.assertTrue(
            pipeline.context_builder.has_stored_result(BuiltGnnAnswerRetriever)
        )


if __name__ == "__main__":
    unittest.main()
