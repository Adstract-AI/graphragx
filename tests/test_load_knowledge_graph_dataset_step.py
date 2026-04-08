"""Tests for preparation-time knowledge graph dataset loading."""

import sys
import unittest
from unittest.mock import patch

import main
from pipeline import (
    BuiltPipelineConfiguration,
    KnowledgeGraphDatasetLoadingException,
    LoadKnowledgeGraphDatasetStep,
    MalformedKnowledgeGraphDatasetException,
    MissingTorchDependencyException,
    MissingTorchGeometricDependencyException,
    Pipeline,
    StepContext,
    UnsupportedKnowledgeGraphDatasetLoaderException,
    LoadedKnowledgeGraphDataset,
)
from pipeline.services import AbstractDatasetLoaderService


class FakeTensor:
    def __init__(self, values):
        self._values = values

    def tolist(self):
        return list(self._values)

    def __getitem__(self, index):
        return FakeTensor(self._values[index])


class FakeData:
    def __init__(self):
        self.edge_index = FakeTensor([[0, 1, 2], [1, 2, 0]])
        self.edge_type = FakeTensor([0, 1, 0])
        self.num_nodes = 3


class FakeDataset:
    def __init__(self, data):
        self._data = data
        self.num_relations = 2

    def __getitem__(self, index):
        if index != 0:
            raise IndexError(index)
        return self._data


class FakeLoaderService(AbstractDatasetLoaderService):
    def __init__(self, dataset=None, data=None):
        self.dataset = dataset or FakeDataset(FakeData())
        self.data = data or self.dataset[0]

    def load_dataset(self, dataset_id: str) -> tuple[FakeDataset, FakeData]:
        return self.dataset, self.data


class LoadKnowledgeGraphDatasetStepTests(unittest.TestCase):
    @staticmethod
    def make_configuration_context(
        dataset_id: str = "FB15K-237",
    ) -> StepContext[BuiltPipelineConfiguration]:
        return StepContext(
            result=BuiltPipelineConfiguration(
                dataset_id=dataset_id,
                main_llm_model="gpt-5.4",
                assistant_llm_model="gpt-5.4-mini",
                subgraph_construction_algorithm="shortest_path",
                context_construction_strategy="textualized",
                gnn_architecture="rgcn",
            )
        )

    def test_successful_load_returns_expected_artifact(self) -> None:
        step = LoadKnowledgeGraphDatasetStep(loader_service=FakeLoaderService())

        result = step.execute(self.make_configuration_context())

        self.assertEqual(result.dataset_id, "FB15K-237")
        self.assertEqual(result.dataset_family, "knowledge_graph")
        self.assertEqual(result.entity_count, 3)
        self.assertEqual(result.relation_count, 2)
        self.assertEqual(result.triple_count, 3)
        self.assertEqual(result.raw_triples[0].head_id, 0)
        self.assertEqual(result.raw_triples[0].relation_id, 0)
        self.assertEqual(result.raw_triples[0].tail_id, 1)
        self.assertIsNotNone(result.torch_geometric_dataset)
        self.assertIsNotNone(result.torch_geometric_data)

    def test_step_consumes_configuration_result(self) -> None:
        step = LoadKnowledgeGraphDatasetStep(loader_service=FakeLoaderService())

        result = step.execute(self.make_configuration_context())

        self.assertEqual(result.dataset_id, "FB15K-237")

    def test_unsupported_dataset_raises(self) -> None:
        step = LoadKnowledgeGraphDatasetStep(loader_service=FakeLoaderService())

        with self.assertRaises(UnsupportedKnowledgeGraphDatasetLoaderException):
            step.execute(self.make_configuration_context(dataset_id="WN18RR"))

    def test_malformed_loaded_dataset_raises(self) -> None:
        class MalformedData:
            edge_index = FakeTensor([[0, 1], [1]])
            edge_type = FakeTensor([0, 1])
            num_nodes = 3

        step = LoadKnowledgeGraphDatasetStep(
            loader_service=FakeLoaderService(
                dataset=FakeDataset(MalformedData()),
                data=MalformedData(),
            )
        )

        with self.assertRaises(MalformedKnowledgeGraphDatasetException):
            step.execute(self.make_configuration_context())

    def test_loading_result_is_stored_in_result_bank(self) -> None:
        pipeline = Pipeline(
            preparation_steps=[
                LoadKnowledgeGraphDatasetStep(loader_service=FakeLoaderService()),
            ]
        )

        result = pipeline.prepare(self.make_configuration_context())

        self.assertTrue(result.success)
        self.assertTrue(pipeline.context_builder.has_stored_result(LoadedKnowledgeGraphDataset))

    def test_full_preparation_pipeline_runs_dataset_selection_configuration_and_loading(self) -> None:
        fake_loader = FakeLoaderService()
        with patch("main.LoadKnowledgeGraphDatasetStep", return_value=LoadKnowledgeGraphDatasetStep(loader_service=fake_loader)):
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
        self.assertEqual(result.steps_executed, 3)
        self.assertEqual(result.final_result.dataset_id, "FB15K-237")


class TorchGeometricLoaderServiceTests(unittest.TestCase):
    def test_missing_torch_raises_dedicated_exception(self) -> None:
        from pipeline.services.dataset_loader import TorchGeometricKnowledgeGraphLoaderService

        service = TorchGeometricKnowledgeGraphLoaderService()
        try:
            original_torch_module = sys.modules.pop("torch", None)
            with self.assertRaises(
                (MissingTorchDependencyException, MissingTorchGeometricDependencyException)
            ):
                service.load_dataset("FB15K-237")
        finally:
            if original_torch_module is not None:
                sys.modules["torch"] = original_torch_module

    def test_missing_torch_geometric_raises_dedicated_exception(self) -> None:
        from pipeline.services.dataset_loader import TorchGeometricKnowledgeGraphLoaderService

        service = TorchGeometricKnowledgeGraphLoaderService()

        import builtins
        original_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "torch":
                return object()
            if name == "torch_geometric.datasets":
                raise ModuleNotFoundError("No module named 'torch_geometric'")
            return original_import(name, globals, locals, fromlist, level)

        with patch("builtins.__import__", side_effect=fake_import):
            with self.assertRaises(MissingTorchGeometricDependencyException):
                service.load_dataset("FB15K-237")

    def test_loader_failure_raises_dataset_loading_exception(self) -> None:
        from pipeline.services.dataset_loader import TorchGeometricKnowledgeGraphLoaderService

        class BrokenDatasetClass:
            def __init__(self, root: str):
                raise RuntimeError("broken")

        class FakeDatasetModule:
            FB15k_237 = BrokenDatasetClass

        service = TorchGeometricKnowledgeGraphLoaderService()

        import builtins
        original_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "torch":
                return object()
            if name == "torch_geometric.datasets":
                return FakeDatasetModule()
            return original_import(name, globals, locals, fromlist, level)

        with patch("builtins.__import__", side_effect=fake_import):
            with self.assertRaises(KnowledgeGraphDatasetLoadingException):
                service.load_dataset("FB15K-237")


if __name__ == "__main__":
    unittest.main()
