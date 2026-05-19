"""Tests for evaluation candidate scoring and shortest path extraction."""

import unittest

from pipeline import (
    CandidateNodeScore,
    EvaluationSample,
    ExtractShortestPathsStep,
    InvalidEvaluationSampleException,
    MockCandidateNodeScoringStep,
    Pipeline,
    ShortestPathExtractionService,
    StepContext,
)


def make_sample() -> EvaluationSample:
    return EvaluationSample.from_webqsp_row(
        {
            "id": "sample-1",
            "question": "what is the name of justin bieber brother",
            "q_entity": ["Justin Bieber"],
            "a_entity": ["Jaxon Bieber"],
            "graph": [
                ["Justin Bieber", "people.person.sibling_s", "m.0gxnnwp"],
                ["m.0gxnnwp", "people.sibling_relationship.sibling", "Jaxon Bieber"],
                ["Justin Bieber", "people.person.place_of_birth", "London"],
                ["Scooter Braun", "music.artist.manager", "Justin Bieber"],
            ],
        }
    )


class EvaluationSampleParsingTests(unittest.TestCase):
    def test_valid_webqsp_row_becomes_evaluation_sample(self) -> None:
        print("\n[test_valid_webqsp_row_becomes_evaluation_sample] Starting.")
        sample = make_sample()

        self.assertEqual(sample.sample_id, "sample-1")
        self.assertEqual(sample.question, "what is the name of justin bieber brother")
        self.assertEqual(sample.q_entities, ["Justin Bieber"])
        self.assertEqual(sample.a_entities, ["Jaxon Bieber"])
        self.assertEqual(len(sample.graph_triples), 4)
        self.assertEqual(sample.graph_triples[0].relation, "people.person.sibling_s")
        print("[test_valid_webqsp_row_becomes_evaluation_sample] Passed.")

    def test_missing_q_entity_fails(self) -> None:
        print("\n[test_missing_q_entity_fails] Starting.")
        with self.assertRaises(InvalidEvaluationSampleException):
            EvaluationSample.from_webqsp_row(
                {
                    "question": "who?",
                    "a_entity": ["answer"],
                    "graph": [["a", "r", "b"]],
                }
            )
        print("[test_missing_q_entity_fails] Passed.")

    def test_malformed_graph_triple_fails(self) -> None:
        print("\n[test_malformed_graph_triple_fails] Starting.")
        with self.assertRaises(InvalidEvaluationSampleException):
            EvaluationSample.from_webqsp_row(
                {
                    "question": "who?",
                    "q_entity": ["topic"],
                    "a_entity": ["answer"],
                    "graph": [["a", "r"]],
                }
            )
        print("[test_malformed_graph_triple_fails] Passed.")


class MockCandidateNodeScoringStepTests(unittest.TestCase):
    def test_mock_candidates_include_gold_then_distractors(self) -> None:
        print("\n[test_mock_candidates_include_gold_then_distractors] Starting.")
        step = MockCandidateNodeScoringStep(top_k=3)
        result = step.execute(StepContext(result=make_sample()))

        self.assertEqual(result.top_k, 3)
        self.assertEqual(result.candidates[0].node_id, "Jaxon Bieber")
        self.assertEqual(result.candidates[0].source, "mock_gold")
        self.assertEqual(len(result.candidates), 3)
        self.assertEqual(len({candidate.node_id for candidate in result.candidates}), 3)
        self.assertTrue(
            all(candidate.source in {"mock_gold", "mock_distractor"} for candidate in result.candidates)
        )
        print("[test_mock_candidates_include_gold_then_distractors] Passed.")


class ShortestPathExtractionServiceTests(unittest.TestCase):
    def test_multi_hop_path_preserves_relation_labels(self) -> None:
        print("\n[test_multi_hop_path_preserves_relation_labels] Starting.")
        service = ShortestPathExtractionService()
        sample = make_sample()
        result = service.extract_paths(
            sample=sample,
            candidates=[
                CandidateNodeScore(node_id="Jaxon Bieber", score=1.0, source="mock_gold")
            ],
        )

        self.assertEqual(result.found_paths, 1)
        self.assertEqual(result.missing_paths, 0)
        self.assertTrue(result.paths[0].path_found)
        self.assertEqual(
            [triple.relation for triple in result.paths[0].triples],
            ["people.person.sibling_s", "people.sibling_relationship.sibling"],
        )
        self.assertIn(
            "Justin Bieber -> people.person.sibling_s -> m.0gxnnwp",
            result.reasoning_paths_text,
        )
        self.assertEqual(
            [triple.relation for triple in result.reasoning_subgraph_triples],
            ["people.person.sibling_s", "people.sibling_relationship.sibling"],
        )
        print("[test_multi_hop_path_preserves_relation_labels] Passed.")

    def test_reasoning_subgraph_deduplicates_shortest_path_triples(self) -> None:
        print("\n[test_reasoning_subgraph_deduplicates_shortest_path_triples] Starting.")
        sample = EvaluationSample.from_webqsp_row(
            {
                "question": "who is connected",
                "q_entity": ["Topic"],
                "a_entity": ["Answer A", "Answer B"],
                "graph": [
                    ["Topic", "r1", "Shared"],
                    ["Shared", "r2", "Answer A"],
                    ["Shared", "r3", "Answer B"],
                ],
            }
        )
        result = ShortestPathExtractionService().extract_paths(
            sample=sample,
            candidates=[
                CandidateNodeScore(node_id="Answer A", score=1.0, source="mock_gold"),
                CandidateNodeScore(node_id="Answer B", score=0.9, source="mock_gold"),
            ],
        )

        self.assertEqual(
            [(triple.source, triple.relation, triple.target) for triple in result.reasoning_subgraph_triples],
            [
                ("Topic", "r1", "Shared"),
                ("Shared", "r2", "Answer A"),
                ("Shared", "r3", "Answer B"),
            ],
        )
        self.assertEqual(result.reasoning_paths_text.count("Topic -> r1 -> Shared"), 1)
        print("[test_reasoning_subgraph_deduplicates_shortest_path_triples] Passed.")

    def test_all_equal_length_shortest_paths_contribute_to_subgraph(self) -> None:
        print("\n[test_all_equal_length_shortest_paths_contribute_to_subgraph] Starting.")
        sample = EvaluationSample.from_webqsp_row(
            {
                "question": "what is the name of justin bieber brother",
                "q_entity": ["Justin Bieber"],
                "a_entity": ["Jaxon Bieber"],
                "graph": [
                    ["Justin Bieber", "people.person.nationality", "Canada"],
                    ["Jaxon Bieber", "people.person.nationality", "Canada"],
                    ["Justin Bieber", "people.person.sibling_s", "m.0gxnnwp"],
                    ["Jaxon Bieber", "people.person.sibling_s", "m.0gxnnwp"],
                    ["m.0gxnnwp", "people.sibling_relationship.sibling", "Justin Bieber"],
                    ["m.0gxnnwp", "people.sibling_relationship.sibling", "Jaxon Bieber"],
                ],
            }
        )
        result = ShortestPathExtractionService().extract_paths(
            sample=sample,
            candidates=[
                CandidateNodeScore(node_id="Jaxon Bieber", score=1.0, source="mock_gold")
            ],
        )

        self.assertEqual(len(result.paths[0].shortest_paths), 5)
        self.assertEqual(
            {
                tuple((triple.source, triple.relation, triple.target) for triple in path)
                for path in result.paths[0].shortest_paths
            },
            {
                (
                    ("Justin Bieber", "people.person.nationality", "Canada"),
                    ("Jaxon Bieber", "people.person.nationality", "Canada"),
                ),
                (
                    ("Justin Bieber", "people.person.sibling_s", "m.0gxnnwp"),
                    ("Jaxon Bieber", "people.person.sibling_s", "m.0gxnnwp"),
                ),
                (
                    ("Justin Bieber", "people.person.sibling_s", "m.0gxnnwp"),
                    ("m.0gxnnwp", "people.sibling_relationship.sibling", "Jaxon Bieber"),
                ),
                (
                    ("m.0gxnnwp", "people.sibling_relationship.sibling", "Justin Bieber"),
                    ("Jaxon Bieber", "people.person.sibling_s", "m.0gxnnwp"),
                ),
                (
                    ("m.0gxnnwp", "people.sibling_relationship.sibling", "Justin Bieber"),
                    ("m.0gxnnwp", "people.sibling_relationship.sibling", "Jaxon Bieber"),
                ),
            },
        )
        self.assertIn(
            "m.0gxnnwp -> people.sibling_relationship.sibling -> Jaxon Bieber",
            result.reasoning_paths_text,
        )
        print("[test_all_equal_length_shortest_paths_contribute_to_subgraph] Passed.")

    def test_direct_reverse_traversal_path_is_found(self) -> None:
        print("\n[test_direct_reverse_traversal_path_is_found] Starting.")
        sample = EvaluationSample.from_webqsp_row(
            {
                "question": "who manages justin bieber",
                "q_entity": ["Justin Bieber"],
                "a_entity": ["Scooter Braun"],
                "graph": [["Scooter Braun", "music.artist.manager", "Justin Bieber"]],
            }
        )
        result = ShortestPathExtractionService().extract_paths(
            sample=sample,
            candidates=[
                CandidateNodeScore(node_id="Scooter Braun", score=1.0, source="mock_gold")
            ],
        )

        self.assertTrue(result.paths[0].path_found)
        self.assertEqual(result.paths[0].triples[0].source, "Scooter Braun")
        self.assertEqual(result.paths[0].triples[0].target, "Justin Bieber")
        print("[test_direct_reverse_traversal_path_is_found] Passed.")

    def test_multiple_q_entities_choose_shortest_available_path(self) -> None:
        print("\n[test_multiple_q_entities_choose_shortest_available_path] Starting.")
        sample = EvaluationSample.from_webqsp_row(
            {
                "question": "who is connected",
                "q_entity": ["Far Topic", "Near Topic"],
                "a_entity": ["Answer"],
                "graph": [
                    ["Far Topic", "r1", "Middle"],
                    ["Middle", "r2", "Answer"],
                    ["Near Topic", "r3", "Answer"],
                ],
            }
        )
        result = ShortestPathExtractionService().extract_paths(
            sample=sample,
            candidates=[CandidateNodeScore(node_id="Answer", score=1.0, source="mock_gold")],
        )

        self.assertEqual(len(result.paths[0].triples), 1)
        self.assertEqual(result.paths[0].triples[0].relation, "r3")
        print("[test_multiple_q_entities_choose_shortest_available_path] Passed.")

    def test_unreachable_candidate_is_recorded_without_abort(self) -> None:
        print("\n[test_unreachable_candidate_is_recorded_without_abort] Starting.")
        result = ShortestPathExtractionService().extract_paths(
            sample=make_sample(),
            candidates=[
                CandidateNodeScore(node_id="Unreachable", score=0.3, source="mock_distractor")
            ],
        )

        self.assertEqual(result.found_paths, 0)
        self.assertEqual(result.missing_paths, 1)
        self.assertFalse(result.paths[0].path_found)
        self.assertIn("No reasoning subgraph found.", result.reasoning_paths_text)
        print("[test_unreachable_candidate_is_recorded_without_abort] Passed.")


class PathExtractionPipelineTests(unittest.TestCase):
    def test_mock_scoring_and_path_extraction_run_as_pipeline_steps(self) -> None:
        print("\n[test_mock_scoring_and_path_extraction_run_as_pipeline_steps] Starting.")
        pipeline = Pipeline(
            evaluation_steps=[
                MockCandidateNodeScoringStep(top_k=3),
                ExtractShortestPathsStep(),
            ],
        )

        result = pipeline.evaluate(StepContext(result=make_sample()))

        self.assertTrue(result.success)
        self.assertEqual(result.steps_executed, 2)
        self.assertEqual(result.final_result.sample.sample_id, "sample-1")
        self.assertGreaterEqual(result.final_result.found_paths, 1)
        self.assertIn("Reasoning subgraph:", result.final_result.reasoning_paths_text)
        print("[test_mock_scoring_and_path_extraction_run_as_pipeline_steps] Passed.")


if __name__ == "__main__":
    unittest.main()
