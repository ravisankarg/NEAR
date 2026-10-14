import tempfile
import unittest
from pathlib import Path

import numpy as np

from near_retrieval.metrics import rank
from near_retrieval.near import unit
from neighbor_retrieval.one_hop import OneHopGraph, score_method, softmax
from neighbor_retrieval.benchmark import better, holm


class OneHopTests(unittest.TestCase):
    def test_exact_graph_self_exclusion_and_ties(self):
        graph = OneHopGraph.fit([[1, 0], [1, 0], [0, 1]], block_size=2)
        np.testing.assert_array_equal(graph.neighbor, [1, 0, 0])
        np.testing.assert_array_equal(graph.indegree, [2, 1, 0])
        np.testing.assert_allclose(graph.similarity, [1, 1, 0])

    def test_graph_matches_brute_force(self):
        x = unit(np.random.default_rng(3).normal(size=(43, 7)))
        graph = OneHopGraph.fit(x, block_size=9)
        matrix = unit(x) @ unit(x).T
        np.fill_diagonal(matrix, -np.inf)
        np.testing.assert_array_equal(graph.neighbor, np.argmax(matrix, axis=1))

    def test_one_hop_does_not_recurse(self):
        graph = OneHopGraph(np.array([1, 2, 3, 2]), np.ones(4), np.array([0, 1, 2, 1]))
        np.testing.assert_array_equal(graph.expanded([0]), [0, 1])
        g = graph.support(np.array([.8, .6, .4, .2]), [0], .1)
        np.testing.assert_array_equal(g, [0, 1, 0, 0])

    def test_weighted_vote_accumulation_by_hand(self):
        graph = OneHopGraph(np.array([2, 2, 1]), np.array([.5, 1., 1.]), np.array([0, 1, 2]))
        # Equal seed scores: 0->2 carries .25, 2->1 carries .5.
        g = graph.support(np.zeros(3), [0, 2], 1.)
        np.testing.assert_allclose(g, [0, 2/3, 1/3])
        corrected = graph.support(np.zeros(3), [0, 2], 1., 1)
        np.testing.assert_allclose(corrected, [0, .8, .2])

    def test_hub_correction_uniform_votes(self):
        graph = OneHopGraph(np.array([1, 0, 1, 1]), np.ones(4), np.array([1, 3, 0, 0]))
        np.testing.assert_allclose(graph.support(np.zeros(4), np.arange(4), 1, 1), [.5, .5, 0, 0])
        np.testing.assert_allclose(graph.support(np.zeros(4), np.arange(4), 1, 0), [.25, .75, 0, 0])

    def test_probability_conservation_and_zero_weight(self):
        graph = OneHopGraph.fit([[1, 0], [.8, .6], [0, 1]])
        scores = np.array([.9, .7, .6])
        for weight in [0, .1, .8, 1]:
            mixed = graph.diffuse(scores, [0, 1], .05, weight, 1)
            self.assertAlmostEqual(mixed.sum(), 1)
            self.assertTrue(np.all(mixed >= 0))
        np.testing.assert_array_equal(rank(graph.diffuse(scores, [0], .05, 0)[None]), rank(scores[None]))

    def test_no_positive_edges_falls_back_to_direct(self):
        graph = OneHopGraph.fit([[1, 0], [-1, 0]])
        np.testing.assert_allclose(graph.diffuse([.8, .2], [0], .05, .4), softmax([.8, .2], .05))

    def test_diffusion_pair_flip_condition(self):
        p, g = np.array([.6, .4]), np.array([0., 1.])
        for weight in [0, .1, 1/6, .2, 1]:
            mixed = (1-weight)*p+weight*g
            self.assertAlmostEqual(mixed[1]-mixed[0], (1-weight)*(-.2)+weight)
            self.assertEqual(mixed[1] > mixed[0]+1e-12, weight > 1/6+1e-12)

    def test_exact_cosine_expansion_cannot_improve_topn(self):
        rng = np.random.default_rng(7)
        x, q = unit(rng.normal(size=(51, 9))), unit(rng.normal(size=(4, 9)))
        graph = OneHopGraph.fit(x)
        scores = q @ x.T
        for row in scores:
            order = rank(row[None], 51)[0]
            for n in [3, 10, 25]:
                union = graph.expanded(order[:n])
                out = union[np.argsort(-row[union], kind="stable")][:n]
                np.testing.assert_array_equal(out, order[:n])

    def test_feedback_control_and_hand_calculated_neighbor(self):
        x = np.array([[1., 0], [0., 1]])
        graph = OneHopGraph(np.array([1, 0]), np.ones(2), np.ones(2))
        q, scores = np.array([1., 0]), np.array([1., 0])
        plain = graph.feedback_query(q, x, scores, np.array([0]), 1, .5, 0)
        paired = graph.feedback_query(q, x, scores, np.array([0]), 1, .5, .5)
        np.testing.assert_allclose(plain, [1., 0])
        np.testing.assert_allclose(paired, unit([.75, .25]))

    def test_round_trip_and_query_batch_consistency(self):
        rng = np.random.default_rng(12)
        x, q = unit(rng.normal(size=(41, 9))), unit(rng.normal(size=(3, 9)))
        graph = OneHopGraph.fit(x)
        with tempfile.TemporaryDirectory() as tmp:
            graph.save(Path(tmp)/"graph.npz")
            restored = OneHopGraph.load(Path(tmp)/"graph.npz")
        for family in ["diffusion", "degree_diffusion", "rocchio", "graph_feedback"]:
            setting = dict(family=family, seeds=3, temperature=.05, weight=.2, neighbor_fraction=.5)
            s, order = q @ x.T, rank(q @ x.T)
            batch = score_method(restored, q, x, s, order, setting)
            separate = np.concatenate([score_method(graph, q[i:i+1], x, s[i:i+1], order[i:i+1], setting) for i in range(3)])
            np.testing.assert_allclose(batch, separate, atol=2e-7)

    def test_recall_first_selection_and_holm(self):
        cfg = dict(primary_metric="recall@10", tie_break_metric="ndcg@10")
        self.assertTrue(better({"recall@10": .9, "ndcg@10": .5}, {"recall@10": .8, "ndcg@10": 1}, cfg))
        self.assertFalse(better({"recall@10": .9, "ndcg@10": .5}, {"recall@10": .9, "ndcg@10": .5}, cfg))
        self.assertEqual(holm(dict(a=.01, b=.02, c=.5)), dict(a=.03, b=.04, c=.5))

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            OneHopGraph.fit([[1, 0]])
        with self.assertRaises(ValueError):
            softmax([1, 2], 0)
        graph = OneHopGraph.fit([[1, 0], [0, 1]])
        with self.assertRaises(ValueError):
            graph.diffuse([1, 0], [0], .1, 2)


if __name__ == "__main__":
    unittest.main()
