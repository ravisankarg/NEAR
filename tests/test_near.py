import tempfile
import unittest

import numpy as np

from near_retrieval.metrics import evaluate, paired_comparison, rank
from near_retrieval.near import NearConfig, NearIndex, local_references, score_scale, unit


class NearTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.corpus = unit(rng.normal(size=(37, 8)))
        self.queries = unit(rng.normal(size=(7, 8)))
        self.config = NearConfig(k=5, anchors=4, max_depth=3, kmeans_restarts=2)

    def test_covariance_scale_matches_all_pairs_including_self(self):
        for x in [self.corpus, self.corpus * np.arange(1, 38)[:, None] / 37]:
            expected = np.sqrt(np.var(x @ x.T, axis=1, ddof=0).mean())
            self.assertAlmostEqual(score_scale(x), expected, places=7)

    def test_self_exclusion_and_k_cap_even_for_duplicate_vectors(self):
        x = np.array([[1., 0.], [1., 0.], [0., 1.]])
        refs, ix = local_references(x, k=50)
        self.assertEqual(ix.shape, (3, 2))
        for i in range(3):
            self.assertNotIn(i, ix[i])
            np.testing.assert_allclose(refs[i], unit(np.delete(x, i, axis=0).mean(axis=0)))

    def test_residual_is_orthogonal_to_local_reference(self):
        index = NearIndex(self.config).fit(self.corpus)
        for layer in index.layers:
            np.testing.assert_allclose(np.sum(layer["residual"]*layer["refs"], axis=1), 0, atol=3e-7)

    def test_nearest_neighbor_direction_by_hand(self):
        x = np.array([[1., 0.], [.8, .6], [-1., 0.]], dtype=np.float32)
        refs, neighbors = local_references(x, k=1)
        np.testing.assert_array_equal(neighbors[:, 0], [1, 0, 1])
        np.testing.assert_allclose(refs, x[[1, 0, 1]])

    def test_query_routes_top_r_anchors_and_softmaxes_by_hand(self):
        x = np.array([[1., 0.], [0., 1.], [-1., 0.]], dtype=np.float32)
        index = NearIndex(NearConfig(k=1, anchors=3, routing=2, beta=2, max_depth=1)).fit(x)
        index.layers[0]["anchors"] = x
        index.layers[0]["values"] = np.array([[.6, .8], [0., 1.], [1., 0.]], dtype=np.float32)
        p = np.exp(2) / (np.exp(2) + 1)
        c = np.array([.6*p, .8*p + (1-p)])
        c /= np.linalg.norm(c)
        expected = x[0] - c[0] * c
        actual = index.query_views(x[:1], normalized=False)[1][0]
        np.testing.assert_allclose(actual, expected, atol=2e-7)
        np.testing.assert_allclose(index.query_views(x[:1], normalized=True)[1][0],
                                   expected / np.linalg.norm(expected), atol=2e-7)

    def test_equation4_matches_explicit_score_all_variants(self):
        index = NearIndex(self.config).fit(self.corpus)
        for normalized in [False, True]:
            scores = index.score_components(self.queries, normalized)
            for depth in [1, 2, 3]:
                for alpha in [.25, 1., 4.]:
                    actual = index.query_keys(self.queries, depth, normalized, alpha) @ index.memory_keys(depth, normalized, alpha).T
                    expected = scores[0] + alpha/depth * sum(scores[1:depth+1])
                    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)

    def test_zero_alpha_recovers_cosine_ranking(self):
        index = NearIndex(self.config).fit(self.corpus)
        actual = index.query_keys(self.queries, alpha=0) @ index.memory_keys(alpha=0).T
        np.testing.assert_array_equal(rank(actual), rank(self.queries @ self.corpus.T))

    def test_tables_are_query_independent_and_round_trip(self):
        index = NearIndex(self.config).fit(self.corpus)
        one = index.query_keys(self.queries[:1])
        batch = index.query_keys(self.queries)
        np.testing.assert_allclose(one, batch[:1], atol=1e-6)
        with tempfile.TemporaryDirectory() as path:
            index.save(path)
            restored = NearIndex.load(path)
            np.testing.assert_array_equal(restored.memory_keys(), index.memory_keys())
            np.testing.assert_array_equal(restored.query_keys(self.queries), batch)

    def test_degenerate_bank_is_finite(self):
        index = NearIndex(self.config).fit(np.ones((6, 8)))
        self.assertTrue(np.isfinite(index.query_keys(self.queries)).all())
        self.assertTrue(np.isfinite(index.memory_keys()).all())

    def test_invalid_input_fails(self):
        for x in [np.zeros((3, 8)), np.ones((1, 8)), np.full((3, 8), np.nan)]:
            with self.assertRaises(ValueError):
                NearIndex(self.config).fit(x)

    def test_hand_computed_graded_ndcg_and_uncapped_recall(self):
        qrels = {"q": {"b": 2, "c": 1}}
        means, _ = evaluate(np.array([[0, 1, 2]]), ["q"], ["a", "b", "c"], qrels)
        expected = (2/np.log2(3) + 1/np.log2(4)) / (2 + 1/np.log2(3))
        self.assertAlmostEqual(means["ndcg@10"], expected)
        self.assertEqual(means["mrr@10"], .5)
        self.assertEqual(means["recall@10"], 1.)
        many = {"q": {str(i): 1 for i in range(20)}}
        m, _ = evaluate(np.array([list(range(10))]), ["q"], [str(i) for i in range(20)], many)
        self.assertEqual(m["recall@10"], .5)
        self.assertEqual(m["capped_recall@10"], 1.)

    def test_paired_bootstrap_identical_is_zero(self):
        values = {str(i): {"ndcg@10": i/10} for i in range(10)}
        result = paired_comparison(values, values, samples=100)["ndcg@10"]
        self.assertEqual(result["paired_bootstrap_95ci"], [0., 0.])
        self.assertEqual(result["unchanged_queries"], 10)


if __name__ == "__main__":
    unittest.main()
