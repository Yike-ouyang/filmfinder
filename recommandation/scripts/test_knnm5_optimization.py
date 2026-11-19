#!/usr/bin/env python3

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "recommandation" / "scripts" / "9.KNNM5Optimization.py"


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


KNN = load_module(SCRIPT, "_knn_m5_tests")


def graded_relevance(ratings, factor):
    ratings = np.asarray(ratings, dtype=np.float64)
    scaled = np.rint(ratings * float(factor)).astype(np.int64)
    return np.maximum(0, scaled - 5)


def ideal_dcg(ratings, factor, k=20):
    relevance = np.sort(graded_relevance(ratings, factor))[::-1][:int(k)]
    if relevance.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, relevance.size + 2))
    gains = np.power(2.0, relevance.astype(np.float64)) - 1.0
    return float(np.sum(gains / discounts))


def ndcg_known(top20, truth_map, ideal, factor, k=20):
    if ideal <= 0:
        return None, 0
    dcg = 0.0
    known = 0
    for position, movie in enumerate(top20[:int(k)], start=1):
        if int(movie) not in truth_map:
            continue
        known += 1
        relevance = float(
            graded_relevance(
                np.array([truth_map[int(movie)]]), factor
            )[0]
        )
        dcg += (2.0 ** relevance - 1.0) / np.log2(position + 1)
    return float(dcg / ideal), known


def ndcg_from_ranked(ranked_cols, truth_map, ideal, factor, k=20):
    if ideal <= 0:
        return None
    dcg = 0.0
    for position, movie in enumerate(ranked_cols[:int(k)], start=1):
        if int(movie) not in truth_map:
            continue
        relevance = float(
            graded_relevance(
                np.array([truth_map[int(movie)]]), factor
            )[0]
        )
        dcg += (2.0 ** relevance - 1.0) / np.log2(position + 1)
    return float(dcg / ideal)


def recall_at_k(top20, relevant, k=20):
    relevant = np.asarray(relevant, dtype=np.int64)
    if relevant.size == 0:
        return None
    return float(
        np.isin(top20[:int(k)], relevant).sum() / relevant.size
    )


def candidate_recall(candidates, relevant):
    relevant = np.asarray(relevant, dtype=np.int64)
    if relevant.size == 0:
        return None
    return float(np.isin(relevant, candidates).sum() / relevant.size)


class ReferenceEval:
    @staticmethod
    def ideal_dcg(ratings, factor, k=20):
        return ideal_dcg(ratings, factor, k)


class EngineStub:
    def predict(self, matrix):
        rows = np.asarray(matrix.sum(axis=1)).ravel()
        return rows * 0.5 + 3.0


class IncrementalUnionTest(unittest.TestCase):
    def test_snapshots_match_reference(self):
        rng = np.random.default_rng(11)
        n_users = 40
        n_movies = 300
        arrays = []
        for _ in range(n_users):
            size = int(rng.integers(0, 120))
            arrays.append(
                rng.choice(n_movies, size=size, replace=False)
                .astype(np.int64)
            )
        topk_values = [5, 10, 25, 50]
        topk_list = arrays
        topk_cache = {
            value: [values[:value] for values in arrays]
            for value in topk_values
        }
        neighbour_values = rng.choice(
            n_users, size=15, replace=False
        ).astype(np.int64)
        catalogue_mask = rng.random(n_movies) > 0.1
        catalogue_mask[:10] = True
        visible_cols = rng.choice(
            n_movies, size=40, replace=False
        ).astype(np.int64)
        popularity = rng.random(n_movies)
        n_values = [1, 3, 7, 15]
        for top_k in topk_values:
            snapshots = KNN.build_knn_union_snapshots(
                neighbour_values,
                n_values,
                top_k,
                topk_list,
                catalogue_mask,
                visible_cols,
                popularity,
            )
            for n_neighbors in n_values:
                reference = KNN.build_knn_union(
                    neighbour_values,
                    n_neighbors,
                    top_k,
                    topk_cache,
                    catalogue_mask,
                    visible_cols,
                    popularity,
                )
                ref_candidates, _, ref_order = reference
                ours_candidates, ours_order = snapshots[n_neighbors]
                if ref_candidates is None:
                    self.assertIsNone(ours_candidates)
                    continue
                np.testing.assert_array_equal(
                    ours_candidates, ref_candidates
                )
                np.testing.assert_array_equal(ours_order, ref_order)

    def test_repeated_movies_across_neighbours(self):
        rng = np.random.default_rng(5)
        n_movies = 50
        arrays = [
            np.array([1, 2, 3], dtype=np.int64),
            np.array([2, 3, 4], dtype=np.int64),
        ]
        neighbours = np.array([0, 1], dtype=np.int64)
        catalogue = np.ones(n_movies, dtype=bool)
        popularity = rng.random(n_movies)
        snapshots = KNN.build_knn_union_snapshots(
            neighbours,
            [1, 2],
            10,
            arrays,
            catalogue,
            np.zeros(0, dtype=np.int64),
            popularity,
        )
        reference = KNN.build_knn_union(
            neighbours,
            2,
            10,
            {10: [values[:10] for values in arrays]},
            catalogue,
            np.zeros(0, dtype=np.int64),
            popularity,
        )
        np.testing.assert_array_equal(
            snapshots[2][0], reference[0]
        )
        np.testing.assert_array_equal(
            snapshots[2][1], reference[2]
        )


class MetricsTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        self.n_movies = 400
        self.hidden_movies = rng.choice(
            self.n_movies, size=80, replace=False
        ).astype(np.int64)
        self.hidden_ratings = rng.uniform(1.0, 10.0, size=80)
        self.factor = 1.0
        self.threshold = 8.0
        self.cutoffs = [5, 10, 20, 30, 50, 100]
        self.truth = KNN.prepare_truth_arrays(
            ReferenceEval,
            self.hidden_movies,
            self.hidden_ratings,
            self.factor,
            self.threshold,
            self.n_movies,
            self.cutoffs,
        )
        self.truth_map = {
            int(movie): float(rating)
            for movie, rating in zip(
                self.hidden_movies, self.hidden_ratings
            )
        }
        self.ranked = rng.permutation(self.n_movies)[:200]

    def test_metrics_match_reference(self):
        for cutoff in self.cutoffs:
            ours_ndcg, ours_known = KNN.ndcg_known_fast(
                self.ranked,
                self.truth,
                self.truth["ideal_by_cutoff"][cutoff],
                cutoff,
            )
            ref_ndcg, ref_known = ndcg_known(
                self.ranked,
                self.truth_map,
                self.truth["ideal_by_cutoff"][cutoff],
                self.factor,
                cutoff,
            )
            self.assertAlmostEqual(ours_ndcg, ref_ndcg, places=10)
            self.assertEqual(ours_known, ref_known)
            self.assertAlmostEqual(
                KNN.recall_at_k_fast(self.ranked, self.truth, cutoff),
                recall_at_k(
                    self.ranked, self.truth["relevant"], cutoff
                ),
                places=10,
            )
            self.assertAlmostEqual(
                KNN.ndcg_from_ranked_fast(
                    self.ranked,
                    self.truth,
                    self.truth["ideal_by_cutoff"][cutoff],
                    cutoff,
                ),
                ndcg_from_ranked(
                    self.ranked,
                    self.truth_map,
                    self.truth["ideal_by_cutoff"][cutoff],
                    self.factor,
                    cutoff,
                ),
                places=10,
            )

    def test_candidate_recall_matches_reference(self):
        candidates = np.arange(0, self.n_movies, 7, dtype=np.int64)
        self.assertAlmostEqual(
            KNN.candidate_recall_fast(candidates, self.truth),
            candidate_recall(candidates, self.truth["relevant"]),
            places=10,
        )

    def test_empty_relevant(self):
        hidden = np.array([1, 2, 3], dtype=np.int64)
        ratings = np.array([1.0, 2.0, 3.0])
        truth = KNN.prepare_truth_arrays(
            ReferenceEval,
            hidden,
            ratings,
            self.factor,
            self.threshold,
            self.n_movies,
            self.cutoffs,
        )
        self.assertIsNone(
            KNN.recall_at_k_fast(np.array([1, 2]), truth, 5)
        )
        self.assertIsNone(
            KNN.candidate_recall_fast(np.array([1, 2]), truth)
        )
        value, known = KNN.ndcg_known_fast(
            np.array([1, 2, 3]), truth, 0.0, 5
        )
        self.assertIsNone(value)
        self.assertEqual(known, 0)


class PredictBatchedTest(unittest.TestCase):
    def test_batched_matches_direct(self):
        rng = np.random.default_rng(9)
        rows = 137
        cols = 23
        matrix = csr_matrix(
            rng.random((rows, cols)) > 0.8
        )
        engine = EngineStub()
        direct = engine.predict(matrix)
        for batch_size in (1, 5, 32, 137, 500):
            batched = KNN.predict_batched(engine, matrix, batch_size)
            np.testing.assert_allclose(batched, direct, atol=1e-12)


class CheckpointTest(unittest.TestCase):
    def test_atomic_state_and_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = KNN.CheckpointStore(root, "fingerprint-a", True)
            store.save(
                "validation_grid",
                {
                    "status": "running",
                    "done_users": [1, 2],
                    "accumulators": {},
                },
            )
            loaded = store.load("validation_grid")
            self.assertEqual(loaded["done_users"], [1, 2])
            self.assertEqual(loaded["fingerprint"], "fingerprint-a")
            other = KNN.CheckpointStore(root, "fingerprint-b", True)
            with self.assertRaises(KNN.CheckpointMismatch):
                other.load("validation_grid")
            fresh = KNN.CheckpointStore(root, "fingerprint-b", False)
            self.assertIsNone(fresh.load("validation_grid"))

    def test_parts_roundtrip(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = KNN.CheckpointStore(root, "fingerprint", True)
            frame = KNN.pd.DataFrame(
                {"a": [1, 2], "b": ["x", "y"]}
            )
            for index in (1, 0):
                KNN.write_csv_atomic(
                    store.part_path("fixed_test", index), frame
                )
            combined = store.combine_parts("fixed_test")
            self.assertEqual(combined.shape[0], 4)
            self.assertEqual(
                list(combined.columns), ["a", "b"]
            )

    def test_accumulator_roundtrip(self):
        accumulators = {"fixed": {}}
        bucket = KNN.new_bucket([5, 20])
        bucket["users"] = 3
        bucket["per_k"][20]["known_sum"] = 1.5
        accumulators["fixed"][(10, "centered|5|20|none")] = bucket
        payload = json.loads(
            json.dumps(
                KNN.serialize_accumulators(accumulators)
            )
        )
        restored = KNN.deserialize_accumulators(payload)
        key = (10, "centered|5|20|none")
        self.assertIn(key, restored["fixed"])
        self.assertEqual(restored["fixed"][key]["users"], 3)
        self.assertIn(20, restored["fixed"][key]["per_k"])


class MergeTest(unittest.TestCase):
    def test_merge_contributions(self):
        accumulators = {"fixed": {}}
        first = {}
        bucket = KNN.new_bucket([5, 20])
        bucket["users"] = 2
        bucket["per_k"][20]["ge"] = 2
        first[(5, "a")] = bucket
        second = {}
        bucket = KNN.new_bucket([5, 20])
        bucket["users"] = 1
        bucket["per_k"][20]["ge"] = 1
        second[(5, "a")] = bucket
        KNN.merge_contributions(accumulators, "fixed", first)
        KNN.merge_contributions(accumulators, "fixed", second)
        merged = accumulators["fixed"][(5, "a")]
        self.assertEqual(merged["users"], 3)
        self.assertEqual(merged["per_k"][20]["ge"], 3)


if __name__ == "__main__":
    unittest.main()
