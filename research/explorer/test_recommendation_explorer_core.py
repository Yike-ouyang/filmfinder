import importlib.util
import os
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


CORE_PATH = Path(__file__).with_name("recommendation_explorer_core.py")
SPEC = importlib.util.spec_from_file_location(
    "recommendation_explorer_core_tests", CORE_PATH
)
CORE = importlib.util.module_from_spec(SPEC)
sys.modules["recommendation_explorer_core_tests"] = CORE
SPEC.loader.exec_module(CORE)


def _same(ours, theirs, tolerance=1e-9):
    ours_missing = ours is None or (
        isinstance(ours, float) and np.isnan(ours)
    )
    theirs_missing = theirs is None or (
        isinstance(theirs, float) and np.isnan(theirs)
    )
    if ours_missing or theirs_missing:
        return bool(ours_missing and theirs_missing)
    return abs(float(ours) - float(theirs)) <= tolerance


class ConfigParsingTest(unittest.TestCase):
    def test_cold_global(self):
        config = CORE._config_from_cold_global(
            {
                "method": "centered",
                "n_neighbors": 50,
                "top_k": 20,
                "pool_size": 100,
            },
            "selected_configs.json",
        )
        self.assertEqual(config.method, "centered")
        self.assertEqual(config.n_neighbors, 50)
        self.assertEqual(config.top_k, 20)
        self.assertEqual(config.pool_size, 100)
        self.assertTrue(config.uses_knn)

    def test_warm(self):
        config = CORE._config_from_warm(
            {
                "approach": "knn_m5",
                "similarity": "centered",
                "n_neighbors": 10,
                "top_k": 20,
            },
            "warm/selected_configs.json",
        )
        self.assertEqual(config.approach, "knn_m5")
        self.assertEqual(config.method, "centered")
        self.assertEqual(config.n_neighbors, 10)
        self.assertEqual(config.top_k, 20)
        self.assertIsNone(config.pool_size)

    def test_unknown_warm_approach(self):
        with self.assertRaises(CORE.ArtifactError):
            CORE._config_from_warm({"approach": "unknown"}, "x")


@unittest.skipUnless(
    os.environ.get("FILMFINDER_EXPLORER_TESTS") == "1",
    "Set FILMFINDER_EXPLORER_TESTS=1 to run artifact-backed tests.",
)
class ArtifactEquivalenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.context = CORE.load_explorer()
        cls.users = cls.context.users

    def test_cold_matches_saved_per_user(self):
        path = (
            self.context.artifacts.knn_dir
            / "test"
            / "fixed_per_user.csv"
        )
        if not path.exists():
            self.skipTest(f"Missing {path}")
        frame = pd.read_csv(path)
        frame = frame[
            (frame["role"] == "global")
            & (frame["cutoff"] == 20)
        ]
        if frame.empty:
            self.skipTest("No global K=20 rows")
        for level in sorted(frame["level"].unique()):
            rows = frame[frame["level"] == level]
            user = int(rows.iloc[0]["user"])
            pipeline = CORE.compute_pipeline(
                self.context, user, level=int(level)
            )
            expected = rows.iloc[0]
            self.assertTrue(
                _same(
                    pipeline.metrics["candidate_recall"],
                    expected["candidate_recall"],
                ),
                f"candidate recall level {level}",
            )
            self.assertTrue(
                _same(
                    pipeline.metrics["recall20"],
                    expected["recall"],
                ),
                f"recall level {level}",
            )
            self.assertTrue(
                _same(
                    pipeline.metrics["ndcg_known"],
                    expected["ndcg_known"],
                ),
                f"ndcg level {level}",
            )
            self.assertEqual(
                int(pipeline.metrics["candidate_size"]),
                int(expected["candidate_size"]),
            )
            self.assertEqual(
                int(pipeline.metrics["known_count"]),
                int(expected["known_count"]),
            )

    def test_warm_matches_saved_per_user(self):
        path = (
            self.context.artifacts.baseline_dir
            / "warm"
            / "final_per_user.csv"
        )
        if not path.exists():
            self.skipTest(f"Missing {path}")
        frame = pd.read_csv(path)
        frame = frame[frame["approach"] == "knn_m5"]
        if frame.empty:
            self.skipTest("No warm knn_m5 rows")
        for _, expected in frame.head(5).iterrows():
            pipeline = CORE.compute_pipeline(
                self.context,
                int(expected["user"]),
                scenario="warm",
                level=0,
            )
            self.assertTrue(
                _same(
                    pipeline.metrics["candidate_recall"],
                    expected["candidate_recall"],
                )
            )
            self.assertTrue(
                _same(
                    pipeline.metrics["recall20"],
                    expected["recall20"],
                )
            )
            self.assertTrue(
                _same(
                    pipeline.metrics["ndcg_known"],
                    expected["ndcg_known"],
                )
            )
            self.assertEqual(
                int(pipeline.metrics["candidate_size"]),
                int(expected["candidate_size"]),
            )
            self.assertEqual(
                int(pipeline.metrics["known_count"]),
                int(expected["known_count"]),
            )

    def test_pipeline_integrity(self):
        for record in list(self.users.values())[:10]:
            pipeline = CORE.compute_pipeline(
                self.context,
                record.user_row,
                level=record.levels[-1],
            )
            self.assertEqual(
                pipeline.stage_counts["hidden_removed_as_seen"], 0
            )
            self.assertLessEqual(
                pipeline.stage_counts["pool_cap"],
                pipeline.stage_counts["eligible_before_pool"],
            )
            self.assertLessEqual(
                pipeline.stage_counts["unique_candidates"],
                pipeline.stage_counts["raw_occurrences"],
            )
            self.assertFalse(
                set(pipeline.visible_cols.tolist())
                & set(pipeline.hidden_movies.tolist())
            )
            if len(pipeline.top20_table):
                movie = int(pipeline.top20_table.iloc[0]["movie_col"])
                tracked = CORE.track_movie(
                    self.context, pipeline, movie
                )
                self.assertTrue(tracked.get("in_top20"))

    def test_cold_zero_is_fallback(self):
        cold = [
            record for record in self.users.values()
            if record.scenario == "cold" and 0 in record.levels
        ]
        if not cold:
            self.skipTest("No cold test users")
        pipeline = CORE.compute_pipeline(
            self.context, cold[0].user_row, level=0
        )
        self.assertEqual(pipeline.source, "fallback")
        self.assertEqual(len(pipeline.neighbor_order), 0)


if __name__ == "__main__":
    unittest.main()
