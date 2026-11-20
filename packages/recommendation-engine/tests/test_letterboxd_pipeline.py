from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = ROOT.parent
SRC_DIR = PACKAGE_ROOT / "src"
for _path in (PACKAGE_ROOT, SRC_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from filmfinder import FilmFinderEngine
from filmfinder.letterboxd_import import load_letterboxd_profile

ZIP_PATH = ROOT / "fixtures" / "letterboxd-eslpej-2026-11-20-10-33-utc.zip"
ARTIFACTS = Path(
    __import__("os").environ.get(
        "FILMFINDER_ARTIFACTS_DIR", PACKAGE_ROOT / "artifacts"
    )
)


@unittest.skipUnless(
    ZIP_PATH.exists() and ARTIFACTS.exists(),
    "Letterboxd fixture or artifacts are unavailable.",
)
class LetterboxdPipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile = load_letterboxd_profile(
            ZIP_PATH, artifacts_dir=ARTIFACTS
        )
        cls.engine = FilmFinderEngine.load(ARTIFACTS)

    def test_import_report_consistency(self):
        report = self.profile.report
        self.assertEqual(report["ratings_total"], 81)
        self.assertGreaterEqual(report["ratings_matched"], 65)
        self.assertGreaterEqual(report["watched_matched"], 120)
        self.assertEqual(
            report["ratings_matched"] + report["ratings_unmatched"]
            + report["ratings_ambiguous"],
            report["ratings_total"],
        )
        self.assertGreater(
            report["excluded_movie_count"],
            report["ratings_matched"],
        )

    def test_rating_scale(self):
        self.assertEqual(self.profile.ratings["la-la-land"], 10.0)
        self.assertEqual(self.profile.ratings["parasite-2019"], 8.0)
        self.assertEqual(self.profile.ratings["groundhog-day"], 7.0)

    def test_watched_only_not_rated(self):
        watched_only = set(self.profile.watched_internal) - set(
            self.profile.ratings_internal
        )
        self.assertTrue(watched_only)
        self.assertNotIn(
            list(watched_only)[0], self.profile.ratings_internal
        )

    def test_recommendations(self):
        recommendations, stats = (
            self.engine.recommend_new_user_with_stats(
                self.profile.ratings,
                top_n=100,
                watched_movie_ids=self.profile.watched,
            )
        )
        self.assertEqual(len(recommendations), 100)
        self.assertTrue(all(
            recommendations[index]["score"]
            >= recommendations[index + 1]["score"]
            for index in range(len(recommendations) - 1)
        ))
        returned = {row["movie_id"] for row in recommendations}
        self.assertTrue(
            returned.isdisjoint(self.profile.ratings)
        )
        self.assertTrue(returned.isdisjoint(self.profile.watched))
        self.assertEqual(len(returned), len(recommendations))
        self.assertTrue(all(
            row["score_type"] == "m5_predicted_rating"
            for row in recommendations
        ))
        self.assertGreaterEqual(stats["pool_candidates"], 100)

    def test_determinism_and_model_integrity(self):
        before_w = self.engine.model.w.copy()
        before_v = self.engine.model.V.copy()
        first, _ = self.engine.recommend_new_user_with_stats(
            self.profile.ratings,
            top_n=100,
            watched_movie_ids=self.profile.watched,
        )
        second, _ = self.engine.recommend_new_user_with_stats(
            self.profile.ratings,
            top_n=100,
            watched_movie_ids=self.profile.watched,
        )
        self.assertEqual(first, second)
        np.testing.assert_array_equal(before_w, self.engine.model.w)
        np.testing.assert_array_equal(before_v, self.engine.model.V)

    def test_unknown_movie_rejected(self):
        with self.assertRaises(ValueError):
            self.engine.recommend_new_user(
                {"not-a-real-movie": 8.0},
                top_n=20,
            )


if __name__ == "__main__":
    unittest.main()
