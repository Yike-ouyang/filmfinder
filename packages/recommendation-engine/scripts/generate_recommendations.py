from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PACKAGE_ROOT / "src"
for _path in (PACKAGE_ROOT, SRC_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from filmfinder import FilmFinderEngine
from filmfinder.letterboxd_import import load_letterboxd_profile


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate FilmFinder recommendations from a Letterboxd zip."
    )
    parser.add_argument("--letterboxd-zip", required=True)
    parser.add_argument(
        "--artifacts",
        default=str(
            Path(__file__).resolve().parents[1] / "artifacts"
        ),
    )
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument(
        "--output",
        default="recommendations.json",
    )
    parser.add_argument(
        "--report",
        default="import_report.json",
    )
    return parser.parse_args()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def main():
    args = parse_args()
    started = time.perf_counter()
    artifacts = Path(args.artifacts)
    if not artifacts.exists():
        raise FileNotFoundError(f"Artifacts not found: {artifacts}")
    profile = load_letterboxd_profile(
        args.letterboxd_zip,
        artifacts_dir=artifacts,
    )
    load_start = time.perf_counter()
    engine = FilmFinderEngine.load(artifacts)
    load_time = time.perf_counter() - load_start
    recommend_start = time.perf_counter()
    recommendations, stats = engine.recommend_new_user_with_stats(
        profile.ratings,
        top_n=args.top_n,
        watched_movie_ids=profile.watched,
    )
    recommend_time = time.perf_counter() - recommend_start
    report = dict(profile.report)
    report.update({
        "artifacts": str(artifacts.resolve()),
        "engine_load_seconds": load_time,
        "recommendation_seconds": recommend_time,
        "total_seconds": time.perf_counter() - started,
        "recommendation_config": engine.store.recommendation_config,
        "model_config": {
            "model_version": engine.store.m5_config.get("model_version"),
            "trained_on": engine.store.m5_config.get("trained_on"),
            "n_features": engine.store.m5_config.get("n_features"),
        },
        "recommendation_stats": stats,
        "requested_top_n": int(args.top_n),
        "returned_recommendations": len(recommendations),
    })
    if len(recommendations) < args.top_n:
        report["limitation"] = (
            f"Only {len(recommendations)} recommendations available for "
            f"top_n={args.top_n} after filtering."
        )
    write_json(args.output, recommendations)
    write_json(args.report, report)

    print("Letterboxd profile loaded.")
    print(f"Ratings found: {profile.report['ratings_total']}")
    print(f"Ratings matched: {profile.report['ratings_matched']}")
    print(f"Ratings unmatched: {profile.report['ratings_unmatched']}")
    print(f"Ratings ambiguous: {profile.report['ratings_ambiguous']}")
    print(f"Watched movies found: {profile.report['watched_total']}")
    print(f"Watched movies matched: {profile.report['watched_matched']}")
    print(f"Watched movies unmatched: {profile.report['watched_unmatched']}")
    print(f"Watched/rated movies excluded: {profile.report['excluded_movie_count']}")
    print(
        "Recommendation strategy: "
        f"{engine.store.recommendation_config.get('algorithm')} / "
        f"{engine.store.recommendation_config.get('ranking')}"
    )
    print(f"Candidates raw: {stats.get('raw_occurrences')}")
    print(f"Candidates unique: {stats.get('unique_candidates')}")
    print(f"Candidates after filtering: {stats.get('after_seen')}")
    print(f"Candidates in pool: {stats.get('pool_candidates')}")
    print(f"{len(recommendations)} recommendations generated.")
    if len(recommendations) < args.top_n:
        print(report["limitation"])
    print(f"Output: {Path(args.output).resolve()}")
    print(f"Report: {Path(args.report).resolve()}")


if __name__ == "__main__":
    main()
