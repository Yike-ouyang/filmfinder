from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from filmfinder import FilmFinderEngine


def pick_existing_user(engine):
    for user_row in range(engine.n_users):
        if engine.recommender.ratings.getrow(user_row).nnz > 0:
            return user_row
    raise RuntimeError("No user with ratings in the exported matrix.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifacts",
        default=str(Path(__file__).resolve().parent / "artifacts"),
    )
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args()

    started = time.perf_counter()
    engine = FilmFinderEngine.load(args.artifacts)
    load_time = time.perf_counter() - started
    user_row = pick_existing_user(engine)
    user_id = engine.store.user_ids[user_row]
    started = time.perf_counter()
    existing = engine.recommend(user_id, top_n=args.top_n)
    existing_time = time.perf_counter() - started
    if not 0 < len(existing) <= args.top_n:
        raise AssertionError("Unexpected number of recommendations.")
    seen = set(
        int(value)
        for value in engine.recommender.ratings.getrow(user_row).indices
    )
    for row in existing:
        movie_col = engine.store.movie_col(row["movie_id"])
        if movie_col in seen:
            raise AssertionError("Already rated movie returned.")
    scores = [row["score"] for row in existing]
    if scores != sorted(scores, reverse=True):
        raise AssertionError("Recommendations are not sorted by score.")

    movie_rows = [
        int(value)
        for value in np.flatnonzero(engine.recommender.catalogue_mask)[:3]
    ]
    ratings = {
        engine.store.movie_ids[movie_col]: rating
        for movie_col, rating in zip(movie_rows, (8.0, 9.0, 6.0))
    }
    before_w = engine.model.w.copy()
    before_v = engine.model.V.copy()
    started = time.perf_counter()
    new_user = engine.recommend_new_user(ratings, top_n=args.top_n)
    new_time = time.perf_counter() - started
    if not 0 < len(new_user) <= args.top_n:
        raise AssertionError("Unexpected number of cold recommendations.")
    cold_seen = set(movie_rows)
    for row in new_user:
        if engine.store.movie_col(row["movie_id"]) in cold_seen:
            raise AssertionError("Cold-start rated movie returned.")
    if not np.array_equal(before_w, engine.model.w):
        raise AssertionError("Global w changed after cold-start inference.")
    if not np.array_equal(before_v, engine.model.V):
        raise AssertionError("Global V changed after cold-start inference.")

    report = {
        "n_users": engine.n_users,
        "n_movies": engine.n_movies,
        "load_time_seconds": load_time,
        "existing_user_seconds": existing_time,
        "new_user_seconds": new_time,
        "existing_top_n": len(existing),
        "new_top_n": len(new_user),
        "artifacts": str(Path(args.artifacts).resolve()),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
