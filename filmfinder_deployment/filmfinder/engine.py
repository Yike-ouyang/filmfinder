from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .features import ArtifactStore, _key
from .model import M5Model
from .recommender import KNNM5Recommender


class FilmFinderEngine:
    def __init__(self, store, model, recommender):
        self.store = store
        self.model = model
        self.recommender = recommender
        self.n_users = int(store.ratings.shape[0])
        self.n_movies = int(store.ratings.shape[1])
        self.config = store.recommendation_config
        self.default_top_n = int(self.config.get("default_top_n", 20))
        self.pool_size = int(self.config.get("candidate_pool_size", 100))

    @classmethod
    def load(cls, artifacts_path):
        store = ArtifactStore.load(Path(artifacts_path))
        model = M5Model(store)
        recommender = KNNM5Recommender(store, model)
        return cls(store, model, recommender)

    def _validate_top_n(self, top_n):
        top_n = int(top_n)
        if top_n < 1:
            raise ValueError("top_n must be positive.")
        if top_n > self.pool_size:
            raise ValueError(
                f"top_n={top_n} exceeds candidate pool size "
                f"{self.pool_size}."
            )
        return top_n

    def recommend(self, user_id, top_n=None):
        top_n = self._validate_top_n(
            self.default_top_n if top_n is None else top_n
        )
        try:
            user_row = self.store.user_row(user_id)
        except KeyError as error:
            raise ValueError(
                f"Unknown user_id {user_id!r}. Use recommend_new_user "
                "for a user outside the training matrix."
            ) from error
        recommendations, stats = self.recommender.recommend_existing(
            user_row, top_n
        )
        return recommendations

    def _watched_cols(self, watched_movie_ids):
        if watched_movie_ids is None:
            return np.zeros(0, dtype=np.int64)
        watched = []
        seen_keys = set()
        for movie_id in watched_movie_ids:
            key = _key(movie_id)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            try:
                watched.append(self.store.movie_col(movie_id))
            except KeyError as error:
                raise ValueError(
                    f"Unknown watched movie_id: {movie_id!r}"
                ) from error
        return np.asarray(watched, dtype=np.int64)

    def recommend_new_user_with_stats(
        self,
        ratings,
        top_n=None,
        watched_movie_ids=None,
    ):
        top_n = self._validate_top_n(
            self.default_top_n if top_n is None else top_n
        )
        if ratings is None:
            ratings = {}
        if not isinstance(ratings, dict):
            raise TypeError("ratings must be a dict movie_id -> rating.")
        movie_cols = []
        values = []
        seen_keys = set()
        for movie_id, rating in ratings.items():
            if _key(movie_id) in seen_keys:
                raise ValueError(f"Duplicate movie_id: {movie_id!r}")
            seen_keys.add(_key(movie_id))
            try:
                movie_col = self.store.movie_col(movie_id)
            except KeyError as error:
                raise ValueError(
                    f"Unknown movie_id: {movie_id!r}"
                ) from error
            value = float(rating)
            if not math.isfinite(value):
                raise ValueError(
                    f"Non-finite rating for movie_id {movie_id!r}."
                )
            movie_cols.append(movie_col)
            values.append(value)
        watched_cols = self._watched_cols(watched_movie_ids)
        recommendations, stats = self.recommender.recommend_new(
            np.asarray(movie_cols, dtype=np.int64),
            np.asarray(values, dtype=np.float64),
            top_n,
            extra_seen=watched_cols,
        )
        stats["requested_top_n"] = int(top_n)
        stats["returned"] = int(len(recommendations))
        return recommendations, stats

    def recommend_new_user(
        self,
        ratings,
        top_n=None,
        watched_movie_ids=None,
    ):
        recommendations, stats = self.recommend_new_user_with_stats(
            ratings,
            top_n=top_n,
            watched_movie_ids=watched_movie_ids,
        )
        return recommendations
