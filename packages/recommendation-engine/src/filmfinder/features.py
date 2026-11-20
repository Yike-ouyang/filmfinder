from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.sparse import load_npz


def _native(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def _key(value):
    return str(_native(value))


class ArtifactStore:
    def __init__(self, root):
        self.root = Path(root)
        self.m5_weights = np.load(self.root / "m5_weights.npz")
        self.m5_config = json.loads(
            (self.root / "m5_config.json").read_text()
        )
        self.recommendation_config = json.loads(
            (self.root / "recommendation_config.json").read_text()
        )
        self.feature_mappings = json.loads(
            (self.root / "feature_mappings.json").read_text()
        )
        self.movie_metadata_raw = json.loads(
            (self.root / "movie_metadata.json").read_text()
        )
        if isinstance(self.movie_metadata_raw, dict):
            movies = self.movie_metadata_raw.get("movies")
        else:
            movies = self.movie_metadata_raw
        self.movies = list(movies or [])
        self.user_ids = np.load(
            self.root / "user_ids.npy", allow_pickle=True
        )
        self.movie_ids = np.load(
            self.root / "movie_ids.npy", allow_pickle=True
        )
        self.ratings = load_npz(self.root / "ratings_matrix.npz").tocsr()
        self.item_features = load_npz(
            self.root / "movie_features.npz"
        ).tocsr()
        self.user_index = {
            _key(value): index
            for index, value in enumerate(self.user_ids)
        }
        self.movie_index = {
            _key(value): index
            for index, value in enumerate(self.movie_ids)
        }
        if len(self.user_index) != len(self.user_ids):
            raise ValueError("Duplicate user IDs in user_ids.npy.")
        if len(self.movie_index) != len(self.movie_ids):
            raise ValueError("Duplicate movie IDs in movie_ids.npy.")
        if self.ratings.shape != (
            len(self.user_ids), len(self.movie_ids)
        ):
            raise ValueError("Ratings matrix shape mismatch.")
        if self.item_features.shape[0] != len(self.movie_ids):
            raise ValueError("Movie feature matrix shape mismatch.")
        if len(self.movies) != len(self.movie_ids):
            raise ValueError("Movie metadata length mismatch.")

    @classmethod
    def load(cls, root):
        return cls(root)

    def user_row(self, user_id):
        key = _key(user_id)
        if key not in self.user_index:
            raise KeyError(f"Unknown user_id: {user_id!r}")
        return self.user_index[key]

    def movie_col(self, movie_id):
        key = _key(movie_id)
        if key not in self.movie_index:
            raise KeyError(f"Unknown movie_id: {movie_id!r}")
        return self.movie_index[key]

    def movie_info(self, movie_col):
        movie_col = int(movie_col)
        row = self.movies[movie_col]
        if "genres" in row and row["genres"] is not None:
            genres = list(row["genres"])
        else:
            genres = []
        source_ids = row.get("source_ids") or {}
        if not isinstance(source_ids, dict):
            source_ids = {}
        return {
            "movie_id": _native(self.movie_ids[movie_col]),
            "internal_movie_id": movie_col,
            "title": row.get(
                "title", str(_native(self.movie_ids[movie_col]))
            ),
            "year": row.get("year"),
            "genres": genres,
            "tmdb_id": row.get("tmdb_id", source_ids.get("tmdb_id")),
            "poster_path": row.get(
                "poster_path", row.get("poster")
            ),
            "backdrop_path": row.get(
                "backdrop_path", row.get("backdrop")
            ),
        }
