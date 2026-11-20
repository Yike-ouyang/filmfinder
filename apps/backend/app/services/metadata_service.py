from __future__ import annotations

import json
from pathlib import Path


class TMDBMetadataService:
    def __init__(self, mapping=None):
        self._movies = dict(mapping or {})

    @classmethod
    def load(cls, path):
        if path is None:
            return cls({})
        path = Path(path)
        if not path.exists():
            return cls({})
        try:
            payload = json.loads(path.read_text())
        except (OSError, ValueError):
            return cls({})
        movies = payload.get("movies") if isinstance(payload, dict) else payload
        if not isinstance(movies, dict):
            movies = {}
        normalized = {
            str(key): value
            for key, value in movies.items()
            if isinstance(value, dict)
        }
        return cls(normalized)

    def enrich(self, recommendation):
        item = dict(recommendation)
        metadata = self._movies.get(str(item.get("internal_movie_id")))
        if metadata:
            for field in ("tmdb_id", "poster_path", "backdrop_path"):
                if metadata.get(field) is not None:
                    item[field] = metadata[field]
        return item

    @property
    def size(self):
        return len(self._movies)
