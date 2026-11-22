from __future__ import annotations

import json
from pathlib import Path


def load_catalogue(path):
    payload = json.loads(Path(path).read_text())
    movies = payload.get("movies") if isinstance(payload, dict) else payload
    if not movies:
        raise ValueError(f"No movies found in {path}")
    return list(movies)


def build_catalogue_index(movies):
    by_source = {}
    by_movie_id = {}
    for movie in movies:
        internal = int(movie["internal_id"])
        movie_id = movie.get("movie_id")
        if movie_id is not None:
            by_movie_id[str(movie_id)] = internal
        for value in (movie.get("source_ids") or {}).values():
            if value is not None:
                by_source[str(value)] = internal
    return by_source, by_movie_id


def field_value(row, name):
    if name is None:
        return None
    return row.get(name)


def detect_column(fieldnames, candidates):
    for candidate in candidates:
        if candidate in fieldnames:
            return candidate
    return None


def match_internal(row, by_source, by_movie_id, id_column, movie_column):
    for column in (id_column, movie_column):
        if column is None:
            continue
        value = field_value(row, column)
        if value is None:
            continue
        key = str(value)
        if key in by_source:
            return by_source[key]
        if key in by_movie_id:
            return by_movie_id[key]
    return None
