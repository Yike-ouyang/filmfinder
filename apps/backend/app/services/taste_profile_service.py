from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from app.core.exceptions import APIError
from app.schemas.taste_profile import (
    TasteProfileAnswer,
    TasteProfileMovie,
    TasteProfileRatingScale,
)


CONFIG_VERSION = 1
EXPECTED_MOVIE_COUNT = 20
ALLOWED_RATINGS = (3, 5, 8)


class TasteProfileConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class TasteProfileSelection:
    ratings: dict[str, float]
    watched_movie_ids: set[str]
    ratings_used: int
    skipped: int
    summary: dict

    @property
    def has_enough_ratings(self) -> bool:
        return self.ratings_used > 0


class TasteProfileService:
    def __init__(self, engine, metadata, payload, config_path: Path):
        self.engine = engine
        self.metadata = metadata
        self.config_path = Path(config_path)
        self.min_ratings = int(payload.get("min_ratings", 3))
        if self.min_ratings < 1:
            raise TasteProfileConfigError(
                "Quick Taste minimum rating count must be positive."
            )
        scale = payload.get("rating_scale") or {}
        self.rating_scale = TasteProfileRatingScale(
            loved=int(scale.get("loved", 8)),
            okay=int(scale.get("okay", 5)),
            disliked=int(scale.get("disliked", 3)),
        )
        movies = payload.get("movies")
        if not isinstance(movies, list):
            raise TasteProfileConfigError(
                "Quick Taste configuration has no movie list."
            )
        if len(movies) != EXPECTED_MOVIE_COUNT:
            raise TasteProfileConfigError(
                "Quick Taste configuration must contain exactly "
                f"{EXPECTED_MOVIE_COUNT} movies."
            )
        self._movies = [
            self._build_movie(entry) for entry in movies
        ]
        self._by_internal = {
            movie.internal_movie_id: movie for movie in self._movies
        }
        if len(self._by_internal) != EXPECTED_MOVIE_COUNT:
            raise TasteProfileConfigError(
                "Quick Taste configuration contains duplicate movies."
            )

    @classmethod
    def load(cls, engine, metadata, path):
        path = Path(path)
        if not path.exists():
            raise TasteProfileConfigError(
                f"Quick Taste configuration not found: {path}"
            )
        try:
            payload = json.loads(path.read_text())
        except (OSError, ValueError) as error:
            raise TasteProfileConfigError(
                f"Quick Taste configuration could not be read: {path}"
            ) from error
        if not isinstance(payload, dict):
            raise TasteProfileConfigError(
                "Quick Taste configuration must be a JSON object."
            )
        if int(payload.get("version", 0)) != CONFIG_VERSION:
            raise TasteProfileConfigError(
                "Unsupported Quick Taste configuration version."
            )
        return cls(engine, metadata, payload, path)

    def _build_movie(self, entry) -> TasteProfileMovie:
        if not isinstance(entry, dict):
            raise TasteProfileConfigError(
                "Quick Taste movie entries must be objects."
            )
        internal_id = int(entry.get("internal_movie_id", -1))
        if internal_id < 0 or internal_id >= len(
            self.engine.store.movies
        ):
            raise TasteProfileConfigError(
                f"Invalid internal_movie_id in configuration: {internal_id}"
            )
        info = self.engine.store.movie_info(internal_id)
        configured_movie_id = str(entry.get("movie_id") or "")
        if configured_movie_id != str(info["movie_id"]):
            raise TasteProfileConfigError(
                "Quick Taste configuration does not match the catalogue "
                f"for internal_movie_id {internal_id}."
            )
        if int(self.engine.store.ratings.getcol(internal_id).getnnz()) == 0:
            raise TasteProfileConfigError(
                f"Movie has no ratings in the catalogue: {internal_id}"
            )
        if self.engine.store.item_features.getrow(internal_id).getnnz() == 0:
            raise TasteProfileConfigError(
                f"Movie has no M5 features: {internal_id}"
            )
        movie = {
            "internal_movie_id": internal_id,
            "movie_id": str(info["movie_id"]),
            "title": str(entry.get("title") or info["title"]),
            "year": entry.get("year", info.get("year")),
            "genres": list(entry.get("genres") or info.get("genres") or []),
            "tmdb_id": entry.get("tmdb_id", info.get("tmdb_id")),
            "poster_path": entry.get("poster_path", info.get("poster_path")),
            "backdrop_path": entry.get(
                "backdrop_path", info.get("backdrop_path")
            ),
        }
        enriched = self.metadata.enrich(movie)
        return TasteProfileMovie(**enriched)

    @property
    def movies(self) -> list[TasteProfileMovie]:
        return list(self._movies)

    @property
    def movie_ids(self) -> set[int]:
        return set(self._by_internal)

    def get_movie(self, internal_movie_id: int) -> Optional[TasteProfileMovie]:
        return self._by_internal.get(int(internal_movie_id))

    def validate_answers(
        self,
        answers: list[TasteProfileAnswer],
    ) -> TasteProfileSelection:
        identifiers = [answer.internal_movie_id for answer in answers]
        if len(set(identifiers)) != len(identifiers):
            raise APIError(
                422,
                "invalid_taste_selection",
                "A movie was submitted more than once.",
            )
        if set(identifiers) != self.movie_ids:
            raise APIError(
                422,
                "invalid_taste_selection",
                "The submitted movies do not match the Taste Profile selection.",
            )
        ratings = {}
        watched = set()
        for answer in answers:
            if answer.rating is None:
                continue
            if int(answer.rating) not in ALLOWED_RATINGS:
                raise APIError(
                    422,
                    "invalid_taste_rating",
                    "Taste Profile ratings must be 3, 5 or 8.",
                )
            movie = self._by_internal[answer.internal_movie_id]
            ratings[movie.movie_id] = float(answer.rating)
            watched.add(movie.movie_id)
        used = len(ratings)
        if used < self.min_ratings:
            raise APIError(
                422,
                "insufficient_taste_ratings",
                "Rate at least "
                f"{self.min_ratings} movies to get recommendations.",
            )
        summary = {
            "ratings_found": EXPECTED_MOVIE_COUNT,
            "ratings_matched": used,
            "watched_movies_matched": used,
            "match_rate": used / EXPECTED_MOVIE_COUNT,
        }
        return TasteProfileSelection(
            ratings=ratings,
            watched_movie_ids=watched,
            ratings_used=used,
            skipped=EXPECTED_MOVIE_COUNT - used,
            summary=summary,
        )
