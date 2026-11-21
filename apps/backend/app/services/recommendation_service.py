from __future__ import annotations

import csv
import io
import logging
import math
import threading
from typing import Optional

from filmfinder.letterboxd_import import (
    LetterboxdImportError,
    build_movie_matcher,
    load_letterboxd_profile,
)
from sqlalchemy.orm import Session

from app.core.exceptions import APIError
from app.core.zip_security import validate_archive
from app.db.models import User
from app.services.profile_service import ProfileService


logger = logging.getLogger(__name__)


class RecommendationService:
    def __init__(self, engine, metadata, settings):
        self.engine = engine
        self.metadata = metadata
        self.settings = settings
        self.matcher = build_movie_matcher(engine.store.movies)
        limit = max(1, int(settings.max_concurrent_recommendations))
        self.semaphore = threading.BoundedSemaphore(limit)

    def parse_letterboxd_archive(self, data):
        validate_archive(
            data,
            self.settings.max_upload_bytes,
            self.settings.max_zip_entries,
            self.settings.max_uncompressed_bytes,
        )
        try:
            return load_letterboxd_profile(
                io.BytesIO(data),
                matcher=self.matcher,
            )
        except LetterboxdImportError as error:
            raise APIError(
                400,
                "invalid_letterboxd_archive",
                "The Letterboxd archive could not be parsed.",
            ) from error
        except (UnicodeDecodeError, csv.Error, ValueError, KeyError) as error:
            raise APIError(
                400,
                "invalid_letterboxd_archive",
                "The Letterboxd archive could not be parsed.",
            ) from error

    def recommend_from_bytes(
        self,
        data,
        limit,
        *,
        db: Optional[Session] = None,
        user: Optional[User] = None,
        profile_service: Optional[ProfileService] = None,
        excluded_movie_ids: Optional[set[str]] = None,
    ):
        self._validate_limit(limit)
        profile = self.parse_letterboxd_archive(data)
        if user is None:
            return self.recommend_from_profile(profile, limit)

        if db is None or profile_service is None:
            raise APIError(
                500,
                "profile_persistence_failed",
                "The imported profile could not be saved.",
            )

        existing = profile_service.load_profile(db, user.id)
        if not existing.ratings and not profile.ratings:
            raise APIError(
                422,
                "no_matched_ratings",
                "No usable Letterboxd ratings could be matched to the catalogue.",
            )
        try:
            profile_service.save_import(db, user, profile)
        except Exception as error:
            logger.exception("Letterboxd profile persistence failed.")
            raise APIError(
                500,
                "profile_persistence_failed",
                "The imported profile could not be saved.",
            ) from error

        merged = profile_service.load_profile(db, user.id)
        if not merged.ratings:
            raise APIError(
                422,
                "no_matched_ratings",
                "No usable Letterboxd ratings could be matched to the catalogue.",
            )
        watched = set(merged.watched_movie_ids)
        watched.update(excluded_movie_ids or set())
        return self.recommend_from_profile(
            profile,
            limit,
            ratings=merged.ratings,
            watched=watched,
        )

    def recommend_from_profile(
        self,
        profile,
        limit,
        *,
        ratings=None,
        watched=None,
    ):
        limit = self._validate_limit(limit)
        used_ratings = profile.ratings if ratings is None else ratings
        used_watched = profile.watched if watched is None else watched
        if not used_ratings:
            raise APIError(
                422,
                "no_matched_ratings",
                "No usable Letterboxd ratings could be matched to the catalogue.",
            )
        report = profile.report
        ratings_total = int(report.get("ratings_total", 0))
        ratings_matched = int(report.get("ratings_matched", 0))
        match_rate = (
            ratings_matched / ratings_total if ratings_total else 0.0
        )
        summary = {
            "ratings_found": ratings_total,
            "ratings_matched": ratings_matched,
            "watched_movies_matched": int(
                report.get("watched_matched", 0)
            ),
            "match_rate": float(match_rate),
        }
        return self._recommend_response(
            used_ratings,
            used_watched,
            limit,
            summary,
        )

    def recommend_for_saved_profile(
        self,
        ratings,
        watched,
        limit,
        summary,
    ):
        limit = self._validate_limit(limit)
        if not ratings:
            raise APIError(
                422,
                "no_saved_profile",
                "No saved ratings are available for this account.",
            )
        return self._recommend_response(ratings, watched, limit, summary)

    def _recommend_response(self, ratings, watched, limit, summary):
        try:
            with self.semaphore:
                recommendations, _stats = (
                    self.engine.recommend_new_user_with_stats(
                        ratings,
                        top_n=limit,
                        watched_movie_ids=watched,
                    )
                )
        except (ValueError, KeyError) as error:
            raise APIError(
                422,
                "recommendation_input_error",
                "The imported profile could not be used for recommendations.",
            ) from error
        except Exception as error:
            raise APIError(
                500,
                "inference_error",
                "The recommendation engine failed.",
            ) from error
        if not recommendations:
            raise APIError(
                422,
                "no_recommendations",
                "No recommendations are available for this profile.",
            )
        for recommendation in recommendations:
            if not math.isfinite(
                float(recommendation.get("score", float("nan")))
            ):
                raise APIError(
                    500,
                    "invalid_score",
                    "The recommendation engine returned an invalid score.",
                )
        enriched = [
            self.metadata.enrich(recommendation)
            for recommendation in recommendations
        ]
        model = self.engine.store.m5_config.get("model", "M5")
        algorithm = self.engine.store.recommendation_config.get(
            "algorithm", "knn_m5"
        )
        return {
            "profile": dict(summary),
            "recommendation_info": {
                "model": str(model),
                "algorithm": str(algorithm),
                "requested_count": int(limit),
                "returned_count": len(enriched),
            },
            "recommendations": enriched,
        }

    def _validate_limit(self, limit):
        limit = int(limit)
        if limit < 1 or limit > self.engine.pool_size:
            raise APIError(
                422,
                "invalid_limit",
                f"limit must be between 1 and {self.engine.pool_size}.",
            )
        return limit
