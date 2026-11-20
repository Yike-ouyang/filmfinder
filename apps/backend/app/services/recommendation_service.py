from __future__ import annotations

import csv
import io
import math
import threading

from filmfinder.letterboxd_import import (
    LetterboxdImportError,
    build_movie_matcher,
    load_letterboxd_profile,
)

from app.core.exceptions import APIError
from app.core.zip_security import validate_archive


class RecommendationService:
    def __init__(self, engine, metadata, settings):
        self.engine = engine
        self.metadata = metadata
        self.settings = settings
        self.matcher = build_movie_matcher(engine.store.movies)
        limit = max(1, int(settings.max_concurrent_recommendations))
        self.semaphore = threading.BoundedSemaphore(limit)

    def recommend_from_bytes(self, data, limit):
        limit = int(limit)
        if limit < 1 or limit > self.engine.pool_size:
            raise APIError(
                422,
                "invalid_limit",
                f"limit must be between 1 and {self.engine.pool_size}.",
            )
        validate_archive(
            data,
            self.settings.max_upload_bytes,
            self.settings.max_zip_entries,
            self.settings.max_uncompressed_bytes,
        )
        try:
            profile = load_letterboxd_profile(
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
        report = profile.report
        if int(report.get("ratings_matched", 0)) < 1:
            raise APIError(
                422,
                "no_matched_ratings",
                "No usable Letterboxd ratings could be matched to the catalogue.",
            )
        try:
            with self.semaphore:
                recommendations, stats = (
                    self.engine.recommend_new_user_with_stats(
                        profile.ratings,
                        top_n=limit,
                        watched_movie_ids=profile.watched,
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
            if not math.isfinite(float(recommendation.get("score", float("nan")))):
                raise APIError(
                    500,
                    "invalid_score",
                    "The recommendation engine returned an invalid score.",
                )
        enriched = [
            self.metadata.enrich(recommendation)
            for recommendation in recommendations
        ]
        ratings_total = int(report.get("ratings_total", 0))
        ratings_matched = int(report.get("ratings_matched", 0))
        match_rate = (
            ratings_matched / ratings_total if ratings_total else 0.0
        )
        model = self.engine.store.m5_config.get("model", "M5")
        algorithm = self.engine.store.recommendation_config.get(
            "algorithm", "knn_m5"
        )
        return {
            "profile": {
                "ratings_found": ratings_total,
                "ratings_matched": ratings_matched,
                "watched_movies_matched": int(
                    report.get("watched_matched", 0)
                ),
                "match_rate": float(match_rate),
            },
            "recommendation_info": {
                "model": str(model),
                "algorithm": str(algorithm),
                "requested_count": int(limit),
                "returned_count": len(enriched),
            },
            "recommendations": enriched,
        }
