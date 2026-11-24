from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.exceptions import APIError
from app.db.models import UserFilm, UserSession
from app.db.session import get_db
from app.dependencies.auth import (
    get_client_ip,
    get_optional_session,
    require_optional_csrf,
)
from app.dependencies.recommendation_engine import (
    get_batch_service,
    get_profile_service,
    get_rating_service,
    get_recommendation_service,
    get_taste_profile_service,
)
from app.dependencies.swipes import get_swipe_service
from app.schemas.recommendations import RecommendationResponse
from app.schemas.taste_profile import (
    TasteProfileMoviesResponse,
    TasteProfileRequest,
)


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/taste-profile",
    tags=["taste-profile"],
)


def _rate_key(
    record: Optional[UserSession],
    request: Request,
) -> str:
    if record is not None:
        return f"taste-user:{record.user_id}"
    return f"taste-ip:{get_client_ip(request) or '-'}"


def _check_limit(request: Request, key: str) -> None:
    settings = request.app.state.settings
    limiter = request.app.state.rate_limiter
    if limiter.is_limited(
        key,
        settings.taste_profile_rate_limit_attempts,
        settings.taste_profile_rate_limit_window_seconds,
    ):
        raise APIError(
            429,
            "rate_limited",
            "Too many Taste Profile requests. Try again later.",
        )


def _record_limit(request: Request, key: str) -> None:
    settings = request.app.state.settings
    request.app.state.rate_limiter.record(
        key,
        settings.taste_profile_rate_limit_window_seconds,
    )


@router.get("/movies", response_model=TasteProfileMoviesResponse)
def taste_profile_movies(
    record: Optional[UserSession] = Depends(get_optional_session),
    db: Session = Depends(get_db),
    service=Depends(get_taste_profile_service),
):
    movies = service.movies
    user_ratings: dict[int, int] = {}
    if record is not None and movies:
        rows = db.execute(
            select(UserFilm.internal_movie_id, UserFilm.rating).where(
                UserFilm.user_id == record.user_id,
                UserFilm.internal_movie_id.in_(
                    [movie.internal_movie_id for movie in movies]
                ),
                UserFilm.rating.is_not(None),
            )
        ).all()
        user_ratings = {
            int(internal_movie_id): int(rating)
            for internal_movie_id, rating in rows
            if rating is not None
        }
    return TasteProfileMoviesResponse(
        movies=[
            movie.model_copy(
                update={
                    "user_rating": user_ratings.get(
                        movie.internal_movie_id
                    )
                }
            )
            for movie in movies
        ],
        min_ratings=service.min_ratings,
        recommended_min_ratings=service.recommended_min_ratings,
    )


@router.post(
    "/recommendations",
    response_model=RecommendationResponse,
)
def taste_profile_recommendations(
    payload: TasteProfileRequest,
    request: Request,
    limit: int = Query(default=100, ge=1, le=100),
    record: Optional[UserSession] = Depends(get_optional_session),
    db: Session = Depends(get_db),
    service=Depends(get_recommendation_service),
    taste_service=Depends(get_taste_profile_service),
    profile_service=Depends(get_profile_service),
    batch_service=Depends(get_batch_service),
    swipe_service=Depends(get_swipe_service),
    rating_service=Depends(get_rating_service),
    _: None = Depends(require_optional_csrf),
):
    key = _rate_key(record, request)
    _check_limit(request, key)
    _record_limit(request, key)

    selection = taste_service.validate_answers(payload.answers)
    if record is None:
        if selection.ratings_used < taste_service.min_ratings:
            raise APIError(
                422,
                "insufficient_taste_ratings",
                "Rate at least "
                f"{taste_service.min_ratings} movies to get recommendations.",
            )
        return service.recommend_from_ratings(
            selection.ratings,
            selection.watched_movie_ids,
            limit,
            selection.summary,
        )

    if selection.ratings_internal:
        try:
            rating_service.set_ratings(
                db,
                record.user,
                {
                    internal_movie_id: int(rating)
                    for internal_movie_id, rating
                    in selection.ratings_internal.items()
                },
            )
        except APIError:
            raise
        except Exception as error:
            logger.exception(
                "Taste Profile ratings could not be saved."
            )
            raise APIError(
                500,
                "profile_persistence_failed",
                "Your Taste Profile ratings could not be saved.",
            ) from error

    merged = profile_service.load_profile(db, record.user.id)
    if len(merged.ratings) < taste_service.min_ratings:
        raise APIError(
            422,
            "insufficient_taste_ratings",
            "Rate at least "
            f"{taste_service.min_ratings} movies to get recommendations.",
        )
    watched = set(merged.watched_movie_ids)
    watched.update(
        swipe_service.excluded_movie_ids(db, record.user.id)
    )
    merged_summary = {
        "ratings_found": merged.ratings_count,
        "ratings_matched": merged.ratings_count,
        "watched_movies_matched": len(watched),
        "match_rate": (
            1.0 if merged.ratings_count else 0.0
        ),
    }
    result = service.recommend_for_saved_profile(
        merged.ratings,
        watched,
        limit,
        merged_summary,
    )
    try:
        batch_service.store_ready_batch(record.user.id, result)
    except Exception:
        logger.exception(
            "Could not store recommendations after Taste Profile."
        )
    return result
