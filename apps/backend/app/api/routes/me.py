from __future__ import annotations

from fastapi import APIRouter, Depends, Path, Query
from sqlalchemy.orm import Session

from app.db.models import UserSession
from app.db.session import get_db
from app.dependencies.auth import get_current_session, require_session_csrf
from app.dependencies.recommendation_engine import get_rating_service
from app.schemas.films import (
    ProfileSummaryResponse,
    RatingRequest,
    RatingResponse,
    UserRatingsResponse,
)
from app.services import film_catalog_service


router = APIRouter(prefix="/api/v1/me", tags=["me"])


@router.get("/ratings", response_model=UserRatingsResponse)
def list_ratings(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    record: UserSession = Depends(get_current_session),
    db: Session = Depends(get_db),
):
    return film_catalog_service.list_user_ratings(
        db,
        record.user_id,
        limit=limit,
        offset=offset,
    )


@router.put(
    "/ratings/{film_id}",
    response_model=RatingResponse,
    dependencies=[Depends(require_session_csrf)],
)
def set_rating(
    payload: RatingRequest,
    film_id: int = Path(..., ge=0),
    record: UserSession = Depends(get_current_session),
    db: Session = Depends(get_db),
    service=Depends(get_rating_service),
):
    return service.set_rating(
        db,
        record.user,
        film_id,
        payload.rating,
    )


@router.delete(
    "/ratings/{film_id}",
    response_model=RatingResponse,
    dependencies=[Depends(require_session_csrf)],
)
def delete_rating(
    film_id: int = Path(..., ge=0),
    record: UserSession = Depends(get_current_session),
    db: Session = Depends(get_db),
    service=Depends(get_rating_service),
):
    return service.delete_rating(db, record.user, film_id)


@router.get("/profile-summary", response_model=ProfileSummaryResponse)
def get_profile_summary(
    record: UserSession = Depends(get_current_session),
    db: Session = Depends(get_db),
):
    return film_catalog_service.profile_summary(db, record.user_id)
