from __future__ import annotations

import logging
from typing import Optional

from fastapi import (
    APIRouter,
    Depends,
    File,
    Query,
    Request,
    Response,
    UploadFile,
)
from sqlalchemy.orm import Session

from app.core.exceptions import APIError
from app.db.models import UserSession
from app.db.session import get_db
from app.dependencies.auth import (
    get_current_session,
    get_optional_session,
    require_optional_csrf,
    require_session_csrf,
)
from app.dependencies.recommendation_engine import (
    get_batch_service,
    get_profile_service,
    get_recommendation_service,
)
from app.dependencies.swipes import get_swipe_service
from app.schemas.recommendations import (
    CurrentRecommendationsResponse,
    RecommendationResponse,
    RefreshResponse,
)


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/recommendations",
    tags=["recommendations"],
)


@router.get("/current", response_model=CurrentRecommendationsResponse)
def current_recommendations(
    response: Response,
    record: UserSession = Depends(get_current_session),
    batch_service=Depends(get_batch_service),
):
    result = batch_service.current(record.user_id)
    if result["status"] == "empty":
        raise APIError(
            404,
            "no_recommendations",
            "No recommendations are available for this account yet.",
        )
    if result["status"] == "pending":
        response.status_code = 202
    return CurrentRecommendationsResponse(**result)


@router.post(
    "/refresh",
    response_model=RefreshResponse,
    dependencies=[Depends(require_session_csrf)],
)
def refresh_recommendations(
    response: Response,
    record: UserSession = Depends(get_current_session),
    batch_service=Depends(get_batch_service),
):
    status = batch_service.request_generation(
        record.user.id, reason="refresh"
    )
    if status == "no_profile":
        raise APIError(
            422,
            "no_saved_profile",
            "Save some film ratings before requesting recommendations.",
        )
    if status == "no_new_candidates":
        return RefreshResponse(status="no_new_candidates")
    if status == "unavailable":
        raise APIError(
            503,
            "engine_unavailable",
            "The recommendation engine is not ready.",
        )
    response.status_code = 202
    return RefreshResponse(status="pending")


@router.post(
    "/from-saved-ratings",
    response_model=RefreshResponse,
    dependencies=[Depends(require_session_csrf)],
)
def recommend_from_saved_ratings(
    response: Response,
    record: UserSession = Depends(get_current_session),
    batch_service=Depends(get_batch_service),
):
    status = batch_service.request_generation(
        record.user.id,
        reason="manual",
        min_ratings=10,
    )
    if status == "no_profile":
        raise APIError(
            422,
            "no_saved_profile",
            "Rate at least 10 films before requesting recommendations.",
        )
    if status == "insufficient_ratings":
        raise APIError(
            422,
            "insufficient_ratings",
            "Rate at least 10 films before requesting recommendations.",
        )
    if status == "unavailable":
        raise APIError(
            503,
            "engine_unavailable",
            "The recommendation engine is not ready.",
        )
    response.status_code = 202
    return RefreshResponse(status="pending")


@router.post("/from-letterboxd", response_model=RecommendationResponse)
def recommend_from_letterboxd(
    request: Request,
    file: UploadFile = File(...),
    limit: int = Query(default=100, ge=1, le=100),
    record: Optional[UserSession] = Depends(get_optional_session),
    db: Session = Depends(get_db),
    service=Depends(get_recommendation_service),
    profile_service=Depends(get_profile_service),
    batch_service=Depends(get_batch_service),
    swipe_service=Depends(get_swipe_service),
    _: None = Depends(require_optional_csrf),
):
    max_bytes = service.settings.max_upload_bytes
    try:
        data = file.file.read(max_bytes + 1)
    finally:
        file.file.close()
    if len(data) > max_bytes:
        raise APIError(
            413,
            "upload_too_large",
            "The uploaded archive exceeds the size limit.",
        )
    result = service.recommend_from_bytes(
        data,
        limit,
        db=db,
        user=record.user if record is not None else None,
        profile_service=profile_service,
        excluded_movie_ids=(
            swipe_service.excluded_movie_ids(db, record.user.id)
            if record is not None
            else None
        ),
    )
    if record is not None:
        try:
            batch_service.store_ready_batch(record.user.id, result)
        except Exception:
            logger.exception(
                "Could not store recommendations after import."
            )
    analytics = getattr(request.app.state, "analytics_service", None)
    if analytics is not None:
        try:
            analytics.record_letterboxd_import(
                db,
                record.user.id if record is not None else None,
            )
        except Exception:
            db.rollback()
            logger.exception(
                "Could not record the Letterboxd import event."
            )
    return result
