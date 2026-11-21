from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, File, Query, UploadFile
from sqlalchemy.orm import Session

from app.core.exceptions import APIError
from app.db.models import UserSession
from app.db.session import get_db
from app.dependencies.auth import (
    get_optional_session,
    require_optional_csrf,
)
from app.dependencies.recommendation_engine import (
    get_profile_service,
    get_recommendation_service,
)
from app.schemas.recommendations import RecommendationResponse


router = APIRouter(
    prefix="/api/v1/recommendations",
    tags=["recommendations"],
)


@router.post("/from-letterboxd", response_model=RecommendationResponse)
def recommend_from_letterboxd(
    file: UploadFile = File(...),
    limit: int = Query(default=100, ge=1, le=100),
    record: Optional[UserSession] = Depends(get_optional_session),
    db: Session = Depends(get_db),
    service=Depends(get_recommendation_service),
    profile_service=Depends(get_profile_service),
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
    )
    return result
