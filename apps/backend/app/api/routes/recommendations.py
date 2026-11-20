from __future__ import annotations

from fastapi import APIRouter, Depends, File, Query, UploadFile

from app.core.exceptions import APIError
from app.dependencies.recommendation_engine import (
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
    service=Depends(get_recommendation_service),
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
    result = service.recommend_from_bytes(data, limit)
    return result
