from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse


router = APIRouter(tags=["health"])


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/api/v1/ready")
def ready(request: Request):
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "message": "Recommendation model artifacts are unavailable.",
            },
        )
    return {
        "status": "ready",
        "model": engine.store.m5_config.get("model", "M5"),
        "recommendation_algorithm": engine.store.recommendation_config.get(
            "algorithm", "knn_m5"
        ),
    }
