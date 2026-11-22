from __future__ import annotations

from fastapi import APIRouter

from app.api.routes import (
    auth,
    health,
    recommendations,
    swipes,
    taste_profile,
)


api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(auth.router)
api_router.include_router(swipes.router)
api_router.include_router(recommendations.router)
api_router.include_router(taste_profile.router)
