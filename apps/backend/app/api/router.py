from __future__ import annotations

from fastapi import APIRouter

from app.api.routes import (
    admin,
    auth,
    films,
    health,
    me,
    recommendations,
    swipes,
    taste_profile,
)


api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(admin.router)
api_router.include_router(films.router)
api_router.include_router(me.router)
api_router.include_router(auth.router)
api_router.include_router(swipes.router)
api_router.include_router(recommendations.router)
api_router.include_router(taste_profile.router)
