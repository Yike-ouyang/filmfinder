from __future__ import annotations

from fastapi import APIRouter

from app.api.routes import auth, films, health, me, recommendations, swipes


api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(films.router)
api_router.include_router(me.router)
api_router.include_router(auth.router)
api_router.include_router(swipes.router)
api_router.include_router(recommendations.router)
