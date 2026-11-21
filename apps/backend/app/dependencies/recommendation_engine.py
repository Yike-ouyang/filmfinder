from __future__ import annotations

from fastapi import Request

from app.core.exceptions import APIError


def get_engine(request: Request):
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        raise APIError(
            503,
            "engine_unavailable",
            "The recommendation engine is not ready.",
        )
    return engine


def get_recommendation_service(request: Request):
    service = getattr(request.app.state, "recommendation_service", None)
    if service is None:
        raise APIError(
            503,
            "engine_unavailable",
            "The recommendation engine is not ready.",
        )
    return service


def get_profile_service(request: Request):
    service = getattr(request.app.state, "profile_service", None)
    if service is None:
        raise APIError(
            503,
            "engine_unavailable",
            "The recommendation engine is not ready.",
        )
    return service
