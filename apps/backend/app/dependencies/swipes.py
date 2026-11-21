from __future__ import annotations

from fastapi import Request

from app.core.exceptions import APIError


def get_swipe_service(request: Request):
    service = getattr(request.app.state, "swipe_service", None)
    if service is None:
        raise APIError(
            503,
            "engine_unavailable",
            "The recommendation engine is not ready.",
        )
    return service
