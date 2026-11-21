from __future__ import annotations

from starlette.responses import Response

from app.core.config import Settings


def set_csrf_cookie(
    response: Response,
    settings: Settings,
    csrf_token: str,
    max_age: int,
) -> None:
    response.set_cookie(
        key=settings.csrf_cookie_name,
        value=csrf_token,
        max_age=max_age,
        httponly=False,
        secure=settings.cookie_secure,
        samesite="lax",
        path="/",
    )


def set_session_cookies(
    response: Response,
    settings: Settings,
    session_token: str,
    csrf_token: str,
    max_age: int,
) -> None:
    response.set_cookie(
        key=settings.session_cookie_name,
        value=session_token,
        max_age=max_age,
        httponly=True,
        secure=settings.cookie_secure,
        samesite="lax",
        path="/",
    )
    set_csrf_cookie(response, settings, csrf_token, max_age)


def clear_session_cookies(response: Response, settings: Settings) -> None:
    for key in (settings.session_cookie_name, settings.csrf_cookie_name):
        response.delete_cookie(
            key=key,
            path="/",
            secure=settings.cookie_secure,
            samesite="lax",
        )
