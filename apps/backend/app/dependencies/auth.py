from __future__ import annotations

from typing import Optional

from fastapi import Depends, Request
from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from app.core.config import Settings
from app.core.exceptions import APIError
from app.core.security import constant_time_equal, hash_token
from app.db.models import User, UserSession, utcnow
from app.db.session import get_db


def get_app_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_client_ip(request: Request) -> Optional[str]:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        candidate = forwarded.split(",", 1)[0].strip()
        if candidate:
            return candidate
    if request.client is not None:
        return request.client.host
    return None


def get_current_session(
    request: Request,
    settings: Settings = Depends(get_app_settings),
    db: Session = Depends(get_db),
) -> UserSession:
    record = _lookup_session(request, settings, db)
    if record is None:
        raise APIError(
            401,
            "not_authenticated",
            "Authentication is required.",
        )
    return record


def get_optional_session(
    request: Request,
    settings: Settings = Depends(get_app_settings),
    db: Session = Depends(get_db),
) -> Optional[UserSession]:
    return _lookup_session(request, settings, db)


def get_current_user(
    record: UserSession = Depends(get_current_session),
) -> User:
    return record.user


def get_current_admin(
    record: UserSession = Depends(get_current_session),
) -> User:
    if not bool(record.user.is_admin):
        raise APIError(
            403,
            "admin_required",
            "Administrator access is required.",
        )
    return record.user


def require_public_csrf(
    request: Request,
    settings: Settings = Depends(get_app_settings),
) -> None:
    _check_csrf(request, settings, None)


def require_session_csrf(
    request: Request,
    settings: Settings = Depends(get_app_settings),
    record: UserSession = Depends(get_current_session),
) -> None:
    _check_csrf(request, settings, record)


def require_optional_csrf(
    request: Request,
    settings: Settings = Depends(get_app_settings),
    record: Optional[UserSession] = Depends(get_optional_session),
) -> None:
    if record is not None:
        _check_csrf(request, settings, record)


def _lookup_session(
    request: Request,
    settings: Settings,
    db: Session,
) -> Optional[UserSession]:
    token = request.cookies.get(settings.session_cookie_name)
    if not token:
        return None
    record = db.scalar(
        select(UserSession)
        .options(joinedload(UserSession.user))
        .where(
            UserSession.token_hash == hash_token(token),
            UserSession.revoked_at.is_(None),
        )
    )
    if record is None:
        return None
    if _expired(record, utcnow()):
        return None
    return record


def _check_csrf(
    request: Request,
    settings: Settings,
    record: Optional[UserSession],
) -> None:
    origin = request.headers.get("origin")
    if origin and origin.rstrip("/") not in {
        allowed.rstrip("/") for allowed in settings.cors_origin_list
    }:
        raise APIError(
            403,
            "csrf_failed",
            "The request origin is not allowed.",
        )
    header_token = request.headers.get("x-csrf-token")
    cookie_token = request.cookies.get(settings.csrf_cookie_name)
    if not header_token or not cookie_token:
        raise APIError(
            403,
            "csrf_failed",
            "Missing CSRF token.",
        )
    if not constant_time_equal(header_token, cookie_token):
        raise APIError(
            403,
            "csrf_failed",
            "Invalid CSRF token.",
        )
    if record is not None and not constant_time_equal(
        hash_token(header_token),
        record.csrf_token_hash,
    ):
        raise APIError(
            403,
            "csrf_failed",
            "Invalid CSRF token.",
        )


def _expired(record: UserSession, now) -> bool:
    expires_at = record.expires_at
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=now.tzinfo)
    return expires_at <= now
