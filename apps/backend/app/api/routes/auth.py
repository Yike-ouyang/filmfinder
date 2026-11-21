from __future__ import annotations

from datetime import timedelta
from typing import Optional

from fastapi import APIRouter, Depends, Request, Response
from sqlalchemy.orm import Session

from app.core.cookies import (
    clear_session_cookies,
    set_csrf_cookie,
    set_session_cookies,
)
from app.core.exceptions import APIError
from app.core.security import (
    generate_opaque_token,
    hash_token,
    normalize_email,
)
from app.db.models import User, UserSession, utcnow
from app.db.session import get_db
from app.dependencies.auth import (
    get_client_ip,
    get_current_session,
    get_optional_session,
    require_public_csrf,
    require_session_csrf,
)
from app.schemas.auth import (
    CsrfResponse,
    DeleteAccountRequest,
    EmailRequest,
    LoginRequest,
    MessageResponse,
    ResetPasswordRequest,
    SignupRequest,
    SignupResponse,
    TokenRequest,
    UserResponse,
)
from app.services.auth_service import (
    ACCOUNT_READY_MESSAGE,
    GENERIC_EMAIL_MESSAGE,
    GENERIC_SIGNUP_MESSAGE,
    AuthService,
)


router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


def _service(request: Request, db: Session) -> AuthService:
    return AuthService(
        db,
        request.app.state.settings,
        request.app.state.email_sender,
    )


def _user_response(user: User) -> UserResponse:
    return UserResponse(
        id=user.id,
        email=user.email,
        email_verified=user.email_verified_at is not None,
        created_at=user.created_at,
    )


def _rate_key(*parts: Optional[str]) -> str:
    return ":".join(part or "-" for part in parts)


def _check_limit(
    request: Request,
    key: str,
    limit: int,
    window_seconds: int,
) -> None:
    limiter = request.app.state.rate_limiter
    if limiter.is_limited(key, limit, window_seconds):
        raise APIError(
            429,
            "rate_limited",
            "Too many attempts. Try again later.",
        )


def _record_limit(request: Request, key: str, window_seconds: int) -> None:
    request.app.state.rate_limiter.record(key, window_seconds)


@router.get("/csrf", response_model=CsrfResponse)
def issue_csrf_token(
    request: Request,
    response: Response,
    record: Optional[UserSession] = Depends(get_optional_session),
    db: Session = Depends(get_db),
):
    settings = request.app.state.settings
    token = generate_opaque_token()
    if record is not None:
        record.csrf_token_hash = hash_token(token)
        db.commit()
        now = utcnow()
        expires_at = record.expires_at
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=now.tzinfo)
        max_age = max(0, int((expires_at - now).total_seconds()))
    else:
        max_age = int(timedelta(days=settings.session_ttl_days).total_seconds())
    set_csrf_cookie(response, settings, token, max_age)
    return CsrfResponse(csrf_token=token)


@router.post(
    "/signup",
    response_model=SignupResponse,
    status_code=201,
    dependencies=[Depends(require_public_csrf)],
)
def signup(
    payload: SignupRequest,
    request: Request,
    db: Session = Depends(get_db),
):
    settings = request.app.state.settings
    ip = get_client_ip(request)
    key = _rate_key("signup", ip)
    _check_limit(
        request,
        key,
        settings.signup_rate_limit_attempts,
        settings.signup_rate_limit_window_seconds,
    )
    _record_limit(
        request,
        key,
        settings.signup_rate_limit_window_seconds,
    )
    _service(request, db).signup(str(payload.email), payload.password)
    message = (
        GENERIC_SIGNUP_MESSAGE
        if settings.require_email_verification
        else ACCOUNT_READY_MESSAGE
    )
    return SignupResponse(
        message=message,
        verification_required=settings.require_email_verification,
    )


@router.post(
    "/verify-email",
    response_model=MessageResponse,
    dependencies=[Depends(require_public_csrf)],
)
def verify_email(
    payload: TokenRequest,
    request: Request,
    db: Session = Depends(get_db),
):
    _service(request, db).verify_email(payload.token)
    return MessageResponse(message="Your email address is verified.")


@router.post(
    "/resend-verification",
    response_model=MessageResponse,
    dependencies=[Depends(require_public_csrf)],
)
def resend_verification(
    payload: EmailRequest,
    request: Request,
    db: Session = Depends(get_db),
):
    settings = request.app.state.settings
    normalized = normalize_email(str(payload.email))
    key = _rate_key("resend", get_client_ip(request), normalized)
    _check_limit(
        request,
        key,
        settings.email_rate_limit_attempts,
        settings.email_rate_limit_window_seconds,
    )
    _record_limit(
        request,
        key,
        settings.email_rate_limit_window_seconds,
    )
    _service(request, db).resend_verification(str(payload.email))
    return MessageResponse(message=GENERIC_EMAIL_MESSAGE)


@router.post(
    "/login",
    response_model=UserResponse,
    dependencies=[Depends(require_public_csrf)],
)
def login(
    payload: LoginRequest,
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
):
    settings = request.app.state.settings
    normalized = normalize_email(str(payload.email))
    ip = get_client_ip(request)
    user_key = _rate_key("login-user", ip, normalized)
    ip_key = _rate_key("login-ip", ip)
    _check_limit(
        request,
        user_key,
        settings.login_rate_limit_attempts,
        settings.login_rate_limit_window_seconds,
    )
    _check_limit(
        request,
        ip_key,
        settings.login_rate_limit_attempts * 4,
        settings.login_rate_limit_window_seconds,
    )
    service = _service(request, db)
    try:
        result = service.login(
            str(payload.email),
            payload.password,
            user_agent=request.headers.get("user-agent"),
            client_ip=ip,
        )
    except APIError as error:
        if error.code == "invalid_credentials":
            _record_limit(
                request,
                user_key,
                settings.login_rate_limit_window_seconds,
            )
            _record_limit(
                request,
                ip_key,
                settings.login_rate_limit_window_seconds,
            )
        raise
    request.app.state.rate_limiter.reset(user_key)
    request.app.state.rate_limiter.reset(ip_key)
    max_age = int(timedelta(days=settings.session_ttl_days).total_seconds())
    set_session_cookies(
        response,
        settings,
        result.session_token,
        result.csrf_token,
        max_age,
    )
    return _user_response(result.user)


@router.post(
    "/logout",
    status_code=204,
    dependencies=[Depends(require_session_csrf)],
)
def logout(
    request: Request,
    record: UserSession = Depends(get_current_session),
    db: Session = Depends(get_db),
):
    _service(request, db).logout(record)
    response = Response(status_code=204)
    clear_session_cookies(response, request.app.state.settings)
    return response


@router.get("/me", response_model=UserResponse)
def me(record: UserSession = Depends(get_current_session)):
    return _user_response(record.user)


@router.post(
    "/forgot-password",
    response_model=MessageResponse,
    dependencies=[Depends(require_public_csrf)],
)
def forgot_password(
    payload: EmailRequest,
    request: Request,
    db: Session = Depends(get_db),
):
    settings = request.app.state.settings
    normalized = normalize_email(str(payload.email))
    key = _rate_key("forgot", get_client_ip(request), normalized)
    _check_limit(
        request,
        key,
        settings.email_rate_limit_attempts,
        settings.email_rate_limit_window_seconds,
    )
    _record_limit(
        request,
        key,
        settings.email_rate_limit_window_seconds,
    )
    _service(request, db).request_password_reset(str(payload.email))
    return MessageResponse(message=GENERIC_EMAIL_MESSAGE)


@router.post(
    "/reset-password",
    response_model=MessageResponse,
    dependencies=[Depends(require_public_csrf)],
)
def reset_password(
    payload: ResetPasswordRequest,
    request: Request,
    db: Session = Depends(get_db),
):
    _service(request, db).reset_password(payload.token, payload.password)
    return MessageResponse(message="Your password has been updated.")


@router.delete(
    "/account",
    status_code=204,
    dependencies=[Depends(require_session_csrf)],
)
def delete_account(
    payload: DeleteAccountRequest,
    request: Request,
    record: UserSession = Depends(get_current_session),
    db: Session = Depends(get_db),
):
    _service(request, db).delete_account(record.user, payload.password)
    response = Response(status_code=204)
    clear_session_cookies(response, request.app.state.settings)
    return response
