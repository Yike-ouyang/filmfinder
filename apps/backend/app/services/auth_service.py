from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, joinedload

from app.core.config import Settings
from app.core.exceptions import APIError
from app.core.security import (
    generate_opaque_token,
    hash_client_ip,
    hash_password,
    hash_token,
    normalize_email,
    verify_dummy_password,
    verify_password,
)
from app.db.models import EmailToken, User, UserSession, utcnow
from app.services.email_service import (
    EmailDeliveryError,
    EmailSender,
    password_reset_email,
    send_message,
    verification_email,
)


logger = logging.getLogger(__name__)

GENERIC_SIGNUP_MESSAGE = (
    "If this email can be used, a verification link has been sent."
)
ACCOUNT_READY_MESSAGE = "Your account is ready. You can now log in."
GENERIC_EMAIL_MESSAGE = (
    "If an account exists for this email, a message has been sent."
)


@dataclass
class LoginResult:
    user: User
    session_token: str
    csrf_token: str
    expires_at: datetime


class AuthService:
    def __init__(
        self,
        db: Session,
        settings: Settings,
        email_sender: EmailSender,
    ):
        self.db = db
        self.settings = settings
        self.email_sender = email_sender

    def signup(self, email: str, password: str) -> Optional[User]:
        self._validate_password(password)
        normalized = normalize_email(email)
        existing = self.db.scalar(
            select(User).where(User.email_normalized == normalized)
        )
        if existing is not None:
            return None
        user = User(
            email=str(email).strip(),
            email_normalized=normalized,
            password_hash=hash_password(password, self.settings),
            email_verified_at=(
                None
                if self.settings.require_email_verification
                else utcnow()
            ),
        )
        self.db.add(user)
        try:
            self.db.flush()
        except IntegrityError:
            self.db.rollback()
            return None
        if self.settings.require_email_verification:
            token = self._issue_email_token(
                user,
                "verify_email",
                timedelta(hours=self.settings.email_verification_ttl_hours),
            )
            self.db.commit()
            self._send_verification(user, token)
        else:
            self.db.commit()
        return user

    def verify_email(self, token: str) -> User:
        record = self._find_token(token, "verify_email")
        now = utcnow()
        if record.used_at is not None:
            raise APIError(
                400,
                "invalid_token",
                "This verification link is invalid or has expired.",
            )
        if self._is_expired(record.expires_at, now):
            raise APIError(
                400,
                "invalid_token",
                "This verification link is invalid or has expired.",
            )
        record.used_at = now
        if record.user.email_verified_at is None:
            record.user.email_verified_at = now
        self.db.commit()
        return record.user

    def resend_verification(self, email: str) -> None:
        if not self.settings.require_email_verification:
            return
        normalized = normalize_email(email)
        user = self.db.scalar(
            select(User).where(User.email_normalized == normalized)
        )
        if user is None or user.email_verified_at is not None:
            return
        token = self._issue_email_token(
            user,
            "verify_email",
            timedelta(hours=self.settings.email_verification_ttl_hours),
        )
        self.db.commit()
        self._send_verification(user, token)

    def login(
        self,
        email: str,
        password: str,
        *,
        user_agent: Optional[str] = None,
        client_ip: Optional[str] = None,
    ) -> LoginResult:
        normalized = normalize_email(email)
        user = self.db.scalar(
            select(User).where(User.email_normalized == normalized)
        )
        if user is None:
            verify_dummy_password(password, self.settings)
            raise APIError(
                401,
                "invalid_credentials",
                "Invalid email or password.",
            )
        if not verify_password(password, user.password_hash, self.settings):
            raise APIError(
                401,
                "invalid_credentials",
                "Invalid email or password.",
            )
        now = utcnow()
        if (
            self.settings.require_email_verification
            and user.email_verified_at is None
        ):
            raise APIError(
                403,
                "email_not_verified",
                "Verify your email address before signing in.",
            )
        if (
            not self.settings.require_email_verification
            and user.email_verified_at is None
        ):
            user.email_verified_at = now
        session_token = generate_opaque_token()
        csrf_token = generate_opaque_token()
        expires_at = now + timedelta(days=self.settings.session_ttl_days)
        record = UserSession(
            user_id=user.id,
            token_hash=hash_token(session_token),
            csrf_token_hash=hash_token(csrf_token),
            user_agent=(user_agent or "")[:255] or None,
            ip_hash=hash_client_ip(client_ip),
            created_at=now,
            last_used_at=now,
            expires_at=expires_at,
        )
        self.db.add(record)
        self.db.commit()
        return LoginResult(
            user=user,
            session_token=session_token,
            csrf_token=csrf_token,
            expires_at=expires_at,
        )

    def logout(self, record: UserSession) -> None:
        record.revoked_at = utcnow()
        self.db.commit()

    def request_password_reset(self, email: str) -> None:
        normalized = normalize_email(email)
        user = self.db.scalar(
            select(User).where(User.email_normalized == normalized)
        )
        if user is None:
            return
        if (
            self.settings.require_email_verification
            and user.email_verified_at is None
        ):
            token = self._issue_email_token(
                user,
                "verify_email",
                timedelta(hours=self.settings.email_verification_ttl_hours),
            )
            self.db.commit()
            self._send_verification(user, token)
            return
        token = self._issue_email_token(
            user,
            "reset_password",
            timedelta(minutes=self.settings.password_reset_ttl_minutes),
        )
        self.db.commit()
        try:
            send_message(
                self.email_sender,
                password_reset_email(self.settings, user.email, token),
            )
        except EmailDeliveryError:
            logger.exception("Password reset email could not be sent.")

    def reset_password(self, token: str, password: str) -> User:
        self._validate_password(password)
        record = self._find_token(token, "reset_password")
        now = utcnow()
        if record.used_at is not None or self._is_expired(
            record.expires_at, now
        ):
            raise APIError(
                400,
                "invalid_token",
                "This password reset link is invalid or has expired.",
            )
        record.used_at = now
        record.user.password_hash = hash_password(password, self.settings)
        self.db.execute(
            update(UserSession)
            .where(UserSession.user_id == record.user_id)
            .values(revoked_at=now)
        )
        self.db.commit()
        return record.user

    def delete_account(self, user: User, password: str) -> None:
        if not verify_password(password, user.password_hash, self.settings):
            raise APIError(
                401,
                "invalid_password",
                "The password is incorrect.",
            )
        self.db.delete(user)
        self.db.commit()

    def _validate_password(self, password: str) -> None:
        if len(password) < self.settings.password_min_length:
            raise APIError(
                422,
                "weak_password",
                "The password is too short.",
            )
        if len(password) > self.settings.password_max_length:
            raise APIError(
                422,
                "weak_password",
                "The password is too long.",
            )

    def _issue_email_token(
        self,
        user: User,
        purpose: str,
        lifetime: timedelta,
    ) -> str:
        now = utcnow()
        self.db.execute(
            update(EmailToken)
            .where(
                EmailToken.user_id == user.id,
                EmailToken.purpose == purpose,
                EmailToken.used_at.is_(None),
            )
            .values(used_at=now)
        )
        token = generate_opaque_token()
        self.db.add(
            EmailToken(
                user_id=user.id,
                purpose=purpose,
                token_hash=hash_token(token),
                created_at=now,
                expires_at=now + lifetime,
            )
        )
        return token

    def _find_token(self, token: str, purpose: str) -> EmailToken:
        record = self.db.scalar(
            select(EmailToken)
            .options(joinedload(EmailToken.user))
            .where(
                EmailToken.token_hash == hash_token(token),
                EmailToken.purpose == purpose,
            )
        )
        if record is None:
            raise APIError(
                400,
                "invalid_token",
                "This link is invalid or has expired.",
            )
        return record

    def _send_verification(self, user: User, token: str) -> None:
        try:
            send_message(
                self.email_sender,
                verification_email(self.settings, user.email, token),
            )
        except EmailDeliveryError:
            logger.exception("Verification email could not be sent.")

    @staticmethod
    def _is_expired(expires_at: datetime, now: datetime) -> bool:
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=now.tzinfo)
        return expires_at <= now
