from __future__ import annotations

import hashlib
import hmac
import secrets
from functools import lru_cache
from typing import Optional

from argon2 import PasswordHasher
from argon2.exceptions import (
    InvalidHashError,
    VerificationError,
    VerifyMismatchError,
)

from app.core.config import Settings, get_settings


def normalize_email(email: str) -> str:
    return str(email).strip().casefold()


def hash_password(
    password: str, settings: Optional[Settings] = None
) -> str:
    return _get_password_hasher(settings).hash(password)


def verify_password(
    password: str,
    password_hash: str,
    settings: Optional[Settings] = None,
) -> bool:
    try:
        return _get_password_hasher(settings).verify(password_hash, password)
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False


def verify_dummy_password(
    password: str, settings: Optional[Settings] = None
) -> None:
    verify_password(password, _get_dummy_hash(settings), settings)


def generate_opaque_token() -> str:
    return secrets.token_urlsafe(32)


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def constant_time_equal(left: str, right: str) -> bool:
    return hmac.compare_digest(left.encode("utf-8"), right.encode("utf-8"))


def hash_client_ip(client_ip: Optional[str]) -> Optional[str]:
    if not client_ip:
        return None
    return hashlib.sha256(client_ip.encode("utf-8")).hexdigest()


def _get_password_hasher(
    settings: Optional[Settings] = None,
) -> PasswordHasher:
    resolved = settings or get_settings()
    return PasswordHasher(
        time_cost=resolved.argon2_time_cost,
        memory_cost=resolved.argon2_memory_cost,
        parallelism=resolved.argon2_parallelism,
    )


@lru_cache(maxsize=1)
def _dummy_hash_value() -> str:
    return _get_password_hasher().hash(
        "filmfinder-dummy-password-for-timing-only"
    )


def _get_dummy_hash(settings: Optional[Settings] = None) -> str:
    if settings is None:
        return _dummy_hash_value()
    return _get_password_hasher(settings).hash(
        "filmfinder-dummy-password-for-timing-only"
    )
