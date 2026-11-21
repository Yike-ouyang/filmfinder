from __future__ import annotations

import re
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.core.config import Settings
from app.core.rate_limit import RateLimiter
from app.core.security import hash_password, hash_token
from app.db.base import build_session_factory, create_db_engine
from app.db.models import EmailToken, User, UserSession, utcnow
from app.db.session import get_db
from app.main import create_app
from app.services.email_service import EmailSender


BACKEND_ROOT = Path(__file__).resolve().parents[1]
PASSWORD = "correct horse battery"
CSRF_HEADER = "X-CSRF-Token"


@dataclass
class SentEmail:
    to: str
    subject: str
    text: str
    html: str


class RecordingEmailSender(EmailSender):
    def __init__(self):
        self.messages: list[SentEmail] = []

    def send(self, *, to, subject, text, html):
        self.messages.append(
            SentEmail(to=to, subject=subject, text=text, html=html)
        )


@dataclass
class AuthContext:
    client: TestClient
    app: object
    session_factory: object
    emails: RecordingEmailSender


def _alembic_config(database_url: str) -> Config:
    config = Config(str(BACKEND_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(BACKEND_ROOT / "migrations"))
    config.set_main_option("sqlalchemy.url", database_url)
    return config


@contextmanager
def _auth_context(tmp_path, require_email_verification: bool):
    db_path = tmp_path / "auth.db"
    database_url = f"sqlite+pysqlite:///{db_path}"
    command.upgrade(_alembic_config(database_url), "head")
    engine = create_db_engine(database_url)
    session_factory = build_session_factory(engine)
    settings = Settings(
        filmfinder_db_path=str(db_path),
        app_base_url="http://localhost:3000",
        email_backend="console",
        email_from="noreply@example.com",
        email_from_name="FilmFinder",
        cors_origins="http://testserver",
        cookie_secure=False,
        session_ttl_days=30,
        password_min_length=10,
        require_email_verification=require_email_verification,
        login_rate_limit_attempts=5,
        login_rate_limit_window_seconds=900,
        signup_rate_limit_attempts=100,
        signup_rate_limit_window_seconds=3600,
        email_rate_limit_attempts=100,
        email_rate_limit_window_seconds=900,
        argon2_memory_cost=19456,
        argon2_time_cost=2,
        argon2_parallelism=1,
    )
    app = create_app(settings, load_engine=False)

    def override_db():
        session = session_factory()
        try:
            yield session
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    app.dependency_overrides[get_db] = override_db
    emails = RecordingEmailSender()
    try:
        with TestClient(app) as client:
            app.state.email_sender = emails
            app.state.rate_limiter = RateLimiter()
            yield AuthContext(client, app, session_factory, emails)
    finally:
        engine.dispose()


@pytest.fixture()
def auth(tmp_path):
    with _auth_context(tmp_path, require_email_verification=True) as context:
        yield context


@pytest.fixture()
def auth_open(tmp_path):
    with _auth_context(tmp_path, require_email_verification=False) as context:
        yield context


def _csrf(client: TestClient) -> str:
    response = client.get("/api/v1/auth/csrf")
    assert response.status_code == 200
    return response.json()["csrf_token"]


def _post(client: TestClient, path: str, payload: dict, token: str | None = None):
    return client.post(
        path,
        json=payload,
        headers={CSRF_HEADER: token if token is not None else _csrf(client)},
    )


def _signup(client: TestClient, email: str = "user@example.com"):
    return _post(
        client,
        "/api/v1/auth/signup",
        {"email": email, "password": PASSWORD},
    )


def _extract_token(text: str) -> str:
    match = re.search(r"https?://\S+", text)
    assert match is not None, text
    query = parse_qs(urlparse(match.group(0)).query)
    return query["token"][0]


def _verify(client: TestClient, auth_context: AuthContext, index: int = -1):
    token = _extract_token(auth_context.emails.messages[index].text)
    return _post(client, "/api/v1/auth/verify-email", {"token": token})


def _verified_user(client: TestClient, auth_context: AuthContext):
    assert _signup(client).status_code == 201
    assert _verify(client, auth_context).status_code == 200


def _login(client: TestClient, email: str = "user@example.com"):
    return _post(
        client,
        "/api/v1/auth/login",
        {"email": email, "password": PASSWORD},
    )


def _logout(client: TestClient):
    return client.post(
        "/api/v1/auth/logout",
        headers={CSRF_HEADER: client.cookies.get("ff_csrf") or ""},
    )


def test_signup_creates_unverified_user_and_sends_verification(auth):
    response = _signup(auth.client)

    assert response.status_code == 201
    assert "verification link" in response.json()["message"]
    assert len(auth.emails.messages) == 1
    assert "verify-email" in auth.emails.messages[0].text

    with auth.session_factory() as session:
        user = session.scalar(select(User))
        assert user is not None
        assert user.email_verified_at is None
        assert user.password_hash.startswith("$argon2id$")
        assert PASSWORD not in user.password_hash


def test_signup_duplicate_returns_generic_message(auth):
    first = _signup(auth.client)
    second = _signup(auth.client)

    assert first.status_code == 201
    assert second.status_code == 201
    assert first.json() == second.json()
    assert len(auth.emails.messages) == 1

    with auth.session_factory() as session:
        assert len(session.scalars(select(User)).all()) == 1


def test_signup_rejects_invalid_email_and_weak_password(auth):
    invalid_email = _post(
        auth.client,
        "/api/v1/auth/signup",
        {"email": "not-an-email", "password": PASSWORD},
    )
    weak_password = _post(
        auth.client,
        "/api/v1/auth/signup",
        {"email": "weak@example.com", "password": "short"},
    )

    assert invalid_email.status_code == 422
    assert weak_password.status_code == 422
    assert weak_password.json()["error"]["code"] == "weak_password"


def test_verify_email_is_single_use(auth):
    first = _signup(auth.client)
    assert first.status_code == 201
    token = _extract_token(auth.emails.messages[0].text)

    assert _post(
        auth.client, "/api/v1/auth/verify-email", {"token": token}
    ).status_code == 200
    assert _post(
        auth.client, "/api/v1/auth/verify-email", {"token": token}
    ).status_code == 400

    with auth.session_factory() as session:
        user = session.scalar(select(User))
        assert user.email_verified_at is not None


def test_verify_email_rejects_expired_and_unknown_tokens(auth):
    with auth.session_factory() as session:
        user = User(
            email="expired@example.com",
            email_normalized="expired@example.com",
            password_hash="unused",
        )
        session.add(user)
        session.flush()
        session.add(
            EmailToken(
                user_id=user.id,
                purpose="verify_email",
                token_hash=hash_token("expired-token-value-1234"),
                expires_at=utcnow() - timedelta(minutes=1),
            )
        )
        session.commit()

    expired = _post(
        auth.client,
        "/api/v1/auth/verify-email",
        {"token": "expired-token-value-1234"},
    )
    unknown = _post(
        auth.client,
        "/api/v1/auth/verify-email",
        {"token": "unknown-token-value-1234"},
    )

    assert expired.status_code == 400
    assert unknown.status_code == 400


def test_login_requires_verified_email(auth):
    assert _signup(auth.client).status_code == 201

    response = _login(auth.client)

    assert response.status_code == 403
    assert response.json()["error"]["code"] == "email_not_verified"
    assert "ff_session" not in response.cookies


def test_login_after_verification_sets_session_and_me(auth):
    _verified_user(auth.client, auth)

    response = _login(auth.client)

    assert response.status_code == 200
    assert response.json()["email"] == "user@example.com"
    assert response.json()["email_verified"] is True
    assert response.cookies.get("ff_session")
    assert response.cookies.get("ff_csrf")

    set_cookie = response.headers.get_list("set-cookie")
    session_cookie = next(
        value for value in set_cookie if value.startswith("ff_session=")
    )
    csrf_cookie = next(
        value for value in set_cookie if value.startswith("ff_csrf=")
    )
    assert "HttpOnly" in session_cookie
    assert "SameSite=lax" in session_cookie
    assert "Path=/" in session_cookie
    assert "HttpOnly" not in csrf_cookie
    assert "SameSite=lax" in csrf_cookie

    with auth.session_factory() as session:
        record = session.scalar(select(UserSession))
        assert record is not None
        assert record.token_hash != response.cookies.get("ff_session")
        assert record.token_hash == hash_token(response.cookies.get("ff_session"))
        assert record.csrf_token_hash == hash_token(
            response.cookies.get("ff_csrf")
        )

    me = auth.client.get("/api/v1/auth/me")
    assert me.status_code == 200
    assert me.json()["email"] == "user@example.com"


def test_me_requires_session(auth):
    assert auth.client.get("/api/v1/auth/me").status_code == 401


def test_logout_revokes_session_and_clears_cookies(auth):
    _verified_user(auth.client, auth)
    assert _login(auth.client).status_code == 200

    response = _logout(auth.client)

    assert response.status_code == 204
    assert auth.client.get("/api/v1/auth/me").status_code == 401
    assert "ff_session=" in " ".join(response.headers.get_list("set-cookie"))
    with auth.session_factory() as session:
        record = session.scalar(select(UserSession))
        assert record.revoked_at is not None


def test_csrf_is_required_and_bound_to_session(auth):
    _verified_user(auth.client, auth)
    assert _login(auth.client).status_code == 200

    missing = auth.client.post("/api/v1/auth/logout")
    assert missing.status_code == 403
    assert missing.json()["error"]["code"] == "csrf_failed"

    mismatch = auth.client.post(
        "/api/v1/auth/logout",
        headers={CSRF_HEADER: "not-the-cookie-token"},
    )
    assert mismatch.status_code == 403

    bad_origin = auth.client.post(
        "/api/v1/auth/logout",
        headers={
            CSRF_HEADER: auth.client.cookies.get("ff_csrf") or "",
            "Origin": "https://evil.example",
        },
    )
    assert bad_origin.status_code == 403


def test_login_rate_limit(auth):
    _verified_user(auth.client, auth)

    for _ in range(5):
        response = _post(
            auth.client,
            "/api/v1/auth/login",
            {"email": "user@example.com", "password": "wrong password value"},
        )
        assert response.status_code == 401

    limited = _post(
        auth.client,
        "/api/v1/auth/login",
        {"email": "user@example.com", "password": "wrong password value"},
    )
    assert limited.status_code == 429
    assert limited.json()["error"]["code"] == "rate_limited"


def test_resend_verification_issues_a_new_valid_token(auth):
    assert _signup(auth.client).status_code == 201
    old_token = _extract_token(auth.emails.messages[0].text)

    response = _post(
        auth.client,
        "/api/v1/auth/resend-verification",
        {"email": "user@example.com"},
    )

    assert response.status_code == 200
    assert len(auth.emails.messages) == 2
    new_token = _extract_token(auth.emails.messages[1].text)
    assert new_token != old_token
    assert _post(
        auth.client, "/api/v1/auth/verify-email", {"token": old_token}
    ).status_code == 400
    assert _post(
        auth.client, "/api/v1/auth/verify-email", {"token": new_token}
    ).status_code == 200


def test_forgot_password_is_generic_and_reset_flow_works(auth):
    _verified_user(auth.client, auth)

    unknown = _post(
        auth.client,
        "/api/v1/auth/forgot-password",
        {"email": "unknown@example.com"},
    )
    assert unknown.status_code == 200
    assert len(auth.emails.messages) == 1

    forgot = _post(
        auth.client,
        "/api/v1/auth/forgot-password",
        {"email": "user@example.com"},
    )
    assert forgot.status_code == 200
    assert len(auth.emails.messages) == 2
    token = _extract_token(auth.emails.messages[1].text)

    reset = _post(
        auth.client,
        "/api/v1/auth/reset-password",
        {"token": token, "password": "new correct horse battery"},
    )
    assert reset.status_code == 200

    assert _login(auth.client).status_code == 401
    assert _post(
        auth.client,
        "/api/v1/auth/login",
        {"email": "user@example.com", "password": "new correct horse battery"},
    ).status_code == 200
    assert _post(
        auth.client,
        "/api/v1/auth/reset-password",
        {"token": token, "password": "another correct horse"},
    ).status_code == 400

    with auth.session_factory() as session:
        tokens = session.scalars(select(EmailToken)).all()
        assert all(token.used_at is not None for token in tokens)


def test_reset_password_revokes_existing_sessions(auth):
    _verified_user(auth.client, auth)
    assert _login(auth.client).status_code == 200
    assert auth.client.get("/api/v1/auth/me").status_code == 200

    assert _post(
        auth.client,
        "/api/v1/auth/forgot-password",
        {"email": "user@example.com"},
    ).status_code == 200
    token = _extract_token(auth.emails.messages[1].text)
    assert _post(
        auth.client,
        "/api/v1/auth/reset-password",
        {"token": token, "password": "new correct horse battery"},
    ).status_code == 200

    assert auth.client.get("/api/v1/auth/me").status_code == 401


def test_delete_account_requires_password_and_cascades(auth):
    _verified_user(auth.client, auth)
    assert _login(auth.client).status_code == 200

    wrong = auth.client.request(
        "DELETE",
        "/api/v1/auth/account",
        json={"password": "wrong password value"},
        headers={CSRF_HEADER: auth.client.cookies.get("ff_csrf") or ""},
    )
    assert wrong.status_code == 401

    response = auth.client.request(
        "DELETE",
        "/api/v1/auth/account",
        json={"password": PASSWORD},
        headers={CSRF_HEADER: auth.client.cookies.get("ff_csrf") or ""},
    )
    assert response.status_code == 204
    assert auth.client.get("/api/v1/auth/me").status_code == 401

    with auth.session_factory() as session:
        assert session.scalar(select(User)) is None
        assert session.scalar(select(UserSession)) is None
        assert session.scalar(select(EmailToken)) is None


def test_expired_session_is_rejected(auth):
    _verified_user(auth.client, auth)
    assert _login(auth.client).status_code == 200
    assert auth.client.cookies.get("ff_session")

    with auth.session_factory() as session:
        record = session.scalar(select(UserSession))
        record.expires_at = utcnow() - timedelta(seconds=1)
        session.commit()

    assert auth.client.get("/api/v1/auth/me").status_code == 401


def test_signup_without_verification_creates_ready_account(auth_open):
    response = _signup(auth_open.client)

    assert response.status_code == 201
    payload = response.json()
    assert payload["verification_required"] is False
    assert "log in" in payload["message"].lower()
    assert auth_open.emails.messages == []

    with auth_open.session_factory() as session:
        user = session.scalar(select(User))
        assert user is not None
        assert user.email_verified_at is not None
        assert session.scalar(select(EmailToken)) is None

    login = _login(auth_open.client)
    assert login.status_code == 200
    assert auth_open.client.get("/api/v1/auth/me").status_code == 200


def test_existing_unverified_user_is_verified_on_first_login(auth_open):
    settings = auth_open.app.state.settings
    with auth_open.session_factory() as session:
        session.add(
            User(
                email="legacy@example.com",
                email_normalized="legacy@example.com",
                password_hash=hash_password(PASSWORD, settings),
            )
        )
        session.commit()

    response = _post(
        auth_open.client,
        "/api/v1/auth/login",
        {"email": "legacy@example.com", "password": PASSWORD},
    )

    assert response.status_code == 200
    with auth_open.session_factory() as session:
        user = session.scalar(select(User))
        assert user.email_verified_at is not None


def test_resend_verification_is_noop_when_disabled(auth_open):
    assert _signup(auth_open.client).status_code == 201

    response = _post(
        auth_open.client,
        "/api/v1/auth/resend-verification",
        {"email": "user@example.com"},
    )

    assert response.status_code == 200
    assert auth_open.emails.messages == []


def test_password_reset_works_without_prior_verification(auth_open):
    assert _signup(auth_open.client).status_code == 201

    response = _post(
        auth_open.client,
        "/api/v1/auth/forgot-password",
        {"email": "user@example.com"},
    )

    assert response.status_code == 200
    assert len(auth_open.emails.messages) == 1
    assert "reset-password" in auth_open.emails.messages[0].text
    token = _extract_token(auth_open.emails.messages[0].text)
    assert _post(
        auth_open.client,
        "/api/v1/auth/reset-password",
        {"token": token, "password": "new correct horse battery"},
    ).status_code == 200
    assert _post(
        auth_open.client,
        "/api/v1/auth/login",
        {"email": "user@example.com", "password": "new correct horse battery"},
    ).status_code == 200


class FakeBatchService:
    def __init__(self):
        self.generation_calls = []
        self.current_payload = {
            "status": "ready",
            "batch_id": 7,
            "generated_at": "2026-11-21T10:00:00",
            "response": {
                "profile": {
                    "ratings_found": 1,
                    "ratings_matched": 1,
                    "watched_movies_matched": 1,
                    "match_rate": 1.0,
                },
                "recommendation_info": {
                    "model": "M5",
                    "algorithm": "knn_m5",
                    "requested_count": 100,
                    "returned_count": 1,
                },
                "recommendations": [
                    {
                        "rank": 1,
                        "movie_id": "movie-1",
                        "internal_movie_id": 1,
                        "title": "Film One",
                        "year": 2000,
                        "genres": ["Drama"],
                        "score": 9.0,
                        "score_type": "m5_predicted_rating",
                        "tmdb_id": 10,
                        "poster_path": "/poster.jpg",
                        "backdrop_path": None,
                    }
                ],
            },
        }
        self.refresh_status = "queued"

    def request_generation(self, user_id, *, reason="login"):
        self.generation_calls.append((user_id, reason))
        return self.refresh_status

    def current(self, user_id):
        return self.current_payload


def test_login_schedules_recommendation_generation(auth):
    fake = FakeBatchService()
    auth.app.state.batch_service = fake
    _verified_user(auth.client, auth)

    response = _login(auth.client)

    assert response.status_code == 200
    assert len(fake.generation_calls) == 1
    assert fake.generation_calls[0][1] == "login"


def test_current_recommendations_endpoint(auth):
    fake = FakeBatchService()
    auth.app.state.batch_service = fake
    _verified_user(auth.client, auth)
    assert _login(auth.client).status_code == 200

    response = auth.client.get("/api/v1/recommendations/current")

    assert response.status_code == 200
    assert response.json()["batch_id"] == 7
    assert response.json()["response"]["recommendations"][0]["title"] == "Film One"


def test_current_recommendations_pending_returns_202(auth):
    fake = FakeBatchService()
    fake.current_payload = {
        "status": "pending",
        "batch_id": 8,
        "generated_at": None,
        "response": None,
    }
    auth.app.state.batch_service = fake
    _verified_user(auth.client, auth)
    assert _login(auth.client).status_code == 200

    response = auth.client.get("/api/v1/recommendations/current")

    assert response.status_code == 202
    assert response.json()["status"] == "pending"


def test_refresh_recommendations_endpoint(auth):
    fake = FakeBatchService()
    auth.app.state.batch_service = fake
    _verified_user(auth.client, auth)
    assert _login(auth.client).status_code == 200

    response = auth.client.post(
        "/api/v1/recommendations/refresh",
        headers={CSRF_HEADER: auth.client.cookies.get("ff_csrf") or ""},
    )

    assert response.status_code == 202
    assert response.json()["status"] == "pending"
    assert fake.generation_calls[-1][1] == "refresh"

    fake.refresh_status = "no_new_candidates"
    no_new = auth.client.post(
        "/api/v1/recommendations/refresh",
        headers={CSRF_HEADER: auth.client.cookies.get("ff_csrf") or ""},
    )
    assert no_new.status_code == 200
    assert no_new.json()["status"] == "no_new_candidates"
