from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import select

from app.core.security import hash_token
from app.db.base import build_session_factory, create_db_engine
from app.db.models import (
    AnalyticsEvent,
    RecommendationBatch,
    User,
    UserFilm,
    UserSession,
    UserSwipe,
)
from app.db.session import get_db
from app.services.analytics_service import AnalyticsService


BACKEND_ROOT = Path(__file__).resolve().parents[1]
NOW = datetime.now(timezone.utc)


def _alembic_config(database_url: str) -> Config:
    config = Config(str(BACKEND_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(BACKEND_ROOT / "migrations"))
    config.set_main_option("sqlalchemy.url", database_url)
    return config


def _create_user(
    session_factory,
    email: str,
    *,
    days_ago: int,
    verified: bool,
    is_admin: bool = False,
    login_days_ago: list[int] | None = None,
):
    created = NOW - timedelta(days=days_ago)
    sessions = []
    with session_factory() as session:
        user = User(
            email=email,
            email_normalized=email,
            password_hash="unused",
            email_verified_at=created if verified else None,
            is_admin=is_admin,
            created_at=created,
            updated_at=created,
        )
        session.add(user)
        session.flush()
        for index, login_days in enumerate(login_days_ago or []):
            token = f"session-{email}-{index}"
            csrf = f"csrf-{email}-{index}"
            sessions.append((token, csrf))
            session.add(
                UserSession(
                    user_id=user.id,
                    token_hash=hash_token(token),
                    csrf_token_hash=hash_token(csrf),
                    created_at=NOW - timedelta(days=login_days),
                    last_used_at=NOW - timedelta(days=login_days),
                    expires_at=NOW + timedelta(days=30),
                )
            )
        session.commit()
        return user.id, sessions


def _authenticate(client, token: str, csrf: str) -> None:
    client.cookies.set("ff_session", token)
    client.cookies.set("ff_csrf", csrf)


@pytest.fixture()
def admin_env(client, tmp_path):
    database_url = f"sqlite+pysqlite:///{tmp_path / 'admin.db'}"
    command.upgrade(_alembic_config(database_url), "head")
    engine = create_db_engine(database_url)
    session_factory = build_session_factory(engine)
    admin_id, admin_sessions = _create_user(
        session_factory,
        "admin@example.com",
        days_ago=2,
        verified=True,
        is_admin=True,
        login_days_ago=[1],
    )
    user1_id, user1_sessions = _create_user(
        session_factory,
        "user1@example.com",
        days_ago=10,
        verified=True,
        login_days_ago=[2, 20],
    )
    user2_id, user2_sessions = _create_user(
        session_factory,
        "user2@example.com",
        days_ago=40,
        verified=False,
        login_days_ago=[35],
    )
    with session_factory() as session:
        session.add_all(
            [
                UserFilm(
                    user_id=user1_id,
                    internal_movie_id=1,
                    movie_id="film-1",
                    title="Film 1",
                    rating=8.0,
                    watched=True,
                ),
                UserFilm(
                    user_id=user1_id,
                    internal_movie_id=2,
                    movie_id="film-2",
                    title="Film 2",
                    rating=6.0,
                    watched=True,
                ),
                UserFilm(
                    user_id=user2_id,
                    internal_movie_id=3,
                    movie_id="film-3",
                    title="Film 3",
                    rating=7.0,
                    watched=True,
                ),
                UserSwipe(
                    user_id=user1_id,
                    internal_movie_id=1,
                    movie_id="film-1",
                    title="Film 1",
                    action="interested",
                ),
                UserSwipe(
                    user_id=user2_id,
                    internal_movie_id=2,
                    movie_id="film-2",
                    title="Film 2",
                    action="interested",
                ),
                UserSwipe(
                    user_id=user1_id,
                    internal_movie_id=3,
                    movie_id="film-3",
                    title="Film 3",
                    action="not_interested",
                ),
                RecommendationBatch(
                    user_id=user1_id,
                    status="ready",
                    created_at=NOW - timedelta(days=1),
                    completed_at=NOW - timedelta(days=1),
                ),
                RecommendationBatch(
                    user_id=user2_id,
                    status="ready",
                    created_at=NOW - timedelta(days=10),
                    completed_at=NOW - timedelta(days=10),
                ),
                RecommendationBatch(
                    user_id=user1_id,
                    status="failed",
                    created_at=NOW - timedelta(days=5),
                ),
                AnalyticsEvent(
                    user_id=user1_id,
                    event_type="letterboxd_imported",
                    occurred_at=NOW - timedelta(days=2),
                ),
                AnalyticsEvent(
                    user_id=None,
                    event_type="letterboxd_imported",
                    occurred_at=NOW - timedelta(days=1),
                ),
                AnalyticsEvent(
                    user_id=user2_id,
                    event_type="letterboxd_imported",
                    occurred_at=NOW - timedelta(days=20),
                ),
            ]
        )
        session.commit()

    def override_db():
        session = session_factory()
        try:
            yield session
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    client.cookies.clear()
    client.app.dependency_overrides[get_db] = override_db
    try:
        yield {
            "client": client,
            "session_factory": session_factory,
            "admin": (admin_id, admin_sessions[0][0], admin_sessions[0][1]),
            "user1": (user1_id, user1_sessions[0][0], user1_sessions[0][1]),
            "user2": (user2_id, user2_sessions[0][0], user2_sessions[0][1]),
        }
    finally:
        client.app.dependency_overrides.pop(get_db, None)
        client.cookies.clear()
        engine.dispose()


def test_admin_access_control(admin_env):
    client = admin_env["client"]
    _authenticate(client, admin_env["user1"][1], admin_env["user1"][2])
    assert client.get("/api/v1/admin/overview").status_code == 403
    client.cookies.clear()
    assert client.get("/api/v1/admin/overview").status_code == 401
    _authenticate(client, admin_env["admin"][1], admin_env["admin"][2])
    assert client.get("/api/v1/admin/overview").status_code == 200


def test_overview_metrics_and_unique_counts(admin_env):
    client = admin_env["client"]
    _authenticate(client, admin_env["admin"][1], admin_env["admin"][2])
    response = client.get("/api/v1/admin/overview")
    assert response.status_code == 200
    body = response.json()
    totals = body["totals"]
    assert totals["accounts"] == 3
    assert totals["verified_profiles"] == 2
    assert totals["ratings"] == 3
    assert totals["rating_users"] == 2
    assert totals["logins"] == 4
    assert totals["login_users"] == 3
    assert totals["recommendation_batches"] == 2
    assert totals["recommendation_users"] == 2
    assert totals["letterboxd_imports"] == 3
    assert totals["import_users"] == 2
    assert totals["interested_films"] == 2
    assert totals["interested_users"] == 2
    assert totals["not_interested_films"] == 1
    assert totals["not_interested_users"] == 1
    assert body["last_7_days"] == {
        "registrations": 1,
        "active_users": 2,
        "logins": 2,
    }
    assert body["last_30_days"] == {
        "registrations": 2,
        "active_users": 2,
        "logins": 3,
    }


def test_activity_periods_and_daily_counts(admin_env):
    client = admin_env["client"]
    _authenticate(client, admin_env["admin"][1], admin_env["admin"][2])
    response = client.get("/api/v1/admin/activity?period=7d")
    assert response.status_code == 200
    body = response.json()
    assert body["period"] == "7d"
    assert len(body["points"]) == 7
    assert any(point["registrations"] == 1 for point in body["points"])
    assert any(point["active_users"] >= 1 for point in body["points"])
    assert any(point["recommendations"] == 1 for point in body["points"])

    month = client.get("/api/v1/admin/activity?period=30d").json()
    assert len(month["points"]) == 30
    all_time = client.get("/api/v1/admin/activity?period=all").json()
    assert len(all_time["points"]) >= 40


def test_users_list_detail_and_privacy(admin_env):
    client = admin_env["client"]
    _authenticate(client, admin_env["admin"][1], admin_env["admin"][2])
    response = client.get("/api/v1/admin/users?page=1&page_size=10")
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 3
    assert len(body["items"]) == 3
    assert all("password_hash" not in item for item in body["items"])
    assert all("token_hash" not in item for item in body["items"])

    search = client.get("/api/v1/admin/users?search=user1").json()
    assert search["total"] == 1
    item = search["items"][0]
    assert item["email"] == "user1@example.com"
    assert item["ratings_count"] == 2
    assert item["recommendation_count"] == 1
    assert item["interested_count"] == 1
    assert item["not_interested_count"] == 1

    detail = client.get(
        f"/api/v1/admin/users/{admin_env['user1'][0]}"
    )
    assert detail.status_code == 200
    data = detail.json()
    assert data["watched_count"] == 2
    assert data["last_login_at"] is not None
    for secret in (
        "password_hash",
        "token_hash",
        "csrf_token_hash",
        "ip_hash",
    ):
        assert secret not in data
    assert client.get("/api/v1/admin/users/999999").status_code == 404


def test_empty_analytics_and_import_tracking(admin_env):
    client = admin_env["client"]
    _authenticate(client, admin_env["admin"][1], admin_env["admin"][2])
    with admin_env["session_factory"]() as session:
        session.query(AnalyticsEvent).delete()
        session.commit()
    overview = client.get("/api/v1/admin/overview").json()
    assert overview["totals"]["letterboxd_imports"] == 0
    assert overview["totals"]["import_users"] == 0
    assert overview["imports_tracking_since"] is None


def test_grant_admin_and_prune(tmp_path):
    from scripts.grant_admin import grant_admin

    database_url = f"sqlite+pysqlite:///{tmp_path / 'grant.db'}"
    command.upgrade(_alembic_config(database_url), "head")
    engine = create_db_engine(database_url)
    session_factory = build_session_factory(engine)
    try:
        with session_factory() as session:
            session.add(
                User(
                    email="ordinary@example.com",
                    email_normalized="ordinary@example.com",
                    password_hash="unused",
                )
            )
            session.commit()
        grant_admin(database_url, "ordinary@example.com")
        with session_factory() as session:
            user = session.scalar(
                select(User).where(
                    User.email_normalized == "ordinary@example.com"
                )
            )
            assert user.is_admin is True
            session.add(
                AnalyticsEvent(
                    event_type="letterboxd_imported",
                    occurred_at=NOW - timedelta(days=400),
                )
            )
            session.add(
                AnalyticsEvent(
                    event_type="letterboxd_imported",
                    occurred_at=NOW - timedelta(days=10),
                )
            )
            session.commit()
            removed = AnalyticsService().prune(session, 365, now=NOW)
            assert removed == 1
    finally:
        engine.dispose()


def test_empty_dashboard_handles_zero_data(admin_env):
    client = admin_env["client"]
    _authenticate(client, admin_env["admin"][1], admin_env["admin"][2])
    admin_id = admin_env["admin"][0]
    with admin_env["session_factory"]() as session:
        session.query(AnalyticsEvent).delete()
        session.query(RecommendationBatch).delete()
        session.query(UserSwipe).delete()
        session.query(UserFilm).delete()
        session.query(UserSession).filter(
            UserSession.user_id != admin_id
        ).delete()
        session.query(User).filter(User.id != admin_id).delete()
        session.commit()
    overview = client.get("/api/v1/admin/overview").json()
    assert overview["totals"]["accounts"] == 1
    assert overview["totals"]["ratings"] == 0
    assert overview["totals"]["recommendation_batches"] == 0
    assert overview["totals"]["letterboxd_imports"] == 0
    users = client.get("/api/v1/admin/users").json()
    assert users["total"] == 1
    activity = client.get("/api/v1/admin/activity?period=7d").json()
    assert len(activity["points"]) == 7
    assert all(point["recommendations"] == 0 for point in activity["points"])
