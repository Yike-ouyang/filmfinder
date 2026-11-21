from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import select

from app.core.security import hash_token
from app.db.base import build_session_factory, create_db_engine
from app.db.models import User, UserSession, UserSwipe, UserSwipeEvent, utcnow
from app.db.session import get_db


BACKEND_ROOT = Path(__file__).resolve().parents[1]


def _alembic_config(database_url: str) -> Config:
    config = Config(str(BACKEND_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(BACKEND_ROOT / "migrations"))
    config.set_main_option("sqlalchemy.url", database_url)
    return config


@pytest.fixture()
def swipe_client(client, tmp_path):
    db_path = tmp_path / "swipes.db"
    database_url = f"sqlite+pysqlite:///{db_path}"
    command.upgrade(_alembic_config(database_url), "head")
    engine = create_db_engine(database_url)
    session_factory = build_session_factory(engine)

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
        yield client, session_factory
    finally:
        client.app.dependency_overrides.pop(get_db, None)
        client.cookies.clear()
        engine.dispose()


def _create_session(session_factory, email: str = "member@example.com"):
    session_token = f"session-{email}"
    csrf_token = f"csrf-{email}"
    now = utcnow()
    with session_factory() as session:
        user = User(
            email=email,
            email_normalized=email,
            password_hash="unused",
            email_verified_at=now,
        )
        session.add(user)
        session.flush()
        session.add(
            UserSession(
                user_id=user.id,
                token_hash=hash_token(session_token),
                csrf_token_hash=hash_token(csrf_token),
                expires_at=now + timedelta(days=1),
            )
        )
        session.commit()
        return user.id, session_token, csrf_token


def _authenticate(client, session_token: str, csrf_token: str) -> None:
    client.cookies.set("ff_session", session_token)
    client.cookies.set("ff_csrf", csrf_token)


def _put(client, internal_movie_id: int, action: str, csrf_token):
    return client.put(
        f"/api/v1/swipes/{internal_movie_id}",
        json={"action": action},
        headers={"X-CSRF-Token": csrf_token},
    )


def _undo(client, internal_movie_id: int, csrf_token):
    return client.post(
        f"/api/v1/swipes/{internal_movie_id}/undo",
        headers={"X-CSRF-Token": csrf_token},
    )


def _delete(client, internal_movie_id: int, csrf_token):
    return client.request(
        "DELETE",
        f"/api/v1/swipes/{internal_movie_id}",
        headers={"X-CSRF-Token": csrf_token},
    )


def _lists(client):
    return client.get("/api/v1/swipes")


def test_swipe_creation_and_lists(swipe_client):
    client, session_factory = swipe_client
    _, session_token, csrf_token = _create_session(session_factory)
    _authenticate(client, session_token, csrf_token)
    movie = client.app.state.engine.store.movie_info(0)

    response = _put(client, 0, "interested", csrf_token)

    assert response.status_code == 200
    payload = response.json()
    assert payload["swipe"]["internal_movie_id"] == 0
    assert payload["swipe"]["action"] == "interested"
    assert payload["swipe"]["title"] == movie["title"]
    assert payload["event_id"] > 0

    lists = _lists(client)
    assert lists.status_code == 200
    body = lists.json()
    assert body["counts"] == {"interested": 1, "not_interested": 0}
    assert [item["internal_movie_id"] for item in body["interested"]] == [0]
    assert body["not_interested"] == []


def test_latest_action_replaces_previous(swipe_client):
    client, session_factory = swipe_client
    user_id, session_token, csrf_token = _create_session(session_factory)
    _authenticate(client, session_token, csrf_token)

    assert _put(client, 0, "interested", csrf_token).status_code == 200
    assert _put(client, 0, "not_interested", csrf_token).status_code == 200

    body = _lists(client).json()
    assert body["counts"] == {"interested": 0, "not_interested": 1}
    assert [item["action"] for item in body["not_interested"]] == [
        "not_interested"
    ]
    with session_factory() as session:
        assert len(
            session.scalars(
                select(UserSwipeEvent).where(
                    UserSwipeEvent.user_id == user_id
                )
            ).all()
        ) == 2


def test_undo_restores_previous_action_then_removes(swipe_client):
    client, session_factory = swipe_client
    user_id, session_token, csrf_token = _create_session(session_factory)
    _authenticate(client, session_token, csrf_token)

    assert _put(client, 0, "interested", csrf_token).status_code == 200
    assert _put(client, 0, "not_interested", csrf_token).status_code == 200

    restored = _undo(client, 0, csrf_token)
    assert restored.status_code == 200
    assert restored.json()["swipe"]["action"] == "interested"
    assert _lists(client).json()["counts"]["interested"] == 1

    removed = _undo(client, 0, csrf_token)
    assert removed.status_code == 200
    assert removed.json()["swipe"] is None
    assert _lists(client).json()["counts"] == {
        "interested": 0,
        "not_interested": 0,
    }

    assert _undo(client, 0, csrf_token).status_code == 404
    with session_factory() as session:
        assert session.scalar(
            select(UserSwipe).where(UserSwipe.user_id == user_id)
        ) is None


def test_delete_removes_swipe_and_events(swipe_client):
    client, session_factory = swipe_client
    user_id, session_token, csrf_token = _create_session(session_factory)
    _authenticate(client, session_token, csrf_token)

    assert _put(client, 0, "interested", csrf_token).status_code == 200
    assert _delete(client, 0, csrf_token).status_code == 204

    assert _lists(client).json()["counts"] == {
        "interested": 0,
        "not_interested": 0,
    }
    with session_factory() as session:
        assert session.scalar(
            select(UserSwipe).where(UserSwipe.user_id == user_id)
        ) is None
        assert session.scalar(
            select(UserSwipeEvent).where(
                UserSwipeEvent.user_id == user_id
            )
        ) is None


def test_already_seen_is_stored_but_hidden_from_lists(swipe_client):
    client, session_factory = swipe_client
    user_id, session_token, csrf_token = _create_session(session_factory)
    _authenticate(client, session_token, csrf_token)

    assert _put(client, 1, "already_seen", csrf_token).status_code == 200

    body = _lists(client).json()
    assert body["interested"] == []
    assert body["not_interested"] == []
    with session_factory() as session:
        swipe = session.scalar(
            select(UserSwipe).where(UserSwipe.user_id == user_id)
        )
        assert swipe is not None
        assert swipe.action == "already_seen"


def test_unknown_movie_is_rejected(swipe_client):
    client, session_factory = swipe_client
    _, session_token, csrf_token = _create_session(session_factory)
    _authenticate(client, session_token, csrf_token)
    unknown = int(client.app.state.engine.n_movies) + 10

    response = _put(client, unknown, "interested", csrf_token)

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "movie_not_found"


def test_csrf_and_auth_are_required(swipe_client):
    client, session_factory = swipe_client
    _, session_token, csrf_token = _create_session(session_factory)
    _authenticate(client, session_token, csrf_token)

    missing = client.put(
        "/api/v1/swipes/0",
        json={"action": "interested"},
    )
    assert missing.status_code == 403

    client.cookies.clear()
    unauthorized = _put(client, 0, "interested", csrf_token)
    assert unauthorized.status_code == 401


def test_user_isolation(swipe_client):
    client, session_factory = swipe_client
    _, token_a, csrf_a = _create_session(session_factory, "a@example.com")
    _, _, _ = _create_session(session_factory, "b@example.com")
    _authenticate(client, token_a, csrf_a)

    assert _put(client, 0, "interested", csrf_a).status_code == 200
    body = _lists(client).json()
    assert body["counts"]["interested"] == 1

    _, token_b, csrf_b = _create_session(session_factory, "b2@example.com")
    client.cookies.clear()
    _authenticate(client, token_b, csrf_b)
    assert _lists(client).json()["counts"]["interested"] == 0


def test_metadata_snapshot(swipe_client):
    client, session_factory = swipe_client
    _, session_token, csrf_token = _create_session(session_factory)
    _authenticate(client, session_token, csrf_token)
    movie = client.app.state.engine.store.movie_info(2)
    metadata = client.app.state.metadata.enrich(
        {"internal_movie_id": 2}
    )

    response = _put(client, 2, "interested", csrf_token)

    item = response.json()["swipe"]
    assert item["movie_id"] == str(movie["movie_id"])
    assert item["year"] == movie["year"]
    assert item["genres"] == list(movie["genres"] or [])
    assert item["tmdb_id"] == metadata["tmdb_id"]
    assert item["poster_path"] == metadata["poster_path"]
    assert item["backdrop_path"] == metadata.get("backdrop_path")


def test_existing_swipe_is_enriched_when_tmdb_metadata_is_missing(
    swipe_client,
):
    client, session_factory = swipe_client
    user_id, session_token, csrf_token = _create_session(session_factory)
    _authenticate(client, session_token, csrf_token)
    movie = client.app.state.engine.store.movie_info(2)
    metadata = client.app.state.metadata.enrich(
        {"internal_movie_id": 2}
    )
    with session_factory() as session:
        session.add(
            UserSwipe(
                user_id=user_id,
                internal_movie_id=2,
                movie_id=str(movie["movie_id"]),
                action="interested",
                title=movie["title"],
                year=movie["year"],
                genres=movie["genres"],
                tmdb_id=None,
                poster_path=None,
                backdrop_path=None,
                updated_at=utcnow(),
            )
        )
        session.commit()

    body = _lists(client).json()

    item = body["interested"][0]
    assert item["tmdb_id"] == metadata["tmdb_id"]
    assert item["poster_path"] == metadata["poster_path"]
    assert item["backdrop_path"] == metadata.get("backdrop_path")
