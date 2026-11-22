from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import select

from app.core.security import hash_token
from app.db.base import build_session_factory, create_db_engine
from app.db.models import (
    CatalogFilm,
    CatalogFilmGenre,
    User,
    UserFilm,
    UserSession,
    UserSwipe,
    utcnow,
)
from app.db.session import get_db
from filmfinder.letterboxd_import import _normalize_title


BACKEND_ROOT = Path(__file__).resolve().parents[1]
FILMS_PATH = "/api/v1/films"
FACETS_PATH = "/api/v1/films/facets"


def _alembic_config(database_url: str) -> Config:
    config = Config(str(BACKEND_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(BACKEND_ROOT / "migrations"))
    config.set_main_option("sqlalchemy.url", database_url)
    return config


@pytest.fixture()
def films_client(client, tmp_path):
    db_path = tmp_path / "films.db"
    database_url = f"sqlite+pysqlite:///{db_path}"
    command.upgrade(_alembic_config(database_url), "head")
    engine = create_db_engine(database_url)
    session_factory = build_session_factory(engine)

    movies = client.app.state.engine.store.movies
    selected = list(range(60)) + [415, 2112, 22806]
    selected = sorted(set(selected))
    with session_factory() as session:
        for index in selected:
            movie = movies[index]
            year = movie.get("year")
            title = str(movie.get("title") or movie["movie_id"])
            session.add(
                CatalogFilm(
                    internal_movie_id=index,
                    movie_id=str(movie["movie_id"]),
                    title=title,
                    title_normalized=_normalize_title(title),
                    year=year,
                    release_date=(
                        date(int(year), 1, 1) if year else None
                    ),
                    genres=list(movie.get("genres") or []),
                    summary=(
                        None
                        if index == 1
                        else f"Summary for {title}"
                    ),
                    tmdb_id=None,
                    poster_path=(
                        None
                        if index == 2
                        else f"/poster-{index}.jpg"
                    ),
                    backdrop_path=None,
                    ratings_count=index,
                    rating_mean=10.0 - (index % 10) * 0.1,
                    weighted_rating=9.0 - (index % 10) * 0.1,
                    in_library=bool(year and year <= 2019),
                )
            )
            for genre in dict.fromkeys(movie.get("genres") or []):
                session.add(
                    CatalogFilmGenre(
                        film_id=index, genre=str(genre)
                    )
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
        yield client, session_factory
    finally:
        client.app.dependency_overrides.pop(get_db, None)
        client.cookies.clear()
        engine.dispose()


def _create_session(
    session_factory,
    email: str = "films@example.com",
):
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


def _put_rating(client, film_id: int, rating: int, csrf_token: str):
    return client.put(
        f"/api/v1/me/ratings/{film_id}",
        json={"rating": rating},
        headers={"X-CSRF-Token": csrf_token},
    )


def _delete_rating(client, film_id: int, csrf_token: str):
    return client.request(
        "DELETE",
        f"/api/v1/me/ratings/{film_id}",
        headers={"X-CSRF-Token": csrf_token},
    )


def test_films_pagination_search_and_temporal_limit(films_client):
    client, _ = films_client
    first = client.get(f"{FILMS_PATH}?page=1")
    second = client.get(f"{FILMS_PATH}?page=2")
    assert first.status_code == 200
    assert second.status_code == 200
    body = first.json()
    assert body["page_size"] == 50
    assert body["temporal_limit"] == 2019
    assert len(body["items"]) == min(50, body["total"])
    assert body["total"] >= 50
    assert all(item["year"] <= 2019 for item in body["items"])
    assert len(second.json()["items"]) <= 50

    search = client.get(f"{FILMS_PATH}?search=SCHNEIDER")
    assert search.status_code == 200
    assert any(
        "schneider" in item["title"].lower()
        for item in search.json()["items"]
    )

    accent = client.get(f"{FILMS_PATH}?search=amelie")
    assert accent.status_code == 200
    assert any(
        item["movie_id"] == "amelie"
        for item in accent.json()["items"]
    )

    empty = client.get(f"{FILMS_PATH}?search=zzzz-no-film")
    assert empty.status_code == 200
    assert empty.json()["items"] == []
    assert empty.json()["total"] == 0


def test_films_sorting_and_genre_filter(films_client):
    client, _ = films_client
    popular = client.get(f"{FILMS_PATH}?sort=popular").json()["items"]
    assert [item["ratings_count"] for item in popular] == sorted(
        [item["ratings_count"] for item in popular],
        reverse=True,
    )

    rated = client.get(f"{FILMS_PATH}?sort=rating").json()["items"]
    assert [item["weighted_rating"] for item in rated] == sorted(
        [item["weighted_rating"] for item in rated],
        reverse=True,
    )

    recent = client.get(
        f"{FILMS_PATH}?sort=release_date&order=desc"
    ).json()["items"]
    assert [item["year"] for item in recent] == sorted(
        [item["year"] for item in recent], reverse=True
    )
    oldest = client.get(
        f"{FILMS_PATH}?sort=release_date&order=asc"
    ).json()["items"]
    assert [item["year"] for item in oldest] == sorted(
        [item["year"] for item in oldest]
    )

    facets = client.get(FACETS_PATH)
    assert facets.status_code == 200
    genres = facets.json()["genres"]
    assert genres
    genre = genres[0]
    filtered = client.get(
        f"{FILMS_PATH}?genre={genre}"
    )
    assert filtered.status_code == 200
    assert all(
        genre in item["genres"]
        for item in filtered.json()["items"]
    )


def test_film_detail_and_missing_summary_poster(films_client):
    client, _ = films_client
    movie = client.app.state.engine.store.movie_info(0)
    detail = client.get(f"{FILMS_PATH}/0")
    assert detail.status_code == 200
    body = detail.json()
    assert body["title"] == movie["title"]
    assert body["summary"] is not None
    assert body["poster_path"]

    missing = client.get(f"{FILMS_PATH}/1").json()
    assert missing["summary"] is None
    assert client.get(f"{FILMS_PATH}/2").json()["poster_path"] is None
    assert client.get(f"{FILMS_PATH}/999999").status_code == 404


def test_user_state_is_added_to_library_and_detail(films_client):
    client, session_factory = films_client
    user_id, session_token, csrf_token = _create_session(session_factory)
    _authenticate(client, session_token, csrf_token)
    with session_factory() as session:
        session.add(
            UserFilm(
                user_id=user_id,
                internal_movie_id=0,
                movie_id="film-0",
                title="Film 0",
                year=1994,
                genres=["Drama"],
                rating=8.0,
                watched=True,
            )
        )
        session.add(
            UserSwipe(
                user_id=user_id,
                internal_movie_id=22806,
                movie_id="film-22806",
                title="Film 22806",
                year=2019,
                genres=["Drama"],
                action="interested",
            )
        )
        session.add(
            UserFilm(
                user_id=user_id,
                internal_movie_id=3,
                movie_id="film-3",
                title="Film 3",
                year=2014,
                genres=["Drama"],
                rating=None,
                watched=True,
            )
        )
        session.commit()

    items = client.get(FILMS_PATH).json()["items"]
    by_id = {item["internal_movie_id"]: item for item in items}
    assert by_id[22806]["user_list_status"] == "watchlist"

    detail = client.get(f"{FILMS_PATH}/0").json()
    assert detail["user_rating"] == 8
    assert detail["user_list_status"] == "seen"
    watched_only = client.get(f"{FILMS_PATH}/3").json()
    assert watched_only["user_rating"] is None
    assert watched_only["user_list_status"] == "seen"


def test_rating_create_update_delete_and_lists(films_client):
    client, session_factory = films_client
    user_id, session_token, csrf_token = _create_session(session_factory)
    _authenticate(client, session_token, csrf_token)

    response = _put_rating(client, 0, 8, csrf_token)
    assert response.status_code == 200
    assert response.json()["rating"] == 8
    assert response.json()["ratings_count"] == 1

    with session_factory() as session:
        row = session.scalar(
            select(UserFilm).where(
                UserFilm.user_id == user_id,
                UserFilm.internal_movie_id == 0,
            )
        )
        assert row.rating == 8.0
        assert row.watched is True

    update = _put_rating(client, 0, 5, csrf_token)
    assert update.status_code == 200
    assert update.json()["rating"] == 5

    with session_factory() as session:
        session.add(
            UserSwipe(
                user_id=user_id,
                internal_movie_id=0,
                movie_id="film-0",
                title="Film 0",
                year=1994,
                genres=["Drama"],
                action="not_interested",
            )
        )
        session.commit()
    assert _put_rating(client, 0, 9, csrf_token).status_code == 200
    with session_factory() as session:
        assert session.scalar(
            select(UserSwipe).where(
                UserSwipe.user_id == user_id,
                UserSwipe.internal_movie_id == 0,
            )
        ) is None

    summary = client.get("/api/v1/me/profile-summary").json()
    assert summary["ratings_count"] == 1
    assert summary["can_generate"] is False

    deleted = _delete_rating(client, 0, csrf_token)
    assert deleted.status_code == 200
    assert deleted.json()["rating"] is None
    with session_factory() as session:
        assert session.scalar(
            select(UserFilm).where(
                UserFilm.user_id == user_id,
                UserFilm.internal_movie_id == 0,
            )
        ) is None


def test_rating_validation_and_authentication(films_client):
    client, session_factory = films_client
    assert _put_rating(client, 0, 8, "bad").status_code == 401
    _, session_token, csrf_token = _create_session(session_factory)
    _authenticate(client, session_token, csrf_token)
    assert _put_rating(client, 0, 0, csrf_token).status_code == 422
    assert _put_rating(client, 0, 11, csrf_token).status_code == 422
    assert _delete_rating(client, 0, csrf_token).status_code == 404
    assert client.get("/api/v1/me/ratings").status_code == 200


def test_ratings_are_isolated_between_users(films_client):
    client, session_factory = films_client
    _, first_session, first_csrf = _create_session(
        session_factory, "first@example.com"
    )
    _authenticate(client, first_session, first_csrf)
    assert _put_rating(client, 0, 8, first_csrf).status_code == 200

    client.cookies.clear()
    _, second_session, second_csrf = _create_session(
        session_factory, "second@example.com"
    )
    _authenticate(client, second_session, second_csrf)
    assert _put_rating(client, 1, 3, second_csrf).status_code == 200

    second = client.get("/api/v1/me/ratings").json()
    assert second["total"] == 1
    assert second["items"][0]["internal_movie_id"] == 1

    client.cookies.clear()
    _authenticate(client, first_session, first_csrf)
    first = client.get("/api/v1/me/ratings").json()
    assert first["total"] == 1
    assert first["items"][0]["internal_movie_id"] == 0


class _FakeBatchService:
    def __init__(self, status: str):
        self.status = status
        self.calls = []

    def request_generation(
        self, user_id, *, reason="login", min_ratings=0
    ):
        self.calls.append((user_id, reason, min_ratings))
        return self.status


def test_manual_generation_endpoint_requires_ten_ratings(films_client):
    client, session_factory = films_client
    _, session_token, csrf_token = _create_session(session_factory)
    _authenticate(client, session_token, csrf_token)
    previous = client.app.state.batch_service
    fake = _FakeBatchService("insufficient_ratings")
    client.app.state.batch_service = fake
    try:
        response = client.post(
            "/api/v1/recommendations/from-saved-ratings",
            headers={"X-CSRF-Token": csrf_token},
        )
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "insufficient_ratings"
        assert fake.calls and fake.calls[0][1] == "manual"
        assert fake.calls[0][2] == 10

        client.app.state.batch_service = _FakeBatchService("queued")
        accepted = client.post(
            "/api/v1/recommendations/from-saved-ratings",
            headers={"X-CSRF-Token": csrf_token},
        )
        assert accepted.status_code == 202
        assert accepted.json() == {"status": "pending"}
    finally:
        client.app.state.batch_service = previous
