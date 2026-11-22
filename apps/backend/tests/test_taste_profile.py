from __future__ import annotations

import json
import importlib.util
import json
from collections import Counter
from datetime import timedelta
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.core.security import hash_token
from app.db.base import build_session_factory, create_db_engine
from app.db.models import User, UserFilm, UserSession, utcnow
from app.db.session import get_db
from filmfinder.letterboxd_import import LetterboxdProfile


BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = (
    REPO_ROOT / "packages" / "recommendation-engine" / "artifacts"
)
SELECTION_SCRIPT = (
    REPO_ROOT
    / "packages"
    / "recommendation-engine"
    / "scripts"
    / "select_quick_taste_movies.py"
)
MOVIES_PATH = "/api/v1/taste-profile/movies"
RECOMMENDATIONS_PATH = "/api/v1/taste-profile/recommendations?limit=20"


@pytest.fixture(autouse=True)
def _clear_rate_limiter(client):
    client.app.state.rate_limiter.clear()
    yield
    client.app.state.rate_limiter.clear()


def _alembic_config(database_url: str) -> Config:
    config = Config(str(BACKEND_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(BACKEND_ROOT / "migrations"))
    config.set_main_option("sqlalchemy.url", database_url)
    return config


class RecordingBatchService:
    def __init__(self):
        self.calls = []

    def store_ready_batch(self, user_id, response):
        self.calls.append((user_id, response))
        return len(self.calls)

    def close(self):
        return None


@pytest.fixture()
def taste_client(client, tmp_path):
    db_path = tmp_path / "taste.db"
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

    batch_service = RecordingBatchService()
    client.cookies.clear()
    client.app.dependency_overrides[get_db] = override_db
    previous_batch_service = client.app.state.batch_service
    client.app.state.batch_service = batch_service
    try:
        yield client, session_factory, batch_service
    finally:
        client.app.state.batch_service = previous_batch_service
        client.app.dependency_overrides.pop(get_db, None)
        client.cookies.clear()
        engine.dispose()


def _create_session(
    session_factory,
    email: str = "taste@example.com",
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


def _movies(client) -> list[dict]:
    response = client.get(MOVIES_PATH)
    assert response.status_code == 200
    return response.json()["movies"]


def _answers(
    movies: list[dict],
    ratings: dict[int, int | None] | None = None,
    *,
    default: int | None = None,
) -> list[dict]:
    ratings = ratings or {}
    return [
        {
            "internal_movie_id": movie["internal_movie_id"],
            "rating": ratings.get(index, default),
        }
        for index, movie in enumerate(movies)
    ]


def _post(client, answers, csrf_token=None):
    headers = {}
    if csrf_token is not None:
        headers["X-CSRF-Token"] = csrf_token
    return client.post(
        RECOMMENDATIONS_PATH,
        json={"answers": answers},
        headers=headers,
    )


def test_movies_endpoint_returns_fixed_selection(client):
    response = client.get(MOVIES_PATH)
    assert response.status_code == 200
    payload = response.json()
    assert payload["min_ratings"] == 3
    assert payload["rating_scale"] == {
        "loved": 8,
        "okay": 5,
        "disliked": 3,
    }
    movies = payload["movies"]
    assert len(movies) == 20
    assert len({movie["internal_movie_id"] for movie in movies}) == 20
    assert len({movie["movie_id"] for movie in movies}) == 20
    for movie in movies:
        assert movie["title"]
        assert movie["year"] is not None
        assert movie["genres"]
        assert movie["poster_path"]
        assert movie["backdrop_path"]


def test_selection_script_reproduces_the_committed_configuration(
    client, tmp_path
):
    spec = importlib.util.spec_from_file_location(
        "select_quick_taste_movies", SELECTION_SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    entries = module.build(
        ARTIFACTS,
        tmp_path / "quick_taste_movies.json",
        tmp_path / "report.md",
    )
    committed = json.loads(
        (
            BACKEND_ROOT
            / "app"
            / "data"
            / "quick_taste_movies.json"
        ).read_text()
    )
    assert entries == committed["movies"]


def test_selection_configuration_is_reproducible_and_compatible(client):
    engine = client.app.state.engine
    payload = client.get(MOVIES_PATH).json()
    movies = payload["movies"]
    internal_ids = [movie["internal_movie_id"] for movie in movies]
    assert len(internal_ids) == 20
    for movie in movies:
        index = engine.store.movie_col(movie["movie_id"])
        assert index == movie["internal_movie_id"]
        assert engine.store.ratings.getcol(index).getnnz() > 0
        assert engine.store.item_features.getrow(index).getnnz() > 0

    config = client.app.state.taste_profile_service
    pools = Counter(
        entry.get("pool")
        for entry in json.loads(config.config_path.read_text())["movies"]
    )
    assert pools == {"popular": 10, "discriminative": 10}
    decades = {f"{(movie['year'] // 10) * 10}s" for movie in movies}
    genres = {genre for movie in movies for genre in movie["genres"]}
    assert len(decades) >= 6
    assert len(genres) >= 10


def test_anonymous_recommendations_exclude_rated_movies(client):
    movies = _movies(client)
    ratings = {
        0: 8,
        3: 8,
        4: 8,
        7: 5,
        9: 3,
    }
    response = _post(client, _answers(movies, ratings))
    assert response.status_code == 200
    payload = response.json()
    assert payload["profile"] == {
        "ratings_found": 20,
        "ratings_matched": 5,
        "watched_movies_matched": 5,
        "match_rate": 0.25,
    }
    returned = {
        recommendation["movie_id"]
        for recommendation in payload["recommendations"]
    }
    rated = {movies[index]["movie_id"] for index in ratings}
    assert returned.isdisjoint(rated)
    scores = [
        recommendation["score"]
        for recommendation in payload["recommendations"]
    ]
    assert scores == sorted(scores, reverse=True)
    assert all(
        recommendation["score_type"] == "m5_predicted_rating"
        for recommendation in payload["recommendations"]
    )


def test_no_opinion_does_not_add_a_rating(client):
    movies = _movies(client)
    response = _post(
        client,
        _answers(movies, {0: 8, 1: 5, 2: 3}, default=None),
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["profile"]["ratings_found"] == 20
    assert payload["profile"]["ratings_matched"] == 3
    assert payload["profile"]["watched_movies_matched"] == 3
    assert payload["profile"]["match_rate"] == 0.15


def test_insufficient_ratings_are_rejected(client):
    movies = _movies(client)
    response = _post(
        client,
        _answers(movies, {0: 8, 1: 5}, default=None),
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "insufficient_taste_ratings"


def test_duplicate_movie_is_rejected(client):
    movies = _movies(client)
    answers = _answers(movies, {0: 8, 1: 5, 2: 3}, default=None)
    answers[-1]["internal_movie_id"] = answers[0]["internal_movie_id"]
    response = _post(client, answers)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_taste_selection"


def test_unknown_movie_is_rejected(client):
    movies = _movies(client)
    answers = _answers(movies, {0: 8, 1: 5, 2: 3}, default=None)
    answers[-1]["internal_movie_id"] = 999999
    response = _post(client, answers)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_taste_selection"


def test_missing_answer_is_rejected(client):
    movies = _movies(client)
    answers = _answers(movies, {0: 8, 1: 5, 2: 3}, default=None)
    response = _post(client, answers[:-1])
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_invalid_rating_is_rejected(client):
    movies = _movies(client)
    answers = _answers(movies, {0: 8, 1: 5, 2: 3}, default=None)
    answers[0]["rating"] = 4
    response = _post(client, answers)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_recommendations_are_deterministic(client):
    movies = _movies(client)
    answers = _answers(movies, {0: 8, 1: 5, 2: 3}, default=None)
    first = _post(client, answers)
    second = _post(client, answers)
    assert first.status_code == 200
    assert first.json() == second.json()


def test_rate_limit_protects_the_endpoint(client, monkeypatch):
    settings = client.app.state.settings
    monkeypatch.setattr(settings, "taste_profile_rate_limit_attempts", 1)
    client.app.state.rate_limiter.clear()
    movies = _movies(client)
    answers = _answers(movies, {0: 8, 1: 5, 2: 3}, default=None)
    try:
        first = _post(client, answers)
        second = _post(client, answers)
        assert first.status_code == 200
        assert second.status_code == 429
        assert second.json()["error"]["code"] == "rate_limited"
    finally:
        client.app.state.rate_limiter.clear()


def test_anonymous_does_not_persist_ratings(taste_client):
    client, session_factory, _ = taste_client
    movies = _movies(client)
    answers = _answers(movies, {0: 8, 1: 5, 2: 3}, default=None)
    response = _post(client, answers)
    assert response.status_code == 200
    with session_factory() as session:
        assert session.scalars(select(UserFilm)).all() == []


def test_authenticated_ratings_are_saved_and_merged(taste_client):
    client, session_factory, batch_service = taste_client
    user_id, session_token, csrf_token = _create_session(session_factory)
    _authenticate(client, session_token, csrf_token)
    movies = _movies(client)

    first = movies[0]
    with session_factory() as session:
        session.add(
            UserFilm(
                user_id=user_id,
                internal_movie_id=first["internal_movie_id"],
                movie_id=first["movie_id"],
                title=first["title"],
                year=first["year"],
                genres=first["genres"],
                rating=9.0,
                watched=True,
            )
        )
        session.commit()

    ratings = {0: 8, 1: 5, 2: 3}
    response = _post(
        client,
        _answers(movies, ratings, default=None),
        csrf_token=csrf_token,
    )
    assert response.status_code == 200
    assert response.json()["profile"] == {
        "ratings_found": 3,
        "ratings_matched": 3,
        "watched_movies_matched": 3,
        "match_rate": 1.0,
    }
    with session_factory() as session:
        rows = session.scalars(
            select(UserFilm).where(UserFilm.user_id == user_id)
        ).all()
        by_movie = {row.movie_id: row for row in rows}
        assert len(rows) == 3
        assert by_movie[first["movie_id"]].rating == 8.0
        assert by_movie[movies[1]["movie_id"]].rating == 5.0
        assert by_movie[movies[2]["movie_id"]].rating == 3.0
    assert len(batch_service.calls) == 1
    assert batch_service.calls[0][0] == user_id


def test_letterboxd_import_preserves_and_replaces_taste_ratings(taste_client):
    client, session_factory, _ = taste_client
    user_id, session_token, csrf_token = _create_session(
        session_factory, email="merge@example.com"
    )
    _authenticate(client, session_token, csrf_token)
    movies = _movies(client)
    ratings = {0: 8, 1: 3, 2: 5}
    response = _post(
        client,
        _answers(movies, ratings, default=None),
        csrf_token=csrf_token,
    )
    assert response.status_code == 200

    profile_service = client.app.state.profile_service
    import_film = movies[0]
    new_film = movies[10]
    profile = LetterboxdProfile(
        username="merge",
        ratings={import_film["movie_id"]: 6.0, new_film["movie_id"]: 9.0},
        ratings_internal={
            import_film["internal_movie_id"]: 6.0,
            new_film["internal_movie_id"]: 9.0,
        },
        watched=set(),
        watched_internal=set(),
        report={},
    )
    with session_factory() as session:
        user = session.get(User, user_id)
        profile_service.save_import(session, user, profile)
    with session_factory() as session:
        rows = session.scalars(
            select(UserFilm).where(UserFilm.user_id == user_id)
        ).all()
        by_movie = {row.movie_id: row.rating for row in rows}
        assert len(rows) == 4
        assert by_movie[import_film["movie_id"]] == 6.0
        assert by_movie[movies[1]["movie_id"]] == 3.0
        assert by_movie[movies[2]["movie_id"]] == 5.0
        assert by_movie[new_film["movie_id"]] == 9.0
