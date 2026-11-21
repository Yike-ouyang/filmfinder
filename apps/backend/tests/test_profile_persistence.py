from __future__ import annotations

import csv
import io
import zipfile
from datetime import timedelta
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import select

from app.core.security import hash_token
from app.db.base import build_session_factory, create_db_engine
from app.db.models import User, UserFilm, UserSession, UserSwipe, utcnow
from app.db.session import get_db


BACKEND_ROOT = Path(__file__).resolve().parents[1]
UPLOAD_PATH = "/api/v1/recommendations/from-letterboxd?limit=100"


def _alembic_config(database_url: str) -> Config:
    config = Config(str(BACKEND_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(BACKEND_ROOT / "migrations"))
    config.set_main_option("sqlalchemy.url", database_url)
    return config


class NoopBatchService:
    def store_ready_batch(self, _user_id, _response):
        return 0

    def close(self):
        return None


@pytest.fixture()
def profile_client(client, tmp_path):
    db_path = tmp_path / "profiles.db"
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
    previous_batch_service = client.app.state.batch_service
    client.app.state.batch_service = NoopBatchService()
    try:
        yield client, session_factory
    finally:
        client.app.state.batch_service = previous_batch_service
        client.app.dependency_overrides.pop(get_db, None)
        client.cookies.clear()
        engine.dispose()


def _create_session(
    session_factory,
    email: str = "member@example.com",
    *,
    expired: bool = False,
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
                expires_at=(
                    now - timedelta(seconds=1)
                    if expired
                    else now + timedelta(days=1)
                ),
            )
        )
        session.commit()
        return user.id, session_token, csrf_token


def _authenticate(client, session_token: str, csrf_token: str) -> None:
    client.cookies.set("ff_session", session_token)
    client.cookies.set("ff_csrf", csrf_token)


def _upload(client, source, csrf_token=None):
    headers = {}
    if csrf_token is not None:
        headers["X-CSRF-Token"] = csrf_token
    if isinstance(source, (str, Path)):
        with open(source, "rb") as handle:
            return client.post(
                UPLOAD_PATH,
                files={"file": ("profile.zip", handle, "application/zip")},
                headers=headers,
            )
    return client.post(
        UPLOAD_PATH,
        files={"file": ("profile.zip", source, "application/zip")},
        headers=headers,
    )


def _fixture_rating_rows(fixture_path: Path, count: int) -> list[dict]:
    with zipfile.ZipFile(fixture_path) as archive:
        raw = archive.read("ratings.csv").decode("utf-8-sig")
    rows = []
    for row in csv.DictReader(io.StringIO(raw)):
        if row.get("Name") and row.get("Rating"):
            rows.append(row)
            if len(rows) == count:
                break
    assert len(rows) == count
    return rows


def _zip_bytes(ratings: list[dict], watched: list[dict] | None = None) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        output = io.StringIO()
        writer = csv.DictWriter(
            output,
            fieldnames=["Date", "Name", "Year", "Letterboxd URI", "Rating"],
        )
        writer.writeheader()
        writer.writerows(ratings)
        archive.writestr("ratings.csv", output.getvalue())
        if watched is not None:
            output = io.StringIO()
            writer = csv.DictWriter(
                output,
                fieldnames=["Date", "Name", "Year", "Letterboxd URI"],
            )
            writer.writeheader()
            writer.writerows(watched)
            archive.writestr("watched.csv", output.getvalue())
    return buffer.getvalue()


def test_authenticated_import_persists_ratings_and_watched(
    profile_client, letterboxd_zip
):
    client, session_factory = profile_client
    user_id, session_token, csrf_token = _create_session(session_factory)
    _authenticate(client, session_token, csrf_token)

    response = _upload(client, letterboxd_zip, csrf_token)

    assert response.status_code == 200
    with session_factory() as session:
        rows = session.scalars(
            select(UserFilm).where(UserFilm.user_id == user_id)
        ).all()
    assert rows
    assert all(row.watched for row in rows)
    assert any(row.rating is not None for row in rows)
    assert all(
        1.0 <= row.rating <= 10.0
        for row in rows
        if row.rating is not None
    )
    assert len({row.internal_movie_id for row in rows}) == len(rows)


def test_anonymous_import_does_not_persist(profile_client, letterboxd_zip):
    client, session_factory = profile_client

    response = _upload(client, letterboxd_zip)

    assert response.status_code == 200
    with session_factory() as session:
        assert session.scalar(select(UserFilm).limit(1)) is None


def test_authenticated_import_requires_csrf(profile_client, letterboxd_zip):
    client, session_factory = profile_client
    user_id, session_token, csrf_token = _create_session(session_factory)
    _authenticate(client, session_token, csrf_token)

    response = _upload(client, letterboxd_zip)

    assert response.status_code == 403
    assert response.json()["error"]["code"] == "csrf_failed"
    with session_factory() as session:
        assert session.scalar(select(UserFilm).limit(1)) is None


def test_successive_imports_merge_and_update(profile_client, letterboxd_zip):
    client, session_factory = profile_client
    user_id, session_token, csrf_token = _create_session(session_factory)
    _authenticate(client, session_token, csrf_token)
    film_a, film_b = _fixture_rating_rows(letterboxd_zip, 2)

    first = dict(film_a, Rating="3.0")
    assert _upload(
        client, _zip_bytes([first]), csrf_token
    ).status_code == 200

    second = [
        dict(film_a, Rating="5.0"),
        dict(film_b, Rating="4.0"),
    ]
    assert _upload(
        client, _zip_bytes(second), csrf_token
    ).status_code == 200

    with session_factory() as session:
        rows = session.scalars(
            select(UserFilm).where(UserFilm.user_id == user_id)
        ).all()
    ratings = sorted(row.rating for row in rows)
    assert len(rows) == 2
    assert ratings == [8.0, 10.0]


def test_watched_only_import_preserves_existing_rating(
    profile_client, letterboxd_zip
):
    client, session_factory = profile_client
    user_id, session_token, csrf_token = _create_session(session_factory)
    _authenticate(client, session_token, csrf_token)
    film_a = _fixture_rating_rows(letterboxd_zip, 1)[0]

    assert _upload(
        client, _zip_bytes([dict(film_a, Rating="3.0")]), csrf_token
    ).status_code == 200
    with session_factory() as session:
        row = session.scalar(
            select(UserFilm).where(UserFilm.user_id == user_id)
        )
        assert row is not None
        assert row.rating == 6.0

    watched = [
        {
            key: film_a[key]
            for key in ("Date", "Name", "Year", "Letterboxd URI")
        }
    ]
    response = _upload(
        client, _zip_bytes([], watched=watched), csrf_token
    )

    assert response.status_code == 200
    with session_factory() as session:
        rows = session.scalars(
            select(UserFilm).where(UserFilm.user_id == user_id)
        ).all()
    assert len(rows) == 1
    assert rows[0].rating == 6.0
    assert rows[0].watched is True


def test_watched_only_import_without_profile_is_rejected_without_writes(
    profile_client, letterboxd_zip
):
    client, session_factory = profile_client
    user_id, session_token, csrf_token = _create_session(session_factory)
    _authenticate(client, session_token, csrf_token)
    film_a = _fixture_rating_rows(letterboxd_zip, 1)[0]
    watched = [
        {
            key: film_a[key]
            for key in ("Date", "Name", "Year", "Letterboxd URI")
        }
    ]

    response = _upload(
        client, _zip_bytes([], watched=watched), csrf_token
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "no_matched_ratings"
    with session_factory() as session:
        assert session.scalar(select(UserFilm).limit(1)) is None


def test_expired_session_falls_back_to_anonymous(
    profile_client, letterboxd_zip
):
    client, session_factory = profile_client
    _, session_token, csrf_token = _create_session(
        session_factory, expired=True
    )
    _authenticate(client, session_token, csrf_token)

    response = _upload(client, letterboxd_zip)

    assert response.status_code == 200
    with session_factory() as session:
        assert session.scalar(select(UserFilm).limit(1)) is None


def test_user_isolation(profile_client, letterboxd_zip):
    client, session_factory = profile_client
    user_a, token_a, csrf_a = _create_session(
        session_factory, "a@example.com"
    )
    user_b, _, _ = _create_session(session_factory, "b@example.com")
    _authenticate(client, token_a, csrf_a)

    assert _upload(client, letterboxd_zip, csrf_a).status_code == 200

    with session_factory() as session:
        rows_a = session.scalars(
            select(UserFilm).where(UserFilm.user_id == user_a)
        ).all()
        rows_b = session.scalars(
            select(UserFilm).where(UserFilm.user_id == user_b)
        ).all()
    assert rows_a
    assert rows_b == []


def test_cumulative_profile_excludes_previous_imports(
    profile_client, letterboxd_zip
):
    client, session_factory = profile_client
    user_id, session_token, csrf_token = _create_session(session_factory)
    _authenticate(client, session_token, csrf_token)
    film_a, film_b = _fixture_rating_rows(letterboxd_zip, 2)

    assert _upload(
        client, _zip_bytes([dict(film_a, Rating="3.0")]), csrf_token
    ).status_code == 200
    with session_factory() as session:
        row_a = session.scalar(
            select(UserFilm).where(UserFilm.user_id == user_id)
        )
        assert row_a is not None
        excluded = {row_a.movie_id}

    response = _upload(
        client, _zip_bytes([dict(film_b, Rating="4.0")]), csrf_token
    )

    assert response.status_code == 200
    with session_factory() as session:
        rows = session.scalars(
            select(UserFilm).where(UserFilm.user_id == user_id)
        ).all()
        excluded |= {row.movie_id for row in rows}
    returned = {
        str(item["movie_id"]) for item in response.json()["recommendations"]
    }
    assert returned.isdisjoint(excluded)


def test_authenticated_import_excludes_swiped_films(
    profile_client, letterboxd_zip
):
    client, session_factory = profile_client
    user_id, session_token, csrf_token = _create_session(session_factory)
    _authenticate(client, session_token, csrf_token)

    first = _upload(client, letterboxd_zip, csrf_token)
    assert first.status_code == 200
    film = first.json()["recommendations"][0]
    with session_factory() as session:
        session.add(
            UserSwipe(
                user_id=user_id,
                internal_movie_id=film["internal_movie_id"],
                movie_id=str(film["movie_id"]),
                action="not_interested",
                title=film["title"],
                year=film["year"],
                genres=film["genres"],
                updated_at=utcnow(),
            )
        )
        session.commit()

    second = _upload(client, letterboxd_zip, csrf_token)

    assert second.status_code == 200
    returned = {
        item["internal_movie_id"]
        for item in second.json()["recommendations"]
    }
    assert film["internal_movie_id"] not in returned
