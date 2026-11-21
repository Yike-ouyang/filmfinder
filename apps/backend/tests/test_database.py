from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import inspect, select
from sqlalchemy.exc import IntegrityError

from app.db.base import build_session_factory, create_db_engine
from app.db.models import (
    EmailToken,
    RecommendationBatch,
    RecommendationItem,
    User,
    UserFilm,
    UserSession,
    UserSwipe,
    UserSwipeEvent,
    utcnow,
)


BACKEND_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_TABLES = {
    "users",
    "sessions",
    "email_tokens",
    "user_films",
    "user_swipes",
    "user_swipe_events",
    "recommendation_batches",
    "recommendation_items",
}


def _alembic_config(database_url: str) -> Config:
    config = Config(str(BACKEND_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(BACKEND_ROOT / "migrations"))
    config.set_main_option("sqlalchemy.url", database_url)
    return config


@pytest.fixture()
def database_url(tmp_path):
    return f"sqlite+pysqlite:///{tmp_path / 'filmfinder-test.db'}"


@pytest.fixture()
def engine(database_url):
    command.upgrade(_alembic_config(database_url), "head")
    engine = create_db_engine(database_url, busy_timeout_ms=5000)
    yield engine
    engine.dispose()


@pytest.fixture()
def session_factory(engine):
    return build_session_factory(engine)


def _make_user(email: str = "user@example.com") -> User:
    return User(
        email=email,
        email_normalized=email.casefold(),
        password_hash="argon2-placeholder",
    )


def _make_swipe(user: User, movie: int = 42) -> UserSwipe:
    return UserSwipe(
        user=user,
        internal_movie_id=movie,
        movie_id=f"movie-{movie}",
        action="interested",
        title="Example Film",
        genres=["Drama"],
    )


def _make_batch(user: User, status: str = "pending") -> RecommendationBatch:
    return RecommendationBatch(user=user, status=status)


def test_migration_creates_expected_tables(engine):
    tables = set(inspect(engine).get_table_names())
    assert tables - {"alembic_version"} == EXPECTED_TABLES


def test_migration_downgrade_removes_tables(database_url):
    config = _alembic_config(database_url)
    command.upgrade(config, "head")
    command.downgrade(config, "base")

    engine = create_db_engine(database_url)
    try:
        remaining = set(inspect(engine).get_table_names())
    finally:
        engine.dispose()
    assert remaining == {"alembic_version"}


def test_sqlite_pragmas_are_applied(engine):
    with engine.connect() as connection:
        foreign_keys = connection.exec_driver_sql(
            "PRAGMA foreign_keys"
        ).scalar()
        journal_mode = connection.exec_driver_sql(
            "PRAGMA journal_mode"
        ).scalar()
        busy_timeout = connection.exec_driver_sql(
            "PRAGMA busy_timeout"
        ).scalar()

    assert foreign_keys == 1
    assert str(journal_mode).lower() == "wal"
    assert busy_timeout == 5000


def test_email_normalized_is_unique(session_factory):
    with session_factory() as session:
        session.add(_make_user("User@Example.com"))
        session.commit()

    with pytest.raises(IntegrityError):
        with session_factory() as session:
            session.add(_make_user("user@example.com"))
            session.commit()


def test_swipe_is_unique_per_user_and_movie(session_factory):
    with session_factory() as session:
        user = _make_user()
        session.add(user)
        session.add(_make_swipe(user))
        session.commit()
        user_id = user.id

    with pytest.raises(IntegrityError):
        with session_factory() as session:
            user = session.get(User, user_id)
            session.add(_make_swipe(user))
            session.commit()


def test_rating_check_constraint(session_factory):
    with session_factory() as session:
        user = _make_user()
        session.add(user)
        session.commit()
        user_id = user.id

    for invalid_rating in (0.0, 10.5):
        with pytest.raises(IntegrityError):
            with session_factory() as session:
                session.add(
                    UserFilm(
                        user_id=user_id,
                        internal_movie_id=int(invalid_rating * 10),
                        movie_id="movie",
                        title="Film",
                        rating=invalid_rating,
                    )
                )
                session.commit()

    with session_factory() as session:
        session.add(
            UserFilm(
                user_id=user_id,
                internal_movie_id=99,
                movie_id="movie-99",
                title="Film",
                rating=7.5,
            )
        )
        session.commit()
        assert session.scalar(select(UserFilm.rating)) == 7.5


def test_partial_unique_index_blocks_concurrent_batches(session_factory):
    with session_factory() as session:
        user = _make_user()
        session.add(user)
        session.add(_make_batch(user, "pending"))
        session.commit()
        user_id = user.id

    with pytest.raises(IntegrityError):
        with session_factory() as session:
            session.add(_make_batch(session.get(User, user_id), "running"))
            session.commit()

    with session_factory() as session:
        batch = session.scalar(select(RecommendationBatch))
        batch.status = "ready"
        session.commit()

    with session_factory() as session:
        session.add(_make_batch(session.get(User, user_id), "pending"))
        session.commit()

    with session_factory() as session:
        session.add(_make_batch(session.get(User, user_id), "ready"))
        session.commit()


def test_deleting_user_cascades_to_all_data(session_factory):
    now = utcnow()
    with session_factory() as session:
        user = _make_user()
        session.add(user)
        session.flush()

        session.add(
            UserSession(
                user=user,
                token_hash="a" * 64,
                csrf_token_hash="b" * 64,
                expires_at=now + timedelta(days=1),
            )
        )
        session.add(
            EmailToken(
                user=user,
                purpose="verify_email",
                token_hash="c" * 64,
                expires_at=now + timedelta(hours=1),
            )
        )
        session.add(
            UserFilm(
                user=user,
                internal_movie_id=1,
                movie_id="movie-1",
                title="Rated Film",
                rating=8.0,
            )
        )
        swipe = _make_swipe(user)
        session.add(swipe)
        session.flush()
        session.add(
            UserSwipeEvent(
                swipe=swipe,
                user_id=user.id,
                internal_movie_id=swipe.internal_movie_id,
                action="interested",
            )
        )
        batch = _make_batch(user, "ready")
        session.add(batch)
        session.flush()
        session.add(
            RecommendationItem(
                batch=batch,
                rank=1,
                internal_movie_id=2,
                movie_id="movie-2",
                title="Recommended Film",
                score=9.0,
                score_type="m5_predicted_rating",
            )
        )
        session.commit()
        user_id = user.id

    with session_factory() as session:
        session.delete(session.get(User, user_id))
        session.commit()

    with session_factory() as session:
        assert session.scalar(select(User).where(User.id == user_id)) is None
        for model in (
            UserSession,
            EmailToken,
            UserFilm,
            UserSwipe,
            UserSwipeEvent,
            RecommendationBatch,
            RecommendationItem,
        ):
            assert session.scalar(select(model).limit(1)) is None


def test_transaction_rollback_discards_changes(session_factory):
    with session_factory() as session:
        session.add(_make_user())
        session.flush()
        session.rollback()

    with session_factory() as session:
        assert session.scalar(select(User).limit(1)) is None


def test_data_persists_after_engine_reopen(database_url, session_factory):
    with session_factory() as session:
        session.add(_make_user("persisted@example.com"))
        session.commit()

    reopened = create_db_engine(database_url)
    try:
        with build_session_factory(reopened)() as session:
            user = session.scalar(
                select(User).where(User.email_normalized == "persisted@example.com")
            )
            assert user is not None
            assert user.email == "persisted@example.com"
    finally:
        reopened.dispose()


def test_foreign_keys_are_enforced(engine):
    session_factory = build_session_factory(engine)
    with pytest.raises(IntegrityError):
        with session_factory() as session:
            session.add(
                RecommendationBatch(user_id=999, status="pending")
            )
            session.commit()


def test_batch_item_rank_is_unique(session_factory):
    with session_factory() as session:
        user = _make_user()
        batch = _make_batch(user, "ready")
        session.add(batch)
        session.flush()
        session.add_all(
            [
                RecommendationItem(
                    batch=batch,
                    rank=1,
                    internal_movie_id=1,
                    movie_id="movie-1",
                    title="Film One",
                    score=9.0,
                    score_type="m5_predicted_rating",
                ),
                RecommendationItem(
                    batch=batch,
                    rank=1,
                    internal_movie_id=2,
                    movie_id="movie-2",
                    title="Film Two",
                    score=8.0,
                    score_type="m5_predicted_rating",
                ),
            ]
        )
        with pytest.raises(IntegrityError):
            session.commit()


def test_wal_checkpoint_preserves_rows(engine, session_factory):
    with session_factory() as session:
        session.add(_make_user())
        session.commit()

    with engine.connect() as connection:
        connection.exec_driver_sql("PRAGMA wal_checkpoint(FULL)")

    with session_factory() as session:
        assert session.scalar(select(User).limit(1)) is not None

