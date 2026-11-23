from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import select

from app.core.exceptions import APIError
from app.db.base import build_session_factory, create_db_engine
from app.db.models import (
    RecommendationBatch,
    RecommendationItem,
    User,
    UserSwipe,
    utcnow,
)
from app.services.profile_service import SavedProfile
from app.services.recommendation_batch_service import (
    RecommendationBatchService,
)
from app.services.swipe_service import SwipeService


BACKEND_ROOT = Path(__file__).resolve().parents[1]


def _alembic_config(database_url: str) -> Config:
    config = Config(str(BACKEND_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(BACKEND_ROOT / "migrations"))
    config.set_main_option("sqlalchemy.url", database_url)
    return config


class ImmediateExecutor:
    def submit(self, function, *args):
        function(*args)
        return None

    def shutdown(self, **_kwargs):
        return None


class CapturingExecutor:
    def __init__(self):
        self.function = None
        self.args = None

    def submit(self, function, *args):
        self.function = function
        self.args = args

    def run(self):
        assert self.function is not None
        self.function(*self.args)

    def shutdown(self, **_kwargs):
        return None


class FakeProfileService:
    def __init__(self, profile: SavedProfile):
        self.profile = profile

    def load_profile(self, _db, _user_id):
        return self.profile


class FakeRecommendationService:
    def __init__(self, error: Exception | None = None):
        self.calls = []
        self.error = error

    def recommend_for_saved_profile(
        self, ratings, watched, limit, summary
    ):
        self.calls.append((dict(ratings), set(watched), int(limit), dict(summary)))
        if self.error is not None:
            raise self.error
        return {
            "profile": summary,
            "recommendation_info": {
                "model": "M5",
                "algorithm": "knn_m5",
                "requested_count": int(limit),
                "returned_count": 2,
            },
            "recommendations": [
                {
                    "rank": 1,
                    "movie_id": "101",
                    "internal_movie_id": 10,
                    "title": "Film Ten",
                    "year": 2001,
                    "genres": ["Drama"],
                    "score": 9.1,
                    "score_type": "m5_predicted_rating",
                    "tmdb_id": 501,
                    "poster_path": "/poster-10.jpg",
                    "backdrop_path": None,
                },
                {
                    "rank": 2,
                    "movie_id": "102",
                    "internal_movie_id": 11,
                    "title": "Film Eleven",
                    "year": 2002,
                    "genres": ["Comedy"],
                    "score": 8.4,
                    "score_type": "m5_predicted_rating",
                    "tmdb_id": 502,
                    "poster_path": "/poster-11.jpg",
                    "backdrop_path": None,
                },
            ],
        }


@pytest.fixture()
def batch_env(tmp_path):
    db_path = tmp_path / "batches.db"
    database_url = f"sqlite+pysqlite:///{db_path}"
    command.upgrade(_alembic_config(database_url), "head")
    engine = create_db_engine(database_url)
    session_factory = build_session_factory(engine)
    with session_factory() as session:
        user = User(
            email="batch@example.com",
            email_normalized="batch@example.com",
            password_hash="unused",
            email_verified_at=utcnow(),
        )
        session.add(user)
        session.commit()
        user_id = user.id
    yield session_factory, user_id
    engine.dispose()


def _service(
    session_factory,
    *,
    profile=None,
    recommendation_service=None,
    executor=None,
):
    return RecommendationBatchService(
        session_factory,
        recommendation_service or FakeRecommendationService(),
        FakeProfileService(
            profile
            or SavedProfile(
                ratings={"1": 8.0},
                watched_movie_ids={"1"},
                ratings_count=1,
                watched_count=1,
            )
        ),
        SwipeService(None),
        executor=executor or ImmediateExecutor(),
        requested_count=3,
    )


def test_login_generation_stores_batch_items(batch_env):
    session_factory, user_id = batch_env
    recommendation = FakeRecommendationService()
    service = _service(
        session_factory,
        recommendation_service=recommendation,
    )

    status = service.request_generation(user_id, reason="login")

    assert status == "queued"
    current = service.current(user_id)
    assert current["status"] == "ready"
    assert current["response"]["recommendations"][0]["title"] == "Film Ten"
    assert len(current["response"]["recommendations"]) == 2
    assert recommendation.calls[0][0] == {"1": 8.0}
    assert recommendation.calls[0][1] == {"1"}
    with session_factory() as session:
        batch = session.scalar(select(RecommendationBatch))
        assert batch.status == "ready"
        assert batch.returned_count == 2
        assert len(session.scalars(select(RecommendationItem)).all()) == 2


def test_swipe_exclusions_are_added_to_watched(batch_env):
    session_factory, user_id = batch_env
    with session_factory() as session:
        session.add_all(
            [
                UserSwipe(
                    user_id=user_id,
                    internal_movie_id=42,
                    movie_id="42",
                    action="not_interested",
                    title="Rejected",
                    updated_at=utcnow(),
                ),
                UserSwipe(
                    user_id=user_id,
                    internal_movie_id=43,
                    movie_id="43",
                    action="already_seen",
                    title="Seen",
                    updated_at=utcnow(),
                ),
                UserSwipe(
                    user_id=user_id,
                    internal_movie_id=44,
                    movie_id="44",
                    action="interested",
                    title="Interested",
                    updated_at=utcnow(),
                ),
            ]
        )
        session.commit()
    recommendation = FakeRecommendationService()
    service = _service(
        session_factory,
        recommendation_service=recommendation,
    )

    assert service.request_generation(user_id) == "queued"

    _, watched, _, _ = recommendation.calls[0]
    assert {"42", "43", "1"}.issubset(watched)
    assert "44" not in watched


def test_refresh_recomputes_even_without_data_changes(batch_env):
    session_factory, user_id = batch_env
    recommendation = FakeRecommendationService()
    service = _service(
        session_factory,
        recommendation_service=recommendation,
    )

    assert service.request_generation(user_id) == "queued"
    assert service.request_generation(user_id, reason="refresh") == "queued"
    assert len(recommendation.calls) == 2
    assert service.current(user_id)["status"] == "ready"


def test_refresh_queues_after_new_exclusion(batch_env):
    session_factory, user_id = batch_env
    recommendation = FakeRecommendationService()
    service = _service(
        session_factory,
        recommendation_service=recommendation,
    )
    assert service.request_generation(user_id) == "queued"
    with session_factory() as session:
        session.add(
            UserSwipe(
                user_id=user_id,
                internal_movie_id=55,
                movie_id="55",
                action="not_interested",
                title="New rejection",
                updated_at=utcnow() + timedelta(seconds=1),
            )
        )
        session.commit()

    assert service.request_generation(user_id, reason="refresh") == "queued"
    assert len(recommendation.calls) == 2
    assert "55" in recommendation.calls[1][1]


def test_active_batch_prevents_duplicate(batch_env):
    session_factory, user_id = batch_env
    executor = CapturingExecutor()
    service = _service(session_factory, executor=executor)

    assert service.request_generation(user_id) == "queued"
    assert service.request_generation(user_id) == "pending"
    assert service.current(user_id)["status"] == "pending"

    executor.run()

    assert service.current(user_id)["status"] == "ready"


def test_store_ready_batch_is_returned_by_current(batch_env):
    session_factory, user_id = batch_env
    service = _service(session_factory)
    response = FakeRecommendationService().recommend_for_saved_profile(
        {"1": 8.0},
        {"1"},
        3,
        {
            "ratings_found": 1,
            "ratings_matched": 1,
            "watched_movies_matched": 1,
            "match_rate": 1.0,
        },
    )

    batch_id = service.store_ready_batch(user_id, response)

    current = service.current(user_id)
    assert current["status"] == "ready"
    assert current["batch_id"] == batch_id
    assert current["response"]["recommendations"][0]["movie_id"] == "101"


def test_generation_failure_marks_batch_failed(batch_env):
    session_factory, user_id = batch_env
    service = _service(
        session_factory,
        recommendation_service=FakeRecommendationService(
            error=APIError(500, "inference_error", "boom")
        ),
    )

    assert service.request_generation(user_id) == "queued"
    assert service.current(user_id)["status"] == "failed"
    with session_factory() as session:
        batch = session.scalar(select(RecommendationBatch))
        assert batch.status == "failed"
        assert batch.error_code == "inference_error"


def test_no_profile_prevents_generation(batch_env):
    session_factory, user_id = batch_env
    service = _service(
        session_factory,
        profile=SavedProfile(
            ratings={},
            watched_movie_ids=set(),
            ratings_count=0,
            watched_count=0,
        ),
    )

    assert service.request_generation(user_id) == "no_profile"
    assert service.current(user_id)["status"] == "empty"


def test_manual_generation_requires_ten_ratings(batch_env):
    session_factory, user_id = batch_env
    service = _service(
        session_factory,
        profile=SavedProfile(
            ratings={str(index): 8.0 for index in range(9)},
            watched_movie_ids={str(index) for index in range(9)},
            ratings_count=9,
            watched_count=9,
        ),
    )

    assert (
        service.request_generation(
            user_id, reason="manual", min_ratings=10
        )
        == "insufficient_ratings"
    )
