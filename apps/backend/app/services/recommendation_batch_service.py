from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from app.core.exceptions import APIError
from app.db.models import RecommendationBatch, RecommendationItem, utcnow
from app.services.profile_service import ProfileService
from app.services.recommendation_service import RecommendationService
from app.services.swipe_service import SwipeService


logger = logging.getLogger(__name__)


class RecommendationBatchService:
    def __init__(
        self,
        session_factory: sessionmaker[Session],
        recommendation_service: RecommendationService,
        profile_service: ProfileService,
        swipe_service: SwipeService,
        *,
        executor=None,
        requested_count: Optional[int] = None,
    ):
        self.session_factory = session_factory
        self.recommendation_service = recommendation_service
        self.profile_service = profile_service
        self.swipe_service = swipe_service
        self.requested_count = int(
            requested_count or recommendation_service.engine.pool_size
        )
        self.executor = executor or ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="filmfinder-reco",
        )

    def close(self) -> None:
        shutdown = getattr(self.executor, "shutdown", None)
        if shutdown is not None:
            shutdown(wait=False, cancel_futures=True)

    def request_generation(self, user_id: int, *, reason: str = "login") -> str:
        batch_id = None
        with self.session_factory() as session:
            active = self._active_batch(session, user_id)
            if active is not None:
                return "pending"
            profile = self.profile_service.load_profile(session, user_id)
            if not profile.ratings:
                return "no_profile"
            batch = RecommendationBatch(
                user_id=user_id,
                status="pending",
                requested_count=self.requested_count,
            )
            session.add(batch)
            try:
                session.commit()
            except IntegrityError:
                session.rollback()
                return "pending"
            batch_id = batch.id
        if batch_id is None:
            return "pending"
        try:
            self.executor.submit(self._run_batch, batch_id)
        except RuntimeError:
            self._mark_failed(batch_id, "generation_unavailable")
            return "unavailable"
        return "queued"

    def store_ready_batch(self, user_id: int, response: dict) -> int:
        recommendations = list(response.get("recommendations") or [])
        info = response.get("recommendation_info") or {}
        now = utcnow()
        with self.session_factory() as session:
            active = self._active_batches(session, user_id)
            for batch in active:
                batch.status = "failed"
                batch.error_code = "superseded"
                batch.completed_at = now
            batch = RecommendationBatch(
                user_id=user_id,
                status="ready",
                model=info.get("model"),
                algorithm=info.get("algorithm"),
                requested_count=info.get("requested_count"),
                returned_count=len(recommendations),
                profile_summary=response.get("profile"),
                created_at=now,
                started_at=now,
                completed_at=now,
            )
            session.add(batch)
            session.flush()
            session.add_all(
                [
                    _item_row(batch.id, index, item)
                    for index, item in enumerate(recommendations, start=1)
                ]
            )
            session.commit()
            return batch.id

    def current(self, user_id: int) -> dict:
        with self.session_factory() as session:
            active = self._active_batch(session, user_id)
            ready = self._latest_ready(session, user_id)
            if ready is not None:
                return {
                    "status": "refreshing" if active is not None else "ready",
                    "batch_id": ready.id,
                    "generated_at": ready.completed_at or ready.created_at,
                    "response": self._response_for_batch(ready),
                }
            if active is not None:
                return {
                    "status": "pending",
                    "batch_id": active.id,
                    "generated_at": None,
                    "response": None,
                }
            latest = self._latest_batch(session, user_id)
            if latest is not None and latest.status == "failed":
                return {
                    "status": "failed",
                    "batch_id": latest.id,
                    "generated_at": latest.completed_at or latest.created_at,
                    "response": None,
                }
            return {
                "status": "empty",
                "batch_id": None,
                "generated_at": None,
                "response": None,
            }

    def _run_batch(self, batch_id: int) -> None:
        try:
            profile = None
            with self.session_factory() as session:
                batch = session.get(RecommendationBatch, batch_id)
                if batch is None or batch.status != "pending":
                    return
                batch.status = "running"
                batch.started_at = utcnow()
                session.commit()
                user_id = batch.user_id
                profile = self.profile_service.load_profile(session, user_id)
                excluded = self.swipe_service.excluded_movie_ids(
                    session, user_id
                )

            if not profile.ratings:
                self._mark_failed(batch_id, "no_saved_profile")
                return

            watched = set(profile.watched_movie_ids) | excluded
            summary = {
                "ratings_found": profile.ratings_count,
                "ratings_matched": profile.ratings_count,
                "watched_movies_matched": len(watched),
                "match_rate": 1.0,
            }
            response = self.recommendation_service.recommend_for_saved_profile(
                profile.ratings,
                watched,
                self.requested_count,
                summary,
            )
            self._store_generated(batch_id, response)
        except APIError as error:
            logger.warning("Recommendation batch %s failed: %s", batch_id, error.code)
            self._mark_failed(batch_id, error.code)
        except Exception:
            logger.exception("Recommendation batch %s failed.", batch_id)
            self._mark_failed(batch_id, "generation_failed")

    def _store_generated(self, batch_id: int, response: dict) -> None:
        recommendations = list(response.get("recommendations") or [])
        info = response.get("recommendation_info") or {}
        now = utcnow()
        with self.session_factory() as session:
            batch = session.get(RecommendationBatch, batch_id)
            if batch is None or batch.status != "running":
                return
            batch.status = "ready"
            batch.model = info.get("model")
            batch.algorithm = info.get("algorithm")
            batch.requested_count = info.get("requested_count")
            batch.returned_count = len(recommendations)
            batch.profile_summary = response.get("profile")
            batch.completed_at = now
            session.add_all(
                [
                    _item_row(batch.id, index, item)
                    for index, item in enumerate(recommendations, start=1)
                ]
            )
            session.commit()

    def _mark_failed(self, batch_id: int, error_code: str) -> None:
        try:
            with self.session_factory() as session:
                batch = session.get(RecommendationBatch, batch_id)
                if batch is None:
                    return
                if batch.status not in {"ready", "failed"}:
                    batch.status = "failed"
                    batch.error_code = error_code
                    batch.completed_at = utcnow()
                    session.commit()
        except Exception:
            logger.exception("Could not mark batch %s as failed.", batch_id)

    def _active_batch(
        self, session: Session, user_id: int
    ) -> Optional[RecommendationBatch]:
        return session.scalar(
            select(RecommendationBatch)
            .where(
                RecommendationBatch.user_id == user_id,
                RecommendationBatch.status.in_(("pending", "running")),
            )
            .order_by(RecommendationBatch.created_at.desc())
        )

    def _active_batches(
        self, session: Session, user_id: int
    ) -> list[RecommendationBatch]:
        return list(
            session.scalars(
                select(RecommendationBatch).where(
                    RecommendationBatch.user_id == user_id,
                    RecommendationBatch.status.in_(("pending", "running")),
                )
            ).all()
        )

    def _latest_batch(
        self, session: Session, user_id: int
    ) -> Optional[RecommendationBatch]:
        return session.scalar(
            select(RecommendationBatch)
            .where(RecommendationBatch.user_id == user_id)
            .order_by(RecommendationBatch.created_at.desc())
        )

    def _latest_ready(
        self, session: Session, user_id: int
    ) -> Optional[RecommendationBatch]:
        return session.scalar(
            select(RecommendationBatch)
            .where(
                RecommendationBatch.user_id == user_id,
                RecommendationBatch.status == "ready",
            )
            .order_by(
                RecommendationBatch.completed_at.desc(),
                RecommendationBatch.created_at.desc(),
            )
        )

    def _response_for_batch(self, batch: RecommendationBatch) -> dict:
        recommendations = [
            {
                "rank": item.rank,
                "movie_id": item.movie_id,
                "internal_movie_id": item.internal_movie_id,
                "title": item.title,
                "year": item.year,
                "genres": list(item.genres or []),
                "score": item.score,
                "score_type": item.score_type,
                "tmdb_id": item.tmdb_id,
                "poster_path": item.poster_path,
                "backdrop_path": item.backdrop_path,
            }
            for item in sorted(batch.items, key=lambda row: row.rank)
        ]
        return {
            "profile": batch.profile_summary or {},
            "recommendation_info": {
                "model": batch.model or "M5",
                "algorithm": batch.algorithm or "knn_m5",
                "requested_count": batch.requested_count
                or self.requested_count,
                "returned_count": batch.returned_count
                if batch.returned_count is not None
                else len(recommendations),
            },
            "recommendations": recommendations,
        }


def _item_row(batch_id: int, rank: int, item: dict) -> RecommendationItem:
    return RecommendationItem(
        batch_id=batch_id,
        rank=rank,
        internal_movie_id=int(item["internal_movie_id"]),
        movie_id=str(item["movie_id"]),
        title=str(item["title"]),
        year=item.get("year"),
        genres=list(item.get("genres") or []),
        score=float(item["score"]),
        score_type=str(item["score_type"]),
        tmdb_id=item.get("tmdb_id"),
        poster_path=item.get("poster_path"),
        backdrop_path=item.get("backdrop_path"),
    )
