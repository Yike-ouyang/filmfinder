from __future__ import annotations

import logging
from typing import Optional

from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.exceptions import APIError
from app.db.models import User, UserSwipe, UserSwipeEvent, utcnow


logger = logging.getLogger(__name__)

SWIPE_ACTIONS = ("interested", "not_interested", "already_seen")
VISIBLE_ACTIONS = ("interested", "not_interested")


class SwipeService:
    def __init__(self, engine, metadata=None):
        self.engine = engine
        self.metadata = metadata

    def upsert_swipe(
        self,
        db: Session,
        user: User,
        internal_movie_id: int,
        action: str,
    ):
        if action not in SWIPE_ACTIONS:
            raise APIError(
                422,
                "invalid_swipe_action",
                "Unsupported swipe action.",
            )
        movie = self._movie_info(internal_movie_id)
        if movie.get("movie_id") is None:
            raise APIError(
                404,
                "movie_not_found",
                "This film is not in the FilmFinder catalogue.",
            )
        now = utcnow()
        values = {
            "user_id": user.id,
            "internal_movie_id": int(internal_movie_id),
            "movie_id": str(movie["movie_id"]),
            "action": action,
            "title": str(movie.get("title") or movie["movie_id"]),
            "year": movie.get("year"),
            "genres": movie.get("genres") or [],
            "tmdb_id": movie.get("tmdb_id"),
            "poster_path": movie.get("poster_path"),
            "backdrop_path": movie.get("backdrop_path"),
            "created_at": now,
            "updated_at": now,
        }
        statement = sqlite_insert(UserSwipe).values(**values)
        statement = statement.on_conflict_do_update(
            index_elements=["user_id", "internal_movie_id"],
            set_={
                "movie_id": statement.excluded.movie_id,
                "action": statement.excluded.action,
                "title": statement.excluded.title,
                "year": statement.excluded.year,
                "genres": statement.excluded.genres,
                "tmdb_id": statement.excluded.tmdb_id,
                "poster_path": statement.excluded.poster_path,
                "backdrop_path": statement.excluded.backdrop_path,
                "updated_at": statement.excluded.updated_at,
            },
        )
        try:
            db.execute(statement)
            swipe = db.scalar(
                select(UserSwipe).where(
                    UserSwipe.user_id == user.id,
                    UserSwipe.internal_movie_id == int(internal_movie_id),
                )
            )
            if swipe is None:
                raise APIError(
                    500,
                    "swipe_persistence_failed",
                    "The swipe could not be saved.",
                )
            event = UserSwipeEvent(
                swipe_id=swipe.id,
                user_id=user.id,
                internal_movie_id=int(internal_movie_id),
                action=action,
                created_at=now,
            )
            db.add(event)
            db.commit()
            db.refresh(event)
            return swipe, event
        except APIError:
            db.rollback()
            raise
        except SQLAlchemyError as error:
            db.rollback()
            logger.exception("Swipe persistence failed.")
            raise APIError(
                500,
                "swipe_persistence_failed",
                "The swipe could not be saved.",
            ) from error

    def undo_last_swipe(
        self,
        db: Session,
        user: User,
        internal_movie_id: int,
    ) -> Optional[UserSwipe]:
        swipe = self._get_swipe(db, user.id, internal_movie_id)
        if swipe is None:
            raise APIError(
                404,
                "swipe_not_found",
                "No decision exists for this film.",
            )
        last_event = db.scalar(
            select(UserSwipeEvent)
            .where(
                UserSwipeEvent.swipe_id == swipe.id,
                UserSwipeEvent.undone_at.is_(None),
            )
            .order_by(UserSwipeEvent.id.desc())
        )
        if last_event is None:
            raise APIError(
                404,
                "swipe_not_found",
                "No decision exists for this film.",
            )
        now = utcnow()
        previous = db.scalar(
            select(UserSwipeEvent)
            .where(
                UserSwipeEvent.swipe_id == swipe.id,
                UserSwipeEvent.undone_at.is_(None),
                UserSwipeEvent.id != last_event.id,
            )
            .order_by(UserSwipeEvent.id.desc())
        )
        last_event.undone_at = now
        try:
            if previous is None:
                db.delete(swipe)
                db.commit()
                return None
            swipe.action = previous.action
            swipe.updated_at = now
            db.commit()
            db.refresh(swipe)
            return swipe
        except SQLAlchemyError as error:
            db.rollback()
            logger.exception("Swipe undo failed.")
            raise APIError(
                500,
                "swipe_persistence_failed",
                "The decision could not be updated.",
            ) from error

    def remove_swipe(
        self,
        db: Session,
        user: User,
        internal_movie_id: int,
    ) -> None:
        swipe = self._get_swipe(db, user.id, internal_movie_id)
        if swipe is None:
            raise APIError(
                404,
                "swipe_not_found",
                "No decision exists for this film.",
            )
        try:
            db.delete(swipe)
            db.commit()
        except SQLAlchemyError as error:
            db.rollback()
            logger.exception("Swipe removal failed.")
            raise APIError(
                500,
                "swipe_persistence_failed",
                "The decision could not be removed.",
            ) from error

    def list_swipes(self, db: Session, user_id: int) -> list[UserSwipe]:
        return list(
            db.scalars(
                select(UserSwipe)
                .where(
                    UserSwipe.user_id == user_id,
                    UserSwipe.action.in_(VISIBLE_ACTIONS),
                )
                .order_by(
                    UserSwipe.updated_at.desc(),
                    UserSwipe.id.desc(),
                )
            ).all()
        )

    def excluded_movie_ids(self, db: Session, user_id: int) -> set[str]:
        rows = db.execute(
            select(UserSwipe.movie_id).where(
                UserSwipe.user_id == user_id,
                UserSwipe.action.in_(("not_interested", "already_seen")),
            )
        ).all()
        return {str(row[0]) for row in rows}

    def to_item(self, swipe: UserSwipe) -> dict:
        item = {
            "internal_movie_id": int(swipe.internal_movie_id),
            "movie_id": str(swipe.movie_id),
            "title": str(swipe.title),
            "year": swipe.year,
            "genres": list(swipe.genres or []),
            "tmdb_id": swipe.tmdb_id,
            "poster_path": swipe.poster_path,
            "backdrop_path": swipe.backdrop_path,
            "action": swipe.action,
            "updated_at": swipe.updated_at,
        }
        if self.metadata is not None:
            item = self.metadata.enrich(item)
        return item

    def _get_swipe(
        self,
        db: Session,
        user_id: int,
        internal_movie_id: int,
    ) -> Optional[UserSwipe]:
        return db.scalar(
            select(UserSwipe).where(
                UserSwipe.user_id == user_id,
                UserSwipe.internal_movie_id == int(internal_movie_id),
            )
        )

    def _movie_info(self, internal_movie_id: int) -> dict:
        try:
            internal = int(internal_movie_id)
        except (TypeError, ValueError) as error:
            raise APIError(
                422,
                "invalid_movie_id",
                "The film identifier is invalid.",
            ) from error
        if internal < 0 or internal >= int(self.engine.n_movies):
            raise APIError(
                404,
                "movie_not_found",
                "This film is not in the FilmFinder catalogue.",
            )
        try:
            info = self.engine.store.movie_info(internal)
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise APIError(
                404,
                "movie_not_found",
                "This film is not in the FilmFinder catalogue.",
            ) from error
        if self.metadata is not None:
            info = self.metadata.enrich(info)
        return info
