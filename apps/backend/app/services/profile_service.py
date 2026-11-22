from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterator, Optional

from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from app.db.models import User, UserFilm, utcnow


UPSERT_CHUNK_SIZE = 500


@dataclass(frozen=True)
class SavedProfile:
    ratings: dict[str, float]
    watched_movie_ids: set[str]
    ratings_count: int
    watched_count: int


class ProfileService:
    def __init__(self, engine):
        self.engine = engine

    def save_import(
        self,
        db: Session,
        user: User,
        profile,
        now: Optional[datetime] = None,
    ) -> None:
        imported_at = now or utcnow()
        rated_internal = dict(profile.ratings_internal or {})
        watched_internal = set(profile.watched_internal or set())

        rated_rows = []
        for internal_id, rating in rated_internal.items():
            row = self._row(
                user.id,
                internal_id,
                float(rating),
                imported_at,
            )
            if row is not None:
                rated_rows.append(row)

        watched_rows = []
        for internal_id in watched_internal:
            if internal_id in rated_internal:
                continue
            row = self._row(user.id, internal_id, None, imported_at)
            if row is not None:
                watched_rows.append(row)

        try:
            for chunk in _chunks(rated_rows):
                db.execute(_rated_upsert(chunk))
            for chunk in _chunks(watched_rows):
                db.execute(_watched_upsert(chunk))
            db.commit()
        except Exception:
            db.rollback()
            raise

    def save_ratings(
        self,
        db: Session,
        user: User,
        ratings_internal: dict,
        now: Optional[datetime] = None,
        commit: bool = True,
    ) -> None:
        saved_at = now or utcnow()
        rows = []
        for internal_id, rating in dict(ratings_internal or {}).items():
            row = self._row(
                user.id,
                int(internal_id),
                float(rating),
                saved_at,
            )
            if row is not None:
                rows.append(row)
        if not rows:
            if commit:
                db.commit()
            return
        try:
            for chunk in _chunks(rows):
                db.execute(_rated_upsert(chunk))
            if commit:
                db.commit()
        except Exception:
            db.rollback()
            raise

    def load_profile(self, db: Session, user_id: int) -> SavedProfile:
        rows = db.execute(
            select(UserFilm.movie_id, UserFilm.rating).where(
                UserFilm.user_id == user_id
            )
        ).all()
        ratings = {
            str(movie_id): float(rating)
            for movie_id, rating in rows
            if rating is not None
        }
        watched = {str(movie_id) for movie_id, _ in rows}
        return SavedProfile(
            ratings=ratings,
            watched_movie_ids=watched,
            ratings_count=len(ratings),
            watched_count=len(watched),
        )

    def _row(
        self,
        user_id: int,
        internal_id: int,
        rating: Optional[float],
        imported_at: datetime,
    ) -> Optional[dict]:
        info = self.engine.store.movie_info(int(internal_id))
        movie_id = info.get("movie_id")
        if movie_id is None:
            return None
        return {
            "user_id": user_id,
            "internal_movie_id": int(internal_id),
            "movie_id": str(movie_id),
            "title": str(info.get("title") or movie_id),
            "year": info.get("year"),
            "genres": info.get("genres") or [],
            "rating": rating,
            "watched": True,
            "last_imported_at": imported_at,
            "updated_at": imported_at,
        }


def _chunks(rows: list[dict]) -> Iterator[list[dict]]:
    for start in range(0, len(rows), UPSERT_CHUNK_SIZE):
        yield rows[start : start + UPSERT_CHUNK_SIZE]


def _rated_upsert(rows: list[dict]):
    statement = sqlite_insert(UserFilm).values(rows)
    return statement.on_conflict_do_update(
        index_elements=["user_id", "internal_movie_id"],
        set_={
            "movie_id": statement.excluded.movie_id,
            "title": statement.excluded.title,
            "year": statement.excluded.year,
            "genres": statement.excluded.genres,
            "rating": statement.excluded.rating,
            "watched": True,
            "last_imported_at": statement.excluded.last_imported_at,
            "updated_at": statement.excluded.updated_at,
        },
    )


def _watched_upsert(rows: list[dict]):
    statement = sqlite_insert(UserFilm).values(rows)
    return statement.on_conflict_do_update(
        index_elements=["user_id", "internal_movie_id"],
        set_={
            "movie_id": statement.excluded.movie_id,
            "title": statement.excluded.title,
            "year": statement.excluded.year,
            "genres": statement.excluded.genres,
            "watched": True,
            "last_imported_at": statement.excluded.last_imported_at,
            "updated_at": statement.excluded.updated_at,
        },
    )
