from __future__ import annotations

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.core.exceptions import APIError
from app.db.models import User, UserFilm, UserSwipe
from app.schemas.films import RatingResponse


class RatingService:
    def __init__(self, engine, profile_service):
        self.engine = engine
        self.profile_service = profile_service

    def set_rating(
        self,
        db: Session,
        user: User,
        film_id: int,
        rating: int,
    ) -> RatingResponse:
        info = self._film_info(film_id)
        try:
            self.profile_service.save_ratings(
                db,
                user,
                {int(film_id): float(rating)},
                commit=False,
            )
            db.execute(
                delete(UserSwipe).where(
                    UserSwipe.user_id == user.id,
                    UserSwipe.internal_movie_id == int(film_id),
                )
            )
            db.commit()
        except Exception as error:
            db.rollback()
            raise APIError(
                500,
                "rating_persistence_failed",
                "Your rating could not be saved.",
            ) from error
        return RatingResponse(
            internal_movie_id=int(film_id),
            movie_id=str(info["movie_id"]),
            title=str(info["title"]),
            rating=int(rating),
            ratings_count=self._count_ratings(db, user.id),
        )

    def delete_rating(
        self,
        db: Session,
        user: User,
        film_id: int,
    ) -> RatingResponse:
        info = self._film_info(film_id)
        row = db.scalar(
            select(UserFilm).where(
                UserFilm.user_id == user.id,
                UserFilm.internal_movie_id == int(film_id),
                UserFilm.rating.is_not(None),
            )
        )
        if row is None:
            raise APIError(
                404,
                "rating_not_found",
                "No rating exists for this film.",
            )
        try:
            db.delete(row)
            db.commit()
        except Exception as error:
            db.rollback()
            raise APIError(
                500,
                "rating_persistence_failed",
                "Your rating could not be removed.",
            ) from error
        return RatingResponse(
            internal_movie_id=int(film_id),
            movie_id=str(info["movie_id"]),
            title=str(info["title"]),
            rating=None,
            ratings_count=self._count_ratings(db, user.id),
        )

    def _count_ratings(self, db: Session, user_id: int) -> int:
        return int(
            db.scalar(
                select(func.count(UserFilm.id)).where(
                    UserFilm.user_id == user_id,
                    UserFilm.rating.is_not(None),
                )
            )
            or 0
        )

    def _film_info(self, film_id: int) -> dict:
        try:
            internal = int(film_id)
        except (TypeError, ValueError) as error:
            raise APIError(
                422,
                "invalid_film_id",
                "The film identifier is invalid.",
            ) from error
        if internal < 0 or internal >= int(self.engine.n_movies):
            raise APIError(
                404,
                "film_not_found",
                "This film is not in the FilmFinder catalogue.",
            )
        try:
            return self.engine.store.movie_info(internal)
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise APIError(
                404,
                "film_not_found",
                "This film is not in the FilmFinder catalogue.",
            ) from error
