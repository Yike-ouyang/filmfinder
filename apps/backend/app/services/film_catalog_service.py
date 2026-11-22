from __future__ import annotations

import math
from typing import Optional

from sqlalchemy import and_, func, select
from sqlalchemy.exc import OperationalError, ProgrammingError
from sqlalchemy.orm import Session

from filmfinder.letterboxd_import import _normalize_title

from app.core.exceptions import APIError
from app.db.models import (
    CatalogFilm,
    CatalogFilmGenre,
    UserFilm,
    UserSwipe,
)
from app.schemas.films import (
    FilmDetail,
    FilmFacetsResponse,
    FilmListItem,
    FilmListResponse,
    ProfileSummaryResponse,
    RatingResponse,
    UserRatingItem,
    UserRatingsResponse,
)


TEMPORAL_LIMIT = 2019
DEFAULT_PAGE_SIZE = 50
RECOMMENDED_MIN_RATINGS = 10
LIST_STATUS = {
    "interested": "watchlist",
    "not_interested": "do_not_watch",
    "already_seen": "seen",
}


def _catalog_unavailable(error: Exception):
    return APIError(
        503,
        "catalog_unavailable",
        "The film catalogue has not been synchronized yet.",
    )


def _status(action, rating, watched=False):
    status = LIST_STATUS.get(action)
    if status is not None:
        return status
    if rating is not None or watched:
        return "seen"
    return None


def _base_filters(search: Optional[str], genre: Optional[str]):
    filters = [CatalogFilm.in_library.is_(True)]
    if search:
        normalized = _normalize_title(search)
        if normalized:
            filters.append(
                CatalogFilm.title_normalized.contains(normalized)
            )
    if genre:
        filters.append(
            CatalogFilm.internal_movie_id.in_(
                select(CatalogFilmGenre.film_id).where(
                    CatalogFilmGenre.genre == genre
                )
            )
        )
    return filters


def _order_by(sort: str, order: str):
    if sort == "rating":
        return (
            CatalogFilm.weighted_rating.desc(),
            CatalogFilm.ratings_count.desc(),
            CatalogFilm.internal_movie_id.asc(),
        )
    if sort == "release_date":
        direction = order == "asc"
        return (
            CatalogFilm.year.asc() if direction else CatalogFilm.year.desc(),
            CatalogFilm.release_date.asc()
            if direction
            else CatalogFilm.release_date.desc(),
            CatalogFilm.internal_movie_id.asc(),
        )
    return (
        CatalogFilm.ratings_count.desc(),
        CatalogFilm.weighted_rating.desc(),
        CatalogFilm.internal_movie_id.asc(),
    )


def _item(row, rating=None, action=None, watched=False) -> FilmListItem:
    return FilmListItem(
        internal_movie_id=int(row.internal_movie_id),
        movie_id=str(row.movie_id),
        title=str(row.title),
        year=row.year,
        release_date=row.release_date,
        genres=list(row.genres or []),
        tmdb_id=row.tmdb_id,
        poster_path=row.poster_path,
        backdrop_path=row.backdrop_path,
        ratings_count=int(row.ratings_count),
        rating_mean=float(row.rating_mean),
        weighted_rating=float(row.weighted_rating),
        user_rating=int(rating) if rating is not None else None,
        user_list_status=_status(action, rating, bool(watched)),
    )


def list_films(
    db: Session,
    *,
    page: int,
    page_size: int,
    search: Optional[str],
    sort: str,
    order: str,
    genre: Optional[str],
    user_id: Optional[int],
) -> FilmListResponse:
    page_size = max(1, min(int(page_size), DEFAULT_PAGE_SIZE))
    filters = _base_filters(search, genre)
    try:
        total = int(
            db.scalar(
                select(func.count(CatalogFilm.internal_movie_id)).where(
                    *filters
                )
            )
            or 0
        )
    except (OperationalError, ProgrammingError) as error:
        raise _catalog_unavailable(error) from error
    total_pages = max(1, math.ceil(total / page_size))
    page = max(1, min(int(page), total_pages))
    statement = select(CatalogFilm).where(*filters)
    rating_column = None
    action_column = None
    if user_id is not None:
        statement = statement.outerjoin(
            UserFilm,
            and_(
                UserFilm.user_id == user_id,
                UserFilm.internal_movie_id
                == CatalogFilm.internal_movie_id,
            ),
        ).outerjoin(
            UserSwipe,
            and_(
                UserSwipe.user_id == user_id,
                UserSwipe.internal_movie_id
                == CatalogFilm.internal_movie_id,
            ),
        )
        rating_column = UserFilm.rating
        action_column = UserSwipe.action
        watched_column = UserFilm.watched
        statement = statement.add_columns(
            rating_column, action_column, watched_column
        )
    statement = (
        statement.order_by(*_order_by(sort, order))
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    try:
        rows = db.execute(statement).all()
    except (OperationalError, ProgrammingError) as error:
        raise _catalog_unavailable(error) from error
    items = []
    for row in rows:
        if rating_column is None:
            items.append(_item(row[0]))
        else:
            items.append(_item(row[0], row[1], row[2], row[3]))
    return FilmListResponse(
        items=items,
        page=page,
        page_size=page_size,
        total=total,
        total_pages=total_pages,
        temporal_limit=TEMPORAL_LIMIT,
    )


def get_film(
    db: Session,
    film_id: int,
    user_id: Optional[int],
) -> FilmDetail:
    statement = select(CatalogFilm).where(
        CatalogFilm.internal_movie_id == int(film_id)
    )
    rating_column = None
    action_column = None
    if user_id is not None:
        statement = statement.outerjoin(
            UserFilm,
            and_(
                UserFilm.user_id == user_id,
                UserFilm.internal_movie_id
                == CatalogFilm.internal_movie_id,
            ),
        ).outerjoin(
            UserSwipe,
            and_(
                UserSwipe.user_id == user_id,
                UserSwipe.internal_movie_id
                == CatalogFilm.internal_movie_id,
            ),
        )
        rating_column = UserFilm.rating
        action_column = UserSwipe.action
        watched_column = UserFilm.watched
        statement = statement.add_columns(
            rating_column, action_column, watched_column
        )
    try:
        row = db.execute(statement).first()
    except (OperationalError, ProgrammingError) as error:
        raise _catalog_unavailable(error) from error
    if row is None:
        raise APIError(
            404,
            "film_not_found",
            "This film is not in the FilmFinder catalogue.",
        )
    item = (
        _item(row[0])
        if rating_column is None
        else _item(row[0], row[1], row[2], row[3])
    )
    return FilmDetail(
        **item.model_dump(),
        summary=row[0].summary,
    )


def facets(db: Session) -> FilmFacetsResponse:
    try:
        genres = list(
            db.scalars(
                select(CatalogFilmGenre.genre)
                .join(
                    CatalogFilm,
                    CatalogFilm.internal_movie_id
                    == CatalogFilmGenre.film_id,
                )
                .where(CatalogFilm.in_library.is_(True))
                .distinct()
                .order_by(CatalogFilmGenre.genre)
            ).all()
        )
        total = int(
            db.scalar(
                select(func.count(CatalogFilm.internal_movie_id)).where(
                    CatalogFilm.in_library.is_(True)
                )
            )
            or 0
        )
    except (OperationalError, ProgrammingError) as error:
        raise _catalog_unavailable(error) from error
    return FilmFacetsResponse(
        genres=[str(genre) for genre in genres],
        temporal_limit=TEMPORAL_LIMIT,
        total=total,
    )


def list_user_ratings(
    db: Session,
    user_id: int,
    *,
    limit: int = 50,
    offset: int = 0,
) -> UserRatingsResponse:
    total = int(
        db.scalar(
            select(func.count(UserFilm.id)).where(
                UserFilm.user_id == user_id,
                UserFilm.rating.is_not(None),
            )
        )
        or 0
    )
    try:
        rows = db.execute(
            select(UserFilm, CatalogFilm.poster_path)
            .outerjoin(
                CatalogFilm,
                CatalogFilm.internal_movie_id
                == UserFilm.internal_movie_id,
            )
            .where(
                UserFilm.user_id == user_id,
                UserFilm.rating.is_not(None),
            )
            .order_by(UserFilm.updated_at.desc(), UserFilm.id.desc())
            .offset(int(offset))
            .limit(int(limit))
        ).all()
    except (OperationalError, ProgrammingError) as error:
        raise _catalog_unavailable(error) from error
    return UserRatingsResponse(
        items=[
            UserRatingItem(
                internal_movie_id=int(row[0].internal_movie_id),
                movie_id=str(row[0].movie_id),
                title=str(row[0].title),
                year=row[0].year,
                poster_path=row[1],
                rating=int(row[0].rating),
                updated_at=(
                    row[0].updated_at.isoformat()
                    if row[0].updated_at is not None
                    else None
                ),
            )
            for row in rows
        ],
        total=total,
    )


def profile_summary(db: Session, user_id: int) -> ProfileSummaryResponse:
    ratings_count = int(
        db.scalar(
            select(func.count(UserFilm.id)).where(
                UserFilm.user_id == user_id,
                UserFilm.rating.is_not(None),
            )
        )
        or 0
    )
    watched_count = int(
        db.scalar(
            select(func.count(UserFilm.id)).where(
                UserFilm.user_id == user_id,
                UserFilm.watched.is_(True),
            )
        )
        or 0
    )
    watchlist_count = int(
        db.scalar(
            select(func.count(UserSwipe.id)).where(
                UserSwipe.user_id == user_id,
                UserSwipe.action == "interested",
            )
        )
        or 0
    )
    do_not_watch_count = int(
        db.scalar(
            select(func.count(UserSwipe.id)).where(
                UserSwipe.user_id == user_id,
                UserSwipe.action == "not_interested",
            )
        )
        or 0
    )
    seen_count = int(
        db.scalar(
            select(func.count(UserSwipe.id)).where(
                UserSwipe.user_id == user_id,
                UserSwipe.action == "already_seen",
            )
        )
        or 0
    )
    return ProfileSummaryResponse(
        ratings_count=ratings_count,
        watched_count=watched_count,
        watchlist_count=watchlist_count,
        do_not_watch_count=do_not_watch_count,
        seen_count=seen_count,
        recommended_min_ratings=RECOMMENDED_MIN_RATINGS,
        can_generate=ratings_count >= RECOMMENDED_MIN_RATINGS,
    )
