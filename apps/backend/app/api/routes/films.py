from __future__ import annotations

from typing import Literal, Optional

from fastapi import APIRouter, Depends, Path, Query
from sqlalchemy.orm import Session

from app.db.models import UserSession
from app.db.session import get_db
from app.dependencies.auth import get_optional_session
from app.schemas.films import (
    FilmDetail,
    FilmFacetsResponse,
    FilmListResponse,
)
from app.services import film_catalog_service


router = APIRouter(prefix="/api/v1/films", tags=["films"])


@router.get("/facets", response_model=FilmFacetsResponse)
def film_facets(db: Session = Depends(get_db)):
    return film_catalog_service.facets(db)


@router.get("", response_model=FilmListResponse)
def list_films(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=50),
    search: Optional[str] = Query(default=None, max_length=100),
    sort: Literal["popular", "rating", "release_date"] = "popular",
    order: Literal["asc", "desc"] = "desc",
    genre: Optional[str] = Query(default=None, max_length=64),
    record: Optional[UserSession] = Depends(get_optional_session),
    db: Session = Depends(get_db),
):
    return film_catalog_service.list_films(
        db,
        page=page,
        page_size=page_size,
        search=search,
        sort=sort,
        order=order,
        genre=genre,
        user_id=record.user_id if record is not None else None,
    )


@router.get("/{film_id}", response_model=FilmDetail)
def get_film(
    film_id: int = Path(..., ge=0),
    record: Optional[UserSession] = Depends(get_optional_session),
    db: Session = Depends(get_db),
):
    return film_catalog_service.get_film(
        db,
        film_id,
        record.user_id if record is not None else None,
    )
