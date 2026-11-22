from __future__ import annotations

from datetime import date
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


FilmListStatus = Literal["watchlist", "do_not_watch", "seen"]


class FilmListItem(BaseModel):
    internal_movie_id: int
    movie_id: str
    title: str
    year: Optional[int] = None
    release_date: Optional[date] = None
    genres: List[str] = Field(default_factory=list)
    tmdb_id: Optional[int] = None
    poster_path: Optional[str] = None
    backdrop_path: Optional[str] = None
    ratings_count: int
    rating_mean: float
    weighted_rating: float
    user_rating: Optional[int] = None
    user_list_status: Optional[FilmListStatus] = None


class FilmDetail(FilmListItem):
    summary: Optional[str] = None


class FilmListResponse(BaseModel):
    items: List[FilmListItem]
    page: int
    page_size: int
    total: int
    total_pages: int
    temporal_limit: int


class FilmFacetsResponse(BaseModel):
    genres: List[str]
    temporal_limit: int
    total: int


class RatingRequest(BaseModel):
    rating: int = Field(ge=1, le=10)


class RatingResponse(BaseModel):
    internal_movie_id: int
    movie_id: str
    title: str
    rating: Optional[int] = None
    ratings_count: int


class UserRatingItem(BaseModel):
    internal_movie_id: int
    movie_id: str
    title: str
    year: Optional[int] = None
    poster_path: Optional[str] = None
    rating: int
    updated_at: Optional[str] = None


class UserRatingsResponse(BaseModel):
    items: List[UserRatingItem]
    total: int


class ProfileSummaryResponse(BaseModel):
    ratings_count: int
    watched_count: int
    watchlist_count: int
    do_not_watch_count: int
    seen_count: int
    recommended_min_ratings: int
    can_generate: bool
