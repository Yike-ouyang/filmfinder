from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class TasteProfileMovie(BaseModel):
    internal_movie_id: int
    movie_id: str
    title: str
    year: Optional[int] = None
    genres: List[str] = Field(default_factory=list)
    tmdb_id: Optional[int] = None
    poster_path: Optional[str] = None
    backdrop_path: Optional[str] = None
    user_rating: Optional[int] = None


class TasteProfileMoviesResponse(BaseModel):
    movies: List[TasteProfileMovie]
    min_ratings: int
    recommended_min_ratings: int


class TasteProfileAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    internal_movie_id: int = Field(ge=0)
    rating: Optional[int] = Field(default=None, ge=1, le=10)


class TasteProfileRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answers: List[TasteProfileAnswer] = Field(
        min_length=20,
        max_length=20,
    )
