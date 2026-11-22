from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


TasteRating = Literal[3, 5, 8]


class TasteProfileRatingScale(BaseModel):
    loved: int
    okay: int
    disliked: int


class TasteProfileMovie(BaseModel):
    internal_movie_id: int
    movie_id: str
    title: str
    year: Optional[int] = None
    genres: List[str] = Field(default_factory=list)
    tmdb_id: Optional[int] = None
    poster_path: Optional[str] = None
    backdrop_path: Optional[str] = None


class TasteProfileMoviesResponse(BaseModel):
    movies: List[TasteProfileMovie]
    min_ratings: int
    rating_scale: TasteProfileRatingScale


class TasteProfileAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    internal_movie_id: int = Field(ge=0)
    rating: Optional[TasteRating] = None


class TasteProfileRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answers: List[TasteProfileAnswer] = Field(
        min_length=20,
        max_length=20,
    )
