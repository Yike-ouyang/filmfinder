from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


SwipeAction = Literal["interested", "not_interested", "already_seen"]


class SwipeRequest(BaseModel):
    action: SwipeAction


class SwipeItem(BaseModel):
    internal_movie_id: int
    movie_id: str
    title: str
    year: Optional[int] = None
    genres: list[str] = Field(default_factory=list)
    tmdb_id: Optional[int] = None
    poster_path: Optional[str] = None
    backdrop_path: Optional[str] = None
    action: SwipeAction
    updated_at: datetime


class SwipeListsResponse(BaseModel):
    interested: list[SwipeItem]
    not_interested: list[SwipeItem]
    counts: dict[str, int]


class SwipeMutationResponse(BaseModel):
    swipe: SwipeItem
    event_id: int


class SwipeUndoResponse(BaseModel):
    swipe: Optional[SwipeItem] = None
