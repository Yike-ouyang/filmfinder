from __future__ import annotations

from datetime import datetime
from typing import Any, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class ProfileSummary(BaseModel):
    ratings_found: int
    ratings_matched: int
    watched_movies_matched: int
    match_rate: float


class RecommendationInfo(BaseModel):
    model: str
    algorithm: str
    requested_count: int
    returned_count: int


class RecommendationItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rank: int
    movie_id: Any
    internal_movie_id: int
    title: str
    year: Optional[int] = None
    genres: List[str] = Field(default_factory=list)
    score: float
    score_type: str
    tmdb_id: Optional[int] = None
    poster_path: Optional[str] = None
    backdrop_path: Optional[str] = None


class RecommendationResponse(BaseModel):
    profile: ProfileSummary
    recommendation_info: RecommendationInfo
    recommendations: List[RecommendationItem]


class CurrentRecommendationsResponse(BaseModel):
    status: Literal["ready", "refreshing", "pending", "failed"]
    batch_id: Optional[int] = None
    generated_at: Optional[datetime] = None
    response: Optional[RecommendationResponse] = None


class RefreshResponse(BaseModel):
    status: Literal["pending", "no_new_candidates"]
