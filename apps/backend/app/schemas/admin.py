from __future__ import annotations

from datetime import date, datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


AdminPeriod = Literal["7d", "30d", "all"]


class AdminTotals(BaseModel):
    accounts: int
    verified_profiles: int
    ratings: int
    rating_users: int
    logins: int
    login_users: int
    recommendation_batches: int
    recommendation_users: int
    letterboxd_imports: int
    import_users: int
    interested_films: int
    interested_users: int
    not_interested_films: int
    not_interested_users: int


class AdminWindow(BaseModel):
    registrations: int
    active_users: int
    logins: int


class AdminOverviewResponse(BaseModel):
    generated_at: datetime
    totals: AdminTotals
    last_7_days: AdminWindow
    last_30_days: AdminWindow
    imports_tracking_since: Optional[datetime] = None


class AdminActivityPoint(BaseModel):
    date: date
    registrations: int
    active_users: int
    recommendations: int


class AdminActivityResponse(BaseModel):
    period: AdminPeriod
    start_date: date
    end_date: date
    points: List[AdminActivityPoint]


class AdminUserItem(BaseModel):
    id: int
    email: str
    created_at: datetime
    email_verified: bool
    is_admin: bool
    last_login_at: Optional[datetime] = None
    ratings_count: int
    recommendation_count: int
    interested_count: int
    not_interested_count: int


class AdminUsersResponse(BaseModel):
    items: List[AdminUserItem]
    page: int
    page_size: int
    total: int
    total_pages: int


class AdminUserDetail(AdminUserItem):
    watched_count: int
    last_activity_at: Optional[datetime] = None
    last_rating_at: Optional[datetime] = None
    last_recommendation_at: Optional[datetime] = None
    last_import_at: Optional[datetime] = None
