from __future__ import annotations

from typing import Literal, Optional

from fastapi import APIRouter, Depends, Path, Query
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.dependencies.auth import get_current_admin
from app.schemas.admin import (
    AdminActivityResponse,
    AdminOverviewResponse,
    AdminUserDetail,
    AdminUsersResponse,
)
from app.services import admin_service


router = APIRouter(
    prefix="/api/v1/admin",
    tags=["admin"],
    dependencies=[Depends(get_current_admin)],
)


@router.get("/overview", response_model=AdminOverviewResponse)
def overview(db: Session = Depends(get_db)):
    return admin_service.overview(db)


@router.get("/activity", response_model=AdminActivityResponse)
def activity(
    period: Literal["7d", "30d", "all"] = Query(default="7d"),
    db: Session = Depends(get_db),
):
    return admin_service.activity(db, period)


@router.get("/users", response_model=AdminUsersResponse)
def list_users(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=100),
    search: Optional[str] = Query(default=None, max_length=200),
    db: Session = Depends(get_db),
):
    return admin_service.list_users(
        db,
        page=page,
        page_size=page_size,
        search=search,
    )


@router.get("/users/{user_id}", response_model=AdminUserDetail)
def user_detail(
    user_id: int = Path(..., ge=1),
    db: Session = Depends(get_db),
):
    return admin_service.user_detail(db, user_id)
