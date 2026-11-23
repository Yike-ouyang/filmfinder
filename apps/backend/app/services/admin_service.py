from __future__ import annotations

import math
from datetime import datetime, time, timedelta, timezone
from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.exceptions import APIError
from app.db.models import (
    AnalyticsEvent,
    RecommendationBatch,
    User,
    UserFilm,
    UserSession,
    UserSwipe,
)
from app.schemas.admin import (
    AdminActivityPoint,
    AdminActivityResponse,
    AdminOverviewResponse,
    AdminTotals,
    AdminUserDetail,
    AdminUserItem,
    AdminUsersResponse,
    AdminWindow,
)
from app.services.analytics_service import LETTERBOXD_IMPORTED


DEFAULT_PAGE_SIZE = 50
MAX_PAGE_SIZE = 100


def _now(value: Optional[datetime] = None) -> datetime:
    return value or datetime.now(timezone.utc)


def _today(value: Optional[datetime] = None):
    return _now(value).astimezone(timezone.utc).date()


def _window_start(days: int, now: Optional[datetime] = None) -> datetime:
    today = _today(now)
    return datetime.combine(
        today - timedelta(days=max(1, days) - 1),
        time.min,
        tzinfo=timezone.utc,
    )


def _utc(value):
    if isinstance(value, datetime) and value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _count(db: Session, statement) -> int:
    return int(db.scalar(statement) or 0)


def _distinct_count(db: Session, column, *filters) -> int:
    return _count(
        db,
        select(func.count(func.distinct(column))).where(*filters),
    )


def overview(
    db: Session,
    now: Optional[datetime] = None,
) -> AdminOverviewResponse:
    current = _now(now)
    week_start = _window_start(7, current)
    month_start = _window_start(30, current)

    accounts = _count(db, select(func.count(User.id)))
    verified = _count(
        db,
        select(func.count(User.id)).where(
            User.email_verified_at.is_not(None)
        ),
    )
    ratings = _count(
        db,
        select(func.count(UserFilm.id)).where(
            UserFilm.rating.is_not(None)
        ),
    )
    rating_users = _distinct_count(
        db,
        UserFilm.user_id,
        UserFilm.rating.is_not(None),
    )
    logins = _count(db, select(func.count(UserSession.id)))
    login_users = _distinct_count(
        db,
        UserSession.user_id,
    )
    recommendation_batches = _count(
        db,
        select(func.count(RecommendationBatch.id)).where(
            RecommendationBatch.status == "ready"
        ),
    )
    recommendation_users = _distinct_count(
        db,
        RecommendationBatch.user_id,
        RecommendationBatch.status == "ready",
    )
    imports = _count(
        db,
        select(func.count(AnalyticsEvent.id)).where(
            AnalyticsEvent.event_type == LETTERBOXD_IMPORTED
        ),
    )
    import_users = _distinct_count(
        db,
        AnalyticsEvent.user_id,
        AnalyticsEvent.event_type == LETTERBOXD_IMPORTED,
        AnalyticsEvent.user_id.is_not(None),
    )
    interested_films = _count(
        db,
        select(func.count(UserSwipe.id)).where(
            UserSwipe.action == "interested"
        ),
    )
    interested_users = _distinct_count(
        db,
        UserSwipe.user_id,
        UserSwipe.action == "interested",
    )
    not_interested_films = _count(
        db,
        select(func.count(UserSwipe.id)).where(
            UserSwipe.action == "not_interested"
        ),
    )
    not_interested_users = _distinct_count(
        db,
        UserSwipe.user_id,
        UserSwipe.action == "not_interested",
    )

    def window(start: datetime) -> AdminWindow:
        return AdminWindow(
            registrations=_count(
                db,
                select(func.count(User.id)).where(
                    User.created_at >= start
                ),
            ),
            active_users=_distinct_count(
                db,
                UserSession.user_id,
                UserSession.created_at >= start,
            ),
            logins=_count(
                db,
                select(func.count(UserSession.id)).where(
                    UserSession.created_at >= start
                ),
            ),
        )

    imports_tracking_since = db.scalar(
        select(func.min(AnalyticsEvent.occurred_at)).where(
            AnalyticsEvent.event_type == LETTERBOXD_IMPORTED
        )
    )
    return AdminOverviewResponse(
        generated_at=current,
        totals=AdminTotals(
            accounts=accounts,
            verified_profiles=verified,
            ratings=ratings,
            rating_users=rating_users,
            logins=logins,
            login_users=login_users,
            recommendation_batches=recommendation_batches,
            recommendation_users=recommendation_users,
            letterboxd_imports=imports,
            import_users=import_users,
            interested_films=interested_films,
            interested_users=interested_users,
            not_interested_films=not_interested_films,
            not_interested_users=not_interested_users,
        ),
        last_7_days=window(week_start),
        last_30_days=window(month_start),
        imports_tracking_since=_utc(imports_tracking_since),
    )


def _activity_start(
    db: Session,
    period: str,
    end_date,
) -> object:
    if period == "7d":
        return end_date - timedelta(days=6)
    if period == "30d":
        return end_date - timedelta(days=29)
    candidates = [
        db.scalar(select(func.min(User.created_at))),
        db.scalar(select(func.min(UserSession.created_at))),
        db.scalar(
            select(func.min(RecommendationBatch.completed_at)).where(
                RecommendationBatch.status == "ready",
                RecommendationBatch.completed_at.is_not(None),
            )
        ),
        db.scalar(
            select(func.min(AnalyticsEvent.occurred_at)).where(
                AnalyticsEvent.event_type == LETTERBOXD_IMPORTED
            )
        ),
    ]
    dates = []
    for value in candidates:
        if value is None:
            continue
        if isinstance(value, datetime):
            if value.tzinfo is None:
                value = value.replace(tzinfo=timezone.utc)
            dates.append(value.astimezone(timezone.utc).date())
        else:
            dates.append(value)
    return min(dates) if dates else end_date


def activity(
    db: Session,
    period: str,
    now: Optional[datetime] = None,
) -> AdminActivityResponse:
    current = _now(now)
    end_date = _today(current)
    start_date = _activity_start(db, period, end_date)
    start_dt = datetime.combine(
        start_date, time.min, tzinfo=timezone.utc
    )
    end_dt = datetime.combine(
        end_date + timedelta(days=1), time.min, tzinfo=timezone.utc
    )

    registrations = {
        str(day): int(count)
        for day, count in db.execute(
            select(
                func.date(User.created_at),
                func.count(User.id),
            )
            .where(
                User.created_at >= start_dt,
                User.created_at < end_dt,
            )
            .group_by(func.date(User.created_at))
        ).all()
    }
    active = {
        str(day): int(count)
        for day, count in db.execute(
            select(
                func.date(UserSession.created_at),
                func.count(func.distinct(UserSession.user_id)),
            )
            .where(
                UserSession.created_at >= start_dt,
                UserSession.created_at < end_dt,
            )
            .group_by(func.date(UserSession.created_at))
        ).all()
    }
    recommendations = {
        str(day): int(count)
        for day, count in db.execute(
            select(
                func.date(RecommendationBatch.completed_at),
                func.count(RecommendationBatch.id),
            )
            .where(
                RecommendationBatch.status == "ready",
                RecommendationBatch.completed_at.is_not(None),
                RecommendationBatch.completed_at >= start_dt,
                RecommendationBatch.completed_at < end_dt,
            )
            .group_by(func.date(RecommendationBatch.completed_at))
        ).all()
    }

    points = []
    cursor = start_date
    while cursor <= end_date:
        key = cursor.isoformat()
        points.append(
            AdminActivityPoint(
                date=cursor,
                registrations=registrations.get(key, 0),
                active_users=active.get(key, 0),
                recommendations=recommendations.get(key, 0),
            )
        )
        cursor += timedelta(days=1)
    return AdminActivityResponse(
        period=period,
        start_date=start_date,
        end_date=end_date,
        points=points,
    )


def _user_aggregates(
    db: Session,
    user_ids: list[int],
) -> dict[int, dict]:
    if not user_ids:
        return {}
    result = {
        user_id: {
            "ratings_count": 0,
            "recommendation_count": 0,
            "interested_count": 0,
            "not_interested_count": 0,
            "last_login_at": None,
            "last_rating_at": None,
            "last_recommendation_at": None,
            "last_import_at": None,
            "watched_count": 0,
        }
        for user_id in user_ids
    }
    for user_id, count in db.execute(
        select(UserFilm.user_id, func.count(UserFilm.id))
        .where(
            UserFilm.user_id.in_(user_ids),
            UserFilm.rating.is_not(None),
        )
        .group_by(UserFilm.user_id)
    ).all():
        result[int(user_id)]["ratings_count"] = int(count)
    for user_id, count in db.execute(
        select(UserFilm.user_id, func.count(UserFilm.id))
        .where(
            UserFilm.user_id.in_(user_ids),
            UserFilm.watched.is_(True),
        )
        .group_by(UserFilm.user_id)
    ).all():
        result[int(user_id)]["watched_count"] = int(count)
    for user_id, count in db.execute(
        select(RecommendationBatch.user_id, func.count(RecommendationBatch.id))
        .where(
            RecommendationBatch.user_id.in_(user_ids),
            RecommendationBatch.status == "ready",
        )
        .group_by(RecommendationBatch.user_id)
    ).all():
        result[int(user_id)]["recommendation_count"] = int(count)
    for user_id, action, count in db.execute(
        select(
            UserSwipe.user_id,
            UserSwipe.action,
            func.count(UserSwipe.id),
        )
        .where(UserSwipe.user_id.in_(user_ids))
        .group_by(UserSwipe.user_id, UserSwipe.action)
    ).all():
        key = (
            "interested_count"
            if action == "interested"
            else "not_interested_count"
            if action == "not_interested"
            else None
        )
        if key is not None:
            result[int(user_id)][key] = int(count)
    for user_id, value in db.execute(
        select(UserSession.user_id, func.max(UserSession.created_at))
        .where(UserSession.user_id.in_(user_ids))
        .group_by(UserSession.user_id)
    ).all():
        result[int(user_id)]["last_login_at"] = value
    for user_id, value in db.execute(
        select(UserFilm.user_id, func.max(UserFilm.updated_at))
        .where(
            UserFilm.user_id.in_(user_ids),
            UserFilm.rating.is_not(None),
        )
        .group_by(UserFilm.user_id)
    ).all():
        result[int(user_id)]["last_rating_at"] = value
    for user_id, value in db.execute(
        select(
            RecommendationBatch.user_id,
            func.max(RecommendationBatch.completed_at),
        )
        .where(
            RecommendationBatch.user_id.in_(user_ids),
            RecommendationBatch.status == "ready",
        )
        .group_by(RecommendationBatch.user_id)
    ).all():
        result[int(user_id)]["last_recommendation_at"] = value
    for user_id, value in db.execute(
        select(
            AnalyticsEvent.user_id,
            func.max(AnalyticsEvent.occurred_at),
        )
        .where(
            AnalyticsEvent.user_id.in_(user_ids),
            AnalyticsEvent.event_type == LETTERBOXD_IMPORTED,
        )
        .group_by(AnalyticsEvent.user_id)
    ).all():
        result[int(user_id)]["last_import_at"] = value
    return result


def _item(user: User, aggregates: dict) -> AdminUserItem:
    return AdminUserItem(
        id=user.id,
        email=user.email,
        created_at=_utc(user.created_at),
        email_verified=user.email_verified_at is not None,
        is_admin=bool(user.is_admin),
        last_login_at=_utc(aggregates["last_login_at"]),
        ratings_count=aggregates["ratings_count"],
        recommendation_count=aggregates["recommendation_count"],
        interested_count=aggregates["interested_count"],
        not_interested_count=aggregates["not_interested_count"],
    )


def list_users(
    db: Session,
    *,
    page: int,
    page_size: int,
    search: Optional[str],
) -> AdminUsersResponse:
    page_size = max(1, min(int(page_size), MAX_PAGE_SIZE))
    filters = []
    if search:
        normalized = search.strip().lower()
        if normalized:
            filters.append(
                func.lower(User.email).contains(normalized)
            )
    total = _count(
        db,
        select(func.count(User.id)).where(*filters),
    )
    total_pages = max(1, math.ceil(total / page_size))
    page = max(1, min(int(page), total_pages))
    users = list(
        db.scalars(
            select(User)
            .where(*filters)
            .order_by(User.created_at.desc(), User.id.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
        ).all()
    )
    aggregates = _user_aggregates(db, [user.id for user in users])
    empty = {
        "ratings_count": 0,
        "recommendation_count": 0,
        "interested_count": 0,
        "not_interested_count": 0,
        "last_login_at": None,
        "last_rating_at": None,
        "last_recommendation_at": None,
        "last_import_at": None,
        "watched_count": 0,
    }
    return AdminUsersResponse(
        items=[
            _item(user, aggregates.get(user.id, empty))
            for user in users
        ],
        page=page,
        page_size=page_size,
        total=total,
        total_pages=total_pages,
    )


def user_detail(db: Session, user_id: int) -> AdminUserDetail:
    user = db.get(User, int(user_id))
    if user is None:
        raise APIError(
            404,
            "user_not_found",
            "This user does not exist.",
        )
    aggregates = _user_aggregates(db, [user.id])[user.id]
    timestamps = [
        _utc(value)
        for value in (
            aggregates["last_login_at"],
            aggregates["last_rating_at"],
            aggregates["last_recommendation_at"],
            aggregates["last_import_at"],
        )
        if value is not None
    ]
    item = _item(user, aggregates)
    return AdminUserDetail(
        **item.model_dump(),
        watched_count=aggregates["watched_count"],
        last_activity_at=max(timestamps) if timestamps else None,
        last_rating_at=_utc(aggregates["last_rating_at"]),
        last_recommendation_at=_utc(
            aggregates["last_recommendation_at"]
        ),
        last_import_at=_utc(aggregates["last_import_at"]),
    )
