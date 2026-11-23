from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import delete
from sqlalchemy.orm import Session

from app.db.models import AnalyticsEvent


LETTERBOXD_IMPORTED = "letterboxd_imported"


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class AnalyticsService:
    def record_letterboxd_import(
        self,
        db: Session,
        user_id: int | None,
        now: datetime | None = None,
    ) -> None:
        db.add(
            AnalyticsEvent(
                user_id=user_id,
                event_type=LETTERBOXD_IMPORTED,
                occurred_at=now or utcnow(),
            )
        )
        db.commit()

    def prune(
        self,
        db: Session,
        retention_days: int,
        now: datetime | None = None,
    ) -> int:
        cutoff = (now or utcnow()) - timedelta(
            days=max(0, int(retention_days))
        )
        result = db.execute(
            delete(AnalyticsEvent).where(
                AnalyticsEvent.occurred_at < cutoff
            )
        )
        db.commit()
        return int(result.rowcount or 0)
