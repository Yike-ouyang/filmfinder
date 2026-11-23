#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.config import get_settings  # noqa: E402
from app.db.base import (  # noqa: E402
    build_session_factory,
    create_db_engine,
)
from app.services.analytics_service import AnalyticsService  # noqa: E402


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="Delete analytics events older than the retention window."
    )
    parser.add_argument(
        "--database-url",
        default=None,
        help="SQLAlchemy database URL (default: FILMFINDER_DB_PATH).",
    )
    parser.add_argument(
        "--retention-days",
        type=int,
        default=settings.analytics_retention_days,
        help="Retention window in days.",
    )
    args = parser.parse_args()
    engine = create_db_engine(
        args.database_url or settings.database_url,
        settings.sqlite_busy_timeout_ms,
    )
    try:
        session_factory = build_session_factory(engine)
        with session_factory() as session:
            removed = AnalyticsService().prune(
                session, args.retention_days
            )
    finally:
        engine.dispose()
    print(f"Removed {removed} analytics events")


if __name__ == "__main__":
    main()
