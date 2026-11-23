#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sqlalchemy import select


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.config import get_settings  # noqa: E402
from app.core.security import normalize_email  # noqa: E402
from app.db.base import (  # noqa: E402
    build_session_factory,
    create_db_engine,
)
from app.db.models import User  # noqa: E402


def grant_admin(
    database_url: str,
    email: str,
    *,
    revoke: bool = False,
) -> bool:
    settings = get_settings()
    engine = create_db_engine(
        database_url, settings.sqlite_busy_timeout_ms
    )
    try:
        session_factory = build_session_factory(engine)
        with session_factory() as session:
            user = session.scalar(
                select(User).where(
                    User.email_normalized == normalize_email(email)
                )
            )
            if user is None:
                raise SystemExit(f"User not found: {email}")
            user.is_admin = not revoke
            session.commit()
    finally:
        engine.dispose()
    return not revoke


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grant or revoke the FilmFinder administrator role."
    )
    parser.add_argument("--email", required=True, help="Existing user email.")
    parser.add_argument(
        "--revoke",
        action="store_true",
        help="Remove the administrator role instead of granting it.",
    )
    parser.add_argument(
        "--database-url",
        default=None,
        help="SQLAlchemy database URL (default: FILMFINDER_DB_PATH).",
    )
    args = parser.parse_args()
    settings = get_settings()
    database_url = args.database_url or settings.database_url
    granted = grant_admin(
        database_url, args.email, revoke=args.revoke
    )
    state = "granted" if granted else "revoked"
    print(f"Administrator role {state} for {args.email}")


if __name__ == "__main__":
    main()
