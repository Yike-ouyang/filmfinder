from __future__ import annotations

from collections.abc import Iterator

from sqlalchemy.orm import Session

from app.db.base import get_session_factory


def get_db() -> Iterator[Session]:
    session = get_session_factory()()
    try:
        yield session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
