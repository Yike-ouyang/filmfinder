from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from sqlalchemy import MetaData, create_engine, event
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import get_settings


NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


def _ensure_sqlite_directory(database_url: str) -> None:
    url = make_url(database_url)
    if not url.drivername.startswith("sqlite"):
        return
    database = url.database
    if not database or database == ":memory:":
        return
    Path(database).expanduser().resolve().parent.mkdir(
        parents=True, exist_ok=True
    )


def _install_sqlite_pragmas(engine: Engine, busy_timeout_ms: int) -> None:
    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, connection_record):
        del connection_record
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute(f"PRAGMA busy_timeout={int(busy_timeout_ms)}")
            cursor.execute("PRAGMA synchronous=NORMAL")
        finally:
            cursor.close()


def create_db_engine(
    database_url: Optional[str] = None,
    busy_timeout_ms: Optional[int] = None,
) -> Engine:
    settings = get_settings()
    url = database_url or settings.database_url
    timeout_ms = (
        settings.sqlite_busy_timeout_ms
        if busy_timeout_ms is None
        else int(busy_timeout_ms)
    )
    _ensure_sqlite_directory(url)
    engine = create_engine(
        url,
        connect_args={
            "check_same_thread": False,
            "timeout": max(0.0, timeout_ms / 1000.0),
        },
        pool_pre_ping=True,
    )
    _install_sqlite_pragmas(engine, timeout_ms)
    return engine


def build_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(
        bind=engine,
        autoflush=False,
        expire_on_commit=False,
        class_=Session,
    )


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    settings = get_settings()
    return create_db_engine(
        settings.database_url,
        settings.sqlite_busy_timeout_ms,
    )


@lru_cache(maxsize=1)
def get_session_factory() -> sessionmaker[Session]:
    return build_session_factory(get_engine())
