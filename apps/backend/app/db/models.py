from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Optional

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


EMAIL_TOKEN_PURPOSES = ("verify_email", "reset_password")
SWIPE_ACTIONS = ("interested", "not_interested", "already_seen")
BATCH_STATUSES = ("pending", "running", "ready", "failed")


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(320), nullable=False)
    email_normalized: Mapped[str] = mapped_column(
        String(320), nullable=False, unique=True
    )
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    email_verified_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utcnow,
        server_default=text("CURRENT_TIMESTAMP"),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utcnow,
        onupdate=utcnow,
        server_default=text("CURRENT_TIMESTAMP"),
    )

    sessions: Mapped[list["UserSession"]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    email_tokens: Mapped[list["EmailToken"]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    films: Mapped[list["UserFilm"]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    swipes: Mapped[list["UserSwipe"]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    recommendation_batches: Mapped[list["RecommendationBatch"]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class UserSession(Base):
    __tablename__ = "sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    token_hash: Mapped[str] = mapped_column(
        String(64), nullable=False, unique=True
    )
    csrf_token_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    user_agent: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    ip_hash: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utcnow,
        server_default=text("CURRENT_TIMESTAMP"),
    )
    last_used_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    revoked_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    user: Mapped["User"] = relationship(back_populates="sessions")


class EmailToken(Base):
    __tablename__ = "email_tokens"
    __table_args__ = (
        CheckConstraint(
            "purpose IN ('verify_email', 'reset_password')",
            name="purpose",
        ),
        Index("ix_email_tokens_user_purpose", "user_id", "purpose"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    purpose: Mapped[str] = mapped_column(String(32), nullable=False)
    token_hash: Mapped[str] = mapped_column(
        String(64), nullable=False, unique=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utcnow,
        server_default=text("CURRENT_TIMESTAMP"),
    )
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    used_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    user: Mapped["User"] = relationship(back_populates="email_tokens")


class UserFilm(Base):
    __tablename__ = "user_films"
    __table_args__ = (
        UniqueConstraint(
            "user_id",
            "internal_movie_id",
            name="uq_user_films_user_movie",
        ),
        CheckConstraint(
            "rating IS NULL OR (rating >= 1.0 AND rating <= 10.0)",
            name="rating_range",
        ),
        Index("ix_user_films_user_rating", "user_id", "rating"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    internal_movie_id: Mapped[int] = mapped_column(Integer, nullable=False)
    movie_id: Mapped[str] = mapped_column(String(64), nullable=False)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    year: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    genres: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    rating: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    watched: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        server_default=text("1"),
    )
    last_imported_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utcnow,
        server_default=text("CURRENT_TIMESTAMP"),
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utcnow,
        server_default=text("CURRENT_TIMESTAMP"),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utcnow,
        onupdate=utcnow,
        server_default=text("CURRENT_TIMESTAMP"),
    )

    user: Mapped["User"] = relationship(back_populates="films")


class UserSwipe(Base):
    __tablename__ = "user_swipes"
    __table_args__ = (
        UniqueConstraint(
            "user_id",
            "internal_movie_id",
            name="uq_user_swipes_user_movie",
        ),
        CheckConstraint(
            "action IN ('interested', 'not_interested', 'already_seen')",
            name="action",
        ),
        Index("ix_user_swipes_user_action", "user_id", "action"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    internal_movie_id: Mapped[int] = mapped_column(Integer, nullable=False)
    movie_id: Mapped[str] = mapped_column(String(64), nullable=False)
    action: Mapped[str] = mapped_column(String(20), nullable=False)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    year: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    genres: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    tmdb_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    poster_path: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True
    )
    backdrop_path: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utcnow,
        server_default=text("CURRENT_TIMESTAMP"),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utcnow,
        onupdate=utcnow,
        server_default=text("CURRENT_TIMESTAMP"),
    )

    user: Mapped["User"] = relationship(back_populates="swipes")
    events: Mapped[list["UserSwipeEvent"]] = relationship(
        back_populates="swipe",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="UserSwipeEvent.id",
    )


class UserSwipeEvent(Base):
    __tablename__ = "user_swipe_events"
    __table_args__ = (
        CheckConstraint(
            "action IN ('interested', 'not_interested', 'already_seen')",
            name="action",
        ),
        Index(
            "ix_user_swipe_events_user_movie",
            "user_id",
            "internal_movie_id",
            "id",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    swipe_id: Mapped[int] = mapped_column(
        ForeignKey("user_swipes.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    internal_movie_id: Mapped[int] = mapped_column(Integer, nullable=False)
    action: Mapped[str] = mapped_column(String(20), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utcnow,
        server_default=text("CURRENT_TIMESTAMP"),
    )
    undone_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    swipe: Mapped["UserSwipe"] = relationship(back_populates="events")


class RecommendationBatch(Base):
    __tablename__ = "recommendation_batches"
    __table_args__ = (
        CheckConstraint(
            "status IN ('pending', 'running', 'ready', 'failed')",
            name="status",
        ),
        Index(
            "uq_recommendation_batches_active_user",
            "user_id",
            unique=True,
            sqlite_where=text("status IN ('pending', 'running')"),
        ),
        Index("ix_recommendation_batches_user_created", "user_id", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    status: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default="pending",
        server_default=text("'pending'"),
    )
    model: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    algorithm: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    error_code: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    requested_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    returned_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    profile_summary: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utcnow,
        server_default=text("CURRENT_TIMESTAMP"),
    )
    started_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    user: Mapped["User"] = relationship(
        back_populates="recommendation_batches"
    )
    items: Mapped[list["RecommendationItem"]] = relationship(
        back_populates="batch",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="RecommendationItem.rank",
    )


class RecommendationItem(Base):
    __tablename__ = "recommendation_items"
    __table_args__ = (
        UniqueConstraint(
            "batch_id", "rank", name="uq_recommendation_items_batch_rank"
        ),
        UniqueConstraint(
            "batch_id",
            "internal_movie_id",
            name="uq_recommendation_items_batch_movie",
        ),
        CheckConstraint("rank >= 1", name="rank"),
        Index("ix_recommendation_items_batch_rank", "batch_id", "rank"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    batch_id: Mapped[int] = mapped_column(
        ForeignKey("recommendation_batches.id", ondelete="CASCADE"),
        nullable=False,
    )
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    internal_movie_id: Mapped[int] = mapped_column(Integer, nullable=False)
    movie_id: Mapped[str] = mapped_column(String(64), nullable=False)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    year: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    genres: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    score_type: Mapped[str] = mapped_column(String(64), nullable=False)
    tmdb_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    poster_path: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True
    )
    backdrop_path: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True
    )

    batch: Mapped["RecommendationBatch"] = relationship(back_populates="items")


class CatalogFilm(Base):
    __tablename__ = "catalog_films"
    __table_args__ = (
        UniqueConstraint("movie_id", name="uq_catalog_films_movie_id"),
        Index("ix_catalog_films_title_normalized", "title_normalized"),
        Index("ix_catalog_films_year", "year"),
        Index("ix_catalog_films_release_date", "release_date"),
        Index("ix_catalog_films_ratings_count", "ratings_count"),
        Index("ix_catalog_films_weighted_rating", "weighted_rating"),
        Index("ix_catalog_films_in_library", "in_library"),
    )

    internal_movie_id: Mapped[int] = mapped_column(
        Integer, primary_key=True
    )
    movie_id: Mapped[str] = mapped_column(String(64), nullable=False)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    title_normalized: Mapped[str] = mapped_column(
        String(512), nullable=False
    )
    year: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    release_date: Mapped[Optional[date]] = mapped_column(
        Date, nullable=True
    )
    genres: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    tmdb_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    poster_path: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True
    )
    backdrop_path: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True
    )
    ratings_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )
    rating_mean: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.0
    )
    weighted_rating: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.0
    )
    in_library: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )
    synced_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utcnow,
        server_default=text("CURRENT_TIMESTAMP"),
    )

    genre_rows: Mapped[list["CatalogFilmGenre"]] = relationship(
        back_populates="film",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class CatalogFilmGenre(Base):
    __tablename__ = "catalog_film_genres"
    __table_args__ = (
        Index("ix_catalog_film_genres_genre", "genre"),
    )

    film_id: Mapped[int] = mapped_column(
        ForeignKey(
            "catalog_films.internal_movie_id",
            ondelete="CASCADE",
        ),
        primary_key=True,
    )
    genre: Mapped[str] = mapped_column(
        String(64), primary_key=True
    )

    film: Mapped["CatalogFilm"] = relationship(
        back_populates="genre_rows"
    )
