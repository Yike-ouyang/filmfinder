"""initial SQLite schema

Revision ID: 0001_initial
Revises:
Create Date: 2026-11-21 00:00:00

"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("email", sa.String(length=320), nullable=False),
        sa.Column("email_normalized", sa.String(length=320), nullable=False),
        sa.Column("password_hash", sa.String(length=255), nullable=False),
        sa.Column("email_verified_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name="pk_users"),
        sa.UniqueConstraint(
            "email_normalized", name="uq_users_email_normalized"
        ),
    )

    op.create_table(
        "sessions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("token_hash", sa.String(length=64), nullable=False),
        sa.Column("csrf_token_hash", sa.String(length=64), nullable=False),
        sa.Column("user_agent", sa.String(length=255), nullable=True),
        sa.Column("ip_hash", sa.String(length=64), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name="fk_sessions_user_id_users",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_sessions"),
        sa.UniqueConstraint("token_hash", name="uq_sessions_token_hash"),
    )
    op.create_index("ix_sessions_user_id", "sessions", ["user_id"])
    op.create_index("ix_sessions_expires_at", "sessions", ["expires_at"])

    op.create_table(
        "email_tokens",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("purpose", sa.String(length=32), nullable=False),
        sa.Column("token_hash", sa.String(length=64), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("used_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "purpose IN ('verify_email', 'reset_password')",
            name="ck_email_tokens_purpose",
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name="fk_email_tokens_user_id_users",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_email_tokens"),
        sa.UniqueConstraint("token_hash", name="uq_email_tokens_token_hash"),
    )
    op.create_index("ix_email_tokens_expires_at", "email_tokens", ["expires_at"])
    op.create_index(
        "ix_email_tokens_user_purpose",
        "email_tokens",
        ["user_id", "purpose"],
    )

    op.create_table(
        "user_films",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("internal_movie_id", sa.Integer(), nullable=False),
        sa.Column("movie_id", sa.String(length=64), nullable=False),
        sa.Column("title", sa.String(length=512), nullable=False),
        sa.Column("year", sa.Integer(), nullable=True),
        sa.Column("genres", sa.JSON(), nullable=True),
        sa.Column("rating", sa.Float(), nullable=True),
        sa.Column(
            "watched",
            sa.Boolean(),
            server_default=sa.text("1"),
            nullable=False,
        ),
        sa.Column(
            "last_imported_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "rating IS NULL OR (rating >= 1.0 AND rating <= 10.0)",
            name="ck_user_films_rating_range",
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name="fk_user_films_user_id_users",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_user_films"),
        sa.UniqueConstraint(
            "user_id",
            "internal_movie_id",
            name="uq_user_films_user_movie",
        ),
    )
    op.create_index(
        "ix_user_films_user_rating", "user_films", ["user_id", "rating"]
    )

    op.create_table(
        "user_swipes",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("internal_movie_id", sa.Integer(), nullable=False),
        sa.Column("movie_id", sa.String(length=64), nullable=False),
        sa.Column("action", sa.String(length=20), nullable=False),
        sa.Column("title", sa.String(length=512), nullable=False),
        sa.Column("year", sa.Integer(), nullable=True),
        sa.Column("genres", sa.JSON(), nullable=True),
        sa.Column("tmdb_id", sa.Integer(), nullable=True),
        sa.Column("poster_path", sa.String(length=255), nullable=True),
        sa.Column("backdrop_path", sa.String(length=255), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "action IN ('interested', 'not_interested', 'already_seen')",
            name="ck_user_swipes_action",
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name="fk_user_swipes_user_id_users",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_user_swipes"),
        sa.UniqueConstraint(
            "user_id",
            "internal_movie_id",
            name="uq_user_swipes_user_movie",
        ),
    )
    op.create_index(
        "ix_user_swipes_user_action", "user_swipes", ["user_id", "action"]
    )

    op.create_table(
        "user_swipe_events",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("swipe_id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("internal_movie_id", sa.Integer(), nullable=False),
        sa.Column("action", sa.String(length=20), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column("undone_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "action IN ('interested', 'not_interested', 'already_seen')",
            name="ck_user_swipe_events_action",
        ),
        sa.ForeignKeyConstraint(
            ["swipe_id"],
            ["user_swipes.id"],
            name="fk_user_swipe_events_swipe_id_user_swipes",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name="fk_user_swipe_events_user_id_users",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_user_swipe_events"),
    )
    op.create_index(
        "ix_user_swipe_events_swipe_id", "user_swipe_events", ["swipe_id"]
    )
    op.create_index(
        "ix_user_swipe_events_user_movie",
        "user_swipe_events",
        ["user_id", "internal_movie_id", "id"],
    )

    op.create_table(
        "recommendation_batches",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column(
            "status",
            sa.String(length=16),
            server_default=sa.text("'pending'"),
            nullable=False,
        ),
        sa.Column("model", sa.String(length=32), nullable=True),
        sa.Column("algorithm", sa.String(length=32), nullable=True),
        sa.Column("error_code", sa.String(length=64), nullable=True),
        sa.Column("requested_count", sa.Integer(), nullable=True),
        sa.Column("returned_count", sa.Integer(), nullable=True),
        sa.Column("profile_summary", sa.JSON(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "status IN ('pending', 'running', 'ready', 'failed')",
            name="ck_recommendation_batches_status",
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name="fk_recommendation_batches_user_id_users",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_recommendation_batches"),
    )
    op.create_index(
        "uq_recommendation_batches_active_user",
        "recommendation_batches",
        ["user_id"],
        unique=True,
        sqlite_where=sa.text("status IN ('pending', 'running')"),
    )
    op.create_index(
        "ix_recommendation_batches_user_created",
        "recommendation_batches",
        ["user_id", "created_at"],
    )

    op.create_table(
        "recommendation_items",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("batch_id", sa.Integer(), nullable=False),
        sa.Column("rank", sa.Integer(), nullable=False),
        sa.Column("internal_movie_id", sa.Integer(), nullable=False),
        sa.Column("movie_id", sa.String(length=64), nullable=False),
        sa.Column("title", sa.String(length=512), nullable=False),
        sa.Column("year", sa.Integer(), nullable=True),
        sa.Column("genres", sa.JSON(), nullable=True),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("score_type", sa.String(length=64), nullable=False),
        sa.Column("tmdb_id", sa.Integer(), nullable=True),
        sa.Column("poster_path", sa.String(length=255), nullable=True),
        sa.Column("backdrop_path", sa.String(length=255), nullable=True),
        sa.CheckConstraint(
            "rank >= 1",
            name="ck_recommendation_items_rank",
        ),
        sa.ForeignKeyConstraint(
            ["batch_id"],
            ["recommendation_batches.id"],
            name="fk_recommendation_items_batch_id_recommendation_batches",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_recommendation_items"),
        sa.UniqueConstraint(
            "batch_id",
            "rank",
            name="uq_recommendation_items_batch_rank",
        ),
        sa.UniqueConstraint(
            "batch_id",
            "internal_movie_id",
            name="uq_recommendation_items_batch_movie",
        ),
    )
    op.create_index(
        "ix_recommendation_items_batch_rank",
        "recommendation_items",
        ["batch_id", "rank"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_recommendation_items_batch_rank",
        table_name="recommendation_items",
    )
    op.drop_table("recommendation_items")

    op.drop_index(
        "ix_recommendation_batches_user_created",
        table_name="recommendation_batches",
    )
    op.drop_index(
        "uq_recommendation_batches_active_user",
        table_name="recommendation_batches",
    )
    op.drop_table("recommendation_batches")

    op.drop_index(
        "ix_user_swipe_events_user_movie",
        table_name="user_swipe_events",
    )
    op.drop_index(
        "ix_user_swipe_events_swipe_id",
        table_name="user_swipe_events",
    )
    op.drop_table("user_swipe_events")

    op.drop_index("ix_user_swipes_user_action", table_name="user_swipes")
    op.drop_table("user_swipes")

    op.drop_index("ix_user_films_user_rating", table_name="user_films")
    op.drop_table("user_films")

    op.drop_index("ix_email_tokens_user_purpose", table_name="email_tokens")
    op.drop_index("ix_email_tokens_expires_at", table_name="email_tokens")
    op.drop_table("email_tokens")

    op.drop_index("ix_sessions_expires_at", table_name="sessions")
    op.drop_index("ix_sessions_user_id", table_name="sessions")
    op.drop_table("sessions")

    op.drop_table("users")
