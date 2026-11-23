"""admin role, analytics events and dashboard indexes

Revision ID: 0003_admin_dashboard
Revises: 0002_catalog_films
Create Date: 2026-11-22 00:00:00

"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0003_admin_dashboard"
down_revision = "0002_catalog_films"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "users",
        sa.Column(
            "is_admin",
            sa.Boolean(),
            server_default=sa.text("0"),
            nullable=False,
        ),
    )
    op.create_index("ix_users_created_at", "users", ["created_at"])

    op.create_table(
        "analytics_events",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=True),
        sa.Column("event_type", sa.String(length=64), nullable=False),
        sa.Column(
            "occurred_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name="fk_analytics_events_user_id_users",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_analytics_events"),
    )
    op.create_index(
        "ix_analytics_events_user_id", "analytics_events", ["user_id"]
    )
    op.create_index(
        "ix_analytics_events_type_time",
        "analytics_events",
        ["event_type", "occurred_at"],
    )
    op.create_index(
        "ix_analytics_events_user_time",
        "analytics_events",
        ["user_id", "occurred_at"],
    )

    op.create_index("ix_sessions_created_at", "sessions", ["created_at"])
    op.create_index(
        "ix_sessions_user_created",
        "sessions",
        ["user_id", "created_at"],
    )
    op.create_index(
        "ix_recommendation_batches_status_completed",
        "recommendation_batches",
        ["status", "completed_at"],
    )
    op.create_index(
        "ix_user_films_rating_not_null",
        "user_films",
        ["rating"],
        sqlite_where=sa.text("rating IS NOT NULL"),
    )
    op.create_index(
        "ix_user_swipes_action", "user_swipes", ["action"]
    )


def downgrade() -> None:
    op.drop_index("ix_user_swipes_action", table_name="user_swipes")
    op.drop_index(
        "ix_user_films_rating_not_null", table_name="user_films"
    )
    op.drop_index(
        "ix_recommendation_batches_status_completed",
        table_name="recommendation_batches",
    )
    op.drop_index(
        "ix_sessions_user_created", table_name="sessions"
    )
    op.drop_index("ix_sessions_created_at", table_name="sessions")
    op.drop_index(
        "ix_analytics_events_user_time", table_name="analytics_events"
    )
    op.drop_index(
        "ix_analytics_events_type_time", table_name="analytics_events"
    )
    op.drop_index(
        "ix_analytics_events_user_id", table_name="analytics_events"
    )
    op.drop_table("analytics_events")
    op.drop_index("ix_users_created_at", table_name="users")
    with op.batch_alter_table("users") as batch:
        batch.drop_column("is_admin")
