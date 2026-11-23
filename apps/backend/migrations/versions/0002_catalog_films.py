"""film catalogue tables

Revision ID: 0002_catalog_films
Revises: 0001_initial
Create Date: 2026-11-22 00:00:00

"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0002_catalog_films"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "catalog_films",
        sa.Column("internal_movie_id", sa.Integer(), nullable=False),
        sa.Column("movie_id", sa.String(length=64), nullable=False),
        sa.Column("title", sa.String(length=512), nullable=False),
        sa.Column(
            "title_normalized", sa.String(length=512), nullable=False
        ),
        sa.Column("year", sa.Integer(), nullable=True),
        sa.Column("release_date", sa.Date(), nullable=True),
        sa.Column("genres", sa.JSON(), nullable=True),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("tmdb_id", sa.Integer(), nullable=True),
        sa.Column(
            "poster_path", sa.String(length=255), nullable=True
        ),
        sa.Column(
            "backdrop_path", sa.String(length=255), nullable=True
        ),
        sa.Column(
            "ratings_count",
            sa.Integer(),
            server_default=sa.text("0"),
            nullable=False,
        ),
        sa.Column(
            "rating_mean",
            sa.Float(),
            server_default=sa.text("0"),
            nullable=False,
        ),
        sa.Column(
            "weighted_rating",
            sa.Float(),
            server_default=sa.text("0"),
            nullable=False,
        ),
        sa.Column(
            "in_library",
            sa.Boolean(),
            server_default=sa.text("0"),
            nullable=False,
        ),
        sa.Column(
            "synced_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint(
            "internal_movie_id", name="pk_catalog_films"
        ),
        sa.UniqueConstraint(
            "movie_id", name="uq_catalog_films_movie_id"
        ),
    )
    op.create_index(
        "ix_catalog_films_title_normalized",
        "catalog_films",
        ["title_normalized"],
    )
    op.create_index(
        "ix_catalog_films_year", "catalog_films", ["year"]
    )
    op.create_index(
        "ix_catalog_films_release_date",
        "catalog_films",
        ["release_date"],
    )
    op.create_index(
        "ix_catalog_films_ratings_count",
        "catalog_films",
        ["ratings_count"],
    )
    op.create_index(
        "ix_catalog_films_weighted_rating",
        "catalog_films",
        ["weighted_rating"],
    )
    op.create_index(
        "ix_catalog_films_in_library",
        "catalog_films",
        ["in_library"],
    )

    op.create_table(
        "catalog_film_genres",
        sa.Column("film_id", sa.Integer(), nullable=False),
        sa.Column("genre", sa.String(length=64), nullable=False),
        sa.ForeignKeyConstraint(
            ["film_id"],
            ["catalog_films.internal_movie_id"],
            name="fk_catalog_film_genres_film_id_catalog_films",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "film_id", "genre", name="pk_catalog_film_genres"
        ),
    )
    op.create_index(
        "ix_catalog_film_genres_genre",
        "catalog_film_genres",
        ["genre"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_catalog_film_genres_genre",
        table_name="catalog_film_genres",
    )
    op.drop_table("catalog_film_genres")

    op.drop_index(
        "ix_catalog_films_in_library",
        table_name="catalog_films",
    )
    op.drop_index(
        "ix_catalog_films_weighted_rating",
        table_name="catalog_films",
    )
    op.drop_index(
        "ix_catalog_films_ratings_count",
        table_name="catalog_films",
    )
    op.drop_index(
        "ix_catalog_films_release_date",
        table_name="catalog_films",
    )
    op.drop_index(
        "ix_catalog_films_year", table_name="catalog_films"
    )
    op.drop_index(
        "ix_catalog_films_title_normalized",
        table_name="catalog_films",
    )
    op.drop_table("catalog_films")
