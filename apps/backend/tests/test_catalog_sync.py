from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import select

from app.db.base import build_session_factory, create_db_engine
from app.db.models import CatalogFilm, CatalogFilmGenre


BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = (
    REPO_ROOT / "packages" / "recommendation-engine" / "artifacts"
)
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from scripts.sync_film_catalog import (  # noqa: E402
    build_catalog_rows,
    write_catalog,
)


def _alembic_config(database_url: str) -> Config:
    config = Config(str(BACKEND_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(BACKEND_ROOT / "migrations"))
    config.set_main_option("sqlalchemy.url", database_url)
    return config


def _movies():
    return json.loads(
        (ARTIFACTS / "movie_metadata.json").read_text()
    )["movies"]


def _csv_fixture(path: Path) -> dict:
    movies = _movies()
    selected = {
        0: [
            ("Short overview", "1994-12-21"),
            ("A much longer overview for the first film", "1994-12-21"),
        ],
        1: [("", "")],
        2: [("Sunset overview", "2016-01-01")],
        25: [("Post 2019 overview", "2020-05-01")],
    }
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "_id",
                "movie_id",
                "overview",
                "release_date",
                "vote_count",
                "vote_average",
            ]
        )
        for index, rows in selected.items():
            movie = movies[index]
            for overview, release_date in rows:
                writer.writerow(
                    [
                        movie["source_ids"]["_id"],
                        movie["movie_id"],
                        overview,
                        release_date,
                        "10",
                        "7.5",
                    ]
                )
    return {movies[index]["internal_id"]: movies[index] for index in selected}


def test_catalog_builder_uses_best_summary_and_temporal_flag(tmp_path):
    csv_path = tmp_path / "catalogue.csv"
    expected = _csv_fixture(csv_path)

    rows, genre_rows, report = build_catalog_rows(
        ARTIFACTS, csv_path, limit=26
    )
    by_id = {row["internal_movie_id"]: row for row in rows}

    first = by_id[expected[0]["internal_id"]]
    assert first["summary"] == "A much longer overview for the first film"
    assert first["release_date"].isoformat() == "1994-12-21"
    assert first["in_library"] is True
    assert first["ratings_count"] > 0
    assert first["title_normalized"] == "00 schneider jagd auf nihil baxter"

    missing = by_id[expected[1]["internal_id"]]
    assert missing["summary"] is None
    assert missing["release_date"] is None
    assert missing["in_library"] is True

    post = by_id[expected[25]["internal_id"]]
    assert post["in_library"] is False
    assert post["year"] == 2020

    assert report["processed"] == 26
    assert report["post_2019"] == 1
    assert report["in_library"] == 25
    assert report["csv_matched"] == 5
    assert any(
        row["film_id"] == expected[0]["internal_id"]
        for row in genre_rows
    )


def test_write_catalog_is_idempotent_and_replaces_previous_rows(tmp_path):
    csv_path = tmp_path / "catalogue.csv"
    _csv_fixture(csv_path)
    database_url = f"sqlite+pysqlite:///{tmp_path / 'catalog.db'}"
    command.upgrade(_alembic_config(database_url), "head")
    engine = create_db_engine(database_url)
    session_factory = build_session_factory(engine)
    try:
        rows, genre_rows, _ = build_catalog_rows(
            ARTIFACTS, csv_path, limit=5
        )
        write_catalog(session_factory, rows, genre_rows)
        with session_factory() as session:
            assert len(session.scalars(select(CatalogFilm)).all()) == 5
            assert len(
                session.scalars(select(CatalogFilmGenre)).all()
            ) > 0

        rows, genre_rows, _ = build_catalog_rows(
            ARTIFACTS, csv_path, limit=2
        )
        write_catalog(session_factory, rows, genre_rows)
        with session_factory() as session:
            assert len(session.scalars(select(CatalogFilm)).all()) == 2
    finally:
        engine.dispose()
