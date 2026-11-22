#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
from sqlalchemy import delete, insert

from filmfinder.features import ArtifactStore
from filmfinder.letterboxd_import import _normalize_title


BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.config import get_settings  # noqa: E402
from app.db.base import (  # noqa: E402
    build_session_factory,
    create_db_engine,
)
from app.db.models import CatalogFilm, CatalogFilmGenre  # noqa: E402
from app.services.catalogue_source import (  # noqa: E402
    build_catalogue_index,
    detect_column,
    load_catalogue,
    match_internal,
)


DEFAULT_ARTIFACTS = (
    REPO_ROOT / "packages" / "recommendation-engine" / "artifacts"
)
DEFAULT_CSV = REPO_ROOT / "movie_data.csv"
CATALOG_YEAR_LIMIT = 2019
WEIGHTED_RATING_PRIOR = 100.0
CHUNK_SIZE = 5000


def _parse_date(value) -> date | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def _should_replace(current: dict | None, candidate: dict) -> bool:
    if current is None:
        return True
    return candidate["score"] > current["score"]


def _select_csv_rows(
    csv_path: Path,
    movies: list[dict],
) -> tuple[dict[int, dict], dict]:
    by_source, by_movie_id = build_catalogue_index(movies)
    report = {
        "csv_rows": 0,
        "csv_matched": 0,
        "csv_unmatched": 0,
    }
    best: dict[int, dict] = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        id_column = detect_column(
            fieldnames, ["_id", "movie_id", "letterboxd_id"]
        )
        movie_column = detect_column(
            fieldnames, ["movie_id", "slug", "letterboxd_slug"]
        )
        for row in reader:
            report["csv_rows"] += 1
            internal = match_internal(
                row,
                by_source,
                by_movie_id,
                id_column,
                movie_column,
            )
            if internal is None:
                report["csv_unmatched"] += 1
                continue
            report["csv_matched"] += 1
            overview = str(row.get("overview") or "").strip()
            release = _parse_date(row.get("release_date"))
            try:
                vote_count = int(float(row.get("vote_count") or 0))
            except (TypeError, ValueError):
                vote_count = 0
            try:
                vote_average = float(row.get("vote_average") or 0)
            except (TypeError, ValueError):
                vote_average = 0.0
            candidate = {
                "score": (
                    len(overview),
                    release is not None,
                    release or date.min,
                    vote_count,
                    vote_average,
                ),
                "summary": overview or None,
                "release_date": release,
            }
            if _should_replace(best.get(internal), candidate):
                best[internal] = candidate
    return best, report


def _rating_metrics(store: ArtifactStore) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    ratings = store.ratings.tocsr()
    counts = np.asarray(
        ratings.getnnz(axis=0), dtype=np.int64
    ).ravel()
    sums = np.asarray(ratings.sum(axis=0)).ravel().astype(np.float64)
    means = np.divide(
        sums,
        counts,
        out=np.zeros_like(sums),
        where=counts > 0,
    )
    total = (
        float(np.asarray(ratings.sum()).ravel()[0])
        if ratings.nnz
        else 0.0
    )
    global_mean = total / ratings.nnz if ratings.nnz else 0.0
    weighted = (
        (counts / (counts + WEIGHTED_RATING_PRIOR)) * means
        + (
            WEIGHTED_RATING_PRIOR
            / (counts + WEIGHTED_RATING_PRIOR)
        )
        * global_mean
    )
    return counts, means, weighted, global_mean


def _load_tmdb(path: Path) -> dict:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    movies = payload.get("movies") if isinstance(payload, dict) else payload
    return movies if isinstance(movies, dict) else {}


def build_catalog_rows(
    artifacts: Path,
    catalogue_csv: Path,
    limit: int | None = None,
) -> tuple[list[dict], list[dict], dict]:
    store = ArtifactStore.load(artifacts)
    tmdb = _load_tmdb(artifacts / "tmdb_metadata.json")
    csv_rows, csv_report = _select_csv_rows(catalogue_csv, store.movies)
    counts, means, weighted, global_mean = _rating_metrics(store)

    rows = []
    genre_rows = []
    report = {
        "catalog_movies": len(store.movies),
        "processed": 0,
        "in_library": 0,
        "post_2019": 0,
        "with_summary": 0,
        "with_release_date": 0,
        "with_poster": 0,
        "with_backdrop": 0,
        "genres": 0,
        "global_mean_rating": round(global_mean, 4),
        "weighted_rating_prior": WEIGHTED_RATING_PRIOR,
        **csv_report,
    }
    processed = 0
    seen_genres = set()
    for index, movie in enumerate(store.movies):
        if limit is not None and processed >= int(limit):
            break
        internal = int(movie["internal_id"])
        movie_id = str(movie.get("movie_id") or "")
        if not movie_id:
            continue
        title = str(movie.get("title") or movie_id)
        year = movie.get("year")
        year = int(year) if year is not None else None
        source = csv_rows.get(internal) or {}
        metadata = tmdb.get(str(internal)) or {}
        genres = list(movie.get("genres") or [])
        summary = source.get("summary")
        release_date = source.get("release_date")
        in_library = year is not None and year <= CATALOG_YEAR_LIMIT
        rows.append({
            "internal_movie_id": internal,
            "movie_id": movie_id,
            "title": title,
            "title_normalized": _normalize_title(title),
            "year": year,
            "release_date": release_date,
            "genres": genres,
            "summary": summary,
            "tmdb_id": metadata.get("tmdb_id"),
            "poster_path": metadata.get("poster_path"),
            "backdrop_path": metadata.get("backdrop_path"),
            "ratings_count": int(counts[internal]),
            "rating_mean": float(means[internal]),
            "weighted_rating": float(weighted[internal]),
            "in_library": in_library,
        })
        for genre in dict.fromkeys(genres):
            genre_rows.append({
                "film_id": internal,
                "genre": str(genre),
            })
            seen_genres.add(str(genre))
        processed += 1
        report["processed"] = processed
        report["in_library"] += int(in_library)
        report["post_2019"] += int(
            year is not None and year > CATALOG_YEAR_LIMIT
        )
        report["with_summary"] += int(bool(summary))
        report["with_release_date"] += int(release_date is not None)
        report["with_poster"] += int(bool(metadata.get("poster_path")))
        report["with_backdrop"] += int(bool(metadata.get("backdrop_path")))
    report["genres"] = len(seen_genres)
    return rows, genre_rows, report


def _chunks(values: list[dict]):
    for start in range(0, len(values), CHUNK_SIZE):
        yield values[start : start + CHUNK_SIZE]


def write_catalog(session_factory, rows, genre_rows) -> None:
    with session_factory() as session:
        session.execute(delete(CatalogFilmGenre))
        session.execute(delete(CatalogFilm))
        for chunk in _chunks(rows):
            session.execute(insert(CatalogFilm), chunk)
        for chunk in _chunks(genre_rows):
            session.execute(insert(CatalogFilmGenre), chunk)
        session.commit()


def sync(
    artifacts: Path,
    catalogue_csv: Path,
    database_url: str | None = None,
    dry_run: bool = False,
    limit: int | None = None,
) -> dict:
    rows, genre_rows, report = build_catalog_rows(
        artifacts, catalogue_csv, limit
    )
    if not dry_run:
        settings = get_settings()
        url = database_url or settings.database_url
        engine = create_db_engine(
            url,
            settings.sqlite_busy_timeout_ms,
        )
        try:
            write_catalog(build_session_factory(engine), rows, genre_rows)
        finally:
            engine.dispose()
    report["dry_run"] = bool(dry_run)
    report["written_rows"] = 0 if dry_run else len(rows)
    report["written_genres"] = 0 if dry_run else len(genre_rows)
    return report


def parse_args():
    parser = argparse.ArgumentParser(
        description="Synchronize the FilmFinder film catalogue into SQLite."
    )
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=DEFAULT_ARTIFACTS,
        help="Production artifacts directory.",
    )
    parser.add_argument(
        "--catalogue-csv",
        type=Path,
        default=DEFAULT_CSV,
        help="Raw catalogue CSV containing overview and release_date.",
    )
    parser.add_argument(
        "--database-url",
        default=None,
        help="SQLAlchemy database URL (default: FILMFINDER_DB_PATH).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of catalogue movies processed.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and report without writing to SQLite.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.catalogue_csv.exists():
        raise SystemExit(
            "Catalogue CSV not found: "
            f"{args.catalogue_csv}. Copy movie_data.csv to the repository "
            "root or pass --catalogue-csv."
        )
    report = sync(
        args.artifacts,
        args.catalogue_csv,
        database_url=args.database_url,
        dry_run=args.dry_run,
        limit=args.limit,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
