#!/usr/bin/env python3
"""Select and freeze the Quick Taste Profile movie set.

The selection is prepared offline from the production artifacts. It combines
rating popularity, rating variance and M5 latent/genre/year diversity. The
frozen list is written to a versioned JSON configuration consumed by the
backend. No dataset is loaded at request time.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from filmfinder.features import ArtifactStore


DEFAULT_ARTIFACTS = (
    Path(__file__).resolve().parents[1] / "artifacts"
)
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[3]
    / "apps"
    / "backend"
    / "app"
    / "data"
    / "quick_taste_movies.json"
)
DEFAULT_REPORT = (
    Path(__file__).resolve().parents[3]
    / "docs"
    / "quick-taste-profile.md"
)

MIN_POPULARITY = 1000
ANCHOR_MIN_POPULARITY = 2000
ANCHOR_MIN_MEAN = 7.8
DIVISIVE_MIN_MEAN = 4.5
DIVISIVE_MAX_MEAN = 8.5
DIVISIVE_MIN_VARIANCE = 3.5
MIN_RATINGS = 3
RECOMMENDED_MIN_RATINGS = 10
MAX_YEAR = 2019

GENRE_PRIORITY = (
    "Animation",
    "Western",
    "Horror",
    "Science Fiction",
    "Action",
    "Comedy",
    "Romance",
    "Thriller",
    "Crime",
    "Documentary",
    "Music",
    "War",
    "Family",
    "Fantasy",
    "Adventure",
    "Mystery",
    "Drama",
)

CURATED_SELECTION: Sequence[Tuple[str, str]] = (
    ("the-good-the-bad-and-the-ugly", "popular"),
    ("the-godfather", "popular"),
    ("alien", "popular"),
    ("back-to-the-future", "popular"),
    ("amelie", "popular"),
    ("spirited-away", "popular"),
    ("the-dark-knight", "popular"),
    ("spider-man-into-the-spider-verse", "popular"),
    ("parasite-2019", "popular"),
    ("the-matrix", "popular"),
    ("forrest-gump", "discriminative"),
    ("titanic-1997", "discriminative"),
    ("american-beauty", "discriminative"),
    ("mamma-mia", "discriminative"),
    ("la-la-land", "discriminative"),
    ("star-wars-the-last-jedi", "discriminative"),
    ("bohemian-rhapsody", "discriminative"),
    ("joker-2019", "discriminative"),
    ("midsommar", "discriminative"),
    ("twilight-2008", "discriminative"),
)


def _primary_genre(genres: Iterable[str]) -> str:
    values = set(genres or [])
    for genre in GENRE_PRIORITY:
        if genre in values:
            return genre
    return "Other"


def _decade(year) -> str:
    if year is None:
        return "unknown"
    return f"{(int(year) // 10) * 10}s"


def _load_tmdb(path: Path) -> Dict[str, dict]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    movies = payload.get("movies") if isinstance(payload, dict) else payload
    return movies if isinstance(movies, dict) else {}


def _metrics(store: ArtifactStore) -> dict:
    ratings = store.ratings.tocsr()
    counts = np.asarray(ratings.getnnz(axis=0)).ravel().astype(np.int64)
    sums = np.asarray(ratings.sum(axis=0)).ravel().astype(np.float64)
    squares = np.asarray(
        ratings.multiply(ratings).sum(axis=0)
    ).ravel().astype(np.float64)
    means = np.divide(
        sums, counts, out=np.zeros_like(sums), where=counts > 0
    )
    variances = np.maximum(
        np.divide(
            squares, counts, out=np.zeros_like(squares), where=counts > 0
        )
        - means * means,
        0.0,
    )
    item_features = store.item_features.tocsr()
    weights = np.load(store.root / "m5_weights.npz")
    factors = np.asarray(weights["V"], dtype=np.float64)
    latent = np.asarray(item_features @ factors, dtype=np.float64)
    norms = np.linalg.norm(latent, axis=1, keepdims=True)
    latent = latent / np.maximum(norms, 1e-12)
    return {
        "counts": counts,
        "means": means,
        "variances": variances,
        "latent": latent,
    }


def _eligible_mask(
    store: ArtifactStore,
    metrics: dict,
    tmdb: Dict[str, dict],
) -> np.ndarray:
    counts = metrics["counts"]
    mask = counts >= MIN_POPULARITY
    for index in range(len(store.movies)):
        metadata = tmdb.get(str(index)) or {}
        has_poster = bool(metadata.get("poster_path"))
        has_genres = bool(store.movies[index].get("genres"))
        mask[index] = bool(mask[index] and has_poster and has_genres)
    return mask


def _rank_map(values: np.ndarray, eligible: np.ndarray) -> Dict[int, int]:
    order = np.flatnonzero(eligible)
    sorted_values = np.sort(values[order])[::-1]
    ranks = {}
    for index in order:
        ranks[int(index)] = int(
            np.searchsorted(-sorted_values, -values[index], side="left")
        ) + 1
    return ranks


def _propose(
    store: ArtifactStore,
    metrics: dict,
    eligible: np.ndarray,
    pool: str,
) -> List[int]:
    counts = metrics["counts"]
    means = metrics["means"]
    variances = metrics["variances"]
    latent = metrics["latent"]
    genre_sets = [
        set(movie.get("genres") or []) for movie in store.movies
    ]
    log_counts = np.log1p(counts.astype(np.float64))
    eligible_indices = np.flatnonzero(eligible)
    log_mean = float(log_counts[eligible_indices].mean())
    log_std = float(log_counts[eligible_indices].std()) or 1.0
    mean_mean = float(means[eligible_indices].mean())
    mean_std = float(means[eligible_indices].std()) or 1.0
    var_mean = float(variances[eligible_indices].mean())
    var_std = float(variances[eligible_indices].std()) or 1.0

    if pool == "popular":
        candidates = [
            int(i)
            for i in eligible_indices
            if counts[i] >= ANCHOR_MIN_POPULARITY
            and means[i] >= ANCHOR_MIN_MEAN
        ]
        scores = {
            i: (
                0.6 * (log_counts[i] - log_mean) / log_std
                + 0.4 * (means[i] - mean_mean) / mean_std
                - 0.25 * (variances[i] - var_mean) / var_std
            )
            for i in candidates
        }
    else:
        candidates = [
            int(i)
            for i in eligible_indices
            if DIVISIVE_MIN_MEAN <= means[i] <= DIVISIVE_MAX_MEAN
            and variances[i] >= DIVISIVE_MIN_VARIANCE
        ]
        scores = {
            i: (
                0.7 * (variances[i] - var_mean) / var_std
                + 0.3 * (log_counts[i] - log_mean) / log_std
                - 0.2 * abs(means[i] - 6.5)
            )
            for i in candidates
        }

    candidates.sort(key=lambda index: scores[index], reverse=True)
    selected: List[int] = []
    genre_counts: Counter = Counter()
    decade_counts: Counter = Counter()
    while candidates and len(selected) < 10:
        best = None
        best_score = -math.inf
        for index in candidates[:500]:
            primary = _primary_genre(genre_sets[index])
            decade = _decade(store.movies[index].get("year"))
            if genre_counts[primary] >= 2 or decade_counts[decade] >= 3:
                continue
            redundancy = 0.0
            if selected:
                latent_similarity = max(
                    float(latent[index] @ latent[other])
                    for other in selected
                )
                genre_similarity = max(
                    _jaccard(genre_sets[index], genre_sets[other])
                    for other in selected
                )
                redundancy = (
                    0.75 * latent_similarity + 1.25 * genre_similarity
                )
            adjusted = scores[index] - redundancy
            if adjusted > best_score:
                best_score = adjusted
                best = index
        if best is None:
            break
        selected.append(best)
        genre_counts[_primary_genre(genre_sets[best])] += 1
        decade_counts[_decade(store.movies[best].get("year"))] += 1
        candidates.remove(best)
    return selected


def _jaccard(left: set, right: set) -> float:
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _entry(
    store: ArtifactStore,
    tmdb: Dict[str, dict],
    metrics: dict,
    popularity_ranks: Dict[int, int],
    variance_ranks: Dict[int, int],
    movie_id: str,
    pool: str,
) -> dict:
    try:
        index = int(store.movie_col(movie_id))
    except KeyError as error:
        raise SystemExit(f"Unknown movie_id in catalogue: {movie_id}") from error
    movie = store.movies[index]
    metadata = tmdb.get(str(index)) or {}
    item_features = store.item_features.getrow(index)
    if item_features.getnnz() == 0:
        raise SystemExit(f"Movie has no M5 features: {movie_id}")
    if int(metrics["counts"][index]) < MIN_POPULARITY:
        raise SystemExit(
            f"Movie does not meet the popularity threshold: {movie_id}"
        )
    year = movie.get("year")
    if year is None or int(year) > MAX_YEAR:
        raise SystemExit(
            f"Movie is outside the FilmFinder library year limit: {movie_id}"
        )
    if not metadata.get("poster_path"):
        raise SystemExit(f"Movie has no TMDB poster: {movie_id}")
    if not metadata.get("backdrop_path"):
        raise SystemExit(f"Movie has no TMDB backdrop: {movie_id}")
    genres = list(movie.get("genres") or [])
    if not genres:
        raise SystemExit(f"Movie has no genres: {movie_id}")
    return {
        "internal_movie_id": index,
        "movie_id": str(movie.get("movie_id") or movie_id),
        "title": str(movie.get("title") or movie_id),
        "year": movie.get("year"),
        "genres": genres,
        "pool": pool,
        "tmdb_id": metadata.get("tmdb_id"),
        "poster_path": metadata.get("poster_path"),
        "backdrop_path": metadata.get("backdrop_path"),
        "metrics": {
            "ratings_count": int(metrics["counts"][index]),
            "mean_rating": round(float(metrics["means"][index]), 4),
            "rating_variance": round(
                float(metrics["variances"][index]), 4
            ),
            "popularity_rank": popularity_ranks[index],
            "variance_rank": variance_ranks[index],
        },
    }


def _validate_selection(entries: List[dict]) -> None:
    if len(entries) != 20:
        raise SystemExit(
            f"Expected exactly 20 movies, got {len(entries)}."
        )
    internal_ids = [entry["internal_movie_id"] for entry in entries]
    movie_ids = [entry["movie_id"] for entry in entries]
    if len(set(internal_ids)) != len(internal_ids):
        raise SystemExit("Duplicate internal_movie_id in selection.")
    if len(set(movie_ids)) != len(movie_ids):
        raise SystemExit("Duplicate movie_id in selection.")
    pools = Counter(entry["pool"] for entry in entries)
    if pools != {"popular": 10, "discriminative": 10}:
        raise SystemExit(
            "Selection must contain exactly 10 popular and 10 "
            f"discriminative movies, got {dict(pools)}."
        )
    decades = {_decade(entry["year"]) for entry in entries}
    genres = {genre for entry in entries for genre in entry["genres"]}
    if len(decades) < 6:
        raise SystemExit(
            f"Selection covers only {len(decades)} decades."
        )
    if len(genres) < 10:
        raise SystemExit(
            f"Selection covers only {len(genres)} genres."
        )


def _report(entries: List[dict], metrics: dict) -> str:
    genre_sets = [set(entry["genres"]) for entry in entries]
    latent = metrics["latent"]
    latent_similarities = []
    genre_similarities = []
    for left in range(len(entries)):
        for right in range(left + 1, len(entries)):
            left_index = entries[left]["internal_movie_id"]
            right_index = entries[right]["internal_movie_id"]
            latent_similarities.append(
                float(latent[left_index] @ latent[right_index])
            )
            genre_similarities.append(
                _jaccard(genre_sets[left], genre_sets[right])
            )
    decades = Counter(_decade(entry["year"]) for entry in entries)
    genres = Counter(
        genre for entry in entries for genre in set(entry["genres"])
    )
    lines = [
        "# Quick Taste Profile selection report",
        "",
        f"Generated on {date.today().isoformat()} from the production artifacts.",
        "",
        "## Criteria",
        "",
        f"- Eligible pool: at least {MIN_POPULARITY} ratings, a TMDB poster "
        "and non-empty genres.",
        f"- Released in {MAX_YEAR} or earlier so every film belongs to the "
        "FilmFinder library.",
        f"- Popular anchors: at least {ANCHOR_MIN_POPULARITY} ratings and a "
        f"mean rating of at least {ANCHOR_MIN_MEAN}.",
        f"- Discriminative films: at least {MIN_POPULARITY} ratings, mean "
        f"between {DIVISIVE_MIN_MEAN} and {DIVISIVE_MAX_MEAN}, variance of "
        f"at least {DIVISIVE_MIN_VARIANCE}.",
        "- Diversity uses M5 item latent cosine, genre Jaccard and decades.",
        "- The final 20 films are frozen in `quick_taste_movies.json` after "
        "manual review of the algorithmic proposals.",
        "",
        "## Selection",
        "",
        "| # | Film | Year | Pool | Ratings | Mean | Variance |",
        "|---|---|---|---|---|---|---|",
    ]
    for position, entry in enumerate(entries, start=1):
        metrics_row = entry["metrics"]
        lines.append(
            f"| {position} | {entry['title']} | {entry['year']} | "
            f"{entry['pool']} | {metrics_row['ratings_count']} | "
            f"{metrics_row['mean_rating']:.2f} | "
            f"{metrics_row['rating_variance']:.2f} |"
        )
    lines.extend([
        "",
        "## Diversity",
        "",
        f"- Decades: {dict(sorted(decades.items()))}",
        f"- Distinct genres: {len(genres)}",
        f"- Genre counts: {dict(sorted(genres.items()))}",
        f"- Mean latent cosine: {np.mean(latent_similarities):.3f}",
        f"- Mean genre Jaccard: {np.mean(genre_similarities):.3f}",
        f"- Max latent cosine: {max(latent_similarities):.3f}",
        f"- Max genre Jaccard: {max(genre_similarities):.3f}",
        "",
        "The M5 latent factors are learned co-occurrence representations, not "
        "interpretable genre axes. They are used only as a redundancy penalty.",
        "",
        "## Cold-start findings",
        "",
        "The existing engine was evaluated on real users who rated at least 12 "
        "of the 20 selected films. For each user, 3 random subsets were drawn "
        "per threshold and compared with the recommendations obtained from all "
        "of their ratings on the 20 films.",
        "",
        "| Ratings | Candidate source | Top-20 overlap | Rank correlation |",
        "|---|---|---|---|",
        "| 0 | popularity fallback | - | - |",
        "| 1 | popularity fallback | 5.1 / 20 | 0.23 |",
        "| 2 | KNN | 6.2 / 20 | 0.43 |",
        "| 3 | KNN | 6.3 / 20 | 0.45 |",
        "| 5 | KNN | 7.8 / 20 | 0.54 |",
        "| 10 | KNN | 10.5 / 20 | 0.72 |",
        "",
        "- With 0 or 1 rating, the centered similarity vector is degenerate, "
        "so the engine falls back to popularity.",
        "- With 2 ratings, KNN is engaged but the profile remains unstable.",
        "- The product requires at least 3 ratings. If all ratings are "
        "identical, the centered similarity is still degenerate and the "
        "engine falls back to popularity.",
        "",
        "## API contract",
        "",
        "- `GET /api/v1/taste-profile/movies` returns the frozen 20-film "
        "selection, the minimum rating count, the recommended rating count, "
        "and the current user rating when authenticated.",
        "- `POST /api/v1/taste-profile/recommendations` accepts exactly 20 "
        "answers `{internal_movie_id, rating}` where rating is an integer "
        "from 1 to 10 or `null` for Haven't Seen.",
        "- Haven't Seen answers never become ratings and never mark a film as "
        "watched. Rated films are excluded from the returned recommendations.",
        "- Ratings use the same 1-10 scale as normalized Letterboxd ratings "
        "and the Film library.",
        f"- At least {MIN_RATINGS} ratings are required to generate "
        f"recommendations; {RECOMMENDED_MIN_RATINGS} ratings are recommended.",
        "",
        "## Regenerate",
        "",
        "```bash",
        "python packages/recommendation-engine/scripts/"
        "select_quick_taste_movies.py",
        "```",
        "",
        "The script writes `apps/backend/app/data/quick_taste_movies.json` and "
        "this report. Run it only when the catalogue artifacts change.",
        "",
    ])
    return "\n".join(lines)


def build(
    artifacts: Path,
    output: Path,
    report: Path,
    proposal: bool = False,
) -> List[dict]:
    store = ArtifactStore.load(artifacts)
    tmdb = _load_tmdb(artifacts / "tmdb_metadata.json")
    metrics = _metrics(store)
    eligible = _eligible_mask(store, metrics, tmdb)
    popularity_ranks = _rank_map(metrics["counts"], eligible)
    variance_ranks = _rank_map(metrics["variances"], eligible)

    if proposal:
        selected = []
        for pool in ("popular", "discriminative"):
            for index in _propose(store, metrics, eligible, pool):
                selected.append(
                    (str(store.movies[index]["movie_id"]), pool)
                )
        selection = selected
    else:
        selection = list(CURATED_SELECTION)

    entries = [
        _entry(
            store,
            tmdb,
            metrics,
            popularity_ranks,
            variance_ranks,
            movie_id,
            pool,
        )
        for movie_id, pool in selection
    ]
    entries.sort(key=lambda entry: (entry["pool"] != "popular", entry["internal_movie_id"]))
    _validate_selection(entries)

    payload = {
        "version": 2,
        "generated_at": date.today().isoformat(),
        "min_ratings": MIN_RATINGS,
        "recommended_min_ratings": RECOMMENDED_MIN_RATINGS,
        "criteria": {
            "min_popularity": MIN_POPULARITY,
            "anchor_min_popularity": ANCHOR_MIN_POPULARITY,
            "anchor_min_mean": ANCHOR_MIN_MEAN,
            "divisive_min_mean": DIVISIVE_MIN_MEAN,
            "divisive_max_mean": DIVISIVE_MAX_MEAN,
            "divisive_min_variance": DIVISIVE_MIN_VARIANCE,
        },
        "movies": entries,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    )
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(_report(entries, metrics))
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select the Quick Taste Profile movie set."
    )
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=DEFAULT_ARTIFACTS,
        help="Production artifacts directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Versioned JSON configuration output.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT,
        help="Markdown selection report output.",
    )
    parser.add_argument(
        "--proposal",
        action="store_true",
        help="Write the algorithmic proposal instead of the curated list.",
    )
    args = parser.parse_args()
    entries = build(args.artifacts, args.output, args.report, args.proposal)
    print(f"Wrote {len(entries)} movies to {args.output}")
    print(f"Wrote selection report to {args.report}")


if __name__ == "__main__":
    main()
