from __future__ import annotations

import csv
import difflib
import io
import json
import re
import unicodedata
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Set, Tuple


LETTERBOXD_MIN_RATING = 0.5
LETTERBOXD_MAX_RATING = 5.0
FUZZY_THRESHOLD = 0.92


class LetterboxdImportError(RuntimeError):
    pass


def _normalize_title(value):
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.casefold()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _parse_year(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _latest_date(value):
    return str(value or "")


def _read_csv_from_zip(archive, filename):
    try:
        raw = archive.read(filename)
    except KeyError as error:
        raise LetterboxdImportError(
            f"Missing {filename} in Letterboxd archive."
        ) from error
    return list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))


def _load_movie_metadata(metadata_path):
    path = Path(metadata_path)
    if not path.exists():
        raise LetterboxdImportError(
            f"Movie metadata not found: {path}"
        )
    with open(path) as handle:
        payload = json.load(handle)
    movies = payload.get("movies") if isinstance(payload, dict) else payload
    if not movies:
        raise LetterboxdImportError(
            f"Movie metadata is empty: {path}"
        )
    return list(movies)


class _MovieMatcher:
    def __init__(self, movies):
        self.movies = list(movies)
        self.exact = {}
        self.by_title = {}
        self.by_year = {}
        self.by_source_id = {}
        self.cache = {}
        for movie in self.movies:
            title = _normalize_title(movie.get("title"))
            year = movie.get("year")
            if title:
                self.by_title.setdefault(title, []).append(movie)
            if title and year is not None:
                self.exact.setdefault((title, int(year)), []).append(movie)
                self.by_year.setdefault(int(year), []).append(
                    (title, movie)
                )
            source_ids = movie.get("source_ids") or {}
            if isinstance(source_ids, dict):
                for value in source_ids.values():
                    if value is None:
                        continue
                    self.by_source_id.setdefault(str(value), movie)
            for key in ("letterboxd_id", "tmdb_id"):
                value = movie.get(key)
                if value is not None:
                    self.by_source_id.setdefault(str(value), movie)

    @staticmethod
    def _result(movie, method, confidence=1.0):
        return {
            "movie": movie,
            "match_method": method,
            "confidence": float(confidence),
        }

    def match(self, title, year, uri=None):
        normalized = _normalize_title(title)
        year = _parse_year(year)
        cache_key = (normalized, year, str(uri or ""))
        if cache_key in self.cache:
            return self.cache[cache_key]
        result = self._match_uncached(normalized, year, uri)
        self.cache[cache_key] = result
        return result

    def _match_uncached(self, normalized, year, uri):
        if uri:
            uri_text = str(uri).strip()
            uri_key = uri_text.rstrip("/").rsplit("/", 1)[-1]
            for key in (uri_text, uri_key):
                movie = self.by_source_id.get(key)
                if movie is not None:
                    return self._result(movie, "stable_id")
        if not normalized:
            return None
        if year is not None:
            exact = self.exact.get((normalized, year), [])
            if len(exact) == 1:
                return self._result(exact[0], "title_year")
            if len(exact) > 1:
                return {
                    "ambiguous": exact,
                    "match_method": "title_year",
                }
        same_title = self.by_title.get(normalized, [])
        if year is not None:
            nearby = [
                movie for movie in same_title
                if movie.get("year") is not None
                and abs(int(movie["year"]) - year) <= 1
            ]
            if len(nearby) == 1:
                return self._result(nearby[0], "title_year_pm1")
            if len(nearby) > 1:
                return {"ambiguous": nearby, "match_method": "title_year_pm1"}
        fuzzy = []
        if year is not None:
            year_pairs = []
            for candidate_year in range(year - 1, year + 2):
                year_pairs.extend(self.by_year.get(candidate_year, []))
        else:
            year_pairs = [
                (candidate_title, movie)
                for candidate_title, candidates in self.by_title.items()
                for movie in candidates
            ]
        seen_movies = set()
        for candidate_title, movie in year_pairs:
            movie_key = int(movie["internal_id"])
            if movie_key in seen_movies:
                continue
            seen_movies.add(movie_key)
            score = difflib.SequenceMatcher(
                None, normalized, candidate_title
            ).ratio()
            if score < FUZZY_THRESHOLD:
                continue
            fuzzy.append((score, movie))
        if fuzzy:
            fuzzy.sort(key=lambda item: (-item[0], item[1].get("title", "")))
            best_score = fuzzy[0][0]
            best = [movie for score, movie in fuzzy if score == best_score]
            if len(best) == 1:
                return self._result(
                    best[0], "fuzzy_title", best_score
                )
            return {
                "ambiguous": best,
                "match_method": "fuzzy_title",
            }
        return None


def build_movie_matcher(movies):
    return _MovieMatcher(movies)


def _match_record(matcher, row):
    return matcher.match(
        row.get("Name"),
        row.get("Year"),
        row.get("Letterboxd URI"),
    )


def _movie_summary(movie):
    return {
        "internal_movie_id": int(movie["internal_id"]),
        "movie_id": movie.get("movie_id"),
        "title": movie.get("title"),
        "year": movie.get("year"),
    }


def _row_summary(row, match, method=None, confidence=None):
    result = {
        "name": row.get("Name"),
        "year": _parse_year(row.get("Year")),
        "letterboxd_uri": row.get("Letterboxd URI"),
    }
    if match is not None:
        result.update(_movie_summary(match))
    if method is not None:
        result["match_method"] = method
    if confidence is not None:
        result["confidence"] = confidence
    return result


@dataclass
class LetterboxdProfile:
    username: Optional[str]
    ratings: Dict[str, float]
    ratings_internal: Dict[int, float]
    watched: Set[str]
    watched_internal: Set[int]
    report: dict


def load_letterboxd_profile(
    zip_source,
    metadata_path=None,
    artifacts_dir=None,
    movie_metadata=None,
    matcher=None,
):
    if matcher is None:
        if movie_metadata is None:
            if metadata_path is None:
                if artifacts_dir is None:
                    artifacts_dir = (
                        Path(__file__).resolve().parents[2] / "artifacts"
                    )
                metadata_path = Path(artifacts_dir) / "movie_metadata.json"
            movie_metadata = _load_movie_metadata(metadata_path)
        matcher = _MovieMatcher(movie_metadata)
    target_max = 10.0
    target_min = 1.0
    source_max = LETTERBOXD_MAX_RATING
    multiplier = target_max / source_max
    report = {
        "source": "letterboxd",
        "rating_scale": {
            "source_min": LETTERBOXD_MIN_RATING,
            "source_max": source_max,
            "target_min": target_min,
            "target_max": target_max,
            "multiplier": multiplier,
        },
        "ratings_total": 0,
        "ratings_matched": 0,
        "ratings_unmatched": 0,
        "ratings_ambiguous": 0,
        "ratings_invalid": 0,
        "duplicates": 0,
        "watched_total": 0,
        "watched_matched": 0,
        "watched_unmatched": 0,
        "watched_ambiguous": 0,
        "matches": [],
        "unmatched_ratings": [],
        "unmatched_watched": [],
        "ambiguous": [],
        "fuzzy_matches": [],
        "duplicate_details": [],
        "warnings": [],
    }
    username = None
    ratings_by_movie = {}
    watched = set()
    watched_internal = set()
    try:
        archive = zipfile.ZipFile(zip_source)
    except (OSError, zipfile.BadZipFile) as error:
        raise LetterboxdImportError(
            f"Invalid Letterboxd ZIP archive: {zip_source!r}"
        ) from error
    with archive:
        if "ratings.csv" not in archive.namelist():
            raise LetterboxdImportError(
                "ratings.csv is required in the Letterboxd archive."
            )
        try:
            profile_rows = _read_csv_from_zip(archive, "profile.csv")
            if profile_rows:
                username = profile_rows[0].get("Username") or None
        except LetterboxdImportError:
            username = None
        rating_rows = _read_csv_from_zip(archive, "ratings.csv")
        try:
            watched_rows = _read_csv_from_zip(archive, "watched.csv")
        except LetterboxdImportError:
            watched_rows = []
            report["warnings"].append(
                "watched.csv absent: watched-only movies cannot be excluded."
            )
        report["ratings_total"] = len(rating_rows)
        if not rating_rows:
            report["warnings"].append(
                "ratings.csv is empty: using a cold-start fallback profile."
            )
        for row in rating_rows:
            raw_rating = row.get("Rating")
            try:
                rating = float(str(raw_rating).strip())
            except (TypeError, ValueError):
                report["ratings_invalid"] += 1
                report["warnings"].append(
                    f"Invalid rating for {row.get('Name')!r}: {raw_rating!r}"
                )
                continue
            if not (
                LETTERBOXD_MIN_RATING <= rating <= LETTERBOXD_MAX_RATING
            ):
                report["ratings_invalid"] += 1
                report["warnings"].append(
                    f"Rating out of range for {row.get('Name')!r}: {rating}"
                )
                continue
            result = _match_record(matcher, row)
            if result is None:
                report["ratings_unmatched"] += 1
                report["unmatched_ratings"].append(_row_summary(row, None))
                continue
            if "ambiguous" in result:
                report["ratings_ambiguous"] += 1
                report["ambiguous"].append({
                    "kind": "rating",
                    "name": row.get("Name"),
                    "year": _parse_year(row.get("Year")),
                    "match_method": result["match_method"],
                    "candidates": [
                        _movie_summary(movie)
                        for movie in result["ambiguous"]
                    ],
                })
                continue
            movie = result["movie"]
            internal = int(movie["internal_id"])
            external = movie.get("movie_id")
            normalized_rating = rating * multiplier
            previous = ratings_by_movie.get(internal)
            if previous is None:
                ratings_by_movie[internal] = (external, normalized_rating)
            else:
                report["duplicates"] += 1
                previous_date = previous[2]
                current_date = _latest_date(row.get("Date"))
                report["duplicate_details"].append({
                    "movie_id": external,
                    "previous_rating": previous[1],
                    "new_rating": normalized_rating,
                })
                if current_date >= previous_date:
                    ratings_by_movie[internal] = (
                        external, normalized_rating, current_date
                    )
                continue
            ratings_by_movie[internal] = (
                external, normalized_rating, _latest_date(row.get("Date"))
            )
            report["matches"].append(
                _row_summary(
                    row,
                    movie,
                    result["match_method"],
                    result["confidence"],
                )
            )
            if result["match_method"].startswith("fuzzy"):
                report["fuzzy_matches"].append(
                    _row_summary(
                        row,
                        movie,
                        result["match_method"],
                        result["confidence"],
                    )
                )
        for row in watched_rows:
            report["watched_total"] += 1
            result = _match_record(matcher, row)
            if result is None:
                report["watched_unmatched"] += 1
                report["unmatched_watched"].append(_row_summary(row, None))
                continue
            if "ambiguous" in result:
                report["watched_ambiguous"] += 1
                report["ambiguous"].append({
                    "kind": "watched",
                    "name": row.get("Name"),
                    "year": _parse_year(row.get("Year")),
                    "match_method": result["match_method"],
                    "candidates": [
                        _movie_summary(movie)
                        for movie in result["ambiguous"]
                    ],
                })
                continue
            movie = result["movie"]
            external = movie.get("movie_id")
            if external is not None:
                watched.add(external)
                watched_internal.add(int(movie["internal_id"]))
                report["watched_matched"] += 1
    ratings = {
        external: float(rating)
        for external, rating, _ in ratings_by_movie.values()
        if external is not None
    }
    ratings_internal = {
        internal: float(values[1])
        for internal, values in ratings_by_movie.items()
    }
    report["ratings_matched"] = len(ratings)
    report["watched_matched"] = len(watched)
    report["excluded_movie_count"] = len(
        set(ratings_internal) | set(watched_internal)
    )
    report["username"] = username
    return LetterboxdProfile(
        username=username,
        ratings=ratings,
        ratings_internal=ratings_internal,
        watched={value for value in watched if value is not None},
        watched_internal=watched_internal,
        report=report,
    )
