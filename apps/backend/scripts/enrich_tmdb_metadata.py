from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from pathlib import Path

import httpx


BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parents[1]
DEFAULT_ARTIFACTS = REPO_ROOT / "packages" / "recommendation-engine" / "artifacts"
DEFAULT_OUTPUT = DEFAULT_ARTIFACTS / "tmdb_metadata.json"
DEFAULT_CSV = REPO_ROOT / "movie_data.csv"
TMDB_URL_RE = re.compile(r"/movie/(\d+)")


def _read_env_value(path, key):
    path = Path(path)
    if not path.exists():
        return None
    for line in path.read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        if name.strip() == key:
            return value.strip().strip('"').strip("'")
    return None


def _api_key(explicit):
    if explicit:
        return explicit
    value = os.environ.get("TMDB_API_KEY")
    if value:
        return value
    for path in (BACKEND_ROOT / ".env", REPO_ROOT / ".env"):
        value = _read_env_value(path, "TMDB_API_KEY")
        if value:
            return value
    return None


def _load_catalogue(path):
    payload = json.loads(Path(path).read_text())
    movies = payload.get("movies") if isinstance(payload, dict) else payload
    if not movies:
        raise ValueError(f"No movies found in {path}")
    return list(movies)


def _catalogue_index(movies):
    by_source = {}
    by_movie_id = {}
    for movie in movies:
        internal = int(movie["internal_id"])
        movie_id = movie.get("movie_id")
        if movie_id is not None:
            by_movie_id[str(movie_id)] = internal
        for value in (movie.get("source_ids") or {}).values():
            if value is not None:
                by_source[str(value)] = internal
    return by_source, by_movie_id


def _field(row, name):
    if name is None:
        return None
    return row.get(name)


def _detect_column(fieldnames, candidates):
    for candidate in candidates:
        if candidate in fieldnames:
            return candidate
    return None


def _parse_tmdb_id(row, id_column, link_column):
    if id_column:
        raw = _field(row, id_column)
        if raw is not None:
            text = str(raw).strip()
            if text.isdigit():
                return int(text)
    if link_column:
        raw = _field(row, link_column)
        if raw:
            match = TMDB_URL_RE.search(str(raw))
            if match:
                return int(match.group(1))
    for value in row.values():
        if not value:
            continue
        match = TMDB_URL_RE.search(str(value))
        if match:
            return int(match.group(1))
    return None


def _match_internal(row, by_source, by_movie_id, id_column, movie_column):
    for column in (id_column, movie_column):
        if column is None:
            continue
        value = _field(row, column)
        if value is None:
            continue
        key = str(value)
        if key in by_source:
            return by_source[key]
        if key in by_movie_id:
            return by_movie_id[key]
    return None


def _tmdb_request(client, api_key, tmdb_id, retries=3):
    headers = {}
    params = {}
    if api_key.startswith("eyJ"):
        headers["Authorization"] = f"Bearer {api_key}"
    else:
        params["api_key"] = api_key
    delay = 1.0
    for attempt in range(retries):
        try:
            response = client.get(
                f"/movie/{int(tmdb_id)}",
                headers=headers,
                params=params,
            )
            if response.status_code == 200:
                data = response.json()
                return {
                    "tmdb_id": int(tmdb_id),
                    "poster_path": data.get("poster_path"),
                    "backdrop_path": data.get("backdrop_path"),
                }
            if response.status_code == 404:
                return None
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                time.sleep(
                    float(retry_after) if retry_after else delay
                )
            else:
                time.sleep(delay)
        except httpx.HTTPError:
            time.sleep(delay)
        delay = min(delay * 2, 30.0)
    return None


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    os.replace(temporary, path)


def enrich(
    catalogue_csv,
    artifacts_dir,
    output,
    api_key=None,
    client=None,
    limit=None,
    resume=False,
    dry_run=False,
    request_delay=0.05,
    checkpoint_every=50,
    timeout=20.0,
    id_column=None,
    movie_column=None,
    tmdb_id_column=None,
    tmdb_link_column=None,
):
    artifacts_dir = Path(artifacts_dir)
    movie_metadata_path = artifacts_dir / "movie_metadata.json"
    movies = _load_catalogue(movie_metadata_path)
    by_source, by_movie_id = _catalogue_index(movies)
    csv_path = Path(catalogue_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Catalogue CSV not found: {csv_path}")
    output = Path(output)
    existing = {}
    if resume and output.exists():
        try:
            payload = json.loads(output.read_text())
            existing = dict(payload.get("movies", {}))
        except (OSError, ValueError):
            existing = {}
    api_key = _api_key(api_key)
    if not dry_run and not api_key:
        raise ValueError(
            "TMDB_API_KEY is required (or pass --api-key)."
        )
    owns_client = client is None
    if client is None:
        client = httpx.Client(
            base_url="https://api.themoviedb.org/3",
            timeout=float(timeout),
        )
    report = {
        "csv": str(csv_path),
        "catalogue_movies": len(movies),
        "rows_read": 0,
        "matched": 0,
        "unmatched": 0,
        "missing_tmdb_id": 0,
        "enriched": 0,
        "not_found": 0,
        "skipped_existing": 0,
        "failed": 0,
        "dry_run": bool(dry_run),
        "unmatched_examples": [],
        "missing_tmdb_examples": [],
    }
    with open(csv_path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        id_column = id_column or _detect_column(
            fieldnames, ["_id", "movie_id", "letterboxd_id"]
        )
        movie_column = movie_column or _detect_column(
            fieldnames, ["movie_id", "slug", "letterboxd_slug"]
        )
        tmdb_id_column = tmdb_id_column or _detect_column(
            fieldnames, ["tmdb_id"]
        )
        tmdb_link_column = tmdb_link_column or _detect_column(
            fieldnames, ["tmdb_link", "themoviedb_url", "tmdb_url"]
        )
        processed = 0
        for row in reader:
            if limit is not None and processed >= int(limit):
                break
            processed += 1
            report["rows_read"] += 1
            internal = _match_internal(
                row,
                by_source,
                by_movie_id,
                id_column,
                movie_column,
            )
            if internal is None:
                report["unmatched"] += 1
                if len(report["unmatched_examples"]) < 20:
                    report["unmatched_examples"].append({
                        "id": _field(row, id_column),
                        "movie_id": _field(row, movie_column),
                    })
                continue
            report["matched"] += 1
            tmdb_id = _parse_tmdb_id(
                row, tmdb_id_column, tmdb_link_column
            )
            if tmdb_id is None:
                report["missing_tmdb_id"] += 1
                if len(report["missing_tmdb_examples"]) < 20:
                    report["missing_tmdb_examples"].append(
                        str(_field(row, movie_column))
                    )
                continue
            key = str(internal)
            if key in existing:
                report["skipped_existing"] += 1
                continue
            if dry_run:
                report["enriched"] += 1
                continue
            metadata = _tmdb_request(client, api_key, tmdb_id)
            if metadata is None:
                report["not_found"] += 1
                continue
            existing[key] = metadata
            report["enriched"] += 1
            if request_delay:
                time.sleep(float(request_delay))
            if checkpoint_every and report["enriched"] % int(checkpoint_every) == 0:
                _atomic_json(output, {"movies": existing})
    if not dry_run:
        _atomic_json(output, {"movies": existing})
    if owns_client:
        client.close()
    report_path = output.with_suffix(".report.json")
    _atomic_json(report_path, report)
    return report


def parse_args():
    parser = argparse.ArgumentParser(
        description="Enrich FilmFinder movies with TMDB image paths."
    )
    parser.add_argument("--catalogue-csv", default=str(DEFAULT_CSV))
    parser.add_argument("--artifacts", default=str(DEFAULT_ARTIFACTS))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--request-delay", type=float, default=0.05)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--id-column", default=None)
    parser.add_argument("--movie-column", default=None)
    parser.add_argument("--tmdb-id-column", default=None)
    parser.add_argument("--tmdb-link-column", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    report = enrich(
        catalogue_csv=args.catalogue_csv,
        artifacts_dir=args.artifacts,
        output=args.output,
        api_key=args.api_key,
        limit=args.limit,
        resume=args.resume,
        dry_run=args.dry_run,
        request_delay=args.request_delay,
        checkpoint_every=args.checkpoint_every,
        timeout=args.timeout,
        id_column=args.id_column,
        movie_column=args.movie_column,
        tmdb_id_column=args.tmdb_id_column,
        tmdb_link_column=args.tmdb_link_column,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
