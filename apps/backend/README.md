# FilmFinder backend

FastAPI POC backend for the existing FilmFinder M5 + KNN recommendation engine.

## Install

From the repository root:

```bash
pip install -e packages/recommendation-engine
pip install -e apps/backend
cp apps/backend/.env.example apps/backend/.env
```

Edit `apps/backend/.env` if the artifacts are not at the default location:

```env
APP_ENV=development
FILMFINDER_ARTIFACTS_DIR=../../packages/recommendation-engine/artifacts
TMDB_METADATA_PATH=../../packages/recommendation-engine/artifacts/tmdb_metadata.json
CORS_ORIGINS=http://localhost:3000
MAX_UPLOAD_SIZE_MB=20
MAX_UNCOMPRESSED_SIZE_MB=100
MAX_ZIP_ENTRIES=64
MAX_CONCURRENT_RECOMMENDATIONS=1
TMDB_API_KEY=
```

`TMDB_API_KEY` is used only by the offline enrichment script, not during
recommendation requests.

## Start

```bash
cd apps/backend
uvicorn app.main:app --reload
```

The model is loaded once during application startup. If artifacts are missing,
the application still starts, `/health` remains available, and `/api/v1/ready`
returns HTTP 503.

## Endpoints

```http
GET /health
GET /api/v1/ready
POST /api/v1/recommendations/from-letterboxd?limit=100
```

Health:

```json
{"status": "ok"}
```

Readiness:

```json
{
  "status": "ready",
  "model": "M5",
  "recommendation_algorithm": "knn_m5"
}
```

## Letterboxd upload

```bash
curl -X POST \
  "http://localhost:8000/api/v1/recommendations/from-letterboxd?limit=100" \
  -F "file=@packages/recommendation-engine/tests/fixtures/letterboxd-eslpej-2026-11-20-10-33-utc.zip"
```

The endpoint:

1. validates the ZIP size, entries, paths and compression;
2. parses `ratings.csv` and `watched.csv` in memory;
3. matches movies against the FilmFinder catalogue;
4. rejects profiles with no matched ratings (`422`);
5. calls the existing cold-start adaptation and KNN + M5 engine;
6. excludes rated and watched movies;
7. returns up to `limit` recommendations (`1..100`), enriched with local TMDB
   metadata when available.

The response contains `profile`, `recommendation_info`, and
`recommendations`. Fields `poster_path`, `backdrop_path`, and `tmdb_id` are
`null` when TMDB metadata is unavailable.

## TMDB metadata enrichment

The offline enrichment script reads `movie_data.csv` and matches rows to the
FilmFinder catalogue using stable IDs (`_id`, then `movie_id`). It stores only
TMDB IDs and image paths; it never downloads images.

```bash
python apps/backend/scripts/enrich_tmdb_metadata.py \
  --catalogue-csv movie_data.csv \
  --artifacts packages/recommendation-engine/artifacts \
  --output packages/recommendation-engine/artifacts/tmdb_metadata.json \
  --resume
```

The script requires `TMDB_API_KEY` in the environment or in
`apps/backend/.env`. Use `--dry-run` to validate matching without API calls.
It writes a coverage report next to the output
(`tmdb_metadata.report.json`) and supports `--limit`, `--request-delay`,
`--checkpoint-every`, `--id-column`, `--movie-column`, `--tmdb-id-column`, and
`--tmdb-link-column`.

The recommendation backend loads the resulting file once at startup. If the
file is missing, requests still succeed with null image fields.

## Tests

```bash
python -m pytest -q apps/backend/tests
```

Tests cover health, readiness, valid/invalid ZIP uploads, limit and size
validation, no matched ratings, watched exclusion, determinism, metadata
lookup, and mocked TMDB enrichment. No real TMDB requests are made by tests.

## Design boundaries

This POC is stateless: no accounts, sessions, database, swipe persistence,
background workers, or model retraining. A future MySQL layer can be added
behind a service/repository boundary without changing the recommendation
engine.
