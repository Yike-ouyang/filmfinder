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
POST /api/v1/recommendations/from-saved-ratings
GET /api/v1/recommendations/current
GET /api/v1/films
GET /api/v1/films/facets
GET /api/v1/films/{film_id}
GET /api/v1/me/ratings
PUT /api/v1/me/ratings/{film_id}
DELETE /api/v1/me/ratings/{film_id}
GET /api/v1/me/profile-summary
POST /api/v1/recommendations/refresh
GET /api/v1/auth/csrf
POST /api/v1/auth/signup
POST /api/v1/auth/verify-email
POST /api/v1/auth/resend-verification
POST /api/v1/auth/login
POST /api/v1/auth/logout
GET /api/v1/auth/me
POST /api/v1/auth/forgot-password
POST /api/v1/auth/reset-password
DELETE /api/v1/auth/account
GET /api/v1/swipes
PUT /api/v1/swipes/{internal_movie_id}
POST /api/v1/swipes/{internal_movie_id}/undo
DELETE /api/v1/swipes/{internal_movie_id}
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

When the request carries a valid session cookie, the imported ratings and
watched films are merged into `user_films` in SQLite. Successive imports update
existing ratings, add new films, and keep films absent from the new archive.
Recommendations are then generated from the cumulative saved profile. The
frontend sends `credentials: include` and `X-CSRF-Token`; anonymous uploads are
unchanged and never write to SQLite. Raw ZIP archives are never stored.

The response contains `profile`, `recommendation_info`, and
`recommendations`. Fields `poster_path`, `backdrop_path`, and `tmdb_id` are
`null` when TMDB metadata is unavailable.

## Film library

The browsable catalogue is stored in SQLite in `catalog_films` and
`catalog_film_genres` (migration `0002_catalog_films`). Synchronize it offline:

```bash
alembic -c apps/backend/alembic.ini upgrade head
python apps/backend/scripts/sync_film_catalog.py \
  --catalogue-csv movie_data.csv \
  --artifacts packages/recommendation-engine/artifacts
```

`movie_data.csv` is the raw 163 MB source and is not versioned. The script
streams it, reuses the stable `_id` / `movie_id` matching, and writes summaries,
precise release dates, TMDB image paths and rating metrics. It never calls
TMDB. The library filters on `in_library = year <= 2019`; 38 705 films qualify
from the current artifacts.

`GET /api/v1/films` uses SQL pagination (50 max), search, genre filtering and
popularity/weighted-rating/date sorting. `GET /api/v1/films/{film_id}` serves
the summary. Both accept an optional session and include personal rating/list
state when signed in.

Manual ratings are integers 1-10. `PUT` creates or replaces a rating and clears
the film's Watchlist / Do Not Watch decision. `DELETE` removes the entire
`user_films` row, as required by the manual profile flow. Manual generation uses
`POST /api/v1/recommendations/from-saved-ratings`, which enforces 10 ratings;
the existing Letterboxd import and `/recommendations/refresh` are unchanged.

Watchlist and Do Not Watch reuse `user_swipes`: `interested` and
`not_interested`. Discover and My Lists use the same rows, so there is one
source of truth for list decisions. See `docs/film-library.md`.

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

## SQLite database

Phase 1 adds the SQLite persistence layer. Migrations are versioned with
Alembic and are never applied automatically when the API starts.

```bash
cd apps/backend
alembic upgrade head
```

The development database defaults to `apps/backend/var/filmfinder.db`
(overridable with `FILMFINDER_DB_PATH`). In production, use a path outside the
repository and outside the public Caddy root, for example
`/var/lib/filmfinder/filmfinder.db`.

Useful commands:

```bash
alembic current
alembic history
alembic downgrade -1
alembic check
```

SQLite connections enable `foreign_keys`, WAL, `busy_timeout`, and
`synchronous=NORMAL`.

## Authentication

Phase 2 adds email/password accounts. Email verification is disabled by
default (`REQUIRE_EMAIL_VERIFICATION=false`), so a new account can log in
immediately with its email and password. Set
`REQUIRE_EMAIL_VERIFICATION=true` to require the verification link again.
Existing unverified accounts are marked verified on their first login while
verification is disabled.

- Passwords are hashed with Argon2id.
- Session and email tokens are random; only their SHA-256 hashes are stored.
- The session cookie is `HttpOnly`, `SameSite=Lax`, and `Secure` when
  `COOKIE_SECURE=true`.
- CSRF uses a double-submit cookie plus the `X-CSRF-Token` header.
- Login attempts are rate limited in memory. With a single Uvicorn worker,
  limits are process-local.
- `EMAIL_BACKEND=console` logs verification and reset links for local
  development. No real email is sent.
- `EMAIL_BACKEND=brevo` sends through the Brevo HTTPS API. Set
  `BREVO_API_KEY`, `EMAIL_FROM`, and `APP_BASE_URL` in production.

The backend never returns raw session or email tokens in JSON responses.

Production checklist:

```env
APP_BASE_URL=https://filmfinder.example
CORS_ORIGINS=https://filmfinder.example
COOKIE_SECURE=true
EMAIL_BACKEND=brevo
BREVO_API_KEY=...
EMAIL_FROM=verified-sender@example.com
FILMFINDER_DB_PATH=/var/lib/filmfinder/filmfinder.db
REQUIRE_EMAIL_VERIFICATION=false
```

`CORS_ORIGINS` must contain the exact frontend origin because the CSRF check
validates the `Origin` header. Run `alembic upgrade head` before starting the
service. Set `REQUIRE_EMAIL_VERIFICATION=true` if you want to enforce email
verification again.

## Server recommendations

Authenticated accounts can use server-side recommendation batches:

- `GET /api/v1/recommendations/current` returns the latest ready batch, or
  `202` while a recalculation is running.
- `POST /api/v1/recommendations/refresh` queues a new batch. The frontend
  compares the new batch with the previous one and shows a no-new message when
  no additional film is available.
- Login triggers a background recalculation when a saved profile exists.
- A single background executor and the existing ML semaphore keep ML
  concurrency at one.
- Batch history is kept in `recommendation_batches` and
  `recommendation_items`.
- `not_interested` and `already_seen` films are excluded. `interested` films
  may reappear.
- An unchanged batch is reported to the frontend instead of being retried in a
  loop.

The public `/from-letterboxd` endpoint remains unchanged for anonymous
visitors. Authenticated imports also persist the generated batch.

## SQLite backups

```bash
python apps/backend/scripts/backup_sqlite.py \
  --source /var/lib/filmfinder/filmfinder.db \
  --output-dir /var/lib/filmfinder/backups \
  --retention 14
```

The script uses SQLite's backup API, so WAL content is included consistently.
A systemd timer or cron entry can run it daily and keep the last 14 backups.

## Swipes and My Lists

Authenticated users can persist three actions: `interested`, `not_interested`,
and `already_seen`. The latest action replaces the previous one. Each change is
also appended to `user_swipe_events` so Undo can restore the previous state or
remove the decision entirely.

`GET /api/v1/swipes` returns the two visible categories (`interested` and
`not_interested`); `already_seen` is stored but not listed. All mutation routes
require the session cookie and `X-CSRF-Token`.

The frontend exposes `/lists` to authenticated users with two tabs. Anonymous
visitors see account creation and login links instead. During Discover,
authenticated decisions are saved before the card advances.

## Tests

```bash
python -m pytest -q apps/backend/tests
```

Tests cover health, readiness, valid/invalid ZIP uploads, limit and size
validation, no matched ratings, watched exclusion, determinism, metadata
lookup, mocked TMDB enrichment, catalogue synchronization, library pagination
and search, manual ratings, list coherence, authentication, profile
persistence, swipe history/undo, recommendation batches, exclusions, and SQLite
backups. No real TMDB requests are made by tests.

## Design boundaries

The public film and recommendation APIs remain usable without an account.
SQLite backs accounts, sessions, Letterboxd profiles, manual ratings, list
decisions, recommendation history, and the derived film catalogue. The API
does not create tables at startup, does not retrain models, and does not depend
on an external database server.
