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

## Tests

```bash
python -m pytest -q apps/backend/tests
```

Tests cover health, readiness, valid/invalid ZIP uploads, limit and size
validation, no matched ratings, watched exclusion, determinism, metadata
lookup, and mocked TMDB enrichment. No real TMDB requests are made by tests.

## Design boundaries

The public recommendation API remains stateless. Phase 1 introduces the
SQLite schema and repositories that will back accounts, sessions, Letterboxd
profiles, swipes, and recommendation history in later phases. The API does not
create tables at startup, does not retrain models, and does not depend on an
external database server.
