# FilmFinder

FilmFinder is a personalized movie recommendation web application built around
a contextual Factorization Machine (M5) and a KNN cold-start candidate
generator. The repository is a monorepo containing the research pipeline, the
production inference package, a FastAPI backend, a static Next.js frontend, and
the deployment notes for a single-VPS setup.

Users can build a profile in three ways:

- import a Letterboxd ZIP export;
- browse the film library and rate films manually;
- complete the Quick Taste Profile by rating 20 selected films.

All three paths feed the same recommendation engine and the same persisted
rating model. Recommendations can then be explored in the swipe-based
Discover experience or managed from the account, lists, and admin pages.

## Feature overview

- Letterboxd ZIP import processed in memory, with no archive storage.
- Browsable SQLite film catalogue with search, genre filters, sorting and
  pagination.
- Manual 1–10 ratings using half-star controls.
- Quick Taste Profile with 20 fixed films, Haven't Seen, Undo and anonymous
  drafting.
- Server-side recommendation batches with refresh and "view more" flows.
- Watchlist and Do Not Watch lists backed by swipe decisions.
- Email/password accounts, optional email verification, password reset,
  server-side sessions and CSRF protection.
- Private admin dashboard for aggregate account and engagement metrics.
- Anonymous recommendation flow where supported.
- Static frontend export, Caddy reverse proxy, systemd-managed FastAPI and
  SQLite persistence.

## Technologies

| Layer | Stack |
|---|---|
| Recommendation engine | Python, NumPy, SciPy, contextual FM (M5), KNN, ridge cold-start adaptation |
| Backend | FastAPI, Pydantic v2, SQLAlchemy 2, Alembic, SQLite, Uvicorn, Argon2, httpx |
| Frontend | Next.js 16 App Router, React 19, TypeScript 5, Tailwind CSS 4, Motion, Lucide |
| Frontend tests | Vitest, Testing Library, jsdom |
| Research | Python, marimo, pandas, NumPy, SciPy, matplotlib |
| Production | Caddy, systemd, SQLite backup API, rsync |

There is no Docker, Redis, Celery, PostgreSQL or permanent Node server in the
current production architecture.

## Repository layout

```text
.
├── apps/
│   ├── backend/                 # FastAPI API, SQLite, accounts, catalogue, recommendations
│   └── frontend/                # Next.js static-export application
├── packages/
│   └── recommendation-engine/   # Installable M5 + KNN inference package and artifacts
├── research/
│   ├── preprocessing/           # Data cleaning pipelines
│   ├── modeling/                # M5 training and evaluation
│   ├── recommendation/          # Recommendation and KNN experiments, artifact export
│   ├── explorer/                # marimo recommendation explorer
│   ├── legacy/                  # Early SVD experiments
│   └── results/                 # Generated experiment outputs (ignored)
├── infrastructure/              # Deployment notes
├── docs/                        # Architecture, deployment, feature documentation
├── prompts/                     # Task specifications (local notes)
├── .env.example
├── README.md
└── pyproject.toml
```

## How recommendations work

The production engine is frozen; it is not retrained during normal requests.

```text
rating profile
  -> centered KNN similarity
  -> neighbour pool
  -> remove rated and watched films
  -> M5 scoring
  -> deterministic top-N recommendations
```

For known users, M5 uses learned user factors. For cold-start users, the engine
fits a ridge adaptation (`theta`) from the submitted ratings. The global model
parameters and artifacts are never modified by a request.

The browsable catalogue is synchronized offline from `movie_data.csv` and the
production artifacts into `catalog_films` and `catalog_film_genres`. Requests
use SQL pagination and never load the full dataset into memory.

## Local setup

### Prerequisites

- Python 3.11 or newer recommended.
- Node.js 22.12 or newer and npm 10 or newer.
- The production artifacts under
  `packages/recommendation-engine/artifacts/`.
- `movie_data.csv` at the repository root for the initial catalogue sync
  (gitignored, not part of the repository).

### 1. Python environment

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e packages/recommendation-engine
pip install -e apps/backend
```

For the research notebooks and training scripts, also install:

```bash
pip install -r research/requirements.txt
```

### 2. Backend configuration

```bash
cp apps/backend/.env.example apps/backend/.env
```

Default local values already target:

```env
FILMFINDER_ARTIFACTS_DIR=../../packages/recommendation-engine/artifacts
TMDB_METADATA_PATH=../../packages/recommendation-engine/artifacts/tmdb_metadata.json
CORS_ORIGINS=http://localhost:3000
APP_BASE_URL=http://localhost:3000
FILMFINDER_DB_PATH=var/filmfinder.db
EMAIL_BACKEND=console
COOKIE_SECURE=false
REQUIRE_EMAIL_VERIFICATION=false
```

### 3. Database and catalogue

```bash
alembic -c apps/backend/alembic.ini upgrade head

python apps/backend/scripts/sync_film_catalog.py \
  --catalogue-csv movie_data.csv \
  --artifacts packages/recommendation-engine/artifacts
```

`sync_film_catalog.py` is idempotent. Use `--dry-run` or `--limit` for
inspection. If the catalogue tables are empty, `/api/v1/films` returns HTTP 503
with `catalog_unavailable`.

### 4. Run the backend

```bash
cd apps/backend
uvicorn app.main:app --reload
```

The backend listens on `http://localhost:8000`. The model is loaded once at
startup. If artifacts are unavailable, the process still starts, `/health`
stays available and `/api/v1/ready` returns HTTP 503.

### 5. Run the frontend

```bash
cd apps/frontend
npm install
cp .env.example .env.local
npm run dev
```

Open `http://localhost:3000`.

For production, the frontend is statically exported:

```bash
cd apps/frontend
NEXT_PUBLIC_API_BASE_URL=https://filmfinder.example npm run build
```

The generated files are in `apps/frontend/out/`. `NEXT_PUBLIC_*` values are
inlined at build time.

## Environment variables

### Backend (`apps/backend/.env`)

| Variable | Purpose |
|---|---|
| `APP_ENV` | Application environment label. |
| `FILMFINDER_ARTIFACTS_DIR` | M5/KNN artifacts directory. |
| `TMDB_METADATA_PATH` | Local TMDB metadata JSON. |
| `CORS_ORIGINS` | Comma-separated allowed frontend origins. |
| `APP_BASE_URL` | Public application URL used in emails and links. |
| `FILMFINDER_DB_PATH` | SQLite database path. |
| `SQLITE_BUSY_TIMEOUT_MS` | SQLite lock timeout. |
| `MAX_UPLOAD_SIZE_MB` | Letterboxd ZIP size limit. |
| `MAX_UNCOMPRESSED_SIZE_MB` | Extracted archive size limit. |
| `MAX_ZIP_ENTRIES` | Maximum archive entry count. |
| `MAX_CONCURRENT_RECOMMENDATIONS` | Inference concurrency limit. |
| `SESSION_COOKIE_NAME`, `CSRF_COOKIE_NAME` | Cookie names. |
| `COOKIE_SECURE` | Set `true` behind HTTPS in production. |
| `REQUIRE_EMAIL_VERIFICATION` | Enable the email verification flow. |
| `EMAIL_BACKEND` | `console` or `brevo`. |
| `BREVO_API_KEY`, `EMAIL_FROM` | Production email settings. |
| `TASTE_PROFILE_PATH` | Quick Taste 20-film configuration. |
| `TASTE_PROFILE_RATE_LIMIT_*` | Quick Taste rate limiting. |
| `ANALYTICS_RETENTION_DAYS` | Analytics pruning window. |

### Frontend (`apps/frontend/.env.local`)

| Variable | Purpose |
|---|---|
| `NEXT_PUBLIC_API_BASE_URL` | FastAPI base URL. |
| `NEXT_PUBLIC_MAX_UPLOAD_SIZE_MB` | Client-side ZIP size guard. |

Keep the frontend upload limit aligned with the backend value.

## Common commands

### Backend

```bash
# Run API
cd apps/backend && uvicorn app.main:app --reload

# Tests
python -m pytest -q apps/backend/tests

# Migrations
alembic -c apps/backend/alembic.ini upgrade head

# Catalogue sync
python apps/backend/scripts/sync_film_catalog.py \
  --catalogue-csv movie_data.csv \
  --artifacts packages/recommendation-engine/artifacts

# Grant admin
python apps/backend/scripts/grant_admin.py --email admin@example.com

# SQLite backup
python apps/backend/scripts/backup_sqlite.py \
  --source apps/backend/var/filmfinder.db \
  --output-dir apps/backend/var/backups \
  --retention 14

# Analytics pruning
python apps/backend/scripts/prune_analytics.py --retention-days 365
```

### Frontend

```bash
cd apps/frontend
npm run dev
npm run test
npm run test:watch
npm run lint
npm run typecheck
npm run build
```

Use `npm run test:watch` for the watch mode.

### Engine and research

```bash
# Backend/engine tests
python -m pytest -q packages/recommendation-engine/tests

# Letterboxd CLI
python packages/recommendation-engine/scripts/generate_recommendations.py \
  --letterboxd-zip packages/recommendation-engine/tests/fixtures/letterboxd-eslpej-2026-11-20-10-33-utc.zip \
  --artifacts packages/recommendation-engine/artifacts \
  --top-n 100 \
  --output recommendations.json \
  --report import_report.json

# Regenerate the Quick Taste selection
python packages/recommendation-engine/scripts/select_quick_taste_movies.py

# Research notebooks
marimo edit research/preprocessing/data_cleaning.py
marimo edit research/modeling/2.Processing.py
marimo edit research/modeling/5.ContextFM.py
marimo edit research/recommendation/8.RecommendationSystem.py
marimo edit research/recommendation/9.KNNM5Optimization.py
```

## API endpoints

All application endpoints are under `/api/v1`. Signed-in mutations require the
session cookie and CSRF token. The frontend sends both automatically.

### Health

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness check. |
| `GET` | `/api/v1/ready` | Engine and model readiness. |

### Authentication

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/v1/auth/csrf` | Fetch/refresh the CSRF token. |
| `POST` | `/api/v1/auth/signup` | Create an account. |
| `POST` | `/api/v1/auth/verify-email` | Verify an email token. |
| `POST` | `/api/v1/auth/resend-verification` | Resend verification. |
| `POST` | `/api/v1/auth/login` | Create a session. |
| `POST` | `/api/v1/auth/logout` | Revoke the session. |
| `GET` | `/api/v1/auth/me` | Current user. |
| `POST` | `/api/v1/auth/forgot-password` | Request a reset link. |
| `POST` | `/api/v1/auth/reset-password` | Reset the password. |
| `DELETE` | `/api/v1/auth/account` | Delete the current account. |

### Films and ratings

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/v1/films` | Catalogue with `page`, `page_size`, `search`, `sort`, `order`, `genre`. |
| `GET` | `/api/v1/films/facets` | Genres and catalogue totals. |
| `GET` | `/api/v1/films/{film_id}` | Film details and current user data. |
| `GET` | `/api/v1/me/ratings` | Paginated user ratings. |
| `PUT` | `/api/v1/me/ratings/{film_id}` | Set or replace a 1–10 rating. |
| `DELETE` | `/api/v1/me/ratings/{film_id}` | Remove a rating. |
| `GET` | `/api/v1/me/profile-summary` | Rating, watched and list counts. |

### Recommendations

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/v1/recommendations/from-letterboxd?limit=100` | Import a Letterboxd ZIP and generate recommendations. |
| `POST` | `/api/v1/recommendations/from-saved-ratings` | Queue a batch from the saved profile. |
| `GET` | `/api/v1/recommendations/current` | Latest ready or active batch. |
| `POST` | `/api/v1/recommendations/refresh` | Request more recommendations. |
| `GET` | `/api/v1/taste-profile/movies` | 20 Quick Taste films and existing user ratings. |
| `POST` | `/api/v1/taste-profile/recommendations?limit=100` | Submit 20 Quick Taste answers and generate recommendations. |

### Swipes and lists

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/v1/swipes` | Watchlist and Do Not Watch decisions. |
| `PUT` | `/api/v1/swipes/{internal_movie_id}` | Set a decision. |
| `POST` | `/api/v1/swipes/{internal_movie_id}/undo` | Undo the latest decision. |
| `DELETE` | `/api/v1/swipes/{internal_movie_id}` | Remove a decision. |

### Admin

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/v1/admin/overview` | Aggregate metrics. |
| `GET` | `/api/v1/admin/activity?period=7d\|30d\|all` | Activity charts. |
| `GET` | `/api/v1/admin/users?page=1&page_size=50&search=` | User list. |
| `GET` | `/api/v1/admin/users/{user_id}` | User detail. |

Admin endpoints require the `is_admin` flag. There is no public endpoint to
grant it.

## Data and artifacts

The training pipeline reads cleaned exports defined by
`FILMFINDER_CLEANING_DIR`:

```text
ratings_matrix.npz
user_ids.npy
movie_ids.npy
movie_context.json
```

Production artifacts live under:

```text
packages/recommendation-engine/artifacts/
```

They are ignored by Git and must be copied to the server separately. They
contain the M5 weights, recommendation configuration, feature matrices,
rating matrix and metadata mappings.

Other important data:

- `movie_data.csv`: raw catalogue source, gitignored, used only for the offline
  catalogue sync and TMDB enrichment.
- `packages/recommendation-engine/artifacts/tmdb_metadata.json`: local image
  metadata, no TMDB call at request time.
- `apps/backend/app/data/quick_taste_movies.json`: frozen 20-film Quick Taste
  selection, tracked in Git.

## VPS migration and deployment

The production target is a single VPS without Docker.

```text
Caddy (HTTPS, static export, reverse proxy)
  |
  +-- /var/www/filmfinder         Next.js static export
  |
  +-- /api/* and /health          FastAPI on 127.0.0.1:8000
                                     |
                                     +-- M5/KNN engine
                                     +-- SQLite in /var/lib/filmfinder
```

### 1. Server prerequisites

- Debian/Ubuntu-like VPS with a non-root deployment user.
- Python 3.11+, Node.js 22.12+, npm, git, rsync.
- Caddy installed and serving HTTPS for the domain.
- The domain DNS record pointing to the VPS.

```bash
sudo adduser --disabled-password --gecos "" filmfinder
sudo mkdir -p /var/www/filmfinder /var/lib/filmfinder /opt/filmfinder
sudo chown -R filmfinder:filmfinder /var/www/filmfinder /var/lib/filmfinder /opt/filmfinder
sudo chmod 750 /var/lib/filmfinder
```

### 2. Deploy the repository

```bash
sudo -u filmfinder git clone <REPOSITORY_URL> /home/filmfinder/filmfinder
cd /home/filmfinder/filmfinder
sudo -u filmfinder python3 -m venv .venv
sudo -u filmfinder .venv/bin/pip install --upgrade pip
sudo -u filmfinder .venv/bin/pip install -e packages/recommendation-engine
sudo -u filmfinder .venv/bin/pip install -e apps/backend
```

Copy the model artifacts and the catalogue CSV to the server. Example using
rsync from a local machine:

```bash
rsync -avh packages/recommendation-engine/artifacts/ \
  filmfinder@VPS:/home/filmfinder/filmfinder/packages/recommendation-engine/artifacts/

rsync -avh movie_data.csv \
  filmfinder@VPS:/home/filmfinder/filmfinder/movie_data.csv
```

The Quick Taste configuration is tracked in the repository and does not need a
separate copy.

### 3. Production environment

Create `/home/filmfinder/filmfinder/apps/backend/.env` from the example:

```bash
cp apps/backend/.env.example apps/backend/.env
```

Set at least:

```env
APP_ENV=production
APP_BASE_URL=https://filmfinder.example
CORS_ORIGINS=https://filmfinder.example
COOKIE_SECURE=true
FILMFINDER_DB_PATH=/var/lib/filmfinder/filmfinder.db
FILMFINDER_ARTIFACTS_DIR=/home/filmfinder/filmfinder/packages/recommendation-engine/artifacts
TMDB_METADATA_PATH=/home/filmfinder/filmfinder/packages/recommendation-engine/artifacts/tmdb_metadata.json
TASTE_PROFILE_PATH=/home/filmfinder/filmfinder/apps/backend/app/data/quick_taste_movies.json
EMAIL_BACKEND=brevo
BREVO_API_KEY=...
EMAIL_FROM=verified-sender@example.com
REQUIRE_EMAIL_VERIFICATION=false
```

`CORS_ORIGINS` must match the public frontend origin exactly because the CSRF
check validates the `Origin` header.

### 4. Database and catalogue

```bash
cd /home/filmfinder/filmfinder
.venv/bin/alembic -c apps/backend/alembic.ini upgrade head

.venv/bin/python apps/backend/scripts/sync_film_catalog.py \
  --catalogue-csv movie_data.csv \
  --artifacts packages/recommendation-engine/artifacts
```

Migrations are not applied automatically at application startup. Back up the
database before a major upgrade.

### 5. Systemd backend service

Create `/etc/systemd/system/filmfinder-api.service`:

```ini
[Unit]
Description=FilmFinder FastAPI backend
After=network.target

[Service]
Type=simple
User=filmfinder
Group=filmfinder
WorkingDirectory=/home/filmfinder/filmfinder/apps/backend
EnvironmentFile=/home/filmfinder/filmfinder/apps/backend/.env
ExecStart=/home/filmfinder/filmfinder/.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
Restart=on-failure
RestartSec=5
NoNewPrivileges=true
PrivateTmp=true

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now filmfinder-api
sudo systemctl status filmfinder-api
```

Use a single Uvicorn worker. ML inference is already serialized internally.

### 6. Build and deploy the frontend

On the build machine or on the VPS:

```bash
cd apps/frontend
npm ci
NEXT_PUBLIC_API_BASE_URL=https://filmfinder.example npm run build
rsync -avh out/ filmfinder@VPS:/var/www/filmfinder/
```

The static export needs no Node process and no restart. Deploy the backend
first when new API routes are introduced.

### 7. Caddy configuration

Example `/etc/caddy/Caddyfile`:

```caddyfile
filmfinder.example {
    encode zstd gzip

    handle /api/* {
        reverse_proxy 127.0.0.1:8000
    }

    handle /health {
        reverse_proxy 127.0.0.1:8000
    }

    handle {
        root * /var/www/filmfinder
        try_files {path} {path}.html {path}/index.html /index.html
        file_server
    }
}
```

Reload Caddy:

```bash
sudo caddy validate --config /etc/caddy/Caddyfile
sudo systemctl reload caddy
```

Never expose `apps/backend/var`, `movie_data.csv`, model artifacts, or `.env`
through a static route.

### 8. Administrator access

Create a normal account through the UI, then grant the admin role on the
server:

```bash
cd /home/filmfinder/filmfinder
.venv/bin/python apps/backend/scripts/grant_admin.py \
  --email admin@example.com
```

Revoke with `--revoke`. The `/admin` page is unavailable until this is done.

### 9. Backups and retention

```bash
cd /home/filmfinder/filmfinder
.venv/bin/python apps/backend/scripts/backup_sqlite.py \
  --source /var/lib/filmfinder/filmfinder.db \
  --output-dir /var/lib/filmfinder/backups \
  --retention 14

.venv/bin/python apps/backend/scripts/prune_analytics.py \
  --retention-days 365
```

Both scripts are suitable for systemd timers. The backup script uses SQLite's
backup API and includes WAL content consistently.

### 10. Verification after deployment

```bash
curl --fail-with-body https://filmfinder.example/health
curl --fail-with-body https://filmfinder.example/api/v1/ready
curl -o /dev/null -w '%{http_code}\n' https://filmfinder.example/login
curl --fail-with-body "https://filmfinder.example/api/v1/films?page=1&page_size=5"
curl -o /dev/null -w '%{http_code}\n' https://filmfinder.example/admin
sudo systemctl status filmfinder-api
sudo journalctl -u filmfinder-api -n 50 --no-pager
```

A successful `/api/v1/ready` response looks like:

```json
{
  "status": "ready",
  "model": "M5",
  "recommendation_algorithm": "knn_m5"
}
```

### 11. Update procedure

```bash
cd /home/filmfinder/filmfinder
sudo systemctl stop filmfinder-api
git pull
.venv/bin/pip install -e packages/recommendation-engine
.venv/bin/pip install -e apps/backend
.venv/bin/alembic -c apps/backend/alembic.ini upgrade head
# Sync the catalogue only if movie_data.csv or artifacts changed.
.venv/bin/python apps/backend/scripts/sync_film_catalog.py \
  --catalogue-csv movie_data.csv \
  --artifacts packages/recommendation-engine/artifacts
sudo systemctl start filmfinder-api

cd apps/frontend
npm ci
NEXT_PUBLIC_API_BASE_URL=https://filmfinder.example npm run build
rsync -avh out/ filmfinder@VPS:/var/www/filmfinder/
```

## Tests and quality checks

Backend and engine:

```bash
python -m pytest -q apps/backend/tests
python -m pytest -q packages/recommendation-engine/tests
```

Frontend:

```bash
cd apps/frontend
npm run test
npm run lint
npm run typecheck
npm run build
```

Artifact-backed tests require the generated artifacts. The Letterboxd tests
also require the ignored fixture ZIP.

## Documentation

- `docs/architecture.md` — components, data flow and model boundaries.
- `docs/development.md` — local research and backend setup.
- `docs/deployment.md` — VPS, systemd, Caddy and backup details.
- `docs/film-library.md` — catalogue, UI and manual rating flow.
- `docs/admin-dashboard.md` — admin metrics and permissions.
- `docs/quick-taste-profile.md` — frozen Quick Taste selection and API contract.
- `docs/repository-migration.md` — historical monorepo migration notes.
- `docs/frontend.md` — frontend structure and interaction details.

## Notes and constraints

- Model artifacts and `movie_data.csv` are not tracked by Git.
- Generated research outputs under `research/results/` are not tracked.
- Do not retrain or modify M5/KNN artifacts during normal deployment.
- The frontend is a static export; `next start` is not used in production.
- SQLite is the production database; no PostgreSQL migration is planned in the
  current architecture.
- No secret, API key, `.env` file or model artifact should be committed.
