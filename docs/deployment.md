# Deployment

FilmFinder runs as a small single-VPS deployment. No Docker, Redis, Celery, or
permanent Node server is required.

```text
Caddy (HTTPS, static files, reverse proxy)
  |
  +-- /var/www/filmfinder        Next.js static export
  |
  +-- /api/* and /health         FastAPI on 127.0.0.1:8000
                                    |
                                    +-- Recommendation engine
                                    +-- SQLite (/var/lib/filmfinder)
```

## Systemd backend

```bash
/home/filmfinder/filmfinder/.venv/bin/uvicorn app.main:app \
  --host 127.0.0.1 \
  --port 8000 \
  --workers 1
```

Working directory:

```text
/home/filmfinder/filmfinder/apps/backend
```

The backend uses a single worker and a single background recommendation
executor. ML inference is serialized by the existing semaphore.

Caddy must only serve `apps/frontend/out` as static files and proxy `/api/*`
and `/health` to FastAPI. Never expose `apps/backend/var`, `movie_data.csv`, or
`packages/recommendation-engine/artifacts` through a static file route.

## Production environment

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
validates the `Origin` header.

## Database

```bash
sudo mkdir -p /var/lib/filmfinder
sudo chown filmfinder:filmfinder /var/lib/filmfinder
sudo chmod 750 /var/lib/filmfinder
cd /home/filmfinder/filmfinder
.venv/bin/alembic -c apps/backend/alembic.ini upgrade head
```

Migrations are never applied automatically at application startup. Stop the
service before a major migration, then restart it after `upgrade head`.

## Film catalogue

The `/films` library reads from the `catalog_films` and `catalog_film_genres`
tables created by `0002_catalog_films`. Synchronize them after migrating:

```bash
cd /home/filmfinder/filmfinder
.venv/bin/python apps/backend/scripts/sync_film_catalog.py \
  --catalogue-csv movie_data.csv \
  --artifacts packages/recommendation-engine/artifacts
```

`movie_data.csv` is not versioned. Copy it to the VPS temporarily for the
sync, or import a dump of the two catalogue tables generated from a local
sync. The script is idempotent and streams the CSV; it does not call TMDB. Use
`--dry-run` first to inspect the report.

If the catalogue tables are empty, `/api/v1/films` returns
`catalog_unavailable` (HTTP 503). The rest of the API remains available.

## Backups

```bash
python apps/backend/scripts/backup_sqlite.py \
  --source /var/lib/filmfinder/filmfinder.db \
  --output-dir /var/lib/filmfinder/backups \
  --retention 14
```

SQLite's backup API includes WAL content consistently. Example systemd timer:

```ini
[Unit]
Description=FilmFinder SQLite backup

[Service]
Type=oneshot
User=filmfinder
WorkingDirectory=/home/filmfinder/filmfinder
ExecStart=/home/filmfinder/filmfinder/.venv/bin/python apps/backend/scripts/backup_sqlite.py --source /var/lib/filmfinder/filmfinder.db --output-dir /var/lib/filmfinder/backups --retention 14
```

Restore by stopping `filmfinder-api`, replacing the database with a backup,
keeping ownership `filmfinder:filmfinder`, then restarting the service.

## Frontend

```bash
cd apps/frontend
NEXT_PUBLIC_API_BASE_URL=https://filmfinder.example npm run build
rsync -avh out/ filmfinder@VPS:/var/www/filmfinder/
```

The static frontend needs no restart. Deploy the backend first when new API
routes are introduced.

## Verification

```bash
curl --fail-with-body https://filmfinder.example/health
curl --fail-with-body https://filmfinder.example/api/v1/ready
curl -o /dev/null -w '%{http_code}\n' https://filmfinder.example/login
curl --fail-with-body "https://filmfinder.example/api/v1/films?page=1&page_size=5"
sudo systemctl status filmfinder-api
sudo journalctl -u filmfinder-api -n 50 --no-pager
```
