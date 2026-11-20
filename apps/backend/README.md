# FilmFinder backend

Minimal FastAPI scaffold for the future FilmFinder application.

```bash
pip install -e packages/recommendation-engine
pip install -e apps/backend
uvicorn app.main:app --reload
```

The health endpoint is:

```http
GET /health
```

```json
{"status": "ok"}
```

The recommendation service is intentionally lazy. It will use the installed
`filmfinder` package and the directory configured by
`FILMFINDER_ARTIFACTS_DIR`.

```bash
export FILMFINDER_ARTIFACTS_DIR=/path/to/recommendation-artifacts
```

No database connection, authentication, or TMDB integration is implemented
yet.
