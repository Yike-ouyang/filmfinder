# Deployment

The target deployment is intentionally documented, not implemented.

```text
Caddy
  |
  +-- Next.js
  |
  +-- FastAPI
        |
        +-- Recommendation Engine
        |
        +-- PostgreSQL
```

Planned components:

- Docker Compose;
- Caddy reverse proxy;
- Next.js frontend;
- FastAPI backend;
- PostgreSQL;
- model artifacts mounted read-only and exposed through
  `FILMFINDER_ARTIFACTS_DIR`.

TMDB images will be loaded from the TMDB CDN. No image storage service is
planned.

CloudLite must be checked for Docker Compose and persistent-volume support
before selecting the hosting setup.
