# Infrastructure

Future deployment target:

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

Planned technologies:

- Docker Compose
- Caddy reverse proxy
- PostgreSQL
- FastAPI backend
- Next.js frontend
- TMDB CDN for images

No production deployment configuration is created yet. CloudLite Docker
support must be verified before choosing the hosting setup.
