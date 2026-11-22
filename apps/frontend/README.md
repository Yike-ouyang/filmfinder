# FilmFinder frontend

Cinematic Next.js frontend for FilmFinder. Import a Letterboxd ZIP export or
rate 20 films with the Quick Taste Profile, then discover recommendations
through a swipe-based interface.

## Stack

- Next.js 16 (App Router)
- React 19
- TypeScript
- Tailwind CSS 4
- Motion for React
- Lucide React
- Vitest and Testing Library

## Requirements

- Node.js 22.12 or newer (Node 22 LTS recommended)
- npm 10 or newer
- The FastAPI backend running with the recommendation artifacts available

## Install and run

```bash
cd apps/frontend
npm install
cp .env.example .env.local
npm run dev
```

Open `http://localhost:3000`.

Start the backend from the repository root in another terminal:

```bash
cd apps/backend
uvicorn app.main:app --reload
```

The backend defaults to `http://localhost:8000` and allows
`http://localhost:3000` through CORS. If the frontend runs on another origin,
update `CORS_ORIGINS` in `apps/backend/.env`.

## Environment variables

```env
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
NEXT_PUBLIC_MAX_UPLOAD_SIZE_MB=20
```

- `NEXT_PUBLIC_API_BASE_URL`: FastAPI server URL. A trailing slash is accepted.
- `NEXT_PUBLIC_MAX_UPLOAD_SIZE_MB`: client-side upload guard. Keep it aligned
  with the backend `MAX_UPLOAD_SIZE_MB` setting.

`NEXT_PUBLIC_*` values are inlined at build time. Restart or rebuild the app
after changing them. No TMDB token is used by the frontend.

## Routes

```text
/               Letterboxd import / Quick Taste entry point
/taste-profile  Rate 20 films without a Letterboxd account
/discover       Swipe-based recommendation discovery
/lists          Saved interested / not-interested films
/account        Account settings
/login /signup  Authentication
```

The floating navigation links both routes. Opening `/discover` without a
saved session shows an import prompt instead of empty cards.

## Backend integration

The frontend calls:

```http
POST /api/v1/recommendations/from-letterboxd?limit=100
GET  /api/v1/taste-profile/movies
POST /api/v1/taste-profile/recommendations?limit=100
```

The multipart field is named `file`. The request is sent once per import and
the recommendation batch is reused during swiping. Swipes never call the
backend.

The implementation is adapted to the real backend schema:

- `internal_movie_id` is used as the stable local key.
- `movie_id` may be a string or a number.
- `poster_path`, `backdrop_path`, and `tmdb_id` may be `null`.
- The API does not return synopses, so the details panel only shows fields
  that actually exist.
- Backend errors use `{ "error": { "code", "message" } }`. The UI maps known
  codes to friendly messages and never displays raw backend or Python errors.
- `returned_count` and the real array length may be lower than the requested
  limit. The progress counter always uses the actual array length.
- The recommendation request is synchronous. The client allows up to 180
  seconds and supports cancellation.

Known backend error codes handled by the client include `invalid_zip`,
`missing_ratings`, `upload_too_large`, `no_matched_ratings`,
`no_recommendations`, `validation_error`, and `engine_unavailable`.

## Local state and privacy

- The current recommendation batch, its profile summary, and local swipe
  decisions are stored in `sessionStorage`.
- The ZIP file is never stored in browser storage.
- Anonymous Letterboxd ratings are processed in memory; signed-in imports are
  merged into SQLite.
- Quick Taste answers are kept in React state. Anonymous answers are sent to
  the backend for inference only; signed-in answers are merged into the saved
  profile. An anonymous draft can be kept in `sessionStorage` until the visitor
  creates an account.
- Session cookies and CSRF cookies are used for authenticated API requests.

## TMDB images

Posters and backdrops are loaded directly from the TMDB image CDN:

```text
https://image.tmdb.org/t/p/w500/{poster_path}
https://image.tmdb.org/t/p/w1280/{backdrop_path}
```

`next.config.ts` allows `image.tmdb.org` through `remotePatterns`. Images are
not downloaded into this repository. Missing or invalid paths fall back to a
locally rendered placeholder. The next three posters and backdrops are
prefetched, avoiding loading the whole recommendation batch at once. The
backend loads `tmdb_metadata.json` at startup, so restart it after regenerating
the metadata file.

TMDB attribution is included in the landing page footer, the discover footer,
and the details panel.

## Scripts

```bash
npm run dev        # development server
npm run build      # production build
npm run start      # production server
npm run lint       # ESLint
npm run typecheck  # TypeScript without emitting
npm test           # Vitest test suite
```

## Tests

The suite covers:

- multipart request construction and API error mapping
- timeouts, cancellation, and malformed responses
- TMDB URL helpers
- upload validation
- session persistence and expiry
- decision history, undo, and end-of-batch state
- upload-to-discover navigation
- details modal and missing-image fallbacks
- keyboard controls and absence of network calls during swiping

## Known limitations

- Recommendation sessions are per browser tab and expire after 24 hours.
- Anonymous Quick Taste drafts live in `sessionStorage`; they are lost when the
  tab closes.
- Account features require the SQLite backend and Alembic migrations.
- The backend must be running and CORS must allow the frontend origin.
- The client-side upload limit must match the backend limit; the backend
  remains the source of truth.
- Poster and backdrop coverage depends on the backend TMDB metadata file.
  Films without metadata use the built-in fallbacks.
- There is intentionally no demo mode or fabricated recommendation fallback.
- The visual reference in `prompts/frontend.md` takes precedence over the
  older `docs/frontend.md` catalogue specification.
