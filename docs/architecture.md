# FilmFinder architecture

## 1. Components

FilmFinder is split into four independent areas:

| Area | Responsibility |
|---|---|
| `research/` | Data preparation, M5 training, evaluation, KNN optimization, explainability |
| `packages/recommendation-engine/` | Production inference engine and model artifacts |
| `apps/backend/` | Future FastAPI API |
| `apps/frontend/` | Future Next.js application |

The research code is not imported by production inference. The production
engine does not depend on marimo, pandas, matplotlib, FastAPI, PostgreSQL or
the research tree.

## 2. Data contract

Cleaning directory (`FILMFINDER_CLEANING_DIR`):

```text
ratings_matrix.npz       sparse CSR user x movie, ratings 1-10
user_ids.npy             row identifiers
movie_ids.npy            column identifiers
movie_context.json       title, year, genres, studios, description
```

Experimental artifacts:

```text
research/results/recommendation_system/
research/results/knn_m5_optimization/
```

Production artifacts:

```text
packages/recommendation-engine/artifacts/
```

Production artifacts are loaded from an explicit path or
`FILMFINDER_ARTIFACTS_DIR`.

The browsable film catalogue is derived into SQLite by
`apps/backend/scripts/sync_film_catalog.py`:

```text
catalog_films          browsable metadata, summaries, ratings metrics
catalog_film_genres    normalized genre relation for filters and facets
```

The raw source is `movie_data.csv` (gitignored). The catalogue is queried with
SQL pagination and never loaded wholesale into the API process.

## 3. Research pipeline

```text
raw CSVs
  -> research/preprocessing/data_cleaning.py
  -> ratings_matrix.npz + mappings + movie_context.json
  -> research/modeling/2.Processing.py (FM / M5 training)
  -> research/modeling/7.EvaluateM5.py (warm/cold evaluation)
  -> research/recommendation/8.RecommendationSystem.py (Top-20 strategies)
  -> research/recommendation/9.KNNM5Optimization.py (KNN + M5 optimization)
  -> research/results/
```

`research/modeling/6.IncrementalVsRetraining.py` contains the shared
`FMEngine`, feature builder and split utilities. `7.EvaluateM5.py` contains the
cold-start ridge helpers.

`research/explorer/` wraps the experimental pipeline for per-user inspection.
It loads the research modules from `research/modeling` and
`research/recommendation`.

## 4. M5 model

M5 is a contextual Factorization Machine:

```text
features = user one-hot
         + film one-hot
         + genres
         + year
         + studios
```

Reference hyperparameters:

```text
n_factors       8
learning_rate   0.01
n_epochs        20
batch_size      65536
reg_bias        0
reg_factors     1e-6
optimizer       Adam
```

The deployment model is retrained on all valid ratings by
`research/recommendation/export_filmfinder_deployment.py`. It is frozen during
inference.

For a known user:

```text
item_base + w[user] + item_latent . V[user]
```

For a cold user:

```text
item_base + theta[0] + item_latent . theta[1:]
```

`theta` is fitted by the existing ridge procedure. No global parameter is
modified.

## 5. Recommendation engine

Package:

```text
packages/recommendation-engine/
├── src/filmfinder/
│   ├── engine.py
│   ├── model.py
│   ├── features.py
│   ├── recommender.py
│   ├── cold_start.py
│   └── letterboxd_import.py
├── scripts/
│   ├── generate_recommendations.py
│   └── verify_export.py
├── tests/
├── artifacts/
├── pyproject.toml
└── README.md
```

Public API:

```python
from filmfinder import FilmFinderEngine

engine = FilmFinderEngine.load(artifacts_path)
engine.recommend(user_id, top_n=20)
engine.recommend_new_user(ratings, top_n=100, watched_movie_ids=watched)
```

Recommendation configuration remains the experimentally selected one:

```text
algorithm       knn_m5
similarity      centered
n_neighbors     50
top_k           20
pool_size       100
shrink          10.0
ranking         M5
```

Flow:

```text
profile -> centered similarity -> 50 neighbors -> 20 movies/neighbor
        -> union/support -> remove rated and watched -> pool 100
        -> M5 scoring -> deterministic top N
```

## 6. Letterboxd import

`letterboxd_import.py`:

- reads `ratings.csv` and `watched.csv` from the ZIP;
- maps Letterboxd ratings 0.5-5 to M5 ratings 1-10;
- matches by normalized title and year, year ±1, then conservative fuzzy title;
- keeps watched movies separate from rated movies;
- reports unmatched and ambiguous movies.

## 7. Backend and frontend

`apps/backend/` is the FastAPI application. It loads the recommendation engine
once during lifespan startup, exposes `GET /health`, `GET /api/v1/ready`,
`POST /api/v1/recommendations/from-letterboxd?limit=100`, the
`/api/v1/films/*`, `/api/v1/me/*`, `/api/v1/admin/*`, and the
`/api/v1/auth/*` endpoints. Letterboxd ZIPs are validated in memory before
reuse of the existing parser.

Phase 1 adds a versioned SQLite schema managed by Alembic. Phase 2 adds
accounts, optional email verification, server-side sessions, CSRF protection,
and password reset. Verification is disabled by default and can be re-enabled
with `REQUIRE_EMAIL_VERIFICATION=true`. The public recommendation endpoint
remains usable without an account.

Phase 3 persists authenticated Letterboxd imports in `user_films`, merges
successive imports, and generates recommendations from the cumulative saved
profile.

Phase 4 persists swipe decisions in `user_swipes` with an append-only
`user_swipe_events` history for Undo. The `My Lists` page exposes the
`interested` and `not_interested` categories.

Phase 5 adds server-side recommendation batches. Login schedules a background
recalculation, and the engine excludes `not_interested`, `already_seen`, rated,
and watched films. Batch history is stored in `recommendation_batches` and
`recommendation_items`. Phase 6 removes the discovery counter and adds a
server-backed "Get more recommendations" action at the end of a batch, with an
explicit no-new-candidates state.

Phase 7 adds the browsable film library and manual profile. A
`catalog_films` table is synchronized offline from `movie_data.csv` and the
production artifacts. `/api/v1/films` provides SQL pagination, search, genre
filters and popularity/rating/date sorting. Signed-in users rate films through
`/api/v1/me/ratings` and reuse `user_swipes` for Watchlist / Do Not Watch.
Manual generation enforces 10 ratings and reuses the existing cold-start +
KNN/M5 batch pipeline. See `docs/film-library.md`.

Phase 8 adds the private admin dashboard. `users.is_admin` is assigned only by
the `grant_admin.py` shell script. `/api/v1/admin/*` endpoints are protected by
`get_current_admin` and expose aggregate account, login, recommendation,
rating and list metrics. `analytics_events` stores only Letterboxd import
events; registrations, logins, recommendations, ratings and lists are derived
from existing tables. Active users are defined as distinct users with a
successful login in the period. The frontend `/admin` page uses native SVG
charts. See `docs/admin-dashboard.md`.

TMDB image metadata is loaded from a local JSON produced by
`apps/backend/scripts/enrich_tmdb_metadata.py`. Recommendation requests never
call TMDB.

`apps/frontend/` is the Next.js application. It uses static export and exposes
`/films` for the catalogue and manual profile, `/discover`, `/lists`, the
account pages, and the private `/admin` dashboard.

## 8. Future deployment

```text
Caddy
  ├── Next.js
  └── FastAPI
        ├── FilmFinder recommendation engine
        └── SQLite
```

TMDB will provide image URLs. No image storage service is planned.

## 9. Dependency boundaries

| Component | Dependencies |
|---|---|
| Recommendation engine | numpy, scipy |
| Backend | fastapi, uvicorn, pydantic, recommendation engine |
| Research | numpy, scipy, pandas, matplotlib, marimo |
| Frontend | none yet |
| Infrastructure | no production manifests yet |
