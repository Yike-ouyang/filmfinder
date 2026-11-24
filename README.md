# FilmFinder

Film recommendation system based on a contextual Factorization Machine (M5)
and a KNN candidate generator. The repository is organized as a monorepo for
research, the production recommendation engine, and the web application.

## Repository layout

```text
.
├── apps/
│   ├── backend/                 # FastAPI API, SQLite, catalogue and recommendations
│   └── frontend/                # Next.js static-export application
├── packages/
│   └── recommendation-engine/   # installable inference package
├── research/
│   ├── preprocessing/           # data cleaning pipelines
│   ├── modeling/                # FM / M5 training and evaluation
│   ├── recommendation/          # recommendation experiments and export
│   ├── explorer/                # marimo recommendation explorer
│   ├── legacy/                  # early SVD experiments
│   └── results/                 # generated experimental outputs (ignored)
├── infrastructure/              # Caddy / systemd deployment notes
├── docs/
│   ├── architecture.md
│   ├── development.md
│   ├── deployment.md
│   ├── film-library.md
│   ├── admin-dashboard.md
│   ├── quick-taste-profile.md
│   ├── repository-migration.md
│   └── frontend.md
├── prompts/                     # task specifications
├── .env.example
├── README.md
└── pyproject.toml
```

## Web application

The frontend offers three ways to build a profile:

- import a Letterboxd ZIP export;
- browse `/films` and rate catalogue films manually;
- complete the `/taste-profile` Quick Taste Profile by rating 20 selected
  films with half-star precision.

All three use the same cold-start + KNN/M5 engine. Quick Taste ratings are
persisted in the same `user_films` table as library ratings. The browsable
library is backed by a SQLite catalogue synchronized from `movie_data.csv` and
the production artifacts. See `docs/film-library.md` and
`docs/quick-taste-profile.md`.

Administrators get a private read-only `/admin` dashboard for account,
activity, recommendation, rating and list metrics. See
`docs/admin-dashboard.md`.

## Data and artifacts

The training pipeline reads the cleaned exports defined by
`FILMFINDER_CLEANING_DIR`:

```text
ratings_matrix.npz
user_ids.npy
movie_ids.npy
movie_context.json
```

Experimental recommendation artifacts are written under:

```text
research/results/recommendation_system/
research/results/knn_m5_optimization/
```

The production artifacts are under:

```text
packages/recommendation-engine/artifacts/
```

They contain the final M5 weights, the recommendation configuration, the
movie feature matrix, the rating matrix, and the ID/metadata mappings. Large
artifacts are ignored by Git. To mount them elsewhere:

```bash
export FILMFINDER_ARTIFACTS_DIR=/path/to/artifacts
```

## Research workflow

Run marimo notebooks with:

```bash
marimo edit research/preprocessing/data_cleaning.py
marimo edit research/modeling/2.Processing.py
marimo edit research/modeling/3.ExploringBias.py
marimo edit research/modeling/4.DebugTraining.py
marimo edit research/modeling/5.ContextFM.py
```

Standalone experiments:

```bash
python research/modeling/6.IncrementalVsRetraining.py \
  --cleaning-dir "$FILMFINDER_CLEANING_DIR" \
  --model-output-dir "$FILMFINDER_MODEL_OUTPUT_DIR"

python research/modeling/7.EvaluateM5.py \
  --cleaning-dir "$FILMFINDER_CLEANING_DIR" \
  --model-output-dir "$FILMFINDER_MODEL_OUTPUT_DIR" \
  --output-dir "$FILMFINDER_MODEL_OUTPUT_DIR/evaluate_m5"

python research/recommendation/8.RecommendationSystem.py \
  --cleaning-dir "$FILMFINDER_CLEANING_DIR" \
  --model-output-dir "$FILMFINDER_MODEL_OUTPUT_DIR" \
  --output-dir research/results/recommendation_system

python research/recommendation/9.KNNM5Optimization.py \
  --baseline-dir research/results/recommendation_system \
  --output-dir research/results/knn_m5_optimization
```

The recommendation explorer is available at:

```bash
marimo edit research/explorer/9.RecommendationExplorer.py
```

## Recommendation engine

Install the production inference package:

```bash
pip install -e packages/recommendation-engine
```

```python
from filmfinder import FilmFinderEngine

engine = FilmFinderEngine.load("packages/recommendation-engine/artifacts")
recommendations = engine.recommend(user_id="USER_ID", top_n=20)
recommendations = engine.recommend_new_user(
    ratings={"MOVIE_ID": 8.0},
    watched_movie_ids={"OTHER_MOVIE_ID"},
    top_n=100,
)
```

Letterboxd CLI:

```bash
python packages/recommendation-engine/scripts/generate_recommendations.py \
  --letterboxd-zip packages/recommendation-engine/tests/fixtures/letterboxd-eslpej-2026-11-20-10-33-utc.zip \
  --artifacts packages/recommendation-engine/artifacts \
  --top-n 100 \
  --output recommendations.json \
  --report import_report.json
```

The engine never retrains M5 globally. New users use the existing ridge
cold-start adaptation; watched and rated movies are excluded before ranking.

## Export the final artifacts

```bash
python research/recommendation/export_filmfinder_deployment.py \
  --cleaning-dir "$FILMFINDER_CLEANING_DIR" \
  --baseline-dir research/results/recommendation_system \
  --knn-dir research/results/knn_m5_optimization \
  --output-dir packages/recommendation-engine
```

The exporter retrains M5 on all ratings, writes artifacts, verifies prediction
and Top-20 equivalence, and writes `EXPORT_REPORT.md`.

## Backend

```bash
pip install -e packages/recommendation-engine
pip install -e apps/backend
uvicorn app.main:app --reload
```

`GET /health` returns `{"status": "ok"}`. The recommendation engine is loaded
once at startup; `GET /api/v1/ready` reports readiness.

Letterboxd upload:

```bash
curl -X POST \
  "http://localhost:8000/api/v1/recommendations/from-letterboxd?limit=100" \
  -F "file=@packages/recommendation-engine/tests/fixtures/letterboxd-eslpej-2026-11-20-10-33-utc.zip"
```

Offline TMDB metadata enrichment uses `movie_data.csv` and writes
`packages/recommendation-engine/artifacts/tmdb_metadata.json`; see the
backend README.

## Frontend

`apps/frontend/` is a placeholder for the future Next.js/React/TypeScript
application. Planned routes and visual direction are documented in
`apps/frontend/README.md` and `docs/frontend.md`.

## Infrastructure

`infrastructure/README.md` documents the future Docker Compose, Caddy,
FastAPI, Next.js, and PostgreSQL deployment. No production deployment is
configured yet.

## Tests

```bash
python -m unittest discover -s packages/recommendation-engine/tests -p "test_*.py" -v

FILMFINDER_EXPLORER_TESTS=1 python -m unittest \
  research/explorer/test_recommendation_explorer_core.py -v

python -m compileall research packages apps
```

Artifact-backed tests require the generated artifacts and, for the Letterboxd
test, the ignored fixture ZIP.

## Migration and architecture

- `docs/architecture.md`: component responsibilities and data flow.
- `docs/repository-migration.md`: old paths, new paths, import changes, tests.
- `docs/development.md`: local setup and commands.
- `docs/deployment.md`: planned deployment architecture.

Generated outputs under `research/results/` and
`packages/recommendation-engine/artifacts/` are not committed.
