# Development

## Repository setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r research/requirements.txt
pip install -e packages/recommendation-engine
pip install -e apps/backend
```

## Data paths

```bash
export FILMFINDER_CLEANING_DIR=/Data/yike-ouyang/ml_training
export FILMFINDER_MODEL_OUTPUT_DIR=/Data/yike-ouyang/ml_training/modeling
export FILMFINDER_RECOMMENDATION_DIR=research/results/recommendation_system
export FILMFINDER_KNN_M5_DIR=research/results/knn_m5_optimization
export FILMFINDER_ARTIFACTS_DIR=packages/recommendation-engine/artifacts
```

See `.env.example`.

## Research

```bash
marimo edit research/modeling/2.Processing.py
marimo edit research/modeling/3.ExploringBias.py
marimo edit research/modeling/4.DebugTraining.py
marimo edit research/modeling/5.ContextFM.py
```

```bash
python research/modeling/6.IncrementalVsRetraining.py \
  --cleaning-dir "$FILMFINDER_CLEANING_DIR" \
  --model-output-dir "$FILMFINDER_MODEL_OUTPUT_DIR"

python research/modeling/7.EvaluateM5.py \
  --cleaning-dir "$FILMFINDER_CLEANING_DIR" \
  --model-output-dir "$FILMFINDER_MODEL_OUTPUT_DIR"
```

## Recommendation experiments

```bash
python research/recommendation/8.RecommendationSystem.py \
  --cleaning-dir "$FILMFINDER_CLEANING_DIR" \
  --model-output-dir "$FILMFINDER_MODEL_OUTPUT_DIR" \
  --output-dir research/results/recommendation_system

python research/recommendation/9.KNNM5Optimization.py \
  --baseline-dir research/results/recommendation_system \
  --output-dir research/results/knn_m5_optimization
```

## Recommendation engine

```bash
pip install -e packages/recommendation-engine
python packages/recommendation-engine/scripts/generate_recommendations.py \
  --letterboxd-zip packages/recommendation-engine/tests/fixtures/letterboxd-eslpej-2026-11-20-10-33-utc.zip \
  --artifacts packages/recommendation-engine/artifacts \
  --top-n 100 \
  --output recommendations.json \
  --report import_report.json
```

## Backend

```bash
pip install -e packages/recommendation-engine
pip install -e apps/backend
cp apps/backend/.env.example apps/backend/.env
alembic -c apps/backend/alembic.ini upgrade head
cd apps/backend
uvicorn app.main:app --reload
```

For local authentication testing, keep `EMAIL_BACKEND=console`,
`COOKIE_SECURE=false`, and `REQUIRE_EMAIL_VERIFICATION=false`. Accounts are
usable immediately after signup. Password reset links are written to the
Uvicorn logs instead of being sent by email. The frontend runs on
`http://localhost:3000` with `CORS_ORIGINS=http://localhost:3000` and
`APP_BASE_URL=http://localhost:3000`. Set
`REQUIRE_EMAIL_VERIFICATION=true` to test the verification flow again.

```bash
curl -X POST \
  "http://localhost:8000/api/v1/recommendations/from-letterboxd?limit=100" \
  -F "file=@packages/recommendation-engine/tests/fixtures/letterboxd-eslpej-2026-11-20-10-33-utc.zip"
```

Offline TMDB enrichment:

```bash
python apps/backend/scripts/enrich_tmdb_metadata.py \
  --catalogue-csv movie_data.csv \
  --artifacts packages/recommendation-engine/artifacts \
  --resume
```

## Tests

```bash
python -m pytest -q apps/backend/tests
python -m pytest -q packages/recommendation-engine/tests
```

```bash
cd apps/frontend
npm run test
npm run typecheck
npm run lint
npm run build
```

```bash
python -m unittest discover -s packages/recommendation-engine/tests -p "test_*.py" -v
FILMFINDER_EXPLORER_TESTS=1 python -m unittest \
  research/explorer/test_recommendation_explorer_core.py -v
```
