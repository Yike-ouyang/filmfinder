# Repository migration

## Backup point

A safety commit was created before the migration:

```text
chore: checkpoint before monorepo restructure
```

Branch:

```text
refactor/monorepo
```

No destructive Git command was used. Tracked files were moved with `git mv`;
ignored artifacts were moved with `mv`.

## Path mapping

| Before | After |
|---|---|
| `scripts/data_cleaning.py` | `research/preprocessing/data_cleaning.py` |
| `scripts/1.Preprocessing.py` | `research/preprocessing/1.Preprocessing.py` |
| `scripts/2.Processing.py` | `research/modeling/2.Processing.py` |
| `scripts/3.ExploringBias.py` | `research/modeling/3.ExploringBias.py` |
| `scripts/4.DebugTraining.py` | `research/modeling/4.DebugTraining.py` |
| `scripts/5.ContextFM.py` | `research/modeling/5.ContextFM.py` |
| `scripts/6.IncrementalVsRetraining.py` | `research/modeling/6.IncrementalVsRetraining.py` |
| `scripts/7.EvaluateM5.py` | `research/modeling/7.EvaluateM5.py` |
| `scripts/legacy/grid_search.py` | `research/legacy/grid_search.py` |
| `scripts/legacy/2.ModelPredicting.py` | `research/legacy/2.ModelPredicting.py` |
| `recommandation/scripts/8.RecommendationSystem.py` | `research/recommendation/8.RecommendationSystem.py` |
| `recommandation/scripts/9.KNNM5Optimization.py` | `research/recommendation/9.KNNM5Optimization.py` |
| `recommandation/scripts/export_filmfinder_deployment.py` | `research/recommendation/export_filmfinder_deployment.py` |
| `recommandation/scripts/9.RecommendationExplorer.py` | `research/explorer/9.RecommendationExplorer.py` |
| `recommandation/scripts/recommendation_explorer_core.py` | `research/explorer/recommendation_explorer_core.py` |
| `recommandation/scripts/test_recommendation_explorer_core.py` | `research/explorer/test_recommendation_explorer_core.py` |
| `filmfinder_deployment/filmfinder/*` | `packages/recommendation-engine/src/filmfinder/*` |
| `filmfinder_deployment/letterboxd_import.py` | `packages/recommendation-engine/src/filmfinder/letterboxd_import.py` |
| `filmfinder_deployment/generate_recommendations.py` | `packages/recommendation-engine/scripts/generate_recommendations.py` |
| `filmfinder_deployment/verify_export.py` | `packages/recommendation-engine/scripts/verify_export.py` |
| `filmfinder_deployment/test_letterboxd_pipeline.py` | `packages/recommendation-engine/tests/test_letterboxd_pipeline.py` |
| `filmfinder_deployment/artifacts/` | `packages/recommendation-engine/artifacts/` |
| `filmfinder_deployment/EXPORT_REPORT.md` | `packages/recommendation-engine/EXPORT_REPORT.md` |
| `results/` | `research/results/` |
| `recommandation/recommendation_system/` | `research/results/recommendation_system/` |
| `recommandation/knn_m5_optimization/` | `research/results/knn_m5_optimization/` |
| `recommandation/prompts/` | `prompts/recommendation/` |
| `requirements.txt` | `research/requirements.txt` |
| `ARCHITECTURE.md` | `docs/architecture.md` |

## Import and default-path changes

- `research/recommendation/8.RecommendationSystem.py`
  - research modules: `research/modeling`;
  - default output: `research/results/recommendation_system`.
- `research/recommendation/9.KNNM5Optimization.py`
  - research modules: `research/modeling`;
  - baseline: `research/results/recommendation_system`;
  - output: `research/results/knn_m5_optimization`.
- `research/explorer/recommendation_explorer_core.py`
  - 6/7 modules: `research/modeling`;
  - 8/9 modules: `research/recommendation`;
  - result defaults: `research/results/...`.
- `research/recommendation/export_filmfinder_deployment.py`
  - research modules: `research/modeling`;
  - 8/9 modules: same directory;
  - default output: `packages/recommendation-engine`.
- `FilmFinderEngine.load()`:
  - explicit path when provided;
  - otherwise `FILMFINDER_ARTIFACTS_DIR`;
  - otherwise the package `artifacts/` directory in a source checkout.
- Package CLI and verifier:
  - default artifacts: `packages/recommendation-engine/artifacts`;
  - development fallback inserts package `src/` in `sys.path`;
  - artifact-backed tests use `tests/fixtures/` and the same artifact default.
- Backend:
  - lazy engine loader uses `FILMFINDER_ARTIFACTS_DIR`;
  - no model load at `/health`.

The historical JSON files under `research/results/` still record the paths used
at the time of the experiments. They were not rewritten, to preserve
experimental outputs. New runs use the new defaults unless explicit paths are
provided.

## Tests performed

### Recommendation engine

```bash
python -m unittest discover -s packages/recommendation-engine/tests -p "test_*.py" -v
```

Result: **6 tests passed**.

Covered:

- Letterboxd parsing;
- score normalization;
- watched-only filtering;
- 100 recommendations;
- deterministic output;
- model-weight integrity;
- unknown-movie rejection.

### Letterboxd CLI

The migrated CLI produced exactly the same 100 recommendations as the
pre-migration sample output:

```text
recommendations.json: byte-identical
```

Artifact hashes before and after migration:

```text
all artifact SHA-256 hashes identical
```

### Backend

```bash
python -m pytest -q apps/backend/tests
```

Result: **1 test passed**.

The health endpoint returned:

```json
{"status": "ok"}
```

The recommendation engine was also loaded successfully through
`apps.backend.app.services.recommendation.get_engine()` when
`FILMFINDER_ARTIFACTS_DIR` was configured.

### Clean package copy

The package was copied to `/tmp` and tested with only:

```bash
PYTHONPATH=/tmp/filmfinder-engine-clean/src
```

Result: **6 tests passed**, and the CLI generated the same 100
recommendations.

### Research

- `research/explorer` unit tests: **7 run, 4 artifact-backed tests skipped**
  because `/Data` is not available locally.
- `research/explorer/recommendation_explorer_core.py` was loaded against a
  synthetic dataset and recomputed a pipeline successfully.
- `research/recommendation/9.KNNM5Optimization.py` was exercised with synthetic
  data and explicit new module paths; its imports resolved correctly. The
  synthetic plotting call hit an unrelated existing matplotlib warning after
  metrics generation.
- `marimo check research/explorer/9.RecommendationExplorer.py`: **passed**.
- `marimo check research/preprocessing/data_cleaning.py`: **passed**.
- Existing notebooks `2.Processing.py`, `3.ExploringBias.py`,
  `4.DebugTraining.py`, `5.ContextFM.py` report pre-existing marimo warnings
  or cell-style issues. They were not rewritten during migration.

## Remaining limitations

- Artifacts and the Letterboxd ZIP fixture are intentionally ignored by Git.
  Artifact-backed tests require those files to exist locally.
- The full research notebooks require the complete `/Data` dataset and were
  not executed end-to-end locally.
- `research/results/` contains historical config paths from before the
  migration; those files were not modified.
- Frontend and infrastructure remain placeholders by design.
- CloudLite Docker support must still be verified before production deployment.
