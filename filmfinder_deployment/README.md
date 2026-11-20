# FilmFinder deployment package

Self-contained inference package for the final FilmFinder KNN + M5
recommender. It does not depend on marimo, matplotlib, pandas, notebooks, or
the experimental project tree.

## Install

```bash
python -m pip install -r requirements.txt
```

Only NumPy and SciPy are required at inference time.

## Load

```python
from filmfinder import FilmFinderEngine

engine = FilmFinderEngine.load("artifacts")
```

## Existing user

```python
recommendations = engine.recommend(user_id="USER_ID", top_n=20)
```

The user must be present in `artifacts/user_ids.npy`. Their complete rating
history is used as the profile. Already rated movies are excluded. M5 scores
use the user-specific bias and latent factors learned during final training.

## New user

```python
recommendations = engine.recommend_new_user(
    ratings={
        "MOVIE_ID_1": 8.0,
        "MOVIE_ID_2": 9.0,
        "MOVIE_ID_3": 6.0,
    },
    top_n=20,
)
```

Movie IDs must be present in `artifacts/movie_ids.npy`. The profile is adapted
with the ridge cold-start procedure; global M5 parameters are frozen and never
retrained. Already rated movies are excluded. An empty profile is allowed and
uses the popularity fallback.

## Output

```python
[
    {
        "movie_id": 123,
        "title": "Example Movie",
        "year": 1999,
        "genres": ["Drama"],
        "score": 9.2,
        "rank": 1,
    }
]
```

The default candidate pool is 100. `top_n` may not exceed this pool size.

## Recommendation configuration

The exported configuration is the global configuration selected experimentally
in `knn_m5_optimization/selection/selected_configs.json`:

```text
algorithm:              knn_m5
similarity:             centered_cosine
n_neighbors:            50
top_k_per_neighbor:     20
candidate_pool_size:    100
similarity_shrink:      10.0
ranking:                M5
fallback:               Bayesian popularity, m=100
cold adaptation:        ridge, bias=0.01, factors=0.1
default_top_n:          20
```

Candidate ordering before the pool cap is:

```text
support -> popularity -> internal movie index
```

The exact configuration and its source path are in
`artifacts/recommendation_config.json`.

## Artifacts

| File | Content |
|---|---|
| `m5_weights.npz` | `w0`, `w`, `V`, active mask and model dimensions |
| `m5_config.json` | final training configuration, feature configuration, version |
| `recommendation_config.json` | KNN + M5 algorithm configuration |
| `feature_mappings.json` | feature order, families, user feature indices, year stats |
| `movie_metadata.json` | internal ID, canonical movie ID, title, year, genres, available source IDs |
| `movie_features.npz` | sparse precomputed non-user M5 features for every movie |
| `user_ids.npy` | user row ordering |
| `movie_ids.npy` | movie column ordering |
| `ratings_matrix.npz` | sparse user-movie ratings used by centered KNN |

`movie_features.npz` has one row per movie and zeroes the user feature family.
For a known user, the score is:

```text
item_base + w[user] + item_latent · V[user]
```

For a new user:

```text
item_base + theta[0] + item_latent · theta[1:]
```

where `theta` comes from the ridge adaptation.

## Final model note

The final M5 model is retrained on all valid ratings, not only the experimental
training split. Validation and test metrics reported for the experimental M5
model do not apply to this deployment model. The training report and final
training diagnostics are in `m5_config.json`.

## Verify

```bash
python verify_export.py --artifacts artifacts
```

The verifier checks loading, existing-user recommendations, new-user
recommendations, rated-movie exclusion and score ordering.

## Export

The artifacts are generated from the experimental project by:

```bash
python recommandation/scripts/export_filmfinder_deployment.py \
  --cleaning-dir /Data/yike-ouyang/ml_training \
  --baseline-dir recommandation/recommendation_system \
  --knn-dir recommandation/knn_m5_optimization \
  --output-dir filmfinder_deployment
```

The exporter retrains M5 on all ratings, writes the artifacts, verifies M5
prediction equivalence and Top-20 agreement against the experimental
implementation, and writes `EXPORT_REPORT.md`.

Posters and backdrops are not included. Only lightweight metadata is exported.
