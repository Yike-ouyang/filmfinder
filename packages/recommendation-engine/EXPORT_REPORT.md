# FilmFinder deployment export

## Artifact sizes

| Artifact | Size |
|---|---:|
| `feature_mappings.json` | 2.05 MiB |
| `m5_config.json` | 0.00 MiB |
| `m5_weights.npz` | 3.44 MiB |
| `movie_features.npz` | 0.46 MiB |
| `movie_ids.npy` | 1.06 MiB |
| `movie_metadata.json` | 7.96 MiB |
| `ratings_matrix.npz` | 22.08 MiB |
| `recommendation_config.json` | 0.00 MiB |
| `user_ids.npy` | 0.14 MiB |
| **Total artifacts** | **37.19 MiB** |

## Runtime

- Engine load time: 1.002 s
- Existing-user recommendation: 0.0758 s
- New-user recommendation: 0.0381 s
- Peak RSS: 4.50 GiB

## Model and data

- Users: 7369
- Movies: 40981
- Ratings: 9728981
- Features: 52375
- Training time: 80.1 s
- Final training epoch: 20
- Training MAE: 1.011696
- Training RMSE: 1.324578

## Verification

- M5 max prediction difference: 3.553e-15
- Existing-user Top-20 matches: 3/3
- New-user Top-20 matches: 1/1
- Largest artifacts: ratings_matrix.npz (22.08 MiB), movie_metadata.json (7.96 MiB), m5_weights.npz (3.44 MiB)

The final M5 model was retrained on all valid ratings. Experimental validation/test metrics do not apply to this model.
