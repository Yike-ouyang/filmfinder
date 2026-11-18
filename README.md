# FilmFinder

Film recommendation system built from a Letterboxd profile.

## Repository layout

```text
.
├── scripts/
│   ├── data_cleaning.py                 # CSV datasets -> npz / npy / json exports
│   ├── 1.Preprocessing.py               # simplified alternative preprocessing
│   ├── 2.Processing.py                  # FM training and diagnostics
│   ├── 3.ExploringBias.py               # bias analysis
│   ├── 4.DebugTraining.py               # regularization experiments
│   ├── 5.ContextFM.py                   # user-film FM vs contextual FM
│   ├── 6.IncrementalVsRetraining.py     # local adaptation vs global retraining
│   ├── 7.EvaluateM5.py                  # warm-start / cold-start evaluation of M5
│   └── legacy/
│       ├── 2.ModelPredicting.py         # old Surprise SVD notebook
│       └── grid_search.py               # old SVD grid search
├── docs/
│   ├── notes.md                         # local notes (ignored by git)
│   ├── filmfinder_flowchart.pdf
│   └── images/                          # documentation screenshots
├── recommandation/
│   ├── prompts/                         # recommendation specifications (ignored by git)
│   ├── scripts/
│   │   └── 8.RecommendationSystem.py    # Top-20 recommendation evaluation
│   └── results/                         # generated outputs (ignored by git)
├── prompts/                             # task specifications (ignored by git)
├── results/                             # generated outputs (ignored by git)
├── README.md
└── requirements.txt
```

## Data

Raw datasets and generated data files are **never committed**.
`.gitignore` excludes `data/`, sparse matrices, NumPy/Parquet files and the
generated `results/` outputs.

The FM pipeline consumes a cleaning directory containing:

```text
ratings_matrix.npz
user_ids.npy
movie_ids.npy
movie_context.json
```

`data_cleaning.py` builds this contract from the raw Letterboxd CSV exports.
`1.Preprocessing.py` is a lighter alternative that exports parquet mappings
instead.

Default location is the current directory. The notebooks also read:

```bash
export FILMFINDER_CLEANING_DIR=/path/to/cleaning
export FILMFINDER_MODEL_OUTPUT_DIR=/path/to/modeling
```

## Input datasets

- [Letterboxd movies](https://www.kaggle.com/datasets/gsimonx37/letterboxd):
  `movies.csv`, `studios.csv`, `genres.csv`.
- [Letterboxd ratings](https://www.kaggle.com/datasets/samlearner/letterboxd-movie-ratings-data):
  `movie_data.csv`, `ratings_export.csv`.

After cleaning and merging: roughly 7.2k users, 25k films and 5M triplets.
The sparse user-film matrix is about 97% empty.

## Model

Factorization Machines:

```text
y(x) = w0 + sum_i w_i x_i + sum_{i<j} <v_i, v_j> x_i x_j
```

- `w0` is the global baseline score.
- `w_i` is the weight of feature `i`.
- `v_i` is the latent vector of feature `i`, of size `k`.

An observation is the one-hot encoding of `(user, film, context)`, and the
target is the rating:

```text
rating = w0 + w_user + w_film + <v_user, v_film>
```

The training loss is:

```text
L = 1/2 * sum_{(x,y)} (y_hat(x) - y)^2
    + lambda_w/2 * ||w||^2
    + lambda_V/2 * ||V||_F^2
```

## Running

Install dependencies:

```bash
pip install -r requirements.txt
```

Marimo notebooks:

```bash
marimo edit scripts/2.Processing.py
marimo edit scripts/3.ExploringBias.py
marimo edit scripts/4.DebugTraining.py
marimo edit scripts/5.ContextFM.py
```

Standalone experiments:

```bash
python scripts/6.IncrementalVsRetraining.py \
  --cleaning-dir "$FILMFINDER_CLEANING_DIR" \
  --model-output-dir "$FILMFINDER_MODEL_OUTPUT_DIR"

python scripts/7.EvaluateM5.py \
  --cleaning-dir "$FILMFINDER_CLEANING_DIR" \
  --model-output-dir "$FILMFINDER_MODEL_OUTPUT_DIR" \
  --output-dir "$FILMFINDER_MODEL_OUTPUT_DIR/evaluate_m5"

python recommandation/scripts/8.RecommendationSystem.py \
  --cleaning-dir "$FILMFINDER_CLEANING_DIR" \
  --model-output-dir "$FILMFINDER_MODEL_OUTPUT_DIR" \
  --output-dir recommandation/results/recommendation_system
```

`7.EvaluateM5.py` trains the contextual FM `M5` once, measures warm-start
accuracy by user activity, calibrates the cold-start ridge on held-out known
users, and evaluates progressively revealed profiles with several seeds.
Use `--fast` for a quick smoke test and `--load-model` to reuse a checkpoint.

`8.RecommendationSystem.py` evaluates four Top-20 strategies (popularity,
M5 direct, KNN + M5, KNN + popularity) on leakage-free warm and cold cohort
splits, tunes recommendation hyperparameters on validation cohorts, and saves
tables, figures and example recommendations. It reuses `FMEngine`, the split
and feature builders from `scripts/6.IncrementalVsRetraining.py`, and the
cold-profile ridge helpers from `scripts/7.EvaluateM5.py`. Use `--fast` for a
smoke test and `--help` for all options.

## Outputs

Generated artifacts are written outside the versioned tree or under
`results/`, which is ignored:

- training histories and learning curves;
- warm-start and cold-start metrics;
- calibration grids;
- comparison tables and figures.

## Legacy

The `scripts/legacy/` directory keeps the early Surprise SVD baseline
(`2.ModelPredicting.py`, `grid_search.py`). The current pipeline uses the
custom FM implementation.
