import marimo

__generated_with = "0.17.6"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # FilmFinder — Factorization Machines and diagnostics

    This notebook trains explicit Factorization Machine models on the CSR
    matrix exported by `1.Preprocessing.py`. It compares user–film models with
    contextual models that use genres, release year and studios, then analyses
    errors, contributions and ablations.

    No file produced by `1.Preprocessing.py` is modified.
    """)
    return


@app.cell
def _():
    import hashlib
    import json
    import os
    import time
    from collections import Counter
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy.sparse import csr_matrix, hstack, load_npz

    return (
        Counter,
        Path,
        csr_matrix,
        hashlib,
        hstack,
        json,
        load_npz,
        mo,
        np,
        os,
        pd,
        plt,
        time,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Configuration and output paths

    The notebook reads the exports from `1.Preprocessing.py`. The modelling
    outputs are written to a dedicated subdirectory.
    """)
    return


@app.cell
def _(Path, os):
    PREPROCESSING_DIR = Path(
        os.environ.get(
            "FILMFINDER_OUTPUT_DIR",
            Path.cwd() / "outputs",
        )
    )
    MODEL_OUTPUT_DIR = Path(
        os.environ.get(
            "FILMFINDER_MODEL_OUTPUT_DIR",
            PREPROCESSING_DIR / "modeling",
        )
    )
    CHECKPOINT_DIR = MODEL_OUTPUT_DIR / "checkpoints"

    INPUT_FILES = {
        "matrix": PREPROCESSING_DIR / "ratings_matrix.npz",
        "users": PREPROCESSING_DIR / "user_mapping.parquet",
        "movies": PREPROCESSING_DIR / "movie_mapping.parquet",
        "context": PREPROCESSING_DIR / "movie_context.parquet",
    }

    missing_inputs = [
        str(path) for path in INPUT_FILES.values() if not path.exists()
    ]

    if missing_inputs:
        raise FileNotFoundError(
            "Missing preprocessed files. Run 1.Preprocessing.py first. "
            f"Missing: {missing_inputs}"
        )

    return CHECKPOINT_DIR, INPUT_FILES, MODEL_OUTPUT_DIR, PREPROCESSING_DIR


@app.cell
def _(INPUT_FILES, load_npz, pd):
    ratings_matrix = load_npz(INPUT_FILES["matrix"]).tocsr()
    user_mapping = pd.read_parquet(INPUT_FILES["users"])
    movie_mapping = pd.read_parquet(INPUT_FILES["movies"])
    movie_context = pd.read_parquet(INPUT_FILES["context"])
    return movie_context, movie_mapping, ratings_matrix, user_mapping


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Loading checks and observations

    Only stored entries of the CSR matrix are used. Explicit zeros are kept;
    absent positions are never interpreted as ratings.
    """)
    return


@app.cell
def _(movie_context, movie_mapping, np, pd, ratings_matrix, user_mapping):
    n_users, n_movies = ratings_matrix.shape

    user_mapping_sorted = (
        user_mapping.sort_values("row_index").reset_index(drop=True)
    )
    movie_mapping_sorted = (
        movie_mapping.sort_values("col_index").reset_index(drop=True)
    )
    context_sorted = (
        movie_context.sort_values("col_index").reset_index(drop=True)
    )

    checks_loading = {
        "matrix_shape": ratings_matrix.shape,
        "user_mapping_rows": len(user_mapping_sorted),
        "movie_mapping_rows": len(movie_mapping_sorted),
        "context_rows": len(context_sorted),
        "user_indices_contiguous": bool(
            np.array_equal(
                user_mapping_sorted["row_index"].to_numpy(),
                np.arange(n_users),
            )
        ),
        "movie_indices_contiguous": bool(
            np.array_equal(
                movie_mapping_sorted["col_index"].to_numpy(),
                np.arange(n_movies),
            )
        ),
        "context_aligned": bool(
            context_sorted["canonical_movie_id"].tolist()
            == movie_mapping_sorted["canonical_movie_id"].tolist()
        ),
        "user_ids_unique": bool(
            user_mapping_sorted["user_id"].is_unique
        ),
        "movie_ids_unique": bool(
            movie_mapping_sorted["canonical_movie_id"].is_unique
        ),
    }

    if not all([
        checks_loading["user_indices_contiguous"],
        checks_loading["movie_indices_contiguous"],
        checks_loading["context_aligned"],
        checks_loading["user_ids_unique"],
        checks_loading["movie_ids_unique"],
    ]):
        raise ValueError(
            f"Inconsistent preprocessed files: {checks_loading}"
        )

    matrix_coo = ratings_matrix.tocoo()
    observations = pd.DataFrame({
        "row": matrix_coo.row.astype(np.int64),
        "col": matrix_coo.col.astype(np.int64),
        "user_id": user_mapping_sorted["user_id"].to_numpy()[
            matrix_coo.row
        ],
        "film_id": movie_mapping_sorted[
            "canonical_movie_id"
        ].to_numpy()[matrix_coo.col],
        "rating": matrix_coo.data.astype(np.float64),
    })

    if observations.duplicated(subset=["row", "col"]).any():
        raise ValueError("Duplicated user-film coordinates in CSR.")

    if not np.isfinite(observations["rating"]).all():
        raise ValueError("Non-finite ratings in CSR.")

    density = len(observations) / (n_users * n_movies)
    data_summary = pd.DataFrame([
        {"indicator": "users", "value": n_users},
        {"indicator": "films", "value": n_movies},
        {"indicator": "stored ratings", "value": len(observations)},
        {"indicator": "explicit zeros", "value": int(
            (observations["rating"] == 0).sum()
        )},
        {"indicator": "density", "value": density},
        {"indicator": "sparsity", "value": 1 - density},
        {"indicator": "min rating", "value": float(
            observations["rating"].min()
        )},
        {"indicator": "max rating", "value": float(
            observations["rating"].max()
        )},
    ])

    checks_loading_df = pd.DataFrame([
        {"check": key, "value": value}
        for key, value in checks_loading.items()
    ])
    return (
        checks_loading_df,
        context_sorted,
        data_summary,
        movie_mapping_sorted,
        observations,
        user_mapping_sorted,
    )


@app.cell
def _(data_summary):
    data_summary
    return


@app.cell
def _(observations, pd):
    observations["rating"].describe(
        percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    ).to_frame("rating")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Interactive configuration

    Hyperparameters apply to all models. The expensive training block can be
    disabled below.
    """)
    return


@app.cell
def _(mo):
    n_factors = mo.ui.slider(
        start=2, stop=64, value=8, step=1,
        label="Nombre de facteurs latents",
    )
    learning_rate = mo.ui.number(
        value=0.01, start=0.0001, stop=1.0, step=0.001,
        label="Learning rate",
    )
    n_epochs = mo.ui.number(
        value=20, start=1, stop=200, step=1,
        label="Nombre maximal d'époques",
    )
    reg_bias = mo.ui.number(
        value=0.02, start=0.0, stop=1.0, step=0.001,
        label="Régularisation des biais",
    )
    reg_factors = mo.ui.number(
        value=0.05, start=0.0, stop=1.0, step=0.001,
        label="Régularisation des facteurs",
    )
    batch_size = mo.ui.number(
        value=65536, start=1024, step=1024,
        label="Taille de batch",
    )
    seed = mo.ui.number(
        value=42, start=0, step=1,
        label="Seed aléatoire",
    )
    train_fraction = mo.ui.number(
        value=0.8, start=0.5, stop=0.95, step=0.01,
        label="Proportion train",
    )
    val_fraction = mo.ui.number(
        value=0.1, start=0.01, stop=0.4, step=0.01,
        label="Proportion validation",
    )
    max_scatter_points = mo.ui.number(
        value=5000, start=100, step=1000,
        label="Points maximaux par scatter",
    )
    run_training = mo.ui.checkbox(
        value=True,
        label="Entraîner les modèles",
    )
    use_checkpoints = mo.ui.checkbox(
        value=True,
        label="Utiliser les checkpoints si disponibles",
    )
    force_retrain = mo.ui.checkbox(
        value=False,
        label="Forcer le réentraînement (ignorer les checkpoints)",
    )

    mo.vstack([
        mo.hstack(
            [n_factors, learning_rate, n_epochs],
            justify="start",
            gap=2,
        ),
        mo.hstack(
            [reg_bias, reg_factors, batch_size],
            justify="start",
            gap=2,
        ),
        mo.hstack(
            [seed, train_fraction, val_fraction],
            justify="start",
            gap=2,
        ),
        mo.hstack(
            [max_scatter_points, run_training],
            justify="start",
            gap=2,
        ),
        mo.hstack(
            [use_checkpoints, force_retrain],
            justify="start",
            gap=2,
        ),
    ])
    return (
        batch_size,
        force_retrain,
        learning_rate,
        max_scatter_points,
        n_epochs,
        n_factors,
        reg_bias,
        reg_factors,
        run_training,
        seed,
        train_fraction,
        use_checkpoints,
        val_fraction,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Train / validation / test split

    The split is performed at interaction level. Validation and test are
    restricted to warm-start interactions: both the user and the film must
    appear in training. Excluded interactions are reported.
    """)
    return


@app.cell
def _(
    np,
    observations,
    pd,
    seed,
    train_fraction,
    val_fraction,
):
    _train_fraction = float(train_fraction.value)
    _val_fraction = float(val_fraction.value)
    _test_fraction = 1.0 - _train_fraction - _val_fraction

    if not (0 < _train_fraction < 1 and 0 < _val_fraction < 1):
        raise ValueError("Invalid train/validation fractions.")
    if _test_fraction <= 0:
        raise ValueError(
            "Train and validation fractions must sum to less than 1."
        )

    _seed = int(seed.value)
    _rng = np.random.default_rng(_seed)
    _n_obs = len(observations)
    _permutation = _rng.permutation(_n_obs)

    _n_train = int(np.floor(_train_fraction * _n_obs))
    _n_val = int(np.floor(_val_fraction * _n_obs))

    train_idx = _permutation[:_n_train]
    val_idx_all = _permutation[_n_train:_n_train + _n_val]
    test_idx_all = _permutation[_n_train + _n_val:]

    _train_users = set(
        observations.iloc[train_idx]["row"].unique().tolist()
    )
    _train_films = set(
        observations.iloc[train_idx]["col"].unique().tolist()
    )

    _val_frame = observations.iloc[val_idx_all]
    _test_frame = observations.iloc[test_idx_all]

    _val_warm_mask = (
        _val_frame["row"].isin(_train_users)
        & _val_frame["col"].isin(_train_films)
    ).to_numpy()
    _test_warm_mask = (
        _test_frame["row"].isin(_train_users)
        & _test_frame["col"].isin(_train_films)
    ).to_numpy()

    val_idx = val_idx_all[_val_warm_mask]
    test_idx = test_idx_all[_test_warm_mask]

    obs_train = observations.iloc[train_idx].reset_index(drop=True)
    obs_val = observations.iloc[val_idx].reset_index(drop=True)
    obs_test = observations.iloc[test_idx].reset_index(drop=True)

    y_train = obs_train["rating"].to_numpy(dtype=np.float64)
    y_val = obs_val["rating"].to_numpy(dtype=np.float64)
    y_test = obs_test["rating"].to_numpy(dtype=np.float64)

    split_summary = pd.DataFrame([
        {"split": "train", "interactions": len(train_idx)},
        {"split": "validation (all)", "interactions": len(val_idx_all)},
        {"split": "validation warm-start", "interactions": len(val_idx)},
        {"split": "validation excluded", "interactions": int(
            (~_val_warm_mask).sum()
        )},
        {"split": "test (all)", "interactions": len(test_idx_all)},
        {"split": "test warm-start", "interactions": len(test_idx)},
        {"split": "test excluded", "interactions": int(
            (~_test_warm_mask).sum()
        )},
        {"split": "test excluded ratio", "interactions": float(
            (~_test_warm_mask).mean()
        )},
    ])

    global_mean = float(y_train.mean())
    sigma_train = float(y_train.std(ddof=0))
    if sigma_train <= 0:
        sigma_train = 1.0

    user_train_counts = obs_train.groupby("row").size()
    film_train_counts = obs_train.groupby("col").size()
    user_train_mean = obs_train.groupby("row")["rating"].mean()
    film_train_mean = obs_train.groupby("col")["rating"].mean()

    train_stats = {
        "global_mean": global_mean,
        "sigma_train": sigma_train,
        "user_counts": user_train_counts,
        "film_counts": film_train_counts,
        "user_mean": user_train_mean,
        "film_mean": film_train_mean,
        "rating_values": np.sort(obs_train["rating"].unique()),
        "n_train": len(obs_train),
    }

    return (
        obs_test,
        obs_train,
        obs_val,
        sigma_train,
        split_summary,
        test_idx,
        train_idx,
        train_stats,
        val_idx,
        y_test,
        y_train,
        y_val,
    )


@app.cell
def _(split_summary):
    split_summary
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Sparse feature engineering

    Part I uses only user and film one-hot features. Part II adds multi-hot
    genres and studios plus a standardized release year and a missing-year
    indicator. Vocabularies and year statistics are computed on training films
    only.
    """)
    return


@app.cell
def _(
    Counter,
    context_sorted,
    csr_matrix,
    hstack,
    movie_mapping_sorted,
    np,
    obs_test,
    obs_train,
    obs_val,
    pd,
    user_mapping_sorted,
):
    _n_users = len(user_mapping_sorted)
    _n_movies = len(movie_mapping_sorted)

    def _as_list(value):
        if value is None:
            return None
        if isinstance(value, (list, tuple, set, np.ndarray, pd.Series)):
            return [
                str(item).strip().casefold()
                for item in value
                if pd.notna(item) and str(item).strip()
            ]
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
        return [str(value).strip().casefold()] if str(value).strip() else []

    context_by_col = (
        context_sorted.set_index("col_index").sort_index()
    )

    _genres_by_film = [
        _as_list(value)
        for value in context_by_col.get(
            "genres",
            pd.Series([None] * _n_movies, index=context_by_col.index),
        )
    ]
    _studios_by_film = [
        _as_list(value)
        for value in context_by_col.get(
            "studios",
            pd.Series([None] * _n_movies, index=context_by_col.index),
        )
    ]

    _train_films = np.sort(obs_train["col"].unique())
    _genre_counter = Counter()
    _studio_counter = Counter()
    for _film in _train_films:
        _genres = _genres_by_film[_film]
        if _genres:
            _genre_counter.update(set(_genres))
        _studios = _studios_by_film[_film]
        if _studios:
            _studio_counter.update(set(_studios))

    genre_vocab = sorted(_genre_counter)
    studio_vocab = sorted(_studio_counter)

    genre_feature_names = (
        ["genre:<UNKNOWN>", "genre:<MISSING>"]
        + [f"genre:{value}" for value in genre_vocab]
    )
    studio_feature_names = (
        ["studio:<UNKNOWN>", "studio:<MISSING>"]
        + [f"studio:{value}" for value in studio_vocab]
    )

    _genre_index = {
        value: index + 2 for index, value in enumerate(genre_vocab)
    }
    _studio_index = {
        value: index + 2 for index, value in enumerate(studio_vocab)
    }

    _genre_rows = []
    _genre_cols = []
    _studio_rows = []
    _studio_cols = []

    for _film in range(_n_movies):
        _genres = _genres_by_film[_film]
        if _genres is None:
            _genre_rows.append(_film)
            _genre_cols.append(1)
        elif len(_genres) == 0:
            pass
        else:
            for _value in set(_genres):
                _genre_rows.append(_film)
                _genre_cols.append(_genre_index.get(_value, 0))

        _studios = _studios_by_film[_film]
        if _studios is None:
            _studio_rows.append(_film)
            _studio_cols.append(1)
        elif len(_studios) == 0:
            pass
        else:
            for _value in set(_studios):
                _studio_rows.append(_film)
                _studio_cols.append(_studio_index.get(_value, 0))

    item_genres = csr_matrix(
        (
            np.ones(len(_genre_rows), dtype=np.float64),
            (_genre_rows, _genre_cols),
        ),
        shape=(_n_movies, len(genre_feature_names)),
    )
    item_studios = csr_matrix(
        (
            np.ones(len(_studio_rows), dtype=np.float64),
            (_studio_rows, _studio_cols),
        ),
        shape=(_n_movies, len(studio_feature_names)),
    )

    _film_years = pd.to_numeric(
        context_by_col.get(
            "year",
            pd.Series([np.nan] * _n_movies, index=context_by_col.index),
        ),
        errors="coerce",
    ).to_numpy(dtype=np.float64)
    _train_year_values = _film_years[_train_films]
    _valid_train_years = _train_year_values[
        np.isfinite(_train_year_values)
    ]
    if len(_valid_train_years):
        year_mean = float(_valid_train_years.mean())
        year_std = float(_valid_train_years.std(ddof=0))
    else:
        year_mean = 0.0
        year_std = 1.0
    if year_std <= 0:
        year_std = 1.0

    def _build_split(obs):
        _n = len(obs)
        _rows = obs["row"].to_numpy(dtype=np.int64)
        _cols = obs["col"].to_numpy(dtype=np.int64)
        _index = np.arange(_n)

        user_block = csr_matrix(
            (
                np.ones(_n, dtype=np.float64),
                (_index, _rows),
            ),
            shape=(_n, _n_users),
        )
        film_block = csr_matrix(
            (
                np.ones(_n, dtype=np.float64),
                (_index, _cols),
            ),
            shape=(_n, _n_movies),
        )
        genre_block = item_genres[_cols]
        studio_block = item_studios[_cols]

        _years = _film_years[_cols]
        _missing = ~np.isfinite(_years)
        _year_z = np.where(
            _missing,
            0.0,
            (_years - year_mean) / year_std,
        )
        _present_rows = _index[~_missing]
        _missing_rows = _index[_missing]
        _year_rows = np.concatenate([_present_rows, _missing_rows])
        _year_cols = np.concatenate([
            np.zeros(len(_present_rows), dtype=np.int64),
            np.ones(len(_missing_rows), dtype=np.int64),
        ])
        _year_data = np.concatenate([
            _year_z[~_missing],
            np.ones(len(_missing_rows), dtype=np.float64),
        ])
        year_block = csr_matrix(
            (_year_data, (_year_rows, _year_cols)),
            shape=(_n, 2),
        )

        x_user_item = hstack(
            [user_block, film_block],
            format="csr",
        )
        x_context = hstack(
            [
                user_block,
                film_block,
                genre_block,
                year_block,
                studio_block,
            ],
            format="csr",
        )
        return {
            "x_user_item": x_user_item,
            "x_context": x_context,
            "y": obs["rating"].to_numpy(dtype=np.float64),
            "obs": obs,
        }

    _splits = {
        "train": _build_split(obs_train),
        "val": _build_split(obs_val),
        "test": _build_split(obs_test),
    }

    _user_names = [
        f"user:{value}" for value in user_mapping_sorted["user_id"]
    ]
    _film_names = [
        f"film:{value}"
        for value in movie_mapping_sorted["canonical_movie_id"]
    ]
    _part1_groups = np.array(
        ["user"] * _n_users + ["film"] * _n_movies,
        dtype=object,
    )
    _part1_names = np.array(
        _user_names + _film_names,
        dtype=object,
    )

    _context_groups = np.array(
        ["user"] * _n_users
        + ["film"] * _n_movies
        + ["genre"] * len(genre_feature_names)
        + ["year"] * 2
        + ["studio"] * len(studio_feature_names),
        dtype=object,
    )
    _context_names = np.array(
        _user_names
        + _film_names
        + genre_feature_names
        + ["year:standardized", "year:missing"]
        + studio_feature_names,
        dtype=object,
    )

    feature_data = {
        "part1": {
            "splits": {
                key: {
                    "x": value["x_user_item"],
                    "y": value["y"],
                    "obs": value["obs"],
                }
                for key, value in _splits.items()
            },
            "groups": _part1_groups,
            "names": _part1_names,
            "n_features": _n_users + _n_movies,
        },
        "context": {
            "splits": {
                key: {
                    "x": value["x_context"],
                    "y": value["y"],
                    "obs": value["obs"],
                }
                for key, value in _splits.items()
            },
            "groups": _context_groups,
            "names": _context_names,
            "n_features": len(_context_names),
        },
        "item_genres": item_genres,
        "item_studios": item_studios,
        "genre_vocab": genre_vocab,
        "studio_vocab": studio_vocab,
        "year_mean": year_mean,
        "year_std": year_std,
        "n_users": _n_users,
        "n_movies": _n_movies,
        "film_years": _film_years,
        "film_genres": _genres_by_film,
        "film_studios": _studios_by_film,
    }

    feature_summary = pd.DataFrame([
        {"feature_set": "user + film", "n_features": _n_users + _n_movies},
        {"feature_set": "context", "n_features": len(_context_names)},
        {"feature_set": "genres", "n_features": len(genre_feature_names)},
        {"feature_set": "studios", "n_features": len(studio_feature_names)},
        {"feature_set": "year", "n_features": 2},
    ])
    return context_by_col, feature_data, feature_summary


@app.cell
def _(feature_summary):
    feature_summary
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Explicit Factorization Machine engine

    The implementation works on sparse CSR feature matrices. It supports
    linear terms, latent interactions, separate regularization and column
    masks used by the ablations.
    """)
    return


@app.cell
def _(csr_matrix, np, pd, time):
    class FMEngine:
        def __init__(
            self,
            n_features,
            n_factors,
            use_linear,
            use_latent,
            active_mask,
            learning_rate,
            reg_bias,
            reg_factors,
            seed,
        ):
            self.n_features = int(n_features)
            self.n_factors = int(n_factors)
            self.use_linear = bool(use_linear)
            self.use_latent = bool(use_latent)
            self.active_mask = np.asarray(active_mask, dtype=bool)
            self.learning_rate = float(learning_rate)
            self.reg_bias = float(reg_bias)
            self.reg_factors = float(reg_factors)

            _rng = np.random.default_rng(seed)
            self.w0 = 0.0
            self.w = np.zeros(self.n_features, dtype=np.float64)
            self.V = _rng.normal(
                0.0,
                0.01,
                size=(self.n_features, self.n_factors),
            )
            if not self.use_linear:
                self.w[:] = 0.0
            if not self.use_latent:
                self.V[:] = 0.0
            self.V[~self.active_mask] = 0.0

            self.m_w0 = 0.0
            self.v_w0 = 0.0
            self.m_w = np.zeros_like(self.w)
            self.v_w = np.zeros_like(self.w)
            self.m_V = np.zeros_like(self.V)
            self.v_V = np.zeros_like(self.V)
            self.step_number = 0

        def predict(self, X):
            _n_rows = X.shape[0]
            _prediction = np.full(
                _n_rows,
                self.w0,
                dtype=np.float64,
            )
            if self.n_features == 0:
                return _prediction

            if self.use_linear:
                _prediction += X @ self.w

            if self.use_latent:
                _XV = X @ self.V
                _X2 = X.multiply(X)
                _V2_sum = np.einsum("ij,ij->i", self.V, self.V)
                _prediction += 0.5 * (
                    np.einsum("ij,ij->i", _XV, _XV)
                    - _X2 @ _V2_sum
                )
            return _prediction

        def step(self, X, y):
            _batch_size = X.shape[0]
            _prediction = self.predict(X)
            _error = _prediction - y

            _grad_w0 = float(_error.mean())
            _grad_w = np.zeros_like(self.w)
            _grad_V = np.zeros_like(self.V)

            if self.use_linear and self.n_features:
                _grad_w = (X.T @ _error) / _batch_size
                _grad_w += self.reg_bias * self.w
                _grad_w *= self.active_mask

            if self.use_latent and self.n_features:
                _XV = X @ self.V
                _X2 = X.multiply(X)
                _G1 = X.T @ (_error[:, None] * _XV)
                _G2 = _X2.T @ _error
                _grad_V = (
                    _G1 - _G2[:, None] * self.V
                ) / _batch_size
                _grad_V += self.reg_factors * self.V
                _grad_V *= self.active_mask[:, None]

            self.step_number += 1
            _t = self.step_number
            _beta1 = 0.9
            _beta2 = 0.999
            _epsilon = 1e-8

            self.m_w0 = _beta1 * self.m_w0 + (1 - _beta1) * _grad_w0
            self.v_w0 = _beta2 * self.v_w0 + (1 - _beta2) * _grad_w0 ** 2
            _m_hat_w0 = self.m_w0 / (1 - _beta1 ** _t)
            _v_hat_w0 = self.v_w0 / (1 - _beta2 ** _t)
            self.w0 -= self.learning_rate * _m_hat_w0 / (
                np.sqrt(_v_hat_w0) + _epsilon
            )

            if self.use_linear and self.n_features:
                self.m_w = _beta1 * self.m_w + (1 - _beta1) * _grad_w
                self.v_w = (
                    _beta2 * self.v_w + (1 - _beta2) * _grad_w ** 2
                )
                _m_hat_w = self.m_w / (1 - _beta1 ** _t)
                _v_hat_w = self.v_w / (1 - _beta2 ** _t)
                self.w -= self.learning_rate * _m_hat_w / (
                    np.sqrt(_v_hat_w) + _epsilon
                )
                self.w[~self.active_mask] = 0.0

            if self.use_latent and self.n_features:
                self.m_V = _beta1 * self.m_V + (1 - _beta1) * _grad_V
                self.v_V = (
                    _beta2 * self.v_V + (1 - _beta2) * _grad_V ** 2
                )
                _m_hat_V = self.m_V / (1 - _beta1 ** _t)
                _v_hat_V = self.v_V / (1 - _beta2 ** _t)
                self.V -= self.learning_rate * _m_hat_V / (
                    np.sqrt(_v_hat_V) + _epsilon
                )
                self.V[~self.active_mask] = 0.0

        def clone_parameters(self):
            return (
                float(self.w0),
                self.w.copy(),
                self.V.copy(),
            )

        def load_parameters(self, parameters):
            self.w0 = float(parameters[0])
            self.w = parameters[1].copy()
            self.V = parameters[2].copy()

    def compute_metrics(y_true, y_pred):
        _error = y_pred - y_true
        _absolute = np.abs(_error)
        return {
            "mae": float(_absolute.mean()),
            "rmse": float(np.sqrt(np.mean(_error ** 2))),
            "bias": float(_error.mean()),
            "std_error": float(_error.std(ddof=0)),
            "median_ae": float(np.median(_absolute)),
            "p90_ae": float(np.quantile(_absolute, 0.90)),
            "pct_lt_0_5": float((_absolute < 0.5).mean()),
            "pct_lt_1": float((_absolute < 1.0).mean()),
        }

    def train_fm_model(
        X_train,
        y_train,
        X_val,
        y_val,
        n_features,
        n_factors,
        use_linear,
        use_latent,
        active_mask,
        learning_rate,
        reg_bias,
        reg_factors,
        n_epochs,
        batch_size,
        seed,
        patience=3,
    ):
        _engine = FMEngine(
            n_features=n_features,
            n_factors=n_factors,
            use_linear=use_linear,
            use_latent=use_latent,
            active_mask=active_mask,
            learning_rate=learning_rate,
            reg_bias=reg_bias,
            reg_factors=reg_factors,
            seed=seed,
        )
        _rng = np.random.default_rng(seed + 1)
        _n_train = len(y_train)
        _history = []
        _best_rmse = np.inf
        _best_epoch = 0
        _best_parameters = None
        _patience_counter = 0
        _start = time.perf_counter()

        for _epoch in range(1, int(n_epochs) + 1):
            if batch_size and int(batch_size) < _n_train:
                _order = _rng.permutation(_n_train)
                for _start_index in range(
                    0,
                    _n_train,
                    int(batch_size),
                ):
                    _batch = _order[
                        _start_index:_start_index + int(batch_size)
                    ]
                    _engine.step(X_train[_batch], y_train[_batch])
            else:
                _engine.step(X_train, y_train)

            _train_pred = _engine.predict(X_train)
            _val_pred = _engine.predict(X_val)
            _train_metrics = compute_metrics(y_train, _train_pred)
            _val_metrics = compute_metrics(y_val, _val_pred)
            _history.append({
                "epoch": _epoch,
                "train_rmse": _train_metrics["rmse"],
                "train_mae": _train_metrics["mae"],
                "val_rmse": _val_metrics["rmse"],
                "val_mae": _val_metrics["mae"],
            })

            if _val_metrics["rmse"] < _best_rmse - 1e-10:
                _best_rmse = _val_metrics["rmse"]
                _best_epoch = _epoch
                _best_parameters = _engine.clone_parameters()
                _patience_counter = 0
            else:
                _patience_counter += 1
                if _patience_counter >= patience:
                    break

        if _best_parameters is not None:
            _engine.load_parameters(_best_parameters)

        _training_time = time.perf_counter() - _start
        return (
            _engine,
            pd.DataFrame(_history),
            _best_epoch,
            int(_history[-1]["epoch"]),
            _training_time,
        )

    def count_parameters(engine):
        _count = 1
        if engine.use_linear:
            _count += int(engine.active_mask.sum())
        if engine.use_latent:
            _count += int(engine.active_mask.sum()) * engine.n_factors
        return int(_count)

    return FMEngine, compute_metrics, count_parameters, train_fm_model


@app.cell
def _(
    CHECKPOINT_DIR,
    FMEngine,
    INPUT_FILES,
    hashlib,
    json,
    np,
    pd,
    time,
):
    CHECKPOINT_VERSION = "fm-checkpoint-v1"

    def _json_default(value):
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.bool_):
            return bool(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        return str(value)

    def atomic_write_json(path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        _temporary = path.with_name(path.stem + ".tmp" + path.suffix)
        with open(_temporary, "w") as _handle:
            json.dump(
                payload,
                _handle,
                indent=2,
                sort_keys=True,
                default=_json_default,
            )
        _temporary.replace(path)

    def atomic_write_csv(frame, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        _temporary = path.with_name(path.stem + ".tmp" + path.suffix)
        frame.to_csv(_temporary, index=False)
        _temporary.replace(path)

    def atomic_savez(path, **arrays):
        path.parent.mkdir(parents=True, exist_ok=True)
        _temporary = path.with_name(path.stem + ".tmp.npz")
        np.savez_compressed(_temporary, **arrays)
        _temporary.replace(path)

    def atomic_write_npy(path, array):
        path.parent.mkdir(parents=True, exist_ok=True)
        _temporary = path.with_name(path.stem + ".tmp.npy")
        np.save(_temporary, array)
        _temporary.replace(path)

    def input_fingerprint(input_files):
        return {
            _key: {
                "size": int(_path.stat().st_size),
                "mtime_ns": int(_path.stat().st_mtime_ns),
            }
            for _key, _path in input_files.items()
        }

    def checkpoint_signature(payload):
        _encoded = json.dumps(
            payload,
            sort_keys=True,
            default=_json_default,
        ).encode("utf-8")
        return hashlib.sha256(_encoded).hexdigest()

    def model_signature(spec, run_context):
        _mask = np.asarray(spec["active_mask"], dtype=np.uint8)
        return checkpoint_signature({
            "version": CHECKPOINT_VERSION,
            "inputs": run_context["inputs"],
            "split": run_context["split"],
            "hyperparameters": run_context["hyperparameters"],
            "model": {
                "name": spec["name"],
                "family": spec["family"],
                "feature_set": spec["feature_set"],
                "n_features": int(spec["n_features"]),
                "use_linear": bool(spec["use_linear"]),
                "use_latent": bool(spec["use_latent"]),
                "active_mask_hash": hashlib.sha256(
                    _mask.tobytes()
                ).hexdigest(),
            },
        })

    def ablation_signature(
        variant,
        active_mask,
        run_context,
        c4_signature,
    ):
        _mask = np.asarray(active_mask, dtype=np.uint8)
        return checkpoint_signature({
            "version": CHECKPOINT_VERSION,
            "inputs": run_context["inputs"],
            "split": run_context["split"],
            "hyperparameters": run_context["hyperparameters"],
            "ablation": {
                "variant": variant,
                "n_features": int(len(_mask)),
                "active_mask_hash": hashlib.sha256(
                    _mask.tobytes()
                ).hexdigest(),
            },
            "c4_signature": c4_signature,
        })

    def checkpoint_paths(kind, name):
        _base = CHECKPOINT_DIR / kind / name
        return {
            "dir": _base,
            "params": _base / "params.npz",
            "history": _base / "history.csv",
            "val_prediction": _base / "val_prediction.npy",
            "test_prediction": _base / "test_prediction.npy",
            "prediction_frame": _base / "prediction_frame.csv",
            "metric": _base / "metric.csv",
            "metadata": _base / "metadata.json",
            "signature": _base / "signature.json",
        }

    def _engine_arrays(engine):
        return {
            "w0": np.array([engine.w0], dtype=np.float64),
            "w": np.asarray(engine.w, dtype=np.float64),
            "V": np.asarray(engine.V, dtype=np.float64),
            "active_mask": np.asarray(
                engine.active_mask,
                dtype=bool,
            ),
            "n_features": np.array(
                [engine.n_features],
                dtype=np.int64,
            ),
            "n_factors": np.array(
                [engine.n_factors],
                dtype=np.int64,
            ),
            "use_linear": np.array(
                [engine.use_linear],
                dtype=bool,
            ),
            "use_latent": np.array(
                [engine.use_latent],
                dtype=bool,
            ),
            "learning_rate": np.array(
                [engine.learning_rate],
                dtype=np.float64,
            ),
            "reg_bias": np.array(
                [engine.reg_bias],
                dtype=np.float64,
            ),
            "reg_factors": np.array(
                [engine.reg_factors],
                dtype=np.float64,
            ),
        }

    def save_model_checkpoint(
        name,
        signature,
        engine,
        history,
        val_prediction,
        test_prediction,
        prediction_frame,
        metric_row,
        metadata,
    ):
        _paths = checkpoint_paths("models", name)
        _paths["dir"].mkdir(parents=True, exist_ok=True)
        atomic_savez(_paths["params"], **_engine_arrays(engine))
        atomic_write_csv(history, _paths["history"])
        atomic_write_npy(_paths["val_prediction"], val_prediction)
        atomic_write_npy(_paths["test_prediction"], test_prediction)
        atomic_write_csv(
            prediction_frame,
            _paths["prediction_frame"],
        )
        atomic_write_csv(
            pd.DataFrame([metric_row]),
            _paths["metric"],
        )
        atomic_write_json(
            _paths["metadata"],
            {
                **metadata,
                "signature": signature,
                "saved_at": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ",
                    time.gmtime(),
                ),
            },
        )
        atomic_write_json(
            _paths["signature"],
            {"signature": signature},
        )
        return _paths

    def load_model_checkpoint(name, expected_signature):
        _paths = checkpoint_paths("models", name)
        _required = [
            _paths["params"],
            _paths["history"],
            _paths["val_prediction"],
            _paths["test_prediction"],
            _paths["prediction_frame"],
            _paths["metric"],
            _paths["signature"],
        ]
        if not all(_path.exists() for _path in _required):
            return None

        try:
            with open(_paths["signature"]) as _handle:
                _signature_payload = json.load(_handle)
            if _signature_payload.get("signature") != expected_signature:
                return None

            with np.load(_paths["params"]) as _params:
                _engine = FMEngine(
                    n_features=int(_params["n_features"][0]),
                    n_factors=int(_params["n_factors"][0]),
                    use_linear=bool(_params["use_linear"][0]),
                    use_latent=bool(_params["use_latent"][0]),
                    active_mask=_params["active_mask"].astype(bool),
                    learning_rate=float(_params["learning_rate"][0]),
                    reg_bias=float(_params["reg_bias"][0]),
                    reg_factors=float(_params["reg_factors"][0]),
                    seed=0,
                )
                _engine.w0 = float(_params["w0"][0])
                _engine.w = _params["w"].copy()
                _engine.V = _params["V"].copy()

            _history = pd.read_csv(_paths["history"])
            _val_prediction = np.load(_paths["val_prediction"])
            _test_prediction = np.load(_paths["test_prediction"])
            _prediction_frame = pd.read_csv(
                _paths["prediction_frame"]
            )
            _metric_row = pd.read_csv(
                _paths["metric"]
            ).iloc[0].to_dict()
        except Exception:
            return None

        return {
            "engine": _engine,
            "history": _history,
            "val_prediction": _val_prediction,
            "test_prediction": _test_prediction,
            "prediction_frame": _prediction_frame,
            "metric_row": _metric_row,
            "paths": _paths,
        }

    def save_ablation_checkpoint(
        variant,
        signature,
        engine,
        history,
        metric_row,
        metadata,
    ):
        _paths = checkpoint_paths("ablations", variant)
        _paths["dir"].mkdir(parents=True, exist_ok=True)
        atomic_savez(_paths["params"], **_engine_arrays(engine))
        atomic_write_csv(history, _paths["history"])
        atomic_write_csv(
            pd.DataFrame([metric_row]),
            _paths["metric"],
        )
        atomic_write_json(
            _paths["metadata"],
            {
                **metadata,
                "signature": signature,
                "saved_at": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ",
                    time.gmtime(),
                ),
            },
        )
        atomic_write_json(
            _paths["signature"],
            {"signature": signature},
        )
        return _paths

    def load_ablation_checkpoint(variant, expected_signature):
        _paths = checkpoint_paths("ablations", variant)
        _required = [
            _paths["history"],
            _paths["metric"],
            _paths["signature"],
        ]
        if not all(_path.exists() for _path in _required):
            return None

        try:
            with open(_paths["signature"]) as _handle:
                _signature_payload = json.load(_handle)
            if _signature_payload.get("signature") != expected_signature:
                return None
            _history = pd.read_csv(_paths["history"])
            _metric_row = pd.read_csv(
                _paths["metric"]
            ).iloc[0].to_dict()
        except Exception:
            return None

        return {
            "history": _history,
            "metric_row": _metric_row,
            "paths": _paths,
        }

    def make_checkpoint_status(
        name,
        status,
        signature,
        paths,
        duration=None,
        reason="",
    ):
        return {
            "name": name,
            "status": status,
            "signature": signature[:16],
            "checkpoint_dir": str(paths["dir"]) if paths else "",
            "duration_seconds": (
                float(duration) if duration is not None else None
            ),
            "reason": reason,
        }

    return (
        CHECKPOINT_VERSION,
        ablation_signature,
        atomic_write_csv,
        atomic_write_json,
        checkpoint_paths,
        checkpoint_signature,
        input_fingerprint,
        load_ablation_checkpoint,
        load_model_checkpoint,
        make_checkpoint_status,
        model_signature,
        save_ablation_checkpoint,
        save_model_checkpoint,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. Training the eight models

    Part I models use user and film identities only. Part II models add genres,
    release year and studios. Every model uses the same split and the same
    validation/test observations.
    """)
    return


@app.cell
def _(
    CHECKPOINT_DIR,
    CHECKPOINT_VERSION,
    FMEngine,
    INPUT_FILES,
    atomic_write_csv,
    atomic_write_json,
    batch_size,
    compute_metrics,
    count_parameters,
    csr_matrix,
    feature_data,
    force_retrain,
    input_fingerprint,
    learning_rate,
    load_model_checkpoint,
    make_checkpoint_status,
    mo,
    model_signature,
    n_epochs,
    n_factors,
    np,
    pd,
    reg_bias,
    reg_factors,
    run_training,
    save_model_checkpoint,
    seed,
    time,
    train_fm_model,
    train_fraction,
    use_checkpoints,
    val_fraction,
    y_test,
    y_train,
    y_val,
):
    _training_enabled = bool(run_training.value)
    _use_checkpoints = (
        bool(use_checkpoints.value)
        and not bool(force_retrain.value)
    )

    if not _training_enabled and not _use_checkpoints:
        mo.stop(
            True,
            mo.md(
                "Entraînement désactivé et lecture des checkpoints "
                "désactivée. Activez « Utiliser les checkpoints » ou "
                "« Entraîner les modèles »."
            ),
        )

    _run_context = {
        "inputs": input_fingerprint(INPUT_FILES),
        "split": {
            "seed": int(seed.value),
            "train_fraction": float(train_fraction.value),
            "val_fraction": float(val_fraction.value),
        },
        "hyperparameters": {
            "n_factors": int(n_factors.value),
            "learning_rate": float(learning_rate.value),
            "reg_bias": float(reg_bias.value),
            "reg_factors": float(reg_factors.value),
            "n_epochs": int(n_epochs.value),
            "batch_size": int(batch_size.value),
        },
    }

    _n_users = feature_data["n_users"]

    _part1_all_mask = np.ones(
        feature_data["part1"]["n_features"],
        dtype=bool,
    )
    _part1_film_mask = np.zeros_like(_part1_all_mask)
    _part1_film_mask[_n_users:] = True

    _context_all_mask = np.ones(
        feature_data["context"]["n_features"],
        dtype=bool,
    )

    _empty_train = csr_matrix((len(y_train), 0))
    _empty_val = csr_matrix((len(y_val), 0))
    _empty_test = csr_matrix((len(y_test), 0))

    _model_specs = [
        {
            "name": "M1_global",
            "family": "user_film",
            "feature_set": "none",
            "n_features": 0,
            "use_linear": False,
            "use_latent": False,
            "active_mask": np.zeros(0, dtype=bool),
        },
        {
            "name": "M2_film_bias",
            "family": "user_film",
            "feature_set": "part1",
            "n_features": feature_data["part1"]["n_features"],
            "use_linear": True,
            "use_latent": False,
            "active_mask": _part1_film_mask,
        },
        {
            "name": "M3_latent",
            "family": "user_film",
            "feature_set": "part1",
            "n_features": feature_data["part1"]["n_features"],
            "use_linear": False,
            "use_latent": True,
            "active_mask": _part1_all_mask,
        },
        {
            "name": "M4_full",
            "family": "user_film",
            "feature_set": "part1",
            "n_features": feature_data["part1"]["n_features"],
            "use_linear": True,
            "use_latent": True,
            "active_mask": _part1_all_mask,
        },
        {
            "name": "C1_global",
            "family": "context",
            "feature_set": "none",
            "n_features": 0,
            "use_linear": False,
            "use_latent": False,
            "active_mask": np.zeros(0, dtype=bool),
        },
        {
            "name": "C2_linear",
            "family": "context",
            "feature_set": "context",
            "n_features": feature_data["context"]["n_features"],
            "use_linear": True,
            "use_latent": False,
            "active_mask": _context_all_mask,
        },
        {
            "name": "C3_latent",
            "family": "context",
            "feature_set": "context",
            "n_features": feature_data["context"]["n_features"],
            "use_linear": False,
            "use_latent": True,
            "active_mask": _context_all_mask,
        },
        {
            "name": "C4_full",
            "family": "context",
            "feature_set": "context",
            "n_features": feature_data["context"]["n_features"],
            "use_linear": True,
            "use_latent": True,
            "active_mask": _context_all_mask,
        },
    ]

    models = {}
    metric_rows = []
    history_frames = []
    prediction_frames = []
    _checkpoint_rows = []

    for _model_index, _spec in enumerate(_model_specs):
        _signature = model_signature(_spec, _run_context)
        _checkpoint = (
            load_model_checkpoint(_spec["name"], _signature)
            if _use_checkpoints
            else None
        )

        if _checkpoint is not None:
            models[_spec["name"]] = {
                "engine": _checkpoint["engine"],
                "history": _checkpoint["history"],
                "spec": _spec,
                "val_prediction": _checkpoint["val_prediction"],
                "test_prediction": _checkpoint["test_prediction"],
                "prediction_frame": _checkpoint["prediction_frame"],
            }
            history_frames.append(_checkpoint["history"])
            prediction_frames.append(_checkpoint["prediction_frame"])
            metric_rows.append(_checkpoint["metric_row"])
            _checkpoint_rows.append(
                make_checkpoint_status(
                    _spec["name"],
                    "loaded",
                    _signature,
                    _checkpoint["paths"],
                )
            )
            continue

        if not _training_enabled:
            mo.stop(
                True,
                mo.md(
                    f"Checkpoint absent ou invalide pour "
                    f"`{_spec['name']}`. Cochez « Entraîner les "
                    "modèles » pour le produire."
                ),
            )

        if _spec["feature_set"] == "none":
            _X_train = _empty_train
            _X_val = _empty_val
            _X_test = _empty_test
        else:
            _bundle = feature_data[_spec["feature_set"]]["splits"]
            _X_train = _bundle["train"]["x"]
            _X_val = _bundle["val"]["x"]
            _X_test = _bundle["test"]["x"]

        _model_seed = int(seed.value) + _model_index * 1000
        (
            _engine,
            _history,
            _best_epoch,
            _epochs_run,
            _training_time,
        ) = train_fm_model(
            _X_train,
            y_train,
            _X_val,
            y_val,
            n_features=_spec["n_features"],
            n_factors=int(n_factors.value),
            use_linear=_spec["use_linear"],
            use_latent=_spec["use_latent"],
            active_mask=_spec["active_mask"],
            learning_rate=float(learning_rate.value),
            reg_bias=float(reg_bias.value),
            reg_factors=float(reg_factors.value),
            n_epochs=int(n_epochs.value),
            batch_size=int(batch_size.value),
            seed=_model_seed,
        )

        _val_pred = _engine.predict(_X_val)
        _test_pred = _engine.predict(_X_test)
        _val_metrics = compute_metrics(y_val, _val_pred)
        _test_metrics = compute_metrics(y_test, _test_pred)

        _history = _history.assign(model=_spec["name"])
        history_frames.append(_history)

        _test_obs = feature_data[_spec["feature_set"]]["splits"]["test"][
            "obs"
        ] if _spec["feature_set"] != "none" else None
        if _test_obs is None:
            from_obs = feature_data["part1"]["splits"]["test"]["obs"]
        else:
            from_obs = _test_obs

        _prediction_frame = pd.DataFrame({
            "model": _spec["name"],
            "row": from_obs["row"].to_numpy(),
            "col": from_obs["col"].to_numpy(),
            "user_id": from_obs["user_id"].to_numpy(),
            "film_id": from_obs["film_id"].to_numpy(),
            "rating_true": y_test,
            "rating_est": _test_pred,
        })
        _prediction_frame["error"] = (
            _prediction_frame["rating_est"]
            - _prediction_frame["rating_true"]
        )
        _prediction_frame["abs_error"] = (
            _prediction_frame["error"].abs()
        )
        prediction_frames.append(_prediction_frame)

        _parameters = count_parameters(_engine)
        models[_spec["name"]] = {
            "engine": _engine,
            "history": _history,
            "spec": _spec,
            "val_prediction": _val_pred,
            "test_prediction": _test_pred,
            "prediction_frame": _prediction_frame,
        }

        _metric_row = {
            "model": _spec["name"],
            "family": _spec["family"],
            "feature_set": _spec["feature_set"],
            "n_features": _spec["n_features"],
            "n_factors": int(n_factors.value),
            "use_linear": _spec["use_linear"],
            "use_latent": _spec["use_latent"],
            "parameters": _parameters,
            "training_time_seconds": _training_time,
            "best_epoch": _best_epoch,
            "epochs_run": _epochs_run,
            "reg_bias": float(reg_bias.value),
            "reg_factors": float(reg_factors.value),
            "val_mae": _val_metrics["mae"],
            "val_rmse": _val_metrics["rmse"],
            "val_bias": _val_metrics["bias"],
            "val_std_error": _val_metrics["std_error"],
            "val_median_ae": _val_metrics["median_ae"],
            "val_p90_ae": _val_metrics["p90_ae"],
            "val_pct_lt_0_5": _val_metrics["pct_lt_0_5"],
            "val_pct_lt_1": _val_metrics["pct_lt_1"],
            "test_mae": _test_metrics["mae"],
            "test_rmse": _test_metrics["rmse"],
            "test_bias": _test_metrics["bias"],
            "test_std_error": _test_metrics["std_error"],
            "test_median_ae": _test_metrics["median_ae"],
            "test_p90_ae": _test_metrics["p90_ae"],
            "test_pct_lt_0_5": _test_metrics["pct_lt_0_5"],
            "test_pct_lt_1": _test_metrics["pct_lt_1"],
        }
        metric_rows.append(_metric_row)

        _paths = save_model_checkpoint(
            name=_spec["name"],
            signature=_signature,
            engine=_engine,
            history=_history,
            val_prediction=_val_pred,
            test_prediction=_test_pred,
            prediction_frame=_prediction_frame,
            metric_row=_metric_row,
            metadata={
                "model": _spec["name"],
                "spec": {
                    "family": _spec["family"],
                    "feature_set": _spec["feature_set"],
                    "n_features": _spec["n_features"],
                    "use_linear": _spec["use_linear"],
                    "use_latent": _spec["use_latent"],
                },
                "run_context": _run_context,
            },
        )
        _checkpoint_rows.append(
            make_checkpoint_status(
                _spec["name"],
                "trained",
                _signature,
                _paths,
                duration=_training_time,
            )
        )

    model_checkpoint_status = pd.DataFrame(_checkpoint_rows)
    atomic_write_csv(
        model_checkpoint_status,
        CHECKPOINT_DIR / "model_checkpoint_status.csv",
    )
    atomic_write_json(
        CHECKPOINT_DIR / "manifest.json",
        {
            "version": CHECKPOINT_VERSION,
            "saved_at": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ",
                time.gmtime(),
            ),
            "run_context": _run_context,
            "models": model_checkpoint_status.to_dict("records"),
        },
    )

    metrics = pd.DataFrame(metric_rows)
    histories = pd.concat(history_frames, ignore_index=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    return (
        histories,
        metrics,
        model_checkpoint_status,
        models,
        predictions,
    )


@app.cell
def _(metrics):
    metrics
    return


@app.cell
def _(mo):
    mo.md(r"""
    Un biais moyen proche de zéro ne signifie pas que les prédictions sont
    précises : la MAE, la RMSE et l'écart-type des erreurs doivent être lus
    conjointement.
    """)
    return


@app.cell
def _(histories, plt):
    _figure, _axes = plt.subplots(2, 2, figsize=(16, 10))
    _model_order = histories["model"].drop_duplicates().tolist()

    for _axis, _metric, _title in [
        (_axes[0, 0], "train_rmse", "RMSE entraînement"),
        (_axes[0, 1], "val_rmse", "RMSE validation"),
        (_axes[1, 0], "train_mae", "MAE entraînement"),
        (_axes[1, 1], "val_mae", "MAE validation"),
    ]:
        for _model_name in _model_order:
            _subset = histories[histories["model"] == _model_name]
            _axis.plot(
                _subset["epoch"],
                _subset[_metric],
                label=_model_name,
                linewidth=1.2,
            )
        _axis.set_title(_title)
        _axis.set_xlabel("Époque")
        _axis.set_ylabel(_metric.upper())
        _axis.grid(True, alpha=0.3)
        _axis.legend(fontsize=7)

    _figure.tight_layout()
    _figure
    return


@app.cell
def _(metrics, plt):
    _ordered = metrics.sort_values("family")
    _labels = _ordered["model"].tolist()
    _x = range(len(_labels))

    _figure, _axis = plt.subplots(figsize=(14, 6))
    _width = 0.38
    _axis.bar(
        [value - _width / 2 for value in _x],
        _ordered["val_mae"],
        width=_width,
        label="Validation MAE",
    )
    _axis.bar(
        [value + _width / 2 for value in _x],
        _ordered["test_mae"],
        width=_width,
        label="Test MAE",
    )
    _axis.set_xticks(list(_x))
    _axis.set_xticklabels(_labels, rotation=35, ha="right")
    _axis.set_ylabel("MAE")
    _axis.set_title("Comparaison des huit modèles")
    _axis.grid(True, axis="y", alpha=0.3)
    _axis.legend()
    _figure.tight_layout()
    _figure
    return


@app.cell
def _(metrics, plt):
    _ordered = metrics.sort_values("family")
    _labels = _ordered["model"].tolist()
    _x = range(len(_labels))

    _figure, _axis = plt.subplots(figsize=(14, 6))
    _width = 0.38
    _axis.bar(
        [value - _width / 2 for value in _x],
        _ordered["val_rmse"],
        width=_width,
        label="Validation RMSE",
    )
    _axis.bar(
        [value + _width / 2 for value in _x],
        _ordered["test_rmse"],
        width=_width,
        label="Test RMSE",
    )
    _axis.set_xticks(list(_x))
    _axis.set_xticklabels(_labels, rotation=35, ha="right")
    _axis.set_ylabel("RMSE")
    _axis.set_title("Comparaison des RMSE")
    _axis.grid(True, axis="y", alpha=0.3)
    _axis.legend()
    _figure.tight_layout()
    _figure
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 8. Error diagnostics

    Select a model to inspect signed, normalized and absolute errors against
    predictions and training activity. Trend curves use all observations;
    scatter plots are subsampled only for display.
    """)
    return


@app.cell
def _(models, mo):
    selected_model = mo.ui.dropdown(
        options=list(models.keys()),
        value="M4_full",
        label="Modèle pour les diagnostics",
    )
    selected_model
    return (selected_model,)


@app.cell
def _(models, selected_model, train_stats):
    _name = selected_model.value
    diagnostic_frame = models[_name]["prediction_frame"].copy()

    diagnostic_frame["norm_error"] = (
        diagnostic_frame["error"] / train_stats["sigma_train"]
    )
    diagnostic_frame["user_n_train"] = (
        diagnostic_frame["row"]
        .map(train_stats["user_counts"])
        .fillna(0)
        .astype(int)
    )
    diagnostic_frame["film_n_train"] = (
        diagnostic_frame["col"]
        .map(train_stats["film_counts"])
        .fillna(0)
        .astype(int)
    )
    diagnostic_frame["user_mean_train"] = (
        diagnostic_frame["row"].map(train_stats["user_mean"])
    )
    diagnostic_frame["film_mean_train"] = (
        diagnostic_frame["col"].map(train_stats["film_mean"])
    )

    return (diagnostic_frame,)


@app.cell
def _(selected_model):
    selected_model
    return


@app.cell
def _(np, pd, plt):
    def subsample_frame(frame, max_points, random_state):
        if len(frame) <= max_points:
            return frame
        return frame.sample(
            n=int(max_points),
            random_state=random_state,
        )

    def compute_bin_curve(frame, x_column, y_column, log_x, n_bins=20):
        _data = frame[[x_column, y_column]].dropna()
        if _data.empty:
            return pd.DataFrame(
                columns=["bin_left", "bin_right", "n", "mean"]
            )

        if log_x:
            _x = np.log10(np.clip(_data[x_column].to_numpy(), 1, None))
            _edges = np.linspace(
                _x.min(),
                _x.max() + 1e-9,
                n_bins + 1,
            )
            _centers = 10 ** ((_edges[:-1] + _edges[1:]) / 2)
        else:
            _unique = _data[x_column].nunique()
            _bins = min(n_bins, max(2, _unique))
            _codes = pd.qcut(
                _data[x_column],
                q=_bins,
                duplicates="drop",
                labels=False,
            )
            _data = _data.assign(_bin=_codes)
            _edges = None
            _centers = None

        if log_x:
            _codes = np.clip(
                np.digitize(_x, _edges) - 1,
                0,
                n_bins - 1,
            )
            _data = _data.assign(_bin=_codes)

        _grouped = (
            _data.groupby("_bin")[y_column]
            .agg(["size", "mean"])
            .reset_index()
        )
        _rows = []
        for _record in _grouped.to_dict("records"):
            _index = int(_record["_bin"])
            if log_x:
                _left = _centers[_index] / 1.2
                _right = _centers[_index] * 1.2
                _x_value = _centers[_index]
            else:
                _subset = _data[_data["_bin"] == _index][x_column]
                _left = float(_subset.min())
                _right = float(_subset.max())
                _x_value = float(_subset.median())
            _rows.append({
                "bin_left": _left,
                "bin_right": _right,
                "x": _x_value,
                "n": int(_record["size"]),
                "mean": float(_record["mean"]),
            })
        return pd.DataFrame(_rows)

    def _plot_panel(
        axis,
        frame,
        x_column,
        y_column,
        title,
        x_label,
        y_label,
        log_x,
        zero_line,
        max_points,
        random_state,
    ):
        _plot_data = subsample_frame(
            frame[[x_column, y_column]].dropna(),
            max_points,
            random_state,
        )
        axis.scatter(
            _plot_data[x_column],
            _plot_data[y_column],
            s=6,
            alpha=0.15,
            edgecolors="none",
        )
        _curve = compute_bin_curve(
            frame,
            x_column,
            y_column,
            log_x,
        )
        if not _curve.empty:
            axis.plot(
                _curve["x"],
                _curve["mean"],
                color="crimson",
                marker="o",
                linewidth=1.4,
                markersize=3,
            )
        if log_x:
            axis.set_xscale("log")
        if zero_line:
            axis.axhline(0, color="black", linestyle="--", linewidth=0.8)
        axis.set_title(title)
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        axis.grid(True, which="both", alpha=0.3)
        return _curve

    def make_diagnostic_figure(prediction_frame, name, max_points, random_state):
        _specs = [
            ("rating_est", "Prédiction", False, False),
            ("film_n_train", "Notes du film (train)", True, False),
            ("user_n_train", "Notes de l'utilisateur (train)", True, False),
        ]
        _figure, _axes = plt.subplots(3, 3, figsize=(18, 14))
        _curves = []

        for _row, (_metric, _y_label, _zero_line) in enumerate([
            ("error", "Erreur signée", True),
            ("norm_error", "Erreur normalisée", True),
            ("abs_error", "Erreur absolue", False),
        ]):
            for _column, (
                _x_column,
                _x_label,
                _log_x,
                _unused,
            ) in enumerate(_specs):
                _curve = _plot_panel(
                    _axes[_row, _column],
                    prediction_frame,
                    _x_column,
                    _metric,
                    f"{_y_label} vs {_x_label}",
                    _x_label,
                    _y_label,
                    _log_x,
                    _zero_line,
                    max_points,
                    random_state,
                )
                if not _curve.empty:
                    _curve = _curve.assign(
                        model=name,
                        metric=_metric,
                        x_variable=_x_column,
                    )
                    _curves.append(_curve)

        _figure.tight_layout()
        _bin_table = (
            pd.concat(_curves, ignore_index=True)
            if _curves
            else pd.DataFrame()
        )
        return _figure, _bin_table

    return (compute_bin_curve, make_diagnostic_figure, subsample_frame)


@app.cell
def _(
    diagnostic_frame,
    make_diagnostic_figure,
    max_scatter_points,
    selected_model,
    seed,
):
    diagnostic_plot_figure, diagnostic_bin_table = make_diagnostic_figure(
        diagnostic_frame,
        selected_model.value,
        int(max_scatter_points.value),
        int(seed.value),
    )
    diagnostic_plot_figure
    return diagnostic_bin_table, diagnostic_plot_figure


@app.cell
def _(diagnostic_bin_table, mo):
    mo.ui.table(diagnostic_bin_table, pagination=True)
    return


@app.cell
def _(diagnostic_frame, np, pd, plt):
    _frame = diagnostic_frame.copy()

    _calibration = _frame.dropna(subset=["rating_est"]).copy()
    _calibration["prediction_bin"] = pd.qcut(
        _calibration["rating_est"],
        q=min(10, _calibration["rating_est"].nunique()),
        duplicates="drop",
    )
    _calibration_curve = (
        _calibration.groupby("prediction_bin", observed=True)
        .agg(
            predicted=("rating_est", "mean"),
            observed=("rating_true", "mean"),
            n=("rating_true", "size"),
        )
        .reset_index(drop=True)
    )

    _rating_errors = (
        _frame.groupby("rating_true")
        .agg(
            bias=("error", "mean"),
            mae=("abs_error", "mean"),
            rmse=("error", lambda values: np.sqrt(
                np.mean(values ** 2)
            )),
            n=("error", "size"),
        )
        .reset_index()
    )

    def _mean_bins(column):
        _data = _frame.dropna(subset=[column]).copy()
        _data["bin"] = pd.qcut(
            _data[column],
            q=min(10, _data[column].nunique()),
            duplicates="drop",
        )
        return (
            _data.groupby("bin", observed=True)
            .agg(
                mean_x=(column, "mean"),
                bias=("error", "mean"),
                mae=("abs_error", "mean"),
                n=("error", "size"),
            )
            .reset_index(drop=True)
        )

    _user_mean_curve = _mean_bins("user_mean_train")
    _film_mean_curve = _mean_bins("film_mean_train")

    _figure, _axes = plt.subplots(2, 2, figsize=(16, 10))

    _axes[0, 0].plot(
        [0, 5], [0, 5], color="black", linestyle="--", label="Calibration parfaite"
    )
    _axes[0, 0].plot(
        _calibration_curve["predicted"],
        _calibration_curve["observed"],
        marker="o",
        color="crimson",
        label="Bins de prédiction",
    )
    _axes[0, 0].set_title("Calibration")
    _axes[0, 0].set_xlabel("Note moyenne prédite")
    _axes[0, 0].set_ylabel("Note moyenne réelle")
    _axes[0, 0].grid(True, alpha=0.3)
    _axes[0, 0].legend()

    _axes[0, 1].plot(
        _rating_errors["rating_true"],
        _rating_errors["bias"],
        marker="o",
        label="Biais",
    )
    _axes[0, 1].plot(
        _rating_errors["rating_true"],
        _rating_errors["mae"],
        marker="o",
        label="MAE",
    )
    _axes[0, 1].plot(
        _rating_errors["rating_true"],
        _rating_errors["rmse"],
        marker="o",
        label="RMSE",
    )
    _axes[0, 1].axhline(0, color="black", linestyle="--", linewidth=0.8)
    _axes[0, 1].set_title("Erreurs selon la note réelle")
    _axes[0, 1].set_xlabel("Note réelle")
    _axes[0, 1].set_ylabel("Erreur")
    _axes[0, 1].grid(True, alpha=0.3)
    _axes[0, 1].legend()

    _axes[1, 0].plot(
        _user_mean_curve["mean_x"],
        _user_mean_curve["bias"],
        marker="o",
        label="Biais",
    )
    _axes[1, 0].plot(
        _user_mean_curve["mean_x"],
        _user_mean_curve["mae"],
        marker="o",
        label="MAE",
    )
    _axes[1, 0].axhline(0, color="black", linestyle="--", linewidth=0.8)
    _axes[1, 0].set_title("Erreurs selon la note moyenne utilisateur (train)")
    _axes[1, 0].set_xlabel("Note moyenne utilisateur")
    _axes[1, 0].set_ylabel("Erreur")
    _axes[1, 0].grid(True, alpha=0.3)
    _axes[1, 0].legend()

    _axes[1, 1].plot(
        _film_mean_curve["mean_x"],
        _film_mean_curve["bias"],
        marker="o",
        label="Biais",
    )
    _axes[1, 1].plot(
        _film_mean_curve["mean_x"],
        _film_mean_curve["mae"],
        marker="o",
        label="MAE",
    )
    _axes[1, 1].axhline(0, color="black", linestyle="--", linewidth=0.8)
    _axes[1, 1].set_title("Erreurs selon la note moyenne film (train)")
    _axes[1, 1].set_xlabel("Note moyenne film")
    _axes[1, 1].set_ylabel("Erreur")
    _axes[1, 1].grid(True, alpha=0.3)
    _axes[1, 1].legend()

    _figure.tight_layout()
    calibration_figure = _figure
    calibration_figure
    return (calibration_figure,)


@app.cell
def _(diagnostic_frame, np, pd, plt):
    _frame = diagnostic_frame.copy()
    _frame["user_activity"] = _frame["user_n_train"]
    _frame["film_popularity"] = _frame["film_n_train"]

    _frame["user_bin"] = pd.qcut(
        _frame["user_activity"].rank(method="first"),
        q=4,
        labels=["Q1", "Q2", "Q3", "Q4"],
    )
    _frame["film_bin"] = pd.qcut(
        _frame["film_popularity"].rank(method="first"),
        q=4,
        labels=["Q1", "Q2", "Q3", "Q4"],
    )

    _heat = (
        _frame.groupby(["user_bin", "film_bin"], observed=True)
        .agg(
            mae=("abs_error", "mean"),
            n=("abs_error", "size"),
        )
        .reset_index()
    )
    _pivot = _heat.pivot(
        index="user_bin",
        columns="film_bin",
        values="mae",
    )
    _counts = _heat.pivot(
        index="user_bin",
        columns="film_bin",
        values="n",
    )

    _figure, _axis = plt.subplots(figsize=(8, 6))
    _image = _axis.imshow(
        _pivot.to_numpy(),
        aspect="auto",
        cmap="viridis",
    )
    _axis.set_xticks(range(_pivot.shape[1]))
    _axis.set_xticklabels(_pivot.columns)
    _axis.set_yticks(range(_pivot.shape[0]))
    _axis.set_yticklabels(_pivot.index)
    _axis.set_xlabel("Popularité du film (quartiles train)")
    _axis.set_ylabel("Activité utilisateur (quartiles train)")
    _axis.set_title("MAE selon activité utilisateur et popularité film")

    for _row in range(_pivot.shape[0]):
        for _column in range(_pivot.shape[1]):
            _value = _pivot.to_numpy()[_row, _column]
            _count = _counts.to_numpy()[_row, _column]
            _axis.text(
                _column,
                _row,
                f"{_value:.2f}\nn={int(_count)}",
                ha="center",
                va="center",
                color="white" if _value < np.nanmax(_pivot.to_numpy()) * 0.6 else "black",
                fontsize=9,
            )

    _figure.colorbar(_image, ax=_axis, label="MAE")
    _figure.tight_layout()
    heatmap_figure = _figure
    heatmap_figure
    return (heatmap_figure,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 9. Contextual metadata diagnostics
    """)
    return


@app.cell
def _(diagnostic_frame, feature_data, np, pd, plt):
    _frame = diagnostic_frame.copy()
    _years = feature_data["film_years"][_frame["col"].to_numpy()]
    _genres = [feature_data["film_genres"][index] for index in _frame["col"]]
    _studios = [feature_data["film_studios"][index] for index in _frame["col"]]

    _frame["year"] = _years
    _frame["decade"] = np.where(
        np.isfinite(_years),
        (np.floor(_years / 10) * 10),
        np.nan,
    )

    _genre_rows = []
    for _index, _row in enumerate(_frame.itertuples(index=False)):
        _values = _genres[_index]
        if _values is None:
            _labels = ["<MISSING>"]
        elif len(_values) == 0:
            _labels = ["<EMPTY>"]
        else:
            _labels = _values
        for _label in _labels:
            _genre_rows.append({
                "genre": _label,
                "error": _row.error,
                "abs_error": _row.abs_error,
            })

    _genre_frame = pd.DataFrame(_genre_rows)
    _genre_stats = (
        _genre_frame.groupby("genre")
        .agg(
            bias=("error", "mean"),
            mae=("abs_error", "mean"),
            n=("abs_error", "size"),
        )
        .sort_values("n", ascending=False)
        .head(20)
        .reset_index()
    )

    _decade_stats = (
        _frame.dropna(subset=["decade"])
        .groupby("decade")
        .agg(
            bias=("error", "mean"),
            mae=("abs_error", "mean"),
            n=("abs_error", "size"),
        )
        .reset_index()
    )

    _availability_rows = []
    for _label, _values in [
        ("genres", _genres),
        ("studios", _studios),
    ]:
        _categories = []
        for _value in _values:
            if _value is None:
                _categories.append("missing")
            elif len(_value) == 0:
                _categories.append("empty")
            else:
                _categories.append("present")
        _temp = _frame[["error", "abs_error"]].copy()
        _temp["category"] = _categories
        for _category, _group in _temp.groupby("category"):
            _availability_rows.append({
                "metadata": _label,
                "category": _category,
                "bias": float(_group["error"].mean()),
                "mae": float(_group["abs_error"].mean()),
                "n": len(_group),
            })

    _year_availability = _frame.copy()
    _year_availability["category"] = np.where(
        _year_availability["year"].notna(),
        "present",
        "missing",
    )
    for _category, _group in _year_availability.groupby("category"):
        _availability_rows.append({
            "metadata": "year",
            "category": _category,
            "bias": float(_group["error"].mean()),
            "mae": float(_group["abs_error"].mean()),
            "n": len(_group),
        })

    _availability_stats = pd.DataFrame(_availability_rows)

    _figure, _axes = plt.subplots(3, 1, figsize=(14, 14))

    _axes[0].bar(
        _genre_stats["genre"],
        _genre_stats["mae"],
        alpha=0.7,
        label="MAE",
    )
    _axes[0].plot(
        _genre_stats["genre"],
        _genre_stats["bias"],
        color="crimson",
        marker="o",
        label="Biais",
    )
    _axes[0].axhline(0, color="black", linestyle="--", linewidth=0.8)
    _axes[0].set_title("MAE et biais par genre")
    _axes[0].set_ylabel("Erreur")
    _axes[0].tick_params(axis="x", rotation=60)
    _axes[0].grid(True, axis="y", alpha=0.3)
    _axes[0].legend()

    _axes[1].bar(
        _decade_stats["decade"].astype(str),
        _decade_stats["mae"],
        alpha=0.7,
        label="MAE",
    )
    _axes[1].plot(
        _decade_stats["decade"].astype(str),
        _decade_stats["bias"],
        color="crimson",
        marker="o",
        label="Biais",
    )
    _axes[1].axhline(0, color="black", linestyle="--", linewidth=0.8)
    _axes[1].set_title("MAE et biais par décennie")
    _axes[1].set_ylabel("Erreur")
    _axes[1].tick_params(axis="x", rotation=45)
    _axes[1].grid(True, axis="y", alpha=0.3)
    _axes[1].legend()

    _labels = [
        f"{row.metadata}:{row.category}"
        for row in _availability_stats.itertuples(index=False)
    ]
    _axes[2].bar(_labels, _availability_stats["mae"], alpha=0.7)
    _axes[2].set_title("MAE selon la disponibilité des métadonnées")
    _axes[2].set_ylabel("MAE")
    _axes[2].tick_params(axis="x", rotation=45)
    _axes[2].grid(True, axis="y", alpha=0.3)

    _figure.tight_layout()
    context_diagnostics_figure = _figure
    context_diagnostics_figure
    return (context_diagnostics_figure,)


@app.cell
def _(metrics, mo):
    compare_a = mo.ui.dropdown(
        options=metrics["model"].tolist(),
        value="M4_full",
        label="Modèle A",
    )
    compare_b = mo.ui.dropdown(
        options=metrics["model"].tolist(),
        value="C4_full",
        label="Modèle B",
    )
    mo.hstack([compare_a, compare_b], justify="start", gap=2)
    return compare_a, compare_b


@app.cell
def _(compare_a, compare_b, metrics, plt):
    _selected = metrics[
        metrics["model"].isin([compare_a.value, compare_b.value])
    ]

    _figure, _axes = plt.subplots(1, 2, figsize=(12, 4))
    _axes[0].bar(_selected["model"], _selected["test_mae"])
    _axes[0].set_title("MAE test")
    _axes[0].set_ylabel("MAE")
    _axes[0].grid(True, axis="y", alpha=0.3)

    _axes[1].bar(_selected["model"], _selected["test_rmse"])
    _axes[1].set_title("RMSE test")
    _axes[1].set_ylabel("RMSE")
    _axes[1].grid(True, axis="y", alpha=0.3)

    _figure.tight_layout()
    _figure
    return


@app.cell
def _(compare_a, compare_b, metrics):
    metrics[metrics["model"].isin([compare_a.value, compare_b.value])]
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 10. Prediction decomposition

    The full contextual model is decomposed into the global bias, linear
    family contributions and latent interactions between families.
    """)
    return


@app.cell
def _(feature_data, models, np, pd, y_test):
    _engine = models["M4_full"]["engine"]
    _X_test = feature_data["part1"]["splits"]["test"]["x"]
    _obs = feature_data["part1"]["splits"]["test"]["obs"]

    _user_index = _obs["row"].to_numpy()
    _film_index = _obs["col"].to_numpy()
    _n_users = feature_data["n_users"]

    _prediction = _engine.predict(_X_test)
    _global_component = np.full(len(_obs), _engine.w0)
    _user_component = _engine.w[_user_index]
    _film_component = _engine.w[_n_users + _film_index]
    _interaction_component = (
        _prediction
        - _global_component
        - _user_component
        - _film_component
    )

    simple_decomposition = pd.DataFrame({
        "prediction": _prediction,
        "global": _global_component,
        "user_bias": _user_component,
        "film_bias": _film_component,
        "interaction": _interaction_component,
    })

    _component_rows = []
    for _component in [
        "global",
        "user_bias",
        "film_bias",
        "interaction",
    ]:
        _values = simple_decomposition[_component].to_numpy()
        _component_rows.append({
            "component": _component,
            "mean_signed": float(_values.mean()),
            "mean_absolute": float(np.abs(_values).mean()),
            "std": float(_values.std(ddof=0)),
            "median_absolute": float(np.median(np.abs(_values))),
            "p90_absolute": float(np.quantile(np.abs(_values), 0.9)),
        })

    simple_component_stats = pd.DataFrame(_component_rows)
    return simple_component_stats, simple_decomposition


@app.cell
def _(simple_component_stats):
    simple_component_stats
    return


@app.cell
def _(simple_decomposition, plt):
    _components = ["user_bias", "film_bias", "interaction"]
    _figure, _axes = plt.subplots(1, 3, figsize=(18, 5))
    for _axis, _component in zip(_axes, _components):
        _axis.hist(
            simple_decomposition[_component],
            bins=60,
            alpha=0.75,
        )
        _axis.set_title(_component)
        _axis.set_xlabel("Contribution")
        _axis.set_ylabel("Nombre d'observations")
        _axis.grid(True, alpha=0.3)
    _figure.tight_layout()
    _figure
    return


@app.cell
def _(feature_data, models, np, pd):
    _engine = models["C4_full"]["engine"]
    _context = feature_data["context"]
    _X_test = _context["splits"]["test"]["x"]
    _groups = _context["groups"]
    _w = _engine.w
    _V = _engine.V
    _prediction = _engine.predict(_X_test)

    _family_names = ["user", "film", "genre", "year", "studio"]
    _linear = {}
    _latent_sums = {}
    _latent_squares = {}
    _latent_diagonals = {}

    for _family in _family_names:
        _columns = np.where(_groups == _family)[0]
        _X_family = _X_test[:, _columns]
        _w_family = _w[_columns]
        _V_family = _V[_columns]
        _linear[_family] = _X_family @ _w_family
        _S_family = _X_family @ _V_family
        _latent_sums[_family] = _S_family
        _latent_squares[_family] = np.einsum(
            "ij,ij->i", _S_family, _S_family
        )
        _latent_diagonals[_family] = _X_family.multiply(
            _X_family
        ) @ np.einsum("ij,ij->i", _V_family, _V_family)

    _contributions = {}
    for _family in _family_names:
        _contributions[f"linear:{_family}"] = _linear[_family]

    _interaction_stats_rows = []
    _interaction_heatmap = np.zeros(
        (len(_family_names), len(_family_names))
    )

    for _first_index, _first in enumerate(_family_names):
        _within = 0.5 * (
            _latent_squares[_first] - _latent_diagonals[_first]
        )
        _contributions[f"interaction:{_first}:{_first}"] = _within
        _interaction_heatmap[_first_index, _first_index] = float(
            np.abs(_within).mean()
        )

        for _second_index in range(_first_index + 1, len(_family_names)):
            _second = _family_names[_second_index]
            _cross = np.einsum(
                "ij,ij->i",
                _latent_sums[_first],
                _latent_sums[_second],
            )
            _key = f"interaction:{_first}:{_second}"
            _contributions[_key] = _cross
            _interaction_heatmap[_first_index, _second_index] = float(
                np.abs(_cross).mean()
            )
            _interaction_heatmap[_second_index, _first_index] = float(
                np.abs(_cross).mean()
            )

    for _key, _values in _contributions.items():
        _interaction_stats_rows.append({
            "component": _key,
            "mean_signed": float(_values.mean()),
            "mean_absolute": float(np.abs(_values).mean()),
            "std": float(_values.std(ddof=0)),
            "median_absolute": float(np.median(np.abs(_values))),
            "p90_absolute": float(np.quantile(np.abs(_values), 0.9)),
        })

    context_contributions = pd.DataFrame(_contributions)
    context_contribution_stats = pd.DataFrame(_interaction_stats_rows)
    context_reconstruction = (
        _engine.w0
        + sum(_linear.values())
        + sum(
            _contributions[key]
            for key in _contributions
            if key.startswith("interaction:")
        )
    )
    context_reconstruction_error = float(
        np.max(np.abs(context_reconstruction - _prediction))
    )
    context_interaction_heatmap = pd.DataFrame(
        _interaction_heatmap,
        index=_family_names,
        columns=_family_names,
    )
    return (
        context_contribution_stats,
        context_contributions,
        context_interaction_heatmap,
        context_reconstruction_error,
    )


@app.cell
def _(context_contribution_stats):
    context_contribution_stats
    return


@app.cell
def _(context_interaction_heatmap, np, plt):
    _figure, _axis = plt.subplots(figsize=(8, 6))
    _image = _axis.imshow(
        context_interaction_heatmap.to_numpy(),
        cmap="magma",
        aspect="auto",
    )
    _axis.set_xticks(range(len(context_interaction_heatmap.columns)))
    _axis.set_xticklabels(
        context_interaction_heatmap.columns,
        rotation=45,
    )
    _axis.set_yticks(range(len(context_interaction_heatmap.index)))
    _axis.set_yticklabels(context_interaction_heatmap.index)
    _axis.set_title("Contribution absolue moyenne des interactions")
    _figure.colorbar(_image, ax=_axis, label="Contribution absolue moyenne")
    _figure.tight_layout()
    interaction_heatmap_figure = _figure
    interaction_heatmap_figure
    return (interaction_heatmap_figure,)


@app.cell
def _(context_reconstruction_error, mo):
    mo.md(
        f"Erreur maximale de reconstruction des contributions : "
        f"`{context_reconstruction_error:.3e}`"
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 11. Ablations

    The full contextual model is compared with variants that remove feature
    families or model components. C2 and C3 are reused as the no-latent and
    no-linear ablations because they use exactly the same split and settings.
    """)
    return


@app.cell
def _(
    CHECKPOINT_DIR,
    CHECKPOINT_VERSION,
    INPUT_FILES,
    ablation_signature,
    atomic_write_csv,
    atomic_write_json,
    batch_size,
    compute_metrics,
    feature_data,
    force_retrain,
    input_fingerprint,
    learning_rate,
    load_ablation_checkpoint,
    make_checkpoint_status,
    metrics,
    mo,
    model_signature,
    models,
    n_epochs,
    n_factors,
    np,
    pd,
    reg_bias,
    reg_factors,
    run_training,
    save_ablation_checkpoint,
    seed,
    time,
    train_fm_model,
    train_fraction,
    use_checkpoints,
    val_fraction,
    y_train,
    y_val,
):
    _training_enabled = bool(run_training.value)
    _use_checkpoints = (
        bool(use_checkpoints.value)
        and not bool(force_retrain.value)
    )

    if not _training_enabled and not _use_checkpoints:
        mo.stop(
            True,
            mo.md(
                "Entraînement désactivé et lecture des checkpoints "
                "désactivée. Activez « Utiliser les checkpoints » ou "
                "« Entraîner les modèles »."
            ),
        )

    _run_context = {
        "inputs": input_fingerprint(INPUT_FILES),
        "split": {
            "seed": int(seed.value),
            "train_fraction": float(train_fraction.value),
            "val_fraction": float(val_fraction.value),
        },
        "hyperparameters": {
            "n_factors": int(n_factors.value),
            "learning_rate": float(learning_rate.value),
            "reg_bias": float(reg_bias.value),
            "reg_factors": float(reg_factors.value),
            "n_epochs": int(n_epochs.value),
            "batch_size": int(batch_size.value),
        },
    }
    _c4_signature = model_signature(
        models["C4_full"]["spec"],
        _run_context,
    )

    _context = feature_data["context"]
    _groups = _context["groups"]

    _ablation_specs = [
        (
            "without_genre",
            _groups != "genre",
        ),
        (
            "without_year",
            _groups != "year",
        ),
        (
            "without_studio",
            _groups != "studio",
        ),
    ]

    _ablation_rows = []
    _ablation_models = {}
    _checkpoint_rows = []

    _full_metrics = metrics.loc[
        metrics["model"] == "C4_full"
    ].iloc[0]

    for _index, (_name, _active_mask) in enumerate(_ablation_specs):
        _signature = ablation_signature(
            _name,
            _active_mask,
            _run_context,
            _c4_signature,
        )
        _checkpoint = (
            load_ablation_checkpoint(_name, _signature)
            if _use_checkpoints
            else None
        )

        if _checkpoint is not None:
            _ablation_rows.append(_checkpoint["metric_row"])
            _checkpoint_rows.append(
                make_checkpoint_status(
                    _name,
                    "loaded",
                    _signature,
                    _checkpoint["paths"],
                )
            )
            continue

        if not _training_enabled:
            mo.stop(
                True,
                mo.md(
                    f"Checkpoint absent ou invalide pour l'ablation "
                    f"`{_name}`. Cochez « Entraîner les modèles » "
                    "pour le produire."
                ),
            )

        _X_train = _context["splits"]["train"]["x"]
        _X_val = _context["splits"]["val"]["x"]
        (
            _engine,
            _history,
            _best_epoch,
            _epochs_run,
            _training_time,
        ) = train_fm_model(
            _X_train,
            y_train,
            _X_val,
            y_val,
            n_features=_context["n_features"],
            n_factors=int(n_factors.value),
            use_linear=True,
            use_latent=True,
            active_mask=_active_mask,
            learning_rate=float(learning_rate.value),
            reg_bias=float(reg_bias.value),
            reg_factors=float(reg_factors.value),
            n_epochs=int(n_epochs.value),
            batch_size=int(batch_size.value),
            seed=int(seed.value) + 5000 + _index,
        )
        _val_metrics = compute_metrics(
            y_val,
            _engine.predict(_X_val),
        )
        _ablation_models[_name] = _engine
        _metric_row = {
            "variant": _name,
            "val_mae": _val_metrics["mae"],
            "val_rmse": _val_metrics["rmse"],
            "delta_mae": _val_metrics["mae"] - _full_metrics["val_mae"],
            "delta_rmse": _val_metrics["rmse"] - _full_metrics["val_rmse"],
            "best_epoch": _best_epoch,
            "epochs_run": _epochs_run,
            "training_time_seconds": _training_time,
        }
        _ablation_rows.append(_metric_row)

        _paths = save_ablation_checkpoint(
            variant=_name,
            signature=_signature,
            engine=_engine,
            history=_history,
            metric_row=_metric_row,
            metadata={
                "variant": _name,
                "run_context": _run_context,
                "c4_signature": _c4_signature,
            },
        )
        _checkpoint_rows.append(
            make_checkpoint_status(
                _name,
                "trained",
                _signature,
                _paths,
                duration=_training_time,
            )
        )

    _no_latent = metrics.loc[metrics["model"] == "C2_linear"].iloc[0]
    _no_linear = metrics.loc[metrics["model"] == "C3_latent"].iloc[0]

    _ablation_rows.extend([
        {
            "variant": "without_latent_interactions",
            "val_mae": _no_latent["val_mae"],
            "val_rmse": _no_latent["val_rmse"],
            "delta_mae": _no_latent["val_mae"] - _full_metrics["val_mae"],
            "delta_rmse": _no_latent["val_rmse"] - _full_metrics["val_rmse"],
            "best_epoch": _no_latent["best_epoch"],
            "epochs_run": _no_latent["epochs_run"],
            "training_time_seconds": _no_latent["training_time_seconds"],
        },
        {
            "variant": "without_linear_biases",
            "val_mae": _no_linear["val_mae"],
            "val_rmse": _no_linear["val_rmse"],
            "delta_mae": _no_linear["val_mae"] - _full_metrics["val_mae"],
            "delta_rmse": _no_linear["val_rmse"] - _full_metrics["val_rmse"],
            "best_epoch": _no_linear["best_epoch"],
            "epochs_run": _no_linear["epochs_run"],
            "training_time_seconds": _no_linear["training_time_seconds"],
        },
    ])

    ablations = pd.DataFrame(_ablation_rows).sort_values("delta_mae")

    ablation_checkpoint_status = pd.DataFrame(_checkpoint_rows)
    atomic_write_csv(
        ablation_checkpoint_status,
        CHECKPOINT_DIR / "ablation_checkpoint_status.csv",
    )
    atomic_write_json(
        CHECKPOINT_DIR / "ablation_manifest.json",
        {
            "version": CHECKPOINT_VERSION,
            "saved_at": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ",
                time.gmtime(),
            ),
            "run_context": _run_context,
            "c4_signature": _c4_signature,
            "ablations": ablation_checkpoint_status.to_dict("records"),
        },
    )
    return (ablations, ablation_checkpoint_status)


@app.cell
def _(
    CHECKPOINT_DIR,
    ablation_checkpoint_status,
    atomic_write_csv,
    checks_loading_df,
    data_summary,
    feature_summary,
    model_checkpoint_status,
    mo,
    split_summary,
):
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    _stage_dir = CHECKPOINT_DIR / "stage_tables"
    atomic_write_csv(
        data_summary,
        _stage_dir / "data_summary.csv",
    )
    atomic_write_csv(
        checks_loading_df,
        _stage_dir / "checks_loading.csv",
    )
    atomic_write_csv(
        split_summary,
        _stage_dir / "split_summary.csv",
    )
    atomic_write_csv(
        feature_summary,
        _stage_dir / "feature_summary.csv",
    )

    _model_loaded = int(
        (model_checkpoint_status["status"] == "loaded").sum()
    )
    _model_trained = int(
        (model_checkpoint_status["status"] == "trained").sum()
    )
    _ablation_loaded = int(
        (ablation_checkpoint_status["status"] == "loaded").sum()
    )
    _ablation_trained = int(
        (ablation_checkpoint_status["status"] == "trained").sum()
    )

    mo.vstack([
        mo.md(
            "## Checkpoints\n\n"
            f"- Modèles rechargés : {_model_loaded} | "
            f"réentraînés : {_model_trained}\n"
            f"- Ablations rechargées : {_ablation_loaded} | "
            f"réentraînées : {_ablation_trained}"
        ),
        mo.ui.table(model_checkpoint_status, pagination=False),
        mo.ui.table(
            ablation_checkpoint_status,
            pagination=False,
        ),
    ])
    return


@app.cell
def _(ablations):
    ablations
    return


@app.cell
def _(ablations, plt):
    _figure, _axis = plt.subplots(figsize=(12, 5))
    _axis.bar(
        ablations["variant"],
        ablations["delta_mae"],
        alpha=0.75,
    )
    _axis.axhline(0, color="black", linestyle="--", linewidth=0.8)
    _axis.set_title("Ablations : dégradation de la MAE de validation")
    _axis.set_ylabel("ΔMAE (variante - modèle complet)")
    _axis.tick_params(axis="x", rotation=30)
    _axis.grid(True, axis="y", alpha=0.3)
    _figure.tight_layout()
    _figure
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 12. Global comparison and automatic diagnosis
    """)
    return


@app.cell
def _(metrics, models, pd):
    _m1 = metrics.loc[metrics["model"] == "M1_global"].iloc[0]
    _m2 = metrics.loc[metrics["model"] == "M2_film_bias"].iloc[0]
    _m3 = metrics.loc[metrics["model"] == "M3_latent"].iloc[0]
    _m4 = metrics.loc[metrics["model"] == "M4_full"].iloc[0]
    _c4 = metrics.loc[metrics["model"] == "C4_full"].iloc[0]

    gains = pd.DataFrame([
        {
            "comparison": "M1 -> M2 (biais film)",
            "delta_mae": _m2["val_mae"] - _m1["val_mae"],
            "delta_rmse": _m2["val_rmse"] - _m1["val_rmse"],
        },
        {
            "comparison": "M1 -> M3 (facteurs latents)",
            "delta_mae": _m3["val_mae"] - _m1["val_mae"],
            "delta_rmse": _m3["val_rmse"] - _m1["val_rmse"],
        },
        {
            "comparison": "M3 -> M4 (biais + interactions)",
            "delta_mae": _m4["val_mae"] - _m3["val_mae"],
            "delta_rmse": _m4["val_rmse"] - _m3["val_rmse"],
        },
        {
            "comparison": "M4 -> C4 (métadonnées)",
            "delta_mae": _c4["val_mae"] - _m4["val_mae"],
            "delta_rmse": _c4["val_rmse"] - _m4["val_rmse"],
        },
    ])

    _best_mae = metrics.sort_values("val_mae").iloc[0]
    _best_rmse = metrics.sort_values("val_rmse").iloc[0]
    _overfit = []
    for _name, _model in models.items():
        _history = _model["history"]
        if len(_history) < 3:
            continue
        _best_row = _history.loc[_history["val_rmse"].idxmin()]
        if (
            _best_row["epoch"] < _history["epoch"].max()
            and _history["train_rmse"].iloc[-1]
            < _history["train_rmse"].iloc[0]
        ):
            _overfit.append(_name)

    diagnosis = pd.DataFrame([
        {"diagnostic": "meilleur modèle (MAE validation)", "value": _best_mae["model"]},
        {"diagnostic": "meilleur modèle (RMSE validation)", "value": _best_rmse["model"]},
        {"diagnostic": "meilleure MAE validation", "value": float(_best_mae["val_mae"])},
        {"diagnostic": "meilleure RMSE validation", "value": float(_best_rmse["val_rmse"])},
        {"diagnostic": "modèles avec early stopping avant la fin", "value": ", ".join(_overfit) if _overfit else "aucun"},
    ])
    return diagnosis, gains


@app.cell
def _(gains):
    gains
    return


@app.cell
def _(diagnosis):
    diagnosis
    return


@app.cell
def _(models, np, pd, train_stats):
    _frame = models["C4_full"]["prediction_frame"].copy()
    _frame["user_activity"] = (
        _frame["row"].map(train_stats["user_counts"]).fillna(0)
    )
    _frame["film_popularity"] = (
        _frame["col"].map(train_stats["film_counts"]).fillna(0)
    )
    _frame["user_group"] = pd.qcut(
        _frame["user_activity"].rank(method="first"),
        q=4,
        labels=["Q1", "Q2", "Q3", "Q4"],
    )
    _frame["film_group"] = pd.qcut(
        _frame["film_popularity"].rank(method="first"),
        q=4,
        labels=["Q1", "Q2", "Q3", "Q4"],
    )
    _frame["rating_group"] = pd.cut(
        _frame["rating_true"],
        bins=[0.0, 2.0, 4.0, 5.0],
        labels=["bas", "milieu", "haut"],
        include_lowest=True,
    )

    _summary_rows = []
    for _column in ["user_group", "film_group", "rating_group"]:
        _grouped = (
            _frame.groupby(_column, observed=True)
            .agg(
                mae=("abs_error", "mean"),
                bias=("error", "mean"),
                n=("abs_error", "size"),
            )
            .reset_index()
        )
        for _row in _grouped.itertuples(index=False):
            _summary_rows.append({
                "population": _column,
                "group": str(getattr(_row, _column)),
                "mae": float(_row.mae),
                "bias": float(_row.bias),
                "n": int(_row.n),
            })

    error_population_summary = pd.DataFrame(_summary_rows)
    error_population_summary
    return (error_population_summary,)


@app.cell
def _(gains, plt):
    _figure, _axis = plt.subplots(figsize=(12, 5))
    _axis.bar(
        gains["comparison"],
        gains["delta_mae"],
        alpha=0.75,
    )
    _axis.axhline(0, color="black", linestyle="--", linewidth=0.8)
    _axis.set_title("Gains de MAE de validation")
    _axis.set_ylabel("ΔMAE")
    _axis.tick_params(axis="x", rotation=25)
    _axis.grid(True, axis="y", alpha=0.3)
    _figure.tight_layout()
    _figure
    return


@app.cell
def _(
    ablations,
    context_contribution_stats,
    diagnosis,
    error_population_summary,
    mo,
):
    _top_ablation = ablations.sort_values("delta_mae", ascending=False).iloc[0]
    _top_family = (
        context_contribution_stats[
            context_contribution_stats["component"].str.startswith("linear:")
        ]
        .sort_values("mean_absolute", ascending=False)
        .iloc[0]
    )
    _summary = mo.md(
        f"""
        ### Synthèse automatique

        - Meilleur modèle selon la MAE de validation : **{diagnosis.iloc[0]['value']}**.
        - Meilleur modèle selon la RMSE de validation : **{diagnosis.iloc[1]['value']}**.
        - Ablation la plus pénalisante : **{_top_ablation['variant']}**
          (ΔMAE = {_top_ablation['delta_mae']:.4f}).
        - Famille linéaire de plus grande amplitude moyenne :
          **{_top_family['component']}**
          ({_top_family['mean_absolute']:.4f}).
        - Population avec la MAE la plus élevée :
          **{error_population_summary.sort_values('mae', ascending=False).iloc[0]['population']}**
          / **{error_population_summary.sort_values('mae', ascending=False).iloc[0]['group']}**
          (MAE = {error_population_summary['mae'].max():.4f}).
        - Early stopping avant la dernière époque :
          **{diagnosis.iloc[4]['value']}**.

        Les résultats sont descriptifs. Ils ne mesurent pas des effets causaux
        et dépendent du découpage, de la couverture des métadonnées et de la
        sparsité des données.
        """
    )
    _summary
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 13. Consistency checks
    """)
    return


@app.cell
def _(
    context_reconstruction_error,
    feature_data,
    metrics,
    models,
    np,
    obs_test,
    pd,
    ratings_matrix,
    train_idx,
    val_idx,
    test_idx,
):
    _checks = []

    _train_set = set(map(int, train_idx))
    _val_set = set(map(int, val_idx))
    _test_set = set(map(int, test_idx))
    _checks.append({
        "check": "splits disjoints",
        "passed": not (
            _train_set & _val_set
            or _train_set & _test_set
            or _val_set & _test_set
        ),
    })
    _checks.append({
        "check": "test warm-start",
        "passed": bool(
            len(feature_data["part1"]["splits"]["test"]["obs"])
            == len(obs_test)
        ),
    })
    _checks.append({
        "check": "mêmes observations test pour tous les modèles",
        "passed": bool(
            all(
                len(model["prediction_frame"]) == len(obs_test)
                for model in models.values()
            )
        ),
    })
    _checks.append({
        "check": "métriques sans NaN",
        "passed": bool(
            not metrics.select_dtypes(include=[np.number]).isna().any().any()
        ),
    })
    _checks.append({
        "check": "prédictions finies",
        "passed": bool(
            all(
                np.isfinite(model["test_prediction"]).all()
                for model in models.values()
            )
        ),
    })
    _checks.append({
        "check": "matrices de features creuses",
        "passed": bool(
            feature_data["part1"]["splits"]["train"]["x"].format == "csr"
            and feature_data["context"]["splits"]["train"]["x"].format
            == "csr"
        ),
    })

    for _name, _model in models.items():
        _spec = _model["spec"]
        if _spec["feature_set"] == "none":
            _X = feature_data["part1"]["splits"]["test"]["x"]
        else:
            _X = feature_data[_spec["feature_set"]]["splits"]["test"]["x"]
        _prediction = _model["engine"].predict(_X)
        if not np.isfinite(_prediction).all():
            _checks.append({
                "check": f"prédictions finies {_name}",
                "passed": False,
            })

    _genre_diagonal = feature_data["item_genres"].tocoo()
    _studio_diagonal = feature_data["item_studios"].tocoo()
    _checks.append({
        "check": "genres multi-label sans doublon par film",
        "passed": bool(
            not pd.DataFrame({
                "row": _genre_diagonal.row,
                "col": _genre_diagonal.col,
            }).duplicated().any()
        ),
    })
    _checks.append({
        "check": "studios multi-label sans doublon par film",
        "passed": bool(
            not pd.DataFrame({
                "row": _studio_diagonal.row,
                "col": _studio_diagonal.col,
            }).duplicated().any()
        ),
    })
    _checks.append({
        "check": "reconstruction exacte des contributions C4",
        "passed": bool(context_reconstruction_error < 1e-6),
    })
    _checks.append({
        "check": "matrice ratings au format CSR",
        "passed": bool(
            hasattr(ratings_matrix, "format")
            and ratings_matrix.format == "csr"
        ),
    })

    checks_final = pd.DataFrame(_checks)
    if not checks_final["passed"].all():
        failed = checks_final.loc[
            ~checks_final["passed"], "check"
        ].tolist()
        raise AssertionError(f"Contrôles échoués: {failed}")
    return (checks_final,)


@app.cell
def _(checks_final):
    checks_final
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 14. Exports
    """)
    return


@app.cell
def _(
    INPUT_FILES,
    MODEL_OUTPUT_DIR,
    ablations,
    batch_size,
    calibration_figure,
    checks_final,
    context_contribution_stats,
    context_contributions,
    context_diagnostics_figure,
    context_interaction_heatmap,
    context_reconstruction_error,
    diagnosis,
    diagnostic_plot_figure,
    error_population_summary,
    feature_data,
    feature_summary,
    gains,
    heatmap_figure,
    histories,
    interaction_heatmap_figure,
    json,
    learning_rate,
    metrics,
    models,
    n_epochs,
    n_factors,
    np,
    pd,
    plt,
    predictions,
    reg_bias,
    reg_factors,
    seed,
    simple_decomposition,
    train_fraction,
    val_fraction,
):
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metrics.to_csv(MODEL_OUTPUT_DIR / "metrics.csv", index=False)
    histories.to_parquet(
        MODEL_OUTPUT_DIR / "training_histories.parquet",
        index=False,
    )
    predictions.to_parquet(
        MODEL_OUTPUT_DIR / "test_predictions.parquet",
        index=False,
    )
    ablations.to_csv(MODEL_OUTPUT_DIR / "ablations.csv", index=False)
    context_contribution_stats.to_parquet(
        MODEL_OUTPUT_DIR / "context_contribution_stats.parquet",
        index=False,
    )
    context_contributions.to_parquet(
        MODEL_OUTPUT_DIR / "context_contributions.parquet",
        index=False,
    )
    simple_decomposition.to_parquet(
        MODEL_OUTPUT_DIR / "simple_decomposition.parquet",
        index=False,
    )
    context_interaction_heatmap.to_csv(
        MODEL_OUTPUT_DIR / "context_interaction_heatmap.csv",
    )
    feature_summary.to_csv(
        MODEL_OUTPUT_DIR / "feature_summary.csv",
        index=False,
    )
    diagnosis.to_csv(
        MODEL_OUTPUT_DIR / "diagnosis.csv",
        index=False,
    )
    error_population_summary.to_csv(
        MODEL_OUTPUT_DIR / "error_population_summary.csv",
        index=False,
    )
    gains.to_csv(MODEL_OUTPUT_DIR / "gains.csv", index=False)
    checks_final.to_csv(
        MODEL_OUTPUT_DIR / "checks.csv",
        index=False,
    )

    _feature_mapping = pd.DataFrame({
        "feature_index": np.arange(
            feature_data["context"]["n_features"]
        ),
        "feature_name": feature_data["context"]["names"],
        "family": feature_data["context"]["groups"],
    })
    _feature_mapping.to_parquet(
        MODEL_OUTPUT_DIR / "feature_mapping.parquet",
        index=False,
    )

    for _name, _model in models.items():
        _engine = _model["engine"]
        np.savez_compressed(
            MODEL_OUTPUT_DIR / f"best_params_{_name}.npz",
            w0=np.array([_engine.w0]),
            w=_engine.w,
            V=_engine.V,
            active_mask=_engine.active_mask,
        )

    _config_payload = {
        "input_files": {
            key: str(value) for key, value in INPUT_FILES.items()
        },
        "hyperparameters": {
            "n_factors": int(n_factors.value),
            "learning_rate": float(learning_rate.value),
            "n_epochs": int(n_epochs.value),
            "reg_bias": float(reg_bias.value),
            "reg_factors": float(reg_factors.value),
            "batch_size": int(batch_size.value),
            "seed": int(seed.value),
            "train_fraction": float(train_fraction.value),
            "val_fraction": float(val_fraction.value),
            "test_fraction": float(
                1 - train_fraction.value - val_fraction.value
            ),
        },
        "context_reconstruction_error": context_reconstruction_error,
        "models": metrics.to_dict("records"),
        "ablations": ablations.to_dict("records"),
    }
    with open(
        MODEL_OUTPUT_DIR / "model_config.json",
        "w",
    ) as _handle:
        json.dump(_config_payload, _handle, indent=2, default=str)

    _metrics_figure, _metrics_axis = plt.subplots(figsize=(14, 6))
    _ordered = metrics.sort_values("family")
    _metrics_axis.bar(
        _ordered["model"],
        _ordered["test_mae"],
        alpha=0.75,
    )
    _metrics_axis.set_title("MAE test des huit modèles")
    _metrics_axis.set_ylabel("MAE")
    _metrics_axis.tick_params(axis="x", rotation=35)
    _metrics_axis.grid(True, axis="y", alpha=0.3)
    _metrics_figure.tight_layout()
    _metrics_figure.savefig(
        MODEL_OUTPUT_DIR / "metrics_comparison.png",
        dpi=150,
        bbox_inches="tight",
    )

    _ablation_figure, _ablation_axis = plt.subplots(figsize=(12, 5))
    _ablation_axis.bar(
        ablations["variant"],
        ablations["delta_mae"],
        alpha=0.75,
    )
    _ablation_axis.axhline(0, color="black", linestyle="--", linewidth=0.8)
    _ablation_axis.set_title("Ablations : delta MAE validation")
    _ablation_axis.set_ylabel("ΔMAE")
    _ablation_axis.tick_params(axis="x", rotation=30)
    _ablation_axis.grid(True, axis="y", alpha=0.3)
    _ablation_figure.tight_layout()
    _ablation_figure.savefig(
        MODEL_OUTPUT_DIR / "ablations.png",
        dpi=150,
        bbox_inches="tight",
    )

    diagnostic_plot_figure.savefig(
        MODEL_OUTPUT_DIR / "diagnostics_selected_model.png",
        dpi=150,
        bbox_inches="tight",
    )
    calibration_figure.savefig(
        MODEL_OUTPUT_DIR / "calibration_and_errors.png",
        dpi=150,
        bbox_inches="tight",
    )
    context_diagnostics_figure.savefig(
        MODEL_OUTPUT_DIR / "context_diagnostics.png",
        dpi=150,
        bbox_inches="tight",
    )
    interaction_heatmap_figure.savefig(
        MODEL_OUTPUT_DIR / "interaction_heatmap.png",
        dpi=150,
        bbox_inches="tight",
    )
    heatmap_figure.savefig(
        MODEL_OUTPUT_DIR / "activity_error_heatmap.png",
        dpi=150,
        bbox_inches="tight",
    )

    pd.DataFrame({
        "export": [
            str(path.name) for path in MODEL_OUTPUT_DIR.iterdir()
        ]
    }).sort_values("export")
    return


if __name__ == "__main__":
    app.run()
