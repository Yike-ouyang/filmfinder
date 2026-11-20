import marimo

__generated_with = "0.17.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import ast
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
        ast,
        csr_matrix,
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
    # FilmFinder — Contextual Factorization Machine

    Comparaison entre une FM utilisateur-film (M4) et une FM contextuelle
    (C4) utilisant genres, année et studios. Les deux modèles sont entraînés
    avec `FMEngine`, sur les mêmes partitions et avec la même seed.
    """)
    return


@app.cell
def _(Path, json, os, pd):
    CLEANING_DIR = Path(
        os.environ.get(
            "FILMFINDER_CLEANING_DIR",
            Path.cwd(),
        )
    )
    MODEL_OUTPUT_DIR = Path(
        os.environ.get(
            "FILMFINDER_MODEL_OUTPUT_DIR",
            CLEANING_DIR / "modeling",
        )
    )
    CHECKPOINT_DIR = MODEL_OUTPUT_DIR / "checkpoints"
    CONTEXT_FM_DIR = MODEL_OUTPUT_DIR / "context_fm"

    INPUT_FILES = {
        "matrix": CLEANING_DIR / "ratings_matrix.npz",
        "users": CLEANING_DIR / "user_ids.npy",
        "movies": CLEANING_DIR / "movie_ids.npy",
        "context": CLEANING_DIR / "movie_context.json",
    }

    _missing_inputs = [
        str(path)
        for path in INPUT_FILES.values()
        if not path.exists()
    ]
    if _missing_inputs:
        raise FileNotFoundError(
            f"Missing input files: {_missing_inputs}"
        )

    _manifest_path = CHECKPOINT_DIR / "manifest.json"
    manifest = None
    if _manifest_path.exists():
        with open(_manifest_path) as _handle:
            manifest = json.load(_handle)

    _config_path = MODEL_OUTPUT_DIR / "model_config.json"
    if _config_path.exists():
        with open(_config_path) as _handle:
            model_config = json.load(_handle)
    elif manifest is not None:
        _run_context = manifest["run_context"]
        model_config = {
            "hyperparameters": _run_context["hyperparameters"],
            "split": _run_context.get("split", {}),
            "feature_config": _run_context.get("features", {}),
        }
    else:
        raise FileNotFoundError(
            "No model_config.json or checkpoints manifest found."
        )

    hyperparameters = model_config["hyperparameters"]
    _split_context = model_config.get("split", {})
    base_seed = int(
        _split_context.get(
            "seed",
            hyperparameters.get("seed", 42),
        )
    )
    train_fraction = float(
        _split_context.get(
            "train_fraction",
            hyperparameters.get("train_fraction", 0.8),
        )
    )
    val_fraction = float(
        _split_context.get(
            "val_fraction",
            hyperparameters.get("val_fraction", 0.1),
        )
    )
    batch_size = int(hyperparameters["batch_size"])
    learning_rate_default = float(
        hyperparameters.get("learning_rate", 0.01)
    )
    n_epochs_default = int(
        hyperparameters.get("n_epochs", 20)
    )
    max_studios = int(
        model_config.get("feature_config", {}).get(
            "max_studios",
            4000,
        )
    )
    model_seed = base_seed + 3000
    REG_VALUES = [0.0, 1e-8, 1e-7, 1e-6, 1e-5]

    _m4_dir = CHECKPOINT_DIR / "models" / "M4_full"
    m4_param_file = _m4_dir / "params.npz"
    m4_metric_file = _m4_dir / "metric.csv"
    m4_val_prediction_file = _m4_dir / "val_prediction.npy"

    _missing_m4 = [
        str(path)
        for path in (
            m4_param_file,
            m4_metric_file,
            m4_val_prediction_file,
        )
        if not path.exists()
    ]
    if _missing_m4:
        raise FileNotFoundError(
            f"Missing M4 checkpoint files: {_missing_m4}"
        )

    return (
        CHECKPOINT_DIR,
        CLEANING_DIR,
        CONTEXT_FM_DIR,
        INPUT_FILES,
        MODEL_OUTPUT_DIR,
        REG_VALUES,
        base_seed,
        batch_size,
        hyperparameters,
        learning_rate_default,
        m4_metric_file,
        m4_param_file,
        m4_val_prediction_file,
        max_studios,
        model_seed,
        n_epochs_default,
        train_fraction,
        val_fraction,
    )


@app.cell
def _(INPUT_FILES, load_npz, np, pd):
    ratings_matrix = load_npz(INPUT_FILES["matrix"]).tocsr()
    n_users, n_movies = ratings_matrix.shape

    user_ids = np.load(INPUT_FILES["users"], allow_pickle=True)
    movie_ids = np.load(INPUT_FILES["movies"], allow_pickle=True)

    if len(user_ids) != n_users or len(movie_ids) != n_movies:
        raise ValueError(
            "Mismatch between matrix shape and ID arrays: "
            f"matrix={ratings_matrix.shape}, "
            f"users={len(user_ids)}, movies={len(movie_ids)}."
        )

    user_mapping = pd.DataFrame({
        "user_id": user_ids,
        "row_index": np.arange(n_users),
    })
    movie_mapping = pd.DataFrame({
        "canonical_movie_id": movie_ids,
        "col_index": np.arange(n_movies),
    })

    movie_context = pd.read_json(INPUT_FILES["context"])
    movie_context = movie_context.rename(columns={
        "movie_id": "canonical_movie_id",
        "date": "year",
        "studio": "studios",
    })
    movie_context["col_index"] = np.arange(len(movie_context))

    if len(movie_context) != n_movies:
        raise ValueError(
            "movie_context row count does not match matrix columns: "
            f"context={len(movie_context)}, movies={n_movies}."
        )

    if (
        movie_context["canonical_movie_id"].astype(str).tolist()
        != movie_mapping["canonical_movie_id"].astype(str).tolist()
    ):
        raise ValueError(
            "movie_context is not aligned with movie_ids.npy."
        )

    user_mapping_sorted = (
        user_mapping.sort_values("row_index").reset_index(drop=True)
    )
    movie_mapping_sorted = (
        movie_mapping.sort_values("col_index").reset_index(drop=True)
    )
    context_sorted = (
        movie_context.sort_values("col_index").reset_index(drop=True)
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

    return (
        context_sorted,
        movie_mapping_sorted,
        n_movies,
        n_users,
        observations,
        user_mapping_sorted,
    )


@app.cell
def _(
    base_seed,
    np,
    observations,
    pd,
    train_fraction,
    val_fraction,
):
    _test_fraction = 1.0 - train_fraction - val_fraction
    if not (0 < train_fraction < 1 and 0 < val_fraction < 1):
        raise ValueError("Invalid train/validation fractions.")
    if _test_fraction <= 0:
        raise ValueError(
            "Train and validation fractions must sum to less than 1."
        )

    _rng = np.random.default_rng(base_seed)
    _n_obs = len(observations)
    _permutation = _rng.permutation(_n_obs)

    _n_train = int(np.floor(train_fraction * _n_obs))
    _n_val = int(np.floor(val_fraction * _n_obs))

    _train_idx = _permutation[:_n_train]
    _val_idx_all = _permutation[_n_train:_n_train + _n_val]
    _test_idx_all = _permutation[_n_train + _n_val:]

    _train_users = set(
        observations.iloc[_train_idx]["row"].unique().tolist()
    )
    _train_films = set(
        observations.iloc[_train_idx]["col"].unique().tolist()
    )

    _val_frame = observations.iloc[_val_idx_all]
    _test_frame = observations.iloc[_test_idx_all]

    _val_warm_mask = (
        _val_frame["row"].isin(_train_users)
        & _val_frame["col"].isin(_train_films)
    ).to_numpy()
    _test_warm_mask = (
        _test_frame["row"].isin(_train_users)
        & _test_frame["col"].isin(_train_films)
    ).to_numpy()

    _val_idx = _val_idx_all[_val_warm_mask]
    _test_idx = _test_idx_all[_test_warm_mask]

    obs_train = observations.iloc[_train_idx].reset_index(drop=True)
    obs_val = observations.iloc[_val_idx].reset_index(drop=True)
    obs_test = observations.iloc[_test_idx].reset_index(drop=True)

    y_train = obs_train["rating"].to_numpy(dtype=np.float64)
    y_val = obs_val["rating"].to_numpy(dtype=np.float64)
    y_test = obs_test["rating"].to_numpy(dtype=np.float64)

    user_idx_val = obs_val["row"].to_numpy(dtype=np.int64)
    film_idx_val = obs_val["col"].to_numpy(dtype=np.int64)
    user_idx_test = obs_test["row"].to_numpy(dtype=np.int64)
    film_idx_test = obs_test["col"].to_numpy(dtype=np.int64)

    split_summary = pd.DataFrame([
        {"split": "train", "interactions": len(obs_train)},
        {"split": "validation", "interactions": len(obs_val)},
        {"split": "test", "interactions": len(obs_test)},
    ])

    return (
        film_idx_test,
        film_idx_val,
        obs_test,
        obs_train,
        obs_val,
        split_summary,
        user_idx_test,
        user_idx_val,
        y_test,
        y_train,
        y_val,
    )


@app.cell
def _(
    Counter,
    ast,
    context_sorted,
    csr_matrix,
    hstack,
    max_studios,
    movie_mapping_sorted,
    n_movies,
    n_users,
    np,
    obs_test,
    obs_train,
    obs_val,
    pd,
    user_mapping_sorted,
):
    def _as_list(value):
        if value is None:
            return None
        if isinstance(value, str):
            _text = value.strip()
            if _text.startswith("[") and _text.endswith("]"):
                try:
                    value = ast.literal_eval(_text)
                except (SyntaxError, ValueError):
                    value = _text
            elif "," in _text:
                value = [
                    _item.strip()
                    for _item in _text.split(",")
                    if _item.strip()
                ]
            else:
                value = _text
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

    _context_by_col = (
        context_sorted.set_index("col_index").sort_index()
    )

    _genres_by_film = [
        _as_list(value)
        for value in _context_by_col.get(
            "genres",
            pd.Series([None] * n_movies, index=_context_by_col.index),
        )
    ]
    _studios_by_film = [
        _as_list(value)
        for value in _context_by_col.get(
            "studios",
            pd.Series([None] * n_movies, index=_context_by_col.index),
        )
    ]

    _train_films = np.sort(obs_train["col"].unique())
    _genre_counter = Counter()
    for _film in _train_films:
        _genres = _genres_by_film[_film]
        if _genres:
            _genre_counter.update(set(_genres))

    _studio_counter = Counter()
    for _studios in _studios_by_film:
        if _studios:
            _studio_counter.update(set(_studios))

    _studio_ranked = sorted(
        _studio_counter.items(),
        key=lambda item: (-item[1], item[0]),
    )
    genre_vocab = sorted(_genre_counter)
    studio_vocab = sorted(
        studio for studio, _ in _studio_ranked[:max_studios]
    )

    _retained_studios = set(studio_vocab)
    _studios_by_film = [
        (
            None
            if _studios is None
            else (
                [
                    _studio
                    for _studio in _studios
                    if _studio in _retained_studios
                ]
                or None
            )
        )
        for _studios in _studios_by_film
    ]

    genre_feature_names = (
        ["genre:<UNKNOWN>", "genre:<MISSING>"]
        + [f"genre:{value}" for value in genre_vocab]
    )
    studio_feature_names = (
        ["studio:<UNKNOWN>", "studio:<MISSING>"]
        + [f"studio:{value}" for value in studio_vocab]
    )

    _genre_index = {
        value: index + 2
        for index, value in enumerate(genre_vocab)
    }
    _studio_index = {
        value: index + 2
        for index, value in enumerate(studio_vocab)
    }

    _genre_rows = []
    _genre_cols = []
    _studio_rows = []
    _studio_cols = []

    for _film in range(n_movies):
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
        shape=(n_movies, len(genre_feature_names)),
    )
    item_studios = csr_matrix(
        (
            np.ones(len(_studio_rows), dtype=np.float64),
            (_studio_rows, _studio_cols),
        ),
        shape=(n_movies, len(studio_feature_names)),
    )

    _film_years = pd.to_numeric(
        _context_by_col.get(
            "year",
            pd.Series([np.nan] * n_movies, index=_context_by_col.index),
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
            shape=(_n, n_users),
        )
        film_block = csr_matrix(
            (
                np.ones(_n, dtype=np.float64),
                (_index, _cols),
            ),
            shape=(_n, n_movies),
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
    _context_groups = np.array(
        ["user"] * n_users
        + ["film"] * n_movies
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

    context_mapping = pd.DataFrame({
        "feature_index": np.arange(len(_context_names)),
        "feature_name": _context_names,
        "family": _context_groups,
    })
    context_families = _context_groups
    context_family_indices = {
        family: np.where(context_families == family)[0]
        for family in ["user", "film", "genre", "year", "studio"]
    }

    part1_train = _splits["train"]["x_user_item"]
    part1_val = _splits["val"]["x_user_item"]
    part1_test = _splits["test"]["x_user_item"]
    context_train = _splits["train"]["x_context"]
    context_val = _splits["val"]["x_context"]
    context_test = _splits["test"]["x_context"]

    context_blocks_train = {
        family: context_train[:, columns]
        for family, columns in context_family_indices.items()
    }
    context_blocks_val = {
        family: context_val[:, columns]
        for family, columns in context_family_indices.items()
    }
    context_blocks_test = {
        family: context_test[:, columns]
        for family, columns in context_family_indices.items()
    }

    model_inputs = {
        "M4": {
            "X_train": part1_train,
            "X_val": part1_val,
            "X_test": part1_test,
            "n_features": n_users + n_movies,
            "active_mask": np.ones(
                n_users + n_movies,
                dtype=bool,
            ),
            "feature_set": "user_film",
        },
        "C4": {
            "X_train": context_train,
            "X_val": context_val,
            "X_test": context_test,
            "n_features": len(_context_names),
            "active_mask": np.ones(
                len(_context_names),
                dtype=bool,
            ),
            "feature_set": "context",
        },
    }

    return (
        context_blocks_test,
        context_blocks_val,
        context_families,
        context_family_indices,
        context_mapping,
        context_test,
        model_inputs,
        part1_test,
        part1_train,
        part1_val,
    )


@app.cell
def _(np, pd, time):
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
    FMEngine,
    compute_metrics,
    m4_metric_file,
    m4_param_file,
    m4_val_prediction_file,
    mo,
    np,
    obs_val,
    part1_val,
    pd,
    y_val,
):
    _val_prediction = np.load(m4_val_prediction_file)
    _metrics = compute_metrics(y_val, _val_prediction)
    _reported = pd.read_csv(m4_metric_file).iloc[0]

    if (
        abs(_metrics["mae"] - float(_reported["val_mae"])) > 1e-6
        or abs(_metrics["rmse"] - float(_reported["val_rmse"])) > 1e-6
    ):
        raise ValueError(
            "M4 checkpoint is not aligned with the reconstructed split."
        )

    with np.load(m4_param_file) as _params:
        _w0 = float(_params["w0"][0])
        _w = _params["w"].copy()
        _V = _params["V"].copy()
        _active_mask = _params["active_mask"].astype(bool)

    _engine = FMEngine(
        n_features=int(_reported["n_features"]),
        n_factors=int(_reported["n_factors"]),
        use_linear=bool(_reported["use_linear"]),
        use_latent=bool(_reported["use_latent"]),
        active_mask=_active_mask,
        learning_rate=0.0,
        reg_bias=0.0,
        reg_factors=0.0,
        seed=0,
    )
    _engine.w0 = _w0
    _engine.w = _w
    _engine.V = _V
    _prediction = _engine.predict(part1_val)
    _difference = float(
        np.max(np.abs(_prediction - _val_prediction))
    )
    if _difference > 1e-6:
        raise ValueError(
            "M4 feature alignment mismatch: "
            f"max difference={_difference}."
        )

    m4_alignment_check = pd.DataFrame([{
        "check": "M4 checkpoint alignment",
        "val_mae_recomputed": _metrics["mae"],
        "val_mae_reported": float(_reported["val_mae"]),
        "max_abs_prediction_difference": _difference,
        "passed": True,
    }])
    mo.ui.table(m4_alignment_check, pagination=False)
    return (m4_alignment_check,)


@app.cell
def _(
    learning_rate_default,
    mo,
    n_epochs_default,
):
    n_factors = mo.ui.slider(
        start=1,
        stop=64,
        value=8,
        step=1,
        label="Nombre de facteurs latents",
    )
    learning_rate = mo.ui.number(
        value=learning_rate_default,
        start=0.0001,
        stop=1.0,
        step=0.001,
        label="Learning rate",
    )
    n_epochs = mo.ui.number(
        value=n_epochs_default,
        start=1,
        stop=200,
        step=1,
        label="Nombre maximal d'époques",
    )
    reg_bias = mo.ui.number(
        value=1e-7,
        start=0.0,
        stop=1.0,
        step=1e-8,
        label="reg_bias initial",
    )
    reg_factors = mo.ui.number(
        value=1e-7,
        start=0.0,
        stop=1.0,
        step=1e-8,
        label="reg_factors initial",
    )
    max_scatter_points = mo.ui.number(
        value=5000,
        start=100,
        step=1000,
        label="Points maximaux par scatter",
    )
    run_search = mo.ui.checkbox(
        value=True,
        label="Exécuter la recherche croisée des régularisations",
    )
    run_reproducibility = mo.ui.checkbox(
        value=True,
        label="Vérifier la reproductibilité",
    )

    mo.vstack([
        mo.hstack(
            [n_factors, learning_rate, n_epochs],
            justify="start",
            gap=2,
        ),
        mo.hstack(
            [reg_bias, reg_factors],
            justify="start",
            gap=2,
        ),
        mo.hstack(
            [max_scatter_points, run_search, run_reproducibility],
            justify="start",
            gap=2,
        ),
    ])
    return (
        learning_rate,
        max_scatter_points,
        n_epochs,
        n_factors,
        reg_bias,
        reg_factors,
        run_reproducibility,
        run_search,
    )


@app.cell
def _(np, pd):
    def subsample_frame(frame, max_points, random_state):
        if len(frame) <= max_points:
            return frame
        return frame.sample(
            n=int(max_points),
            random_state=random_state,
        )

    def compute_bin_curve(frame, x_column, y_column, n_bins=20):
        _empty = pd.DataFrame(
            columns=["bin_left", "bin_right", "x", "n", "mean"]
        )
        _data = frame[[x_column, y_column]].dropna()
        if _data.empty or _data[x_column].nunique() < 2:
            return _empty

        _bins = min(n_bins, max(2, _data[x_column].nunique()))
        _codes = pd.qcut(
            _data[x_column],
            q=_bins,
            duplicates="drop",
            labels=False,
        )
        _data = _data.assign(_bin=_codes).dropna(subset=["_bin"])
        if _data.empty:
            return _empty

        _grouped = (
            _data.groupby("_bin")[y_column]
            .agg(["size", "mean"])
            .reset_index()
        )
        _rows = []
        for _record in _grouped.to_dict("records"):
            _bin = int(_record["_bin"])
            _subset = _data[_data["_bin"] == _bin]
            _rows.append({
                "bin_left": float(_subset[x_column].min()),
                "bin_right": float(_subset[x_column].max()),
                "x": float(_subset[x_column].median()),
                "n": int(_record["size"]),
                "mean": float(_record["mean"]),
            })
        return pd.DataFrame(_rows)

    def scatter_with_curve(
        axis,
        frame,
        x_column,
        y_column,
        title,
        x_label,
        y_label,
        max_points=5000,
        random_state=42,
    ):
        _plot_data = frame[[x_column, y_column]].dropna()
        _plot_data = subsample_frame(
            _plot_data,
            max_points,
            random_state,
        )
        axis.scatter(
            _plot_data[x_column],
            _plot_data[y_column],
            s=6,
            alpha=0.12,
            edgecolors="none",
        )
        _curve = compute_bin_curve(frame, x_column, y_column)
        if not _curve.empty:
            axis.plot(
                _curve["x"],
                _curve["mean"],
                color="crimson",
                marker="o",
                linewidth=1.4,
                markersize=3,
            )
        axis.set_title(title)
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        axis.grid(True, alpha=0.3)
        return _curve

    return compute_bin_curve, scatter_with_curve, subsample_frame


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Entraînement et recherche de régularisation
    """)
    return


@app.cell
def _(
    CONTEXT_FM_DIR,
    FMEngine,
    REG_VALUES,
    batch_size,
    compute_metrics,
    context_blocks_val,
    context_family_indices,
    film_idx_val,
    learning_rate,
    mo,
    model_inputs,
    model_seed,
    n_epochs,
    n_factors,
    n_users,
    np,
    pd,
    reg_bias,
    reg_factors,
    run_search,
    train_fm_model,
    user_idx_val,
    y_train,
    y_val,
):
    CONTEXT_FM_DIR.mkdir(parents=True, exist_ok=True)
    _search_path = CONTEXT_FM_DIR / "search_results.csv"

    _columns = [
        "source",
        "model",
        "reg_bias",
        "reg_factors",
        "n_factors",
        "learning_rate",
        "n_epochs",
        "batch_size",
        "seed",
        "train_mae",
        "val_mae",
        "val_rmse",
        "val_bias",
        "std_pred",
        "mean_abs_user_bias",
        "mean_abs_film_bias",
        "mean_abs_user_film_dot",
        "mean_v_norm",
        "n_parameters",
        "training_time_seconds",
        "best_epoch",
        "epochs_run",
    ]

    _numeric_columns = [
        "reg_bias",
        "reg_factors",
        "n_factors",
        "learning_rate",
        "n_epochs",
        "batch_size",
        "seed",
        "train_mae",
        "val_mae",
        "val_rmse",
        "val_bias",
        "std_pred",
        "mean_abs_user_bias",
        "mean_abs_film_bias",
        "mean_abs_user_film_dot",
        "mean_v_norm",
        "n_parameters",
        "training_time_seconds",
        "best_epoch",
        "epochs_run",
    ]

    def _coerce_search_results(frame):
        _frame = frame.copy()
        for _column in _numeric_columns:
            if _column in _frame.columns:
                _frame[_column] = pd.to_numeric(
                    _frame[_column],
                    errors="coerce",
                )
        return _frame

    if _search_path.exists():
        _existing = pd.read_csv(_search_path)
        if set(_columns).issubset(_existing.columns):
            _search_results = _existing.copy().reset_index(drop=True)
        else:
            _search_results = pd.DataFrame(columns=_columns)
    else:
        _search_results = pd.DataFrame(columns=_columns)

    _search_results = _coerce_search_results(_search_results)

    _config_mask = (
        (_search_results["n_factors"] == int(n_factors.value))
        & (
            _search_results["learning_rate"].round(12)
            == round(float(learning_rate.value), 12)
        )
        & (_search_results["n_epochs"] == int(n_epochs.value))
        & (_search_results["batch_size"] == int(batch_size))
        & (_search_results["seed"] == int(model_seed))
    )
    _completed = set()
    for _record in _search_results[_config_mask].to_dict("records"):
        _completed.add((
            _record["model"],
            round(float(_record["reg_bias"]), 12),
            round(float(_record["reg_factors"]), 12),
        ))

    _session_engines = {}

    def _save_search_results():
        _temporary = _search_path.with_name(
            _search_path.stem + ".tmp.csv"
        )
        _search_results.to_csv(_temporary, index=False)
        _temporary.replace(_search_path)

    def _amplitude_metrics(model_name, engine):
        if model_name == "M4":
            _user_bias = engine.w[user_idx_val]
            _film_bias = engine.w[n_users + film_idx_val]
            _dot = np.einsum(
                "ij,ij->i",
                engine.V[user_idx_val],
                engine.V[n_users + film_idx_val],
            )
        else:
            _user_cols = context_family_indices["user"]
            _film_cols = context_family_indices["film"]
            _X_user = context_blocks_val["user"]
            _X_film = context_blocks_val["film"]
            _user_bias = _X_user @ engine.w[_user_cols]
            _film_bias = _X_film @ engine.w[_film_cols]
            _S_user = _X_user @ engine.V[_user_cols]
            _S_film = _X_film @ engine.V[_film_cols]
            _dot = np.einsum("ij,ij->i", _S_user, _S_film)
        return {
            "mean_abs_user_bias": float(np.abs(_user_bias).mean()),
            "mean_abs_film_bias": float(np.abs(_film_bias).mean()),
            "mean_abs_user_film_dot": float(np.abs(_dot).mean()),
        }

    def _run_configuration(
        model_name,
        reg_bias_value,
        reg_factors_value,
        source,
        force_retrain=False,
    ):
        _key = (
            model_name,
            round(float(reg_bias_value), 12),
            round(float(reg_factors_value), 12),
        )
        _matching = _search_results[
            (_search_results["model"] == model_name)
            & (
                _search_results["reg_bias"].round(12)
                == _key[1]
            )
            & (
                _search_results["reg_factors"].round(12)
                == _key[2]
            )
            & (_search_results["n_factors"] == int(n_factors.value))
            & (
                _search_results["learning_rate"].round(12)
                == round(float(learning_rate.value), 12)
            )
            & (_search_results["n_epochs"] == int(n_epochs.value))
            & (_search_results["batch_size"] == int(batch_size))
            & (_search_results["seed"] == int(model_seed))
        ]
        if not force_retrain and not _matching.empty:
            return _matching.iloc[0].to_dict(), _session_engines.get(_key)

        _data = model_inputs[model_name]
        (
            _engine,
            _history,
            _best_epoch,
            _epochs_run,
            _training_time,
        ) = train_fm_model(
            _data["X_train"],
            y_train,
            _data["X_val"],
            y_val,
            n_features=_data["n_features"],
            n_factors=int(n_factors.value),
            use_linear=True,
            use_latent=True,
            active_mask=_data["active_mask"],
            learning_rate=float(learning_rate.value),
            reg_bias=float(reg_bias_value),
            reg_factors=float(reg_factors_value),
            n_epochs=int(n_epochs.value),
            batch_size=int(batch_size),
            seed=int(model_seed),
        )

        _train_pred = _engine.predict(_data["X_train"])
        _val_pred = _engine.predict(_data["X_val"])
        _train_metrics = compute_metrics(y_train, _train_pred)
        _val_metrics = compute_metrics(y_val, _val_pred)
        _amplitudes = _amplitude_metrics(model_name, _engine)
        _mean_v_norm = float(
            np.linalg.norm(
                _engine.V,
                axis=1,
            )[_data["active_mask"]].mean()
        )

        _row = {
            "source": source,
            "model": model_name,
            "reg_bias": float(reg_bias_value),
            "reg_factors": float(reg_factors_value),
            "n_factors": int(n_factors.value),
            "learning_rate": float(learning_rate.value),
            "n_epochs": int(n_epochs.value),
            "batch_size": int(batch_size),
            "seed": int(model_seed),
            "train_mae": _train_metrics["mae"],
            "val_mae": _val_metrics["mae"],
            "val_rmse": _val_metrics["rmse"],
            "val_bias": _val_metrics["bias"],
            "std_pred": float(np.std(_val_pred, ddof=0)),
            "mean_abs_user_bias": _amplitudes[
                "mean_abs_user_bias"
            ],
            "mean_abs_film_bias": _amplitudes[
                "mean_abs_film_bias"
            ],
            "mean_abs_user_film_dot": _amplitudes[
                "mean_abs_user_film_dot"
            ],
            "mean_v_norm": _mean_v_norm,
            "n_parameters": int(
                1
                + (
                    int(_engine.active_mask.sum())
                    if _engine.use_linear
                    else 0
                )
                + (
                    int(_engine.active_mask.sum()) * _engine.n_factors
                    if _engine.use_latent
                    else 0
                )
            ),
            "training_time_seconds": _training_time,
            "best_epoch": _best_epoch,
            "epochs_run": _epochs_run,
        }

        _search_results = pd.concat(
            [_search_results, pd.DataFrame([_row])],
            ignore_index=True,
        )
        _search_results = _coerce_search_results(_search_results)
        _session_engines[_key] = _engine
        _save_search_results()
        print(
            f"{model_name} reg_bias={reg_bias_value:g} "
            f"reg_factors={reg_factors_value:g} "
            f"val_mae={_row['val_mae']:.6f}"
        )
        return _row, _engine

    _initial_rows = []
    for _model_name in ["M4", "C4"]:
        _row, _engine = _run_configuration(
            _model_name,
            float(reg_bias.value),
            float(reg_factors.value),
            "initial",
        )
        _initial_rows.append(_row)
    initial_comparison = pd.DataFrame(_initial_rows)
    initial_comparison.to_csv(
        CONTEXT_FM_DIR / "initial_comparison.csv",
        index=False,
    )

    if run_search.value:
        for _model_name in ["M4", "C4"]:
            for _rb in REG_VALUES:
                for _rf in REG_VALUES:
                    _run_configuration(
                        _model_name,
                        _rb,
                        _rf,
                        "search",
                    )

    _current_config = _search_results[
        (_search_results["n_factors"] == int(n_factors.value))
        & (
            _search_results["learning_rate"].round(12)
            == round(float(learning_rate.value), 12)
        )
        & (_search_results["n_epochs"] == int(n_epochs.value))
        & (_search_results["batch_size"] == int(batch_size))
        & (_search_results["seed"] == int(model_seed))
    ]

    _reg_values_rounded = {round(float(value), 12) for value in REG_VALUES}
    _grid_candidates = _current_config[
        _current_config["reg_bias"].round(12).isin(_reg_values_rounded)
        & _current_config["reg_factors"].round(12).isin(
            _reg_values_rounded
        )
    ]

    best_models = {}
    for _model_name in ["M4", "C4"]:
        if run_search.value and not _grid_candidates[
            _grid_candidates["model"] == _model_name
        ].empty:
            _candidates = _grid_candidates[
                _grid_candidates["model"] == _model_name
            ]
        else:
            _candidates = initial_comparison[
                initial_comparison["model"] == _model_name
            ]
        _best_row = _candidates.sort_values(
            ["val_mae", "val_rmse"]
        ).iloc[0].to_dict()
        _best_key = (
            _model_name,
            round(float(_best_row["reg_bias"]), 12),
            round(float(_best_row["reg_factors"]), 12),
        )
        _best_engine = _session_engines.get(_best_key)
        if _best_engine is None:
            _best_row, _best_engine = _run_configuration(
                _model_name,
                float(_best_row["reg_bias"]),
                float(_best_row["reg_factors"]),
                "best_reload",
                force_retrain=True,
            )
        best_models[_model_name] = {
            "engine": _best_engine,
            "row": _best_row,
        }
        np.savez_compressed(
            CONTEXT_FM_DIR / f"best_params_{_model_name}.npz",
            w0=np.array([_best_engine.w0]),
            w=_best_engine.w,
            V=_best_engine.V,
            active_mask=_best_engine.active_mask,
            reg_bias=np.array([float(_best_row["reg_bias"])]),
            reg_factors=np.array([float(_best_row["reg_factors"])]),
            n_factors=np.array([int(_best_row["n_factors"])]),
            learning_rate=np.array(
                [float(_best_row["learning_rate"])]
            ),
            n_epochs=np.array([int(_best_row["n_epochs"])]),
            batch_size=np.array([int(_best_row["batch_size"])]),
            seed=np.array([int(_best_row["seed"])]),
        )

    search_results = _current_config.copy()
    return (
        best_models,
        initial_comparison,
        search_results,
    )


@app.cell
def _(initial_comparison, mo):
    mo.vstack([
        mo.md("### Comparaison initiale"),
        mo.ui.table(initial_comparison, pagination=False),
    ])
    return


@app.cell
def _(CONTEXT_FM_DIR, plt, search_results):
    _results = search_results.copy()
    _figure, _axes = plt.subplots(1, 2, figsize=(16, 6))

    for _axis, _model_name in zip(_axes, ["M4", "C4"]):
        _subset = _results[_results["model"] == _model_name]
        _pivot = _subset.pivot_table(
            index="reg_bias",
            columns="reg_factors",
            values="val_mae",
            aggfunc="mean",
        )
        _image = _axis.imshow(
            _pivot.to_numpy(),
            aspect="auto",
            cmap="viridis",
        )
        _axis.set_xticks(range(_pivot.shape[1]))
        _axis.set_xticklabels(
            [f"{value:g}" for value in _pivot.columns],
            rotation=45,
        )
        _axis.set_yticks(range(_pivot.shape[0]))
        _axis.set_yticklabels(
            [f"{value:g}" for value in _pivot.index]
        )
        _axis.set_xlabel("reg_factors")
        _axis.set_ylabel("reg_bias")
        _axis.set_title(f"MAE validation — {_model_name}")
        _figure.colorbar(
            _image,
            ax=_axis,
            label="MAE validation",
        )

    _figure.tight_layout()
    _figure.savefig(
        CONTEXT_FM_DIR / "regularization_search.png",
        dpi=150,
        bbox_inches="tight",
    )
    _figure


@app.cell
def _(
    CONTEXT_FM_DIR,
    best_models,
    compute_metrics,
    model_inputs,
    np,
    obs_test,
    pd,
    y_test,
    y_train,
    y_val,
):
    _rows = []
    _val_predictions = {}
    _test_predictions = {}
    evaluated_models = {}

    for _model_name in ["M4", "C4"]:
        _engine = best_models[_model_name]["engine"]
        _row = best_models[_model_name]["row"]
        _data = model_inputs[_model_name]

        _train_pred = _engine.predict(_data["X_train"])
        _val_pred = _engine.predict(_data["X_val"])
        _test_pred = _engine.predict(_data["X_test"])
        _train_metrics = compute_metrics(y_train, _train_pred)
        _val_metrics = compute_metrics(y_val, _val_pred)
        _test_metrics = compute_metrics(y_test, _test_pred)

        _rows.append({
            "model": _model_name,
            "reg_bias": float(_row["reg_bias"]),
            "reg_factors": float(_row["reg_factors"]),
            "n_factors": int(_row["n_factors"]),
            "learning_rate": float(_row["learning_rate"]),
            "n_epochs": int(_row["n_epochs"]),
            "train_mae": _train_metrics["mae"],
            "val_mae": _val_metrics["mae"],
            "test_mae": _test_metrics["mae"],
            "val_rmse": _val_metrics["rmse"],
            "test_rmse": _test_metrics["rmse"],
            "val_bias": _val_metrics["bias"],
            "test_bias": _test_metrics["bias"],
            "test_std_pred": float(np.std(_test_pred, ddof=0)),
            "test_std_true": float(np.std(y_test, ddof=0)),
            "n_parameters": int(_row["n_parameters"]),
            "training_time_seconds": float(
                _row["training_time_seconds"]
            ),
            "best_epoch": int(_row["best_epoch"]),
            "epochs_run": int(_row["epochs_run"]),
        })

        _val_predictions[_model_name] = _val_pred
        _test_predictions[_model_name] = _test_pred
        evaluated_models[_model_name] = {
            "engine": _engine,
            "row": _row,
            "val_prediction": _val_pred,
            "test_prediction": _test_pred,
        }

        pd.DataFrame({
            "model": _model_name,
            "row": obs_test["row"].to_numpy(),
            "col": obs_test["col"].to_numpy(),
            "rating_true": y_test,
            "rating_est": _test_pred,
        }).to_csv(
            CONTEXT_FM_DIR / f"test_predictions_{_model_name}.csv.gz",
            index=False,
            compression="gzip",
        )
        pd.DataFrame({
            "model": _model_name,
            "rating_true": y_val,
            "rating_est": _val_pred,
        }).to_csv(
            CONTEXT_FM_DIR / f"val_predictions_{_model_name}.csv.gz",
            index=False,
            compression="gzip",
        )

    best_models_metrics = pd.DataFrame(_rows)
    _m4_mae = float(
        best_models_metrics.loc[
            best_models_metrics["model"] == "M4", "test_mae"
        ].iloc[0]
    )
    _c4_mae = float(
        best_models_metrics.loc[
            best_models_metrics["model"] == "C4", "test_mae"
        ].iloc[0]
    )
    gain_mae = (
        (_m4_mae - _c4_mae) / _m4_mae * 100.0
        if _m4_mae > 0
        else np.nan
    )
    best_models_metrics.to_csv(
        CONTEXT_FM_DIR / "best_models_metrics.csv",
        index=False,
    )

    return (
        evaluated_models,
        best_models_metrics,
        gain_mae,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Diagnostics des erreurs
    """)
    return


@app.cell
def _(
    CONTEXT_FM_DIR,
    evaluated_models,
    compute_bin_curve,
    max_scatter_points,
    mo,
    np,
    obs_test,
    obs_train,
    pd,
    plt,
    scatter_with_curve,
    y_test,
):
    _user_counts = obs_train.groupby("row").size()
    _film_counts = obs_train.groupby("col").size()
    _user_mean = obs_train.groupby("row")["rating"].mean()
    _film_mean = obs_train.groupby("col")["rating"].mean()

    _diagnostic_frames = {}
    _bin_rows = []
    for _model_name in ["M4", "C4"]:
        _frame = pd.DataFrame({
            "row": obs_test["row"].to_numpy(dtype=np.int64),
            "col": obs_test["col"].to_numpy(dtype=np.int64),
            "rating_true": y_test,
            "rating_est": evaluated_models[_model_name]["test_prediction"],
        })
        _frame["error"] = (
            _frame["rating_est"] - _frame["rating_true"]
        )
        _frame["abs_error"] = _frame["error"].abs()
        _frame["user_n_train"] = (
            _frame["row"].map(_user_counts).fillna(0).astype(int)
        )
        _frame["film_n_train"] = (
            _frame["col"].map(_film_counts).fillna(0).astype(int)
        )
        _frame["user_mean_train"] = _frame["row"].map(_user_mean)
        _frame["film_mean_train"] = _frame["col"].map(_film_mean)
        _diagnostic_frames[_model_name] = _frame

    _variables = [
        ("rating_est", "error", "Erreur signée vs prédiction", "Prédiction"),
        ("user_n_train", "abs_error", "MAE vs notes utilisateur", "Notes utilisateur (train)"),
        ("film_n_train", "abs_error", "MAE vs notes film", "Notes film (train)"),
        ("film_mean_train", "error", "Erreur vs note moyenne film", "Note moyenne film (train)"),
        ("user_mean_train", "error", "Erreur vs note moyenne utilisateur", "Note moyenne utilisateur (train)"),
    ]

    _figure, _axes = plt.subplots(
        2,
        5,
        figsize=(26, 9),
        sharex="col",
        sharey="col",
    )

    for _row_index, _model_name in enumerate(["M4", "C4"]):
        _frame = _diagnostic_frames[_model_name]
        for _col_index, (
            _x_column,
            _y_column,
            _title,
            _x_label,
        ) in enumerate(_variables):
            _axis = _axes[_row_index, _col_index]
            _curve = scatter_with_curve(
                _axis,
                _frame,
                _x_column,
                _y_column,
                f"{_model_name} — {_title}",
                _x_label,
                "Erreur",
                max_points=int(max_scatter_points.value),
            )
            if not _curve.empty:
                for _record in _curve.to_dict("records"):
                    _bin_rows.append({
                        "model": _model_name,
                        "variable": _x_column,
                        "bin_left": _record["bin_left"],
                        "bin_right": _record["bin_right"],
                        "n": _record["n"],
                        "mean": _record["mean"],
                    })

    _figure.tight_layout()
    _figure.savefig(
        CONTEXT_FM_DIR / "error_diagnostics.png",
        dpi=150,
        bbox_inches="tight",
    )

    _distribution_figure, _distribution_axis = plt.subplots(
        figsize=(14, 6)
    )
    _distribution_axis.hist(
        y_test,
        bins=60,
        density=True,
        alpha=0.4,
        label="notes réelles",
    )
    for _model_name in ["M4", "C4"]:
        _distribution_axis.hist(
            evaluated_models[_model_name]["test_prediction"],
            bins=60,
            density=True,
            alpha=0.35,
            label=_model_name,
        )
    _distribution_axis.set_title(
        "Distribution des prédictions de test"
    )
    _distribution_axis.set_xlabel("Note")
    _distribution_axis.set_ylabel("Densité")
    _distribution_axis.grid(True, alpha=0.3)
    _distribution_axis.legend()
    _distribution_figure.tight_layout()
    _distribution_figure.savefig(
        CONTEXT_FM_DIR / "prediction_distribution.png",
        dpi=150,
        bbox_inches="tight",
    )

    diagnostic_bin_counts = pd.DataFrame(_bin_rows)
    diagnostic_bin_counts.to_csv(
        CONTEXT_FM_DIR / "diagnostic_bin_counts.csv",
        index=False,
    )

    mo.vstack([
        _figure,
        _distribution_figure,
        mo.ui.table(diagnostic_bin_counts, pagination=True),
    ])
    return (diagnostic_bin_counts,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Contributions des variables

    Les amplitudes de contributions sont descriptives. Elles ne mesurent pas
    une importance prédictive causale et ne remplacent pas une ablation.
    Les interactions "autres" regroupent les termes latents restants :
    film×genre, film×année, film×studio, genre×année, genre×studio,
    année×studio et les termes intra-famille.
    """)
    return


@app.cell
def _(
    CONTEXT_FM_DIR,
    evaluated_models,
    context_blocks_test,
    context_family_indices,
    film_idx_test,
    mo,
    n_movies,
    n_users,
    np,
    pd,
    plt,
    user_idx_test,
):
    _rows = []
    _reconstruction_rows = []

    _m4_engine = evaluated_models["M4"]["engine"]
    _m4_user = _m4_engine.w[user_idx_test]
    _m4_film = _m4_engine.w[n_users + film_idx_test]
    _m4_dot = np.einsum(
        "ij,ij->i",
        _m4_engine.V[user_idx_test],
        _m4_engine.V[n_users + film_idx_test],
    )
    _m4_prediction = evaluated_models["M4"]["test_prediction"]
    _m4_reconstruction = (
        _m4_engine.w0 + _m4_user + _m4_film + _m4_dot
    )
    _reconstruction_rows.append({
        "model": "M4",
        "max_abs_reconstruction_error": float(
            np.max(np.abs(_m4_reconstruction - _m4_prediction))
        ),
    })
    for _component, _values in [
        ("user_bias", _m4_user),
        ("film_bias", _m4_film),
        ("user_film_interaction", _m4_dot),
    ]:
        _rows.append({
            "model": "M4",
            "component": _component,
            "mean_absolute": float(np.abs(_values).mean()),
            "mean_signed": float(_values.mean()),
            "std": float(_values.std(ddof=0)),
            "p95_absolute": float(np.quantile(np.abs(_values), 0.95)),
        })

    _c4_engine = evaluated_models["C4"]["engine"]
    _c4_w = _c4_engine.w
    _c4_V = _c4_engine.V
    _families = ["user", "film", "genre", "year", "studio"]
    _linear = {}
    _S = {}
    _within = {}
    for _family in _families:
        _columns = context_family_indices[_family]
        _X = context_blocks_test[_family]
        _w_family = _c4_w[_columns]
        _V_family = _c4_V[_columns]
        _linear[_family] = _X @ _w_family
        _S[_family] = _X @ _V_family
        _within[_family] = 0.5 * (
            np.einsum("ij,ij->i", _S[_family], _S[_family])
            - _X.multiply(_X)
            @ np.einsum("ij,ij->i", _V_family, _V_family)
        )

    _cross = {}
    for _first_index, _first in enumerate(_families):
        for _second in _families[_first_index + 1:]:
            _cross[(_first, _second)] = np.einsum(
                "ij,ij->i",
                _S[_first],
                _S[_second],
            )

    _linear_total = sum(_linear.values())
    _latent_total = sum(_within.values()) + sum(_cross.values())
    _c4_prediction = evaluated_models["C4"]["test_prediction"]
    _c4_reconstruction = (
        _c4_engine.w0 + _linear_total + _latent_total
    )
    _reconstruction_rows.append({
        "model": "C4",
        "max_abs_reconstruction_error": float(
            np.max(np.abs(_c4_reconstruction - _c4_prediction))
        ),
    })

    for _family in _families:
        _rows.append({
            "model": "C4",
            "component": f"linear_{_family}",
            "mean_absolute": float(np.abs(_linear[_family]).mean()),
            "mean_signed": float(_linear[_family].mean()),
            "std": float(_linear[_family].std(ddof=0)),
            "p95_absolute": float(
                np.quantile(np.abs(_linear[_family]), 0.95)
            ),
        })

    _named_interactions = {
        "latent_user_film": _cross[("user", "film")],
        "latent_user_genre": _cross[("user", "genre")],
        "latent_user_studio": _cross[("user", "studio")],
        "latent_user_year": _cross[("user", "year")],
    }
    for _component, _values in _named_interactions.items():
        _rows.append({
            "model": "C4",
            "component": _component,
            "mean_absolute": float(np.abs(_values).mean()),
            "mean_signed": float(_values.mean()),
            "std": float(_values.std(ddof=0)),
            "p95_absolute": float(np.quantile(np.abs(_values), 0.95)),
        })

    _named_latent_total = sum(_named_interactions.values())
    _other_latent = _latent_total - _named_latent_total
    _rows.append({
        "model": "C4",
        "component": "latent_other_context",
        "mean_absolute": float(np.abs(_other_latent).mean()),
        "mean_signed": float(_other_latent.mean()),
        "std": float(_other_latent.std(ddof=0)),
        "p95_absolute": float(
            np.quantile(np.abs(_other_latent), 0.95)
        ),
    })

    contribution_summary = pd.DataFrame(_rows)
    reconstruction_check = pd.DataFrame(_reconstruction_rows)
    contribution_summary.to_csv(
        CONTEXT_FM_DIR / "contribution_summary.csv",
        index=False,
    )
    reconstruction_check.to_csv(
        CONTEXT_FM_DIR / "reconstruction_check.csv",
        index=False,
    )

    _figure, _axis = plt.subplots(figsize=(16, 8))
    _ordered = contribution_summary.sort_values("mean_absolute")
    _labels = [
        f"{row.model}:{row.component}"
        for row in _ordered.itertuples(index=False)
    ]
    _axis.barh(_labels, _ordered["mean_absolute"])
    _axis.set_xlabel("Contribution absolue moyenne")
    _axis.set_title("Amplitudes des contributions")
    _axis.grid(True, axis="x", alpha=0.3)
    _figure.tight_layout()
    _figure.savefig(
        CONTEXT_FM_DIR / "contribution_comparison.png",
        dpi=150,
        bbox_inches="tight",
    )

    mo.vstack([
        mo.ui.table(contribution_summary, pagination=True),
        mo.ui.table(reconstruction_check, pagination=False),
        _figure,
    ])
    return contribution_summary, reconstruction_check


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Diagnostic final
    """)
    return


@app.cell
def _(
    CONTEXT_FM_DIR,
    best_models_metrics,
    contribution_summary,
    gain_mae,
    mo,
    pd,
):
    _m4 = best_models_metrics[
        best_models_metrics["model"] == "M4"
    ].iloc[0]
    _c4 = best_models_metrics[
        best_models_metrics["model"] == "C4"
    ].iloc[0]

    _latent = contribution_summary[
        (contribution_summary["model"] == "C4")
        & contribution_summary["component"].str.startswith(
            "latent_"
        )
    ].sort_values("mean_absolute", ascending=False)

    _rows = [
        {
            "question": "C4 vs M4 : MAE test",
            "metric": "gain_mae_percent",
            "value": float(gain_mae),
        },
        {
            "question": "C4 vs M4 : RMSE test",
            "metric": "delta_test_rmse_C4_minus_M4",
            "value": float(_c4["test_rmse"] - _m4["test_rmse"]),
        },
        {
            "question": "Interaction latente dominante C4",
            "metric": _latent.iloc[0]["component"],
            "value": float(_latent.iloc[0]["mean_absolute"]),
        },
        {
            "question": "Compression M4",
            "metric": "test_std_pred_over_std_true",
            "value": float(
                _m4["test_std_pred"] / _m4["test_std_true"]
            ),
        },
        {
            "question": "Compression C4",
            "metric": "test_std_pred_over_std_true",
            "value": float(
                _c4["test_std_pred"] / _c4["test_std_true"]
            ),
        },
    ]
    final_diagnostic = pd.DataFrame(_rows)
    final_diagnostic.to_csv(
        CONTEXT_FM_DIR / "final_diagnostic.csv",
        index=False,
    )

    mo.vstack([
        mo.md(
            "### Synthèse quantitative\n\n"
            "Les valeurs ci-dessous sont descriptives. Une amélioration "
            "n'est pas attribuée causalement à une variable sans ablation."
        ),
        mo.ui.table(best_models_metrics, pagination=False),
        mo.ui.table(final_diagnostic, pagination=False),
    ])
    return (final_diagnostic,)


@app.cell
def _(
    CONTEXT_FM_DIR,
    evaluated_models,
    compute_metrics,
    model_inputs,
    mo,
    np,
    pd,
    run_reproducibility,
    train_fm_model,
    y_test,
    y_train,
    y_val,
):
    if not run_reproducibility.value:
        mo.md("Vérification de reproductibilité désactivée.")
    else:
        _rows = []
        for _model_name in ["M4", "C4"]:
            _row = evaluated_models[_model_name]["row"]
            _data = model_inputs[_model_name]
            (
                _engine,
                _history,
                _best_epoch,
                _epochs_run,
                _training_time,
            ) = train_fm_model(
                _data["X_train"],
                y_train,
                _data["X_val"],
                y_val,
                n_features=_data["n_features"],
                n_factors=int(_row["n_factors"]),
                use_linear=True,
                use_latent=True,
                active_mask=_data["active_mask"],
                learning_rate=float(_row["learning_rate"]),
                reg_bias=float(_row["reg_bias"]),
                reg_factors=float(_row["reg_factors"]),
                n_epochs=int(_row["n_epochs"]),
                batch_size=int(_row["batch_size"]),
                seed=int(_row["seed"]),
            )
            _first_val = evaluated_models[_model_name]["val_prediction"]
            _second_val = _engine.predict(_data["X_val"])
            _second_test = _engine.predict(_data["X_test"])
            _first_test = evaluated_models[_model_name]["test_prediction"]
            _rows.append({
                "model": _model_name,
                "max_abs_val_difference": float(
                    np.max(np.abs(_first_val - _second_val))
                ),
                "max_abs_test_difference": float(
                    np.max(np.abs(_first_test - _second_test))
                ),
                "first_val_mae": float(
                    compute_metrics(y_val, _first_val)["mae"]
                ),
                "second_val_mae": float(
                    compute_metrics(y_val, _second_val)["mae"]
                ),
                "first_test_mae": float(
                    compute_metrics(y_test, _first_test)["mae"]
                ),
                "second_test_mae": float(
                    compute_metrics(y_test, _second_test)["mae"]
                ),
                "identical": bool(
                    np.max(
                        np.abs(_first_test - _second_test)
                    ) <= 1e-12
                ),
            })
        reproducibility_check = pd.DataFrame(_rows)
        reproducibility_check.to_csv(
            CONTEXT_FM_DIR / "reproducibility_check.csv",
            index=False,
        )
        mo.ui.table(reproducibility_check, pagination=False)
        return (reproducibility_check,)


@app.cell
def _(
    CONTEXT_FM_DIR,
    REG_VALUES,
    base_seed,
    evaluated_models,
    best_models_metrics,
    hyperparameters,
    json,
    max_studios,
    model_seed,
):
    _config_payload = {
        "base_seed": base_seed,
        "model_seed": model_seed,
        "max_studios": max_studios,
        "hyperparameters": hyperparameters,
        "reg_values": [float(value) for value in REG_VALUES],
        "best_models": {
            name: {
                "reg_bias": float(value["row"]["reg_bias"]),
                "reg_factors": float(value["row"]["reg_factors"]),
                "n_factors": int(value["row"]["n_factors"]),
                "learning_rate": float(
                    value["row"]["learning_rate"]
                ),
                "n_epochs": int(value["row"]["n_epochs"]),
            }
            for name, value in evaluated_models.items()
        },
        "metrics": best_models_metrics.to_dict("records"),
    }
    with open(
        CONTEXT_FM_DIR / "experiment_config.json",
        "w",
    ) as _handle:
        json.dump(_config_payload, _handle, indent=2, default=str)


if __name__ == "__main__":
    app.run()
