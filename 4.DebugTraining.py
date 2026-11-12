import marimo

__generated_with = "0.17.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import os
    import time
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy.sparse import csr_matrix, hstack, load_npz

    return (
        Path,
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
    # FilmFinder — Debug de l'entraînement

    Ce notebook teste l'effet de la régularisation sur les biais et les
    interactions latentes, puis compare une baseline de biais utilisateur-film
    aux modèles sauvegardés. Aucun fichier de `2.Processing.py` n'est modifié.
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
    DEBUG_DIR = MODEL_OUTPUT_DIR / "debug_training"

    INPUT_FILES = {
        "matrix": CLEANING_DIR / "ratings_matrix.npz",
        "users": CLEANING_DIR / "user_ids.npy",
        "movies": CLEANING_DIR / "movie_ids.npy",
    }

    _required_inputs = list(INPUT_FILES.values())
    _missing_inputs = [
        str(path) for path in _required_inputs if not path.exists()
    ]
    if _missing_inputs:
        raise FileNotFoundError(
            "Missing input files: "
            f"{_missing_inputs}"
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
        _model_dirs = sorted(
            path
            for path in (CHECKPOINT_DIR / "models").iterdir()
            if path.is_dir()
        )
        if not _model_dirs:
            raise FileNotFoundError(
                "No model checkpoints found in "
                f"{CHECKPOINT_DIR / 'models'}."
            )
        _metadata_path = _model_dirs[0] / "metadata.json"
        with open(_metadata_path) as _handle:
            _metadata = json.load(_handle)
        _run_context = _metadata["run_context"]
        model_config = {
            "hyperparameters": _run_context["hyperparameters"],
            "split": _run_context.get("split", {}),
            "feature_config": _run_context.get("features", {}),
        }

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
    n_factors = int(hyperparameters["n_factors"])
    learning_rate = float(hyperparameters["learning_rate"])
    n_epochs = int(hyperparameters["n_epochs"])
    batch_size = int(hyperparameters["batch_size"])

    existing_model_names = [
        "M1_global",
        "M2_film_bias",
        "M4_full",
    ]
    existing_param_files = {
        name: CHECKPOINT_DIR / "models" / name / "params.npz"
        for name in existing_model_names
    }
    existing_metric_files = {
        name: CHECKPOINT_DIR / "models" / name / "metric.csv"
        for name in existing_model_names
    }
    existing_val_prediction_files = {
        name: CHECKPOINT_DIR / "models" / name / "val_prediction.npy"
        for name in existing_model_names
    }

    _missing_existing = [
        str(path)
        for path in (
            list(existing_param_files.values())
            + list(existing_metric_files.values())
            + list(existing_val_prediction_files.values())
        )
        if not path.exists()
    ]
    if _missing_existing:
        raise FileNotFoundError(
            "Missing checkpoints for M1/M2/M4: "
            f"{_missing_existing}"
        )

    return (
        CHECKPOINT_DIR,
        CLEANING_DIR,
        DEBUG_DIR,
        INPUT_FILES,
        MODEL_OUTPUT_DIR,
        base_seed,
        batch_size,
        existing_metric_files,
        existing_model_names,
        existing_param_files,
        existing_val_prediction_files,
        hyperparameters,
        learning_rate,
        n_epochs,
        n_factors,
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

    matrix_coo = ratings_matrix.tocoo()
    observations = pd.DataFrame({
        "row": matrix_coo.row.astype(np.int64),
        "col": matrix_coo.col.astype(np.int64),
        "user_id": user_mapping["user_id"].to_numpy()[
            matrix_coo.row
        ],
        "film_id": movie_mapping["canonical_movie_id"].to_numpy()[
            matrix_coo.col
        ],
        "rating": matrix_coo.data.astype(np.float64),
    })

    if observations.duplicated(subset=["row", "col"]).any():
        raise ValueError("Duplicated user-film coordinates in CSR.")

    if not np.isfinite(observations["rating"]).all():
        raise ValueError("Non-finite ratings in CSR.")

    return n_movies, n_users, observations


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

    split_summary = pd.DataFrame([
        {"split": "train", "interactions": len(obs_train)},
        {"split": "validation", "interactions": len(obs_val)},
        {"split": "test", "interactions": len(obs_test)},
    ])

    return (
        obs_test,
        obs_train,
        obs_val,
        split_summary,
        y_test,
        y_train,
        y_val,
    )


@app.cell
def _(
    csr_matrix,
    hstack,
    n_movies,
    n_users,
    np,
    obs_test,
    obs_train,
    obs_val,
    y_test,
    y_train,
    y_val,
):
    def _build_part1(obs):
        _n_rows = len(obs)
        _rows = obs["row"].to_numpy(dtype=np.int64)
        _cols = obs["col"].to_numpy(dtype=np.int64)
        _index = np.arange(_n_rows)

        _user_block = csr_matrix(
            (
                np.ones(_n_rows, dtype=np.float64),
                (_index, _rows),
            ),
            shape=(_n_rows, n_users),
        )
        _film_block = csr_matrix(
            (
                np.ones(_n_rows, dtype=np.float64),
                (_index, _cols),
            ),
            shape=(_n_rows, n_movies),
        )
        return hstack(
            [_user_block, _film_block],
            format="csr",
        )

    X_train = _build_part1(obs_train)
    X_val = _build_part1(obs_val)
    X_test = _build_part1(obs_test)

    n_features_part1 = n_users + n_movies
    active_mask = np.ones(n_features_part1, dtype=bool)

    user_idx_train = obs_train["row"].to_numpy(dtype=np.int64)
    film_idx_train = obs_train["col"].to_numpy(dtype=np.int64)
    user_idx_val = obs_val["row"].to_numpy(dtype=np.int64)
    film_idx_val = obs_val["col"].to_numpy(dtype=np.int64)
    user_idx_test = obs_test["row"].to_numpy(dtype=np.int64)
    film_idx_test = obs_test["col"].to_numpy(dtype=np.int64)

    return (
        X_test,
        X_train,
        X_val,
        active_mask,
        film_idx_test,
        film_idx_train,
        film_idx_val,
        n_features_part1,
        user_idx_test,
        user_idx_train,
        user_idx_val,
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
def _(mo):
    mo.md(r"""
    ## Audit de la loss et de la régularisation

    Dans `2.Processing.py`, la mise à jour utilisée est :

    \[
    \text{loss} \approx
    \frac12\mathbb E[(\hat r-r)^2]
    +\frac{\lambda_w}{2}\|w\|^2
    +\frac{\lambda_V}{2}\|V\|_F^2
    \]

    Le biais global `w0` n'est pas régularisé. Pour les autres paramètres, le
    gradient de régularisation est ajouté à **chaque mini-batch** :

    ```python
    grad_w += reg_bias * w
    grad_V += reg_factors * V
    ```

    Ce terme s'applique à tous les paramètres, même ceux absents du batch
    courant. Avec Adam, la pénalisation est couplée au gradient adaptatif.
    Une feature rare reçoit donc une pénalisation à chaque pas, alors que son
    gradient de données n'apparaît que dans les batches où elle est présente.

    **Correction proposée, non appliquée ici :** régulariser uniquement les
    features présentes dans le batch, normaliser par la fréquence, ou utiliser
    un weight decay découplé (AdamW).
    """)
    return


@app.cell
def _(
    DEBUG_DIR,
    batch_size,
    mo,
    n_epochs,
    n_movies,
    n_users,
    np,
    obs_train,
    pd,
):
    _row_train = obs_train["row"].to_numpy(dtype=np.int64)
    _col_train = obs_train["col"].to_numpy(dtype=np.int64)
    _user_frequency = np.bincount(
        _row_train,
        minlength=n_users,
    )
    _film_frequency = np.bincount(
        _col_train,
        minlength=n_movies,
    )
    _n_batches = int(np.ceil(len(obs_train) / batch_size))

    _rows = []
    for _family, _frequency in [
        ("user", _user_frequency),
        ("film", _film_frequency),
    ]:
        _rows.append({
            "family": _family,
            "n_features": len(_frequency),
            "min_interactions": int(_frequency.min()),
            "p25_interactions": float(
                np.quantile(_frequency, 0.25)
            ),
            "median_interactions": float(
                np.median(_frequency)
            ),
            "p75_interactions": float(
                np.quantile(_frequency, 0.75)
            ),
            "max_interactions": int(_frequency.max()),
            "batches_per_epoch": _n_batches,
            "steps_per_epoch": _n_batches,
            "steps_total": _n_batches * n_epochs,
        })

    feature_frequency = pd.DataFrame(_rows)
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    feature_frequency.to_csv(
        DEBUG_DIR / "feature_frequency.csv",
        index=False,
    )
    mo.ui.table(feature_frequency, pagination=False)
    return (feature_frequency,)


@app.cell
def _():
    experiment_specs = {
        "A": (0.0, 0.0),
        "B": (1e-7, 1e-7),
        "C": (1e-5, 1e-5),
        "D": (1e-3, 1e-3),
        "E": (0.02, 0.05),
    }
    return (experiment_specs,)


@app.cell
def _(mo):
    run_regularization = mo.ui.checkbox(
        value=True,
        label="Exécuter les expériences de régularisation",
    )
    selected_experiments = mo.ui.multiselect(
        options=["A", "B", "C", "D", "E"],
        value=["A", "B", "C", "D", "E"],
        label="Expériences à exécuter",
    )
    lambda_u = mo.ui.number(
        value=10.0,
        start=0.0,
        step=1.0,
        label="lambda_u baseline",
    )
    lambda_i = mo.ui.number(
        value=10.0,
        start=0.0,
        step=1.0,
        label="lambda_i baseline",
    )
    max_iterations = mo.ui.number(
        value=50,
        start=1,
        stop=500,
        step=1,
        label="Itérations max baseline",
    )
    tolerance = mo.ui.number(
        value=1e-8,
        start=1e-12,
        stop=1e-2,
        step=1e-9,
        label="Tolérance baseline",
    )
    run_reproducibility = mo.ui.checkbox(
        value=True,
        label="Vérifier la reproductibilité",
    )

    mo.vstack([
        mo.hstack(
            [run_regularization, selected_experiments],
            justify="start",
            gap=2,
        ),
        mo.hstack(
            [lambda_u, lambda_i],
            justify="start",
            gap=2,
        ),
        mo.hstack(
            [max_iterations, tolerance, run_reproducibility],
            justify="start",
            gap=2,
        ),
    ])
    return (
        lambda_i,
        lambda_u,
        max_iterations,
        run_regularization,
        run_reproducibility,
        selected_experiments,
        tolerance,
    )


@app.cell
def _(
    DEBUG_DIR,
    X_train,
    X_val,
    active_mask,
    base_seed,
    batch_size,
    compute_metrics,
    experiment_specs,
    film_idx_val,
    learning_rate,
    mo,
    n_epochs,
    n_factors,
    n_features_part1,
    n_movies,
    n_users,
    np,
    obs_val,
    pd,
    run_regularization,
    selected_experiments,
    train_fm_model,
    user_idx_val,
    y_train,
    y_val,
):
    if not run_regularization.value:
        mo.stop(
            True,
            mo.md(
                "Expériences de régularisation désactivées."
            ),
        )

    _selected = list(selected_experiments.value)
    if not _selected:
        mo.stop(
            True,
            mo.md("Sélectionne au moins une expérience."),
        )

    _model_seed = base_seed + 3000
    _rows = []
    _histories = []
    _prediction_frames = []
    _engines = {}

    for _label in _selected:
        _reg_bias, _reg_factors = experiment_specs[_label]
        (
            _engine,
            _history,
            _best_epoch,
            _epochs_run,
            _training_time,
        ) = train_fm_model(
            X_train,
            y_train,
            X_val,
            y_val,
            n_features=n_features_part1,
            n_factors=n_factors,
            use_linear=True,
            use_latent=True,
            active_mask=active_mask,
            learning_rate=learning_rate,
            reg_bias=_reg_bias,
            reg_factors=_reg_factors,
            n_epochs=n_epochs,
            batch_size=batch_size,
            seed=_model_seed,
        )

        _train_pred = _engine.predict(X_train)
        _val_pred = _engine.predict(X_val)
        _train_metrics = compute_metrics(y_train, _train_pred)
        _val_metrics = compute_metrics(y_val, _val_pred)

        _user_bias = _engine.w[user_idx_val]
        _film_bias = _engine.w[n_users + film_idx_val]
        _user_film_dot = np.einsum(
            "ij,ij->i",
            _engine.V[user_idx_val],
            _engine.V[n_users + film_idx_val],
        )
        _mean_v_norm = float(
            np.linalg.norm(
                _engine.V,
                axis=1,
            )[active_mask].mean()
        )

        _rows.append({
            "experiment": _label,
            "reg_bias": _reg_bias,
            "reg_factors": _reg_factors,
            "train_mae": _train_metrics["mae"],
            "val_mae": _val_metrics["mae"],
            "val_rmse": _val_metrics["rmse"],
            "val_bias": _val_metrics["bias"],
            "std_pred": float(np.std(_val_pred, ddof=0)),
            "mean_abs_user_bias": float(
                np.abs(_user_bias).mean()
            ),
            "mean_abs_film_bias": float(
                np.abs(_film_bias).mean()
            ),
            "mean_abs_user_film_dot": float(
                np.abs(_user_film_dot).mean()
            ),
            "mean_v_norm": _mean_v_norm,
            "best_epoch": _best_epoch,
            "epochs_run": _epochs_run,
            "training_time_seconds": _training_time,
        })
        _histories.append(_history.assign(experiment=_label))
        _prediction_frames.append(pd.DataFrame({
            "experiment": _label,
            "row": obs_val["row"].to_numpy(),
            "col": obs_val["col"].to_numpy(),
            "rating_true": y_val,
            "rating_est": _val_pred,
        }))
        _engines[_label] = _engine

    regularization_results = pd.DataFrame(_rows)
    regularization_histories = pd.concat(
        _histories,
        ignore_index=True,
    )
    regularization_val_predictions = pd.concat(
        _prediction_frames,
        ignore_index=True,
    )

    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    regularization_results.to_csv(
        DEBUG_DIR / "regularization_results.csv",
        index=False,
    )
    regularization_histories.to_csv(
        DEBUG_DIR / "regularization_histories.csv",
        index=False,
    )
    regularization_val_predictions.to_csv(
        DEBUG_DIR / "regularization_val_predictions.csv.gz",
        index=False,
        compression="gzip",
    )

    best_m4_label = regularization_results.sort_values(
        "val_mae"
    ).iloc[0]["experiment"]
    best_m4_result = regularization_results[
        regularization_results["experiment"] == best_m4_label
    ].iloc[0]
    best_m4_engine = _engines[best_m4_label]

    return (
        best_m4_engine,
        best_m4_label,
        best_m4_result,
        regularization_histories,
        regularization_results,
        regularization_val_predictions,
    )


@app.cell
def _(DEBUG_DIR, plt, regularization_results):
    _results = regularization_results.reset_index(drop=True)
    _x = range(len(_results))

    _figure, _axes = plt.subplots(1, 2, figsize=(16, 6))

    _axes[0].plot(
        _x,
        _results["train_mae"],
        marker="o",
        label="MAE train",
    )
    _axes[0].plot(
        _x,
        _results["val_mae"],
        marker="o",
        label="MAE validation",
    )
    _axes[0].set_xticks(list(_x))
    _axes[0].set_xticklabels(_results["experiment"])
    _axes[0].set_xlabel("Expérience de régularisation")
    _axes[0].set_ylabel("MAE")
    _axes[0].set_title("MAE selon la régularisation")
    _axes[0].grid(True, alpha=0.3)
    _axes[0].legend()

    _axes[1].plot(
        _x,
        _results["mean_abs_user_bias"],
        marker="o",
        label="E[|b_u|]",
    )
    _axes[1].plot(
        _x,
        _results["mean_abs_film_bias"],
        marker="o",
        label="E[|b_i|]",
    )
    _axes[1].plot(
        _x,
        _results["mean_abs_user_film_dot"],
        marker="o",
        label="E[|p_u^T q_i|]",
    )
    _axes[1].set_xticks(list(_x))
    _axes[1].set_xticklabels(_results["experiment"])
    _axes[1].set_xlabel("Expérience de régularisation")
    _axes[1].set_ylabel("Amplitude moyenne absolue")
    _axes[1].set_title("Amplitude des contributions")
    _axes[1].grid(True, alpha=0.3)
    _axes[1].legend()

    _figure.tight_layout()
    _figure.savefig(
        DEBUG_DIR / "regularization_diagnostics.png",
        dpi=150,
        bbox_inches="tight",
    )
    _figure


@app.cell
def _(
    DEBUG_DIR,
    compute_metrics,
    lambda_i,
    lambda_u,
    max_iterations,
    n_movies,
    n_users,
    np,
    obs_train,
    obs_val,
    pd,
    tolerance,
    y_train,
    y_val,
):
    _row_train = obs_train["row"].to_numpy(dtype=np.int64)
    _col_train = obs_train["col"].to_numpy(dtype=np.int64)
    _row_val = obs_val["row"].to_numpy(dtype=np.int64)
    _col_val = obs_val["col"].to_numpy(dtype=np.int64)

    _mu = float(y_train.mean())
    _b_u = np.zeros(n_users, dtype=np.float64)
    _b_i = np.zeros(n_movies, dtype=np.float64)
    _count_u = np.bincount(_row_train, minlength=n_users)
    _count_i = np.bincount(_col_train, minlength=n_movies)

    def _safe_divide(numerator, denominator):
        return np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator, dtype=np.float64),
            where=denominator != 0,
        )

    _history = []
    for _iteration in range(1, int(max_iterations.value) + 1):
        _residual_i = y_train - _mu - _b_u[_row_train]
        _sum_i = np.bincount(
            _col_train,
            weights=_residual_i,
            minlength=n_movies,
        )
        _new_b_i = _safe_divide(
            _sum_i,
            _count_i + float(lambda_i.value),
        )

        _residual_u = y_train - _mu - _new_b_i[_col_train]
        _sum_u = np.bincount(
            _row_train,
            weights=_residual_u,
            minlength=n_users,
        )
        _new_b_u = _safe_divide(
            _sum_u,
            _count_u + float(lambda_u.value),
        )

        _delta = max(
            float(np.max(np.abs(_new_b_i - _b_i), initial=0.0)),
            float(np.max(np.abs(_new_b_u - _b_u), initial=0.0)),
        )
        _b_i = _new_b_i
        _b_u = _new_b_u
        _history.append({
            "iteration": _iteration,
            "max_delta": _delta,
        })
        if _delta < float(tolerance.value):
            break

    _train_prediction = _mu + _b_u[_row_train] + _b_i[_col_train]
    _val_prediction = _mu + _b_u[_row_val] + _b_i[_col_val]
    _train_metrics = compute_metrics(y_train, _train_prediction)
    _val_metrics = compute_metrics(y_val, _val_prediction)

    baseline_history = pd.DataFrame(_history)
    baseline_results = pd.DataFrame([{
        "model": "baseline_bias_reg",
        "val_mae": _val_metrics["mae"],
        "val_rmse": _val_metrics["rmse"],
        "val_bias": _val_metrics["bias"],
        "std_pred": float(np.std(_val_prediction, ddof=0)),
        "mean_abs_user_bias": float(np.abs(_b_u).mean()),
        "mean_abs_film_bias": float(np.abs(_b_i).mean()),
        "mean_abs_user_film_dot": 0.0,
        "train_mae": _train_metrics["mae"],
        "iterations": len(_history),
        "mu": _mu,
        "lambda_u": float(lambda_u.value),
        "lambda_i": float(lambda_i.value),
    }])

    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    baseline_history.to_csv(
        DEBUG_DIR / "baseline_history.csv",
        index=False,
    )
    baseline_results.to_csv(
        DEBUG_DIR / "baseline_results.csv",
        index=False,
    )
    np.savez_compressed(
        DEBUG_DIR / "baseline_params.npz",
        mu=np.array([_mu]),
        b_u=_b_u,
        b_i=_b_i,
        lambda_u=np.array([lambda_u.value]),
        lambda_i=np.array([lambda_i.value]),
    )
    pd.DataFrame({
        "row": _row_val,
        "col": _col_val,
        "rating_true": y_val,
        "rating_est": _val_prediction,
    }).to_csv(
        DEBUG_DIR / "baseline_val_predictions.csv.gz",
        index=False,
        compression="gzip",
    )

    baseline_b_i = _b_i
    baseline_b_u = _b_u
    baseline_mu = _mu
    baseline_val_prediction = _val_prediction

    return (
        baseline_b_i,
        baseline_b_u,
        baseline_history,
        baseline_mu,
        baseline_results,
        baseline_val_prediction,
    )


@app.cell
def _(
    FMEngine,
    X_val,
    compute_metrics,
    existing_metric_files,
    existing_model_names,
    existing_param_files,
    existing_val_prediction_files,
    film_idx_val,
    n_movies,
    n_users,
    np,
    obs_val,
    pd,
    user_idx_val,
    y_val,
):
    _rows = []
    existing_val_predictions = {}
    existing_model_params = {}

    for _name in existing_model_names:
        _val_prediction = np.load(
            existing_val_prediction_files[_name]
        )
        if len(_val_prediction) != len(obs_val):
            raise ValueError(
                f"Validation prediction length mismatch for {_name}: "
                f"{len(_val_prediction)} vs {len(obs_val)}."
            )

        _metrics = compute_metrics(y_val, _val_prediction)
        _reported = pd.read_csv(
            existing_metric_files[_name]
        ).iloc[0]
        if (
            abs(_metrics["mae"] - float(_reported["val_mae"])) > 1e-6
            or abs(
                _metrics["rmse"] - float(_reported["val_rmse"])
            ) > 1e-6
        ):
            raise ValueError(
                f"Checkpoint {_name} is not aligned with the "
                "reconstructed validation split."
            )

        with np.load(existing_param_files[_name]) as _params:
            _w0 = float(_params["w0"][0])
            _w = _params["w"].copy()
            _V = _params["V"].copy()
            _active_mask = _params["active_mask"].astype(bool)

        _reconstructed_engine = FMEngine(
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
        _reconstructed_engine.w0 = _w0
        _reconstructed_engine.w = _w
        _reconstructed_engine.V = _V
        _reconstructed_prediction = _reconstructed_engine.predict(X_val)
        _reconstruction_difference = float(
            np.max(
                np.abs(
                    _reconstructed_prediction - _val_prediction
                )
            )
        )
        if _reconstruction_difference > 1e-6:
            raise ValueError(
                f"Feature alignment mismatch for {_name}: "
                f"max difference={_reconstruction_difference}."
            )

        if len(_w) == 0:
            _user_bias = np.zeros(len(obs_val), dtype=np.float64)
            _film_bias = np.zeros(len(obs_val), dtype=np.float64)
        else:
            _user_bias = _w[user_idx_val]
            _film_bias = _w[n_users + film_idx_val]

        if _V.shape[0] == 0:
            _user_film_dot = np.zeros(
                len(obs_val),
                dtype=np.float64,
            )
        else:
            _user_film_dot = np.einsum(
                "ij,ij->i",
                _V[user_idx_val],
                _V[n_users + film_idx_val],
            )

        _rows.append({
            "model": _name,
            "val_mae": _metrics["mae"],
            "val_rmse": _metrics["rmse"],
            "val_bias": _metrics["bias"],
            "std_pred": float(
                np.std(_val_prediction, ddof=0)
            ),
            "mean_abs_user_bias": float(
                np.abs(_user_bias).mean()
            ),
            "mean_abs_film_bias": float(
                np.abs(_film_bias).mean()
            ),
            "mean_abs_user_film_dot": float(
                np.abs(_user_film_dot).mean()
            ),
        })
        existing_val_predictions[_name] = _val_prediction
        existing_model_params[_name] = {
            "w0": _w0,
            "w": _w,
            "V": _V,
            "active_mask": _active_mask,
        }

    existing_model_results = pd.DataFrame(_rows)
    return (
        existing_model_params,
        existing_model_results,
        existing_val_predictions,
    )


@app.cell
def _(
    DEBUG_DIR,
    X_val,
    baseline_results,
    baseline_val_prediction,
    best_m4_engine,
    best_m4_label,
    compute_metrics,
    existing_model_results,
    existing_val_predictions,
    film_idx_val,
    n_movies,
    n_users,
    np,
    pd,
    user_idx_val,
    y_val,
):
    _best_prediction = best_m4_engine.predict(X_val)
    _best_metrics = compute_metrics(y_val, _best_prediction)
    _best_user_bias = best_m4_engine.w[user_idx_val]
    _best_film_bias = best_m4_engine.w[n_users + film_idx_val]
    _best_dot = np.einsum(
        "ij,ij->i",
        best_m4_engine.V[user_idx_val],
        best_m4_engine.V[n_users + film_idx_val],
    )

    _best_row = pd.DataFrame([{
        "model": f"best_M4_{best_m4_label}",
        "val_mae": _best_metrics["mae"],
        "val_rmse": _best_metrics["rmse"],
        "val_bias": _best_metrics["bias"],
        "std_pred": float(np.std(_best_prediction, ddof=0)),
        "mean_abs_user_bias": float(
            np.abs(_best_user_bias).mean()
        ),
        "mean_abs_film_bias": float(
            np.abs(_best_film_bias).mean()
        ),
        "mean_abs_user_film_dot": float(
            np.abs(_best_dot).mean()
        ),
    }])

    _baseline_row = baseline_results[[
        "model",
        "val_mae",
        "val_rmse",
        "val_bias",
        "std_pred",
        "mean_abs_user_bias",
        "mean_abs_film_bias",
        "mean_abs_user_film_dot",
    ]]

    model_comparison = pd.concat(
        [
            existing_model_results,
            _best_row,
            _baseline_row,
        ],
        ignore_index=True,
    )

    model_val_predictions = dict(existing_val_predictions)
    model_val_predictions[f"best_M4_{best_m4_label}"] = _best_prediction
    model_val_predictions["baseline_bias_reg"] = baseline_val_prediction

    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    model_comparison.to_csv(
        DEBUG_DIR / "model_comparison.csv",
        index=False,
    )

    return model_comparison, model_val_predictions


@app.cell
def _(
    DEBUG_DIR,
    baseline_b_i,
    baseline_b_u,
    existing_model_params,
    mo,
    model_comparison,
    model_val_predictions,
    n_movies,
    n_users,
    plt,
    y_val,
):
    _figure_predictions, _axis_predictions = plt.subplots(
        figsize=(14, 6)
    )
    for _name, _prediction in model_val_predictions.items():
        _axis_predictions.hist(
            _prediction,
            bins=60,
            alpha=0.35,
            density=True,
            label=_name,
        )
    _axis_predictions.set_title(
        "Distribution des prédictions de validation"
    )
    _axis_predictions.set_xlabel("Prédiction")
    _axis_predictions.set_ylabel("Densité")
    _axis_predictions.grid(True, alpha=0.3)
    _axis_predictions.legend(fontsize=8)
    _figure_predictions.tight_layout()
    _figure_predictions.savefig(
        DEBUG_DIR / "prediction_distributions.png",
        dpi=150,
        bbox_inches="tight",
    )

    _figure_bias, _axes_bias = plt.subplots(
        1,
        2,
        figsize=(16, 5),
    )
    _axes_bias[0].hist(
        baseline_b_u,
        bins=60,
        alpha=0.5,
        density=True,
        label="baseline b_u",
    )
    if len(existing_model_params["M4_full"]["w"]) > 0:
        _axes_bias[0].hist(
            existing_model_params["M4_full"]["w"][:n_users],
            bins=60,
            alpha=0.5,
            density=True,
            label="M4 user bias",
        )
    _axes_bias[0].set_title("Biais utilisateurs")
    _axes_bias[0].set_xlabel("Biais")
    _axes_bias[0].set_ylabel("Densité")
    _axes_bias[0].grid(True, alpha=0.3)
    _axes_bias[0].legend()

    _axes_bias[1].hist(
        baseline_b_i,
        bins=60,
        alpha=0.5,
        density=True,
        label="baseline b_i",
    )
    if len(existing_model_params["M2_film_bias"]["w"]) > 0:
        _axes_bias[1].hist(
            existing_model_params["M2_film_bias"]["w"][
                n_users:n_users + n_movies
            ],
            bins=60,
            alpha=0.5,
            density=True,
            label="M2 film bias",
        )
    if len(existing_model_params["M4_full"]["w"]) > 0:
        _axes_bias[1].hist(
            existing_model_params["M4_full"]["w"][
                n_users:n_users + n_movies
            ],
            bins=60,
            alpha=0.5,
            density=True,
            label="M4 film bias",
        )
    _axes_bias[1].set_title("Biais films")
    _axes_bias[1].set_xlabel("Biais")
    _axes_bias[1].set_ylabel("Densité")
    _axes_bias[1].grid(True, alpha=0.3)
    _axes_bias[1].legend()

    _figure_bias.tight_layout()
    _figure_bias.savefig(
        DEBUG_DIR / "bias_distributions.png",
        dpi=150,
        bbox_inches="tight",
    )

    mo.vstack([
        mo.ui.table(model_comparison, pagination=False),
        _figure_predictions,
        _figure_bias,
    ])
    return


@app.cell
def _(
    DEBUG_DIR,
    X_train,
    X_val,
    active_mask,
    base_seed,
    batch_size,
    best_m4_engine,
    best_m4_label,
    compute_metrics,
    experiment_specs,
    learning_rate,
    mo,
    n_epochs,
    n_factors,
    n_features_part1,
    np,
    pd,
    run_reproducibility,
    train_fm_model,
    y_train,
    y_val,
):
    if not run_reproducibility.value:
        mo.md("Vérification de reproductibilité non demandée.")
    else:
        _reg_bias, _reg_factors = experiment_specs[best_m4_label]
        _seed = base_seed + 3000
        (
            _repro_engine,
            _history,
            _best_epoch,
            _epochs_run,
            _training_time,
        ) = train_fm_model(
            X_train,
            y_train,
            X_val,
            y_val,
            n_features=n_features_part1,
            n_factors=n_factors,
            use_linear=True,
            use_latent=True,
            active_mask=active_mask,
            learning_rate=learning_rate,
            reg_bias=_reg_bias,
            reg_factors=_reg_factors,
            n_epochs=n_epochs,
            batch_size=batch_size,
            seed=_seed,
        )

        _first_prediction = best_m4_engine.predict(X_val)
        _second_prediction = _repro_engine.predict(X_val)
        _max_difference = float(
            np.max(
                np.abs(_first_prediction - _second_prediction)
            )
        )
        _first_metrics = compute_metrics(y_val, _first_prediction)
        _second_metrics = compute_metrics(
            y_val,
            _second_prediction,
        )

        reproducibility_check = pd.DataFrame([{
            "experiment": best_m4_label,
            "seed": _seed,
            "max_abs_prediction_difference": _max_difference,
            "first_val_mae": _first_metrics["mae"],
            "second_val_mae": _second_metrics["mae"],
            "identical": _max_difference <= 1e-12,
        }])
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        reproducibility_check.to_csv(
            DEBUG_DIR / "reproducibility_check.csv",
            index=False,
        )
        mo.ui.table(reproducibility_check, pagination=False)
        return (reproducibility_check,)


@app.cell
def _(
    DEBUG_DIR,
    base_seed,
    experiment_specs,
    hyperparameters,
    json,
    lambda_i,
    lambda_u,
    max_iterations,
    model_comparison,
    mo,
    pd,
    regularization_results,
    tolerance,
):
    _rows = []
    _a = regularization_results[
        regularization_results["experiment"] == "A"
    ]
    _e = regularization_results[
        regularization_results["experiment"] == "E"
    ]
    if not _a.empty and not _e.empty:
        _a_row = _a.iloc[0]
        _e_row = _e.iloc[0]
        _rows.extend([
            {
                "question": "A -> E : régularisation",
                "metric": "delta_val_mae_E_minus_A",
                "value": float(
                    _e_row["val_mae"] - _a_row["val_mae"]
                ),
            },
            {
                "question": "A -> E : régularisation",
                "metric": "delta_mean_abs_user_bias_A_minus_E",
                "value": float(
                    _a_row["mean_abs_user_bias"]
                    - _e_row["mean_abs_user_bias"]
                ),
            },
            {
                "question": "A -> E : régularisation",
                "metric": "delta_mean_abs_film_bias_A_minus_E",
                "value": float(
                    _a_row["mean_abs_film_bias"]
                    - _e_row["mean_abs_film_bias"]
                ),
            },
            {
                "question": "A -> E : régularisation",
                "metric": "delta_mean_abs_user_film_dot_A_minus_E",
                "value": float(
                    _a_row["mean_abs_user_film_dot"]
                    - _e_row["mean_abs_user_film_dot"]
                ),
            },
        ])

    _m2 = model_comparison[
        model_comparison["model"] == "M2_film_bias"
    ]
    _best = model_comparison[
        model_comparison["model"].str.startswith("best_M4_")
    ]
    _baseline = model_comparison[
        model_comparison["model"] == "baseline_bias_reg"
    ]

    if not _m2.empty and not _best.empty:
        _m2_row = _m2.iloc[0]
        _best_row = _best.iloc[0]
        _rows.append({
            "question": "Meilleur M4 vs M2",
            "metric": "delta_val_mae_bestM4_minus_M2",
            "value": float(
                _best_row["val_mae"] - _m2_row["val_mae"]
            ),
        })
        _rows.append({
            "question": "Meilleur M4 vs M2",
            "metric": "delta_val_rmse_bestM4_minus_M2",
            "value": float(
                _best_row["val_rmse"] - _m2_row["val_rmse"]
            ),
        })

    if not _baseline.empty and not _best.empty:
        _baseline_row = _baseline.iloc[0]
        _best_row = _best.iloc[0]
        _rows.append({
            "question": "Baseline biais vs meilleur M4",
            "metric": "delta_val_mae_baseline_minus_bestM4",
            "value": float(
                _baseline_row["val_mae"]
                - _best_row["val_mae"]
            ),
        })
        _rows.append({
            "question": "Baseline biais vs meilleur M4",
            "metric": "delta_val_rmse_baseline_minus_bestM4",
            "value": float(
                _baseline_row["val_rmse"]
                - _best_row["val_rmse"]
            ),
        })

    final_diagnostic = pd.DataFrame(_rows)
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    final_diagnostic.to_csv(
        DEBUG_DIR / "final_diagnostic.csv",
        index=False,
    )

    _config_payload = {
        "base_seed": base_seed,
        "hyperparameters": hyperparameters,
        "experiment_specs": {
            key: {
                "reg_bias": float(value[0]),
                "reg_factors": float(value[1]),
            }
            for key, value in experiment_specs.items()
        },
        "baseline": {
            "lambda_u": float(lambda_u.value),
            "lambda_i": float(lambda_i.value),
            "max_iterations": int(max_iterations.value),
            "tolerance": float(tolerance.value),
        },
    }
    with open(
        DEBUG_DIR / "experiment_config.json",
        "w",
    ) as _handle:
        json.dump(_config_payload, _handle, indent=2)

    mo.vstack([
        mo.md(
            "### Synthèse quantitative\n\n"
            "Les valeurs ci-dessous comparent les amplitudes et les "
            "métriques sans modifier l'entraînement."
        ),
        mo.ui.table(final_diagnostic, pagination=False),
        mo.md(
            "### Tableau comparatif final\n\n"
            "Résultats de validation pour M1, M2, M4, le meilleur M4 "
            "régularisé et la baseline de biais."
        ),
        mo.ui.table(model_comparison, pagination=False),
    ])
    return


if __name__ == "__main__":
    app.run()
