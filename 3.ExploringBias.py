import marimo

__generated_with = "0.17.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import ast
    import json
    import os
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy.sparse import csr_matrix, load_npz

    return (
        Path,
        ast,
        csr_matrix,
        json,
        load_npz,
        mo,
        np,
        os,
        pd,
        plt,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # FilmFinder — Exploration des biais

    Ce notebook analyse les modèles sauvegardés par `2.Processing.py` sans
    réentraîner. Il reconstruit le split de test et les features à partir des
    fichiers existants, puis mesure les paramètres appris et les contributions
    effectives aux prédictions.
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
    EXPLORING_BIAS_DIR = MODEL_OUTPUT_DIR / "exploring_bias"

    INPUT_FILES = {
        "matrix": CLEANING_DIR / "ratings_matrix.npz",
        "users": CLEANING_DIR / "user_ids.npy",
        "movies": CLEANING_DIR / "movie_ids.npy",
        "context": CLEANING_DIR / "movie_context.json",
    }

    _required_files = [
        MODEL_OUTPUT_DIR / "model_config.json",
        MODEL_OUTPUT_DIR / "metrics.csv",
        MODEL_OUTPUT_DIR / "test_predictions.csv.gz",
        MODEL_OUTPUT_DIR / "feature_mapping.csv",
        *INPUT_FILES.values(),
    ]
    _missing = [
        str(path) for path in _required_files if not path.exists()
    ]
    if _missing:
        raise FileNotFoundError(
            "Missing files required for bias analysis: "
            f"{_missing}"
        )

    with open(MODEL_OUTPUT_DIR / "model_config.json") as _handle:
        model_config = json.load(_handle)

    metrics = pd.read_csv(MODEL_OUTPUT_DIR / "metrics.csv")
    test_predictions = pd.read_csv(
        MODEL_OUTPUT_DIR / "test_predictions.csv.gz"
    )
    feature_mapping_saved = pd.read_csv(
        MODEL_OUTPUT_DIR / "feature_mapping.csv"
    )
    model_names = metrics["model"].tolist()

    best_param_files = {
        name: MODEL_OUTPUT_DIR / f"best_params_{name}.npz"
        for name in model_names
    }
    _missing_params = [
        str(path)
        for path in best_param_files.values()
        if not path.exists()
    ]
    if _missing_params:
        raise FileNotFoundError(
            "Missing model parameter files: "
            f"{_missing_params}"
        )

    return (
        CLEANING_DIR,
        EXPLORING_BIAS_DIR,
        INPUT_FILES,
        MODEL_OUTPUT_DIR,
        best_param_files,
        feature_mapping_saved,
        metrics,
        model_config,
        model_names,
        test_predictions,
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
def _(model_config, model_names, np, observations, pd, test_predictions):
    _hyperparameters = model_config["hyperparameters"]
    _seed = int(_hyperparameters["seed"])
    _train_fraction = float(_hyperparameters["train_fraction"])
    _val_fraction = float(_hyperparameters["val_fraction"])
    _test_fraction = 1.0 - _train_fraction - _val_fraction

    if not (0 < _train_fraction < 1 and 0 < _val_fraction < 1):
        raise ValueError("Invalid train/validation fractions.")
    if _test_fraction <= 0:
        raise ValueError(
            "Train and validation fractions must sum to less than 1."
        )

    _rng = np.random.default_rng(_seed)
    _n_obs = len(observations)
    _permutation = _rng.permutation(_n_obs)

    _n_train = int(np.floor(_train_fraction * _n_obs))
    _n_val = int(np.floor(_val_fraction * _n_obs))

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
    obs_test = observations.iloc[_test_idx].reset_index(drop=True)

    split_summary = pd.DataFrame([
        {"split": "train", "interactions": len(_train_idx)},
        {"split": "validation warm-start", "interactions": len(_val_idx)},
        {"split": "validation excluded", "interactions": int(
            (~_val_warm_mask).sum()
        )},
        {"split": "test warm-start", "interactions": len(_test_idx)},
        {"split": "test excluded", "interactions": int(
            (~_test_warm_mask).sum()
        )},
    ])

    for _name in model_names:
        _prediction_frame = test_predictions[
            test_predictions["model"] == _name
        ].reset_index(drop=True)
        if (
            len(_prediction_frame) != len(obs_test)
            or not np.array_equal(
                _prediction_frame["row"].to_numpy(),
                obs_test["row"].to_numpy(),
            )
            or not np.array_equal(
                _prediction_frame["col"].to_numpy(),
                obs_test["col"].to_numpy(),
            )
        ):
            raise ValueError(
                "Saved test predictions are not aligned with the "
                f"reconstructed split for model {_name}."
            )

    return obs_test, obs_train, split_summary


@app.cell
def _(
    ast,
    context_sorted,
    csr_matrix,
    feature_mapping_saved,
    movie_mapping_sorted,
    n_movies,
    n_users,
    np,
    obs_test,
    obs_train,
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

    _film_years = pd.to_numeric(
        _context_by_col.get(
            "year",
            pd.Series([np.nan] * n_movies, index=_context_by_col.index),
        ),
        errors="coerce",
    ).to_numpy(dtype=np.float64)

    _train_films = np.sort(obs_train["col"].unique())
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

    context_mapping = (
        feature_mapping_saved
        .sort_values("feature_index")
        .reset_index(drop=True)
    )
    if not np.array_equal(
        context_mapping["feature_index"].to_numpy(),
        np.arange(len(context_mapping)),
    ):
        raise ValueError("feature_mapping.csv is not contiguous.")

    _groups = context_mapping["family"].astype(str).to_numpy()
    _families = ["user", "film", "genre", "year", "studio"]
    _family_columns = {
        family: np.where(_groups == family)[0]
        for family in _families
    }

    if (
        len(_family_columns["user"]) != n_users
        or len(_family_columns["film"]) != n_movies
    ):
        raise ValueError(
            "feature_mapping.csv is not aligned with user/film counts."
        )

    _feature_names = context_mapping["feature_name"].astype(str).to_numpy()

    def _local_special(prefix, label, columns):
        for _local, _name in enumerate(_feature_names[columns]):
            if _name == f"{prefix}:{label}":
                return _local
        return None

    _genre_columns = _family_columns["genre"]
    _genre_unknown = _local_special(
        "genre", "<UNKNOWN>", _genre_columns
    )
    _genre_missing = _local_special(
        "genre", "<MISSING>", _genre_columns
    )
    _genre_index = {}
    for _local, _name in enumerate(_feature_names[_genre_columns]):
        if _name.startswith("genre:") and _name not in (
            "genre:<UNKNOWN>",
            "genre:<MISSING>",
        ):
            _genre_index[_name[len("genre:"):]] = _local

    _studio_columns = _family_columns["studio"]
    _studio_unknown = _local_special(
        "studio", "<UNKNOWN>", _studio_columns
    )
    _studio_missing = _local_special(
        "studio", "<MISSING>", _studio_columns
    )
    _studio_index = {}
    for _local, _name in enumerate(_feature_names[_studio_columns]):
        if _name.startswith("studio:") and _name not in (
            "studio:<UNKNOWN>",
            "studio:<MISSING>",
        ):
            _studio_index[_name[len("studio:"):]] = _local

    _year_columns = _family_columns["year"]
    _year_standardized = _local_special(
        "year", "standardized", _year_columns
    )
    _year_missing = _local_special(
        "year", "missing", _year_columns
    )

    _n_test = len(obs_test)
    _index = np.arange(_n_test)
    _rows = obs_test["row"].to_numpy(dtype=np.int64)
    _cols = obs_test["col"].to_numpy(dtype=np.int64)

    _user_block = csr_matrix(
        (np.ones(_n_test), (_index, _rows)),
        shape=(_n_test, n_users),
    )
    _film_block = csr_matrix(
        (np.ones(_n_test), (_index, _cols)),
        shape=(_n_test, n_movies),
    )

    _genre_rows = []
    _genre_block_cols = []
    _genre_data = []
    for _row_index, _film in enumerate(_cols):
        _values = _genres_by_film[_film]
        if _values is None:
            if _genre_missing is not None:
                _genre_rows.append(_row_index)
                _genre_block_cols.append(_genre_missing)
                _genre_data.append(1.0)
        elif len(_values) == 0:
            pass
        else:
            for _value in set(_values):
                _local = _genre_index.get(_value, _genre_unknown)
                if _local is not None:
                    _genre_rows.append(_row_index)
                    _genre_block_cols.append(_local)
                    _genre_data.append(1.0)

    _genre_block = csr_matrix(
        (_genre_data, (_genre_rows, _genre_block_cols)),
        shape=(_n_test, len(_genre_columns)),
    )

    _studio_rows = []
    _studio_block_cols = []
    _studio_data = []
    for _row_index, _film in enumerate(_cols):
        _values = _studios_by_film[_film]
        if _values is None or len(_values) == 0:
            if _studio_missing is not None:
                _studio_rows.append(_row_index)
                _studio_block_cols.append(_studio_missing)
                _studio_data.append(1.0)
        else:
            _retained = [
                _studio_index[_value]
                for _value in set(_values)
                if _value in _studio_index
            ]
            if _retained:
                for _local in _retained:
                    _studio_rows.append(_row_index)
                    _studio_block_cols.append(_local)
                    _studio_data.append(1.0)
            elif _studio_missing is not None:
                _studio_rows.append(_row_index)
                _studio_block_cols.append(_studio_missing)
                _studio_data.append(1.0)

    _studio_block = csr_matrix(
        (_studio_data, (_studio_rows, _studio_block_cols)),
        shape=(_n_test, len(_studio_columns)),
    )

    _year_values = _film_years[_cols]
    _year_is_missing = ~np.isfinite(_year_values)
    _year_z = np.where(
        _year_is_missing,
        0.0,
        (_year_values - year_mean) / year_std,
    )

    if (
        _year_standardized is None
        or _year_missing is None
    ):
        _year_block = csr_matrix(
            (_n_test, len(_year_columns))
        )
    else:
        _present_rows = _index[~_year_is_missing]
        _missing_rows = _index[_year_is_missing]
        _year_rows = np.concatenate([_present_rows, _missing_rows])
        _year_block_cols = np.concatenate([
            np.full(len(_present_rows), _year_standardized),
            np.full(len(_missing_rows), _year_missing),
        ])
        _year_data = np.concatenate([
            _year_z[~_year_is_missing],
            np.ones(len(_missing_rows)),
        ])
        _year_block = csr_matrix(
            (_year_data, (_year_rows, _year_block_cols)),
            shape=(_n_test, len(_year_columns)),
        )

    test_blocks = {
        "user": _user_block,
        "film": _film_block,
        "genre": _genre_block,
        "year": _year_block,
        "studio": _studio_block,
    }

    part1_mapping = pd.DataFrame({
        "feature_index": np.arange(n_users + n_movies),
        "feature_name": (
            [f"user:{value}" for value in user_mapping_sorted["user_id"]]
            + [
                f"film:{value}"
                for value in movie_mapping_sorted["canonical_movie_id"]
            ]
        ),
        "family": ["user"] * n_users + ["film"] * n_movies,
    })

    return context_mapping, part1_mapping, test_blocks


@app.cell
def _(
    best_param_files,
    context_mapping,
    metrics,
    model_names,
    np,
    part1_mapping,
    pd,
):
    _metrics_by_model = metrics.set_index("model")
    model_params = {}
    model_specs = {}
    model_feature_mapping = {}

    for _name in model_names:
        with np.load(best_param_files[_name]) as _handle:
            _w0 = float(_handle["w0"][0])
            _w = _handle["w"].copy()
            _V = _handle["V"].copy()
            _active_mask = _handle["active_mask"].astype(bool)

        _metric_row = _metrics_by_model.loc[_name]
        _feature_set = str(_metric_row["feature_set"])
        _n_features = int(_metric_row["n_features"])

        if _feature_set == "none":
            _mapping = pd.DataFrame(
                columns=["feature_index", "feature_name", "family"]
            )
        elif _feature_set == "part1":
            _mapping = part1_mapping
        elif _feature_set == "context":
            _mapping = context_mapping
        else:
            raise ValueError(
                f"Unknown feature_set for model {_name}: {_feature_set}"
            )

        if (
            len(_w) != _n_features
            or len(_active_mask) != _n_features
            or _V.shape[0] != _n_features
            or len(_mapping) != _n_features
        ):
            raise ValueError(
                f"Inconsistent parameter shapes for model {_name}: "
                f"n_features={_n_features}, len(w)={len(_w)}, "
                f"len(active_mask)={len(_active_mask)}, "
                f"len(mapping)={len(_mapping)}, V={_V.shape}."
            )

        model_params[_name] = {
            "w0": _w0,
            "w": _w,
            "V": _V,
            "active_mask": _active_mask,
        }
        model_specs[_name] = {
            "family": str(_metric_row["family"]),
            "feature_set": _feature_set,
            "n_features": _n_features,
            "n_factors": int(_metric_row["n_factors"]),
            "use_linear": bool(_metric_row["use_linear"]),
            "use_latent": bool(_metric_row["use_latent"]),
        }
        model_feature_mapping[_name] = _mapping.reset_index(drop=True)

    return model_feature_mapping, model_params, model_specs


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Distribution des paramètres appris
    """)
    return


@app.cell
def _(
    mo,
    model_feature_mapping,
    model_names,
    model_params,
    model_specs,
    np,
    pd,
):
    _rows = []
    _families = ["user", "film", "genre", "year", "studio"]

    for _name in model_names:
        _spec = model_specs[_name]
        if _spec["feature_set"] == "none":
            continue

        _mapping = model_feature_mapping[_name]
        _frame = _mapping.copy()
        _frame["abs_w"] = np.abs(model_params[_name]["w"])
        _frame["v_norm"] = np.linalg.norm(
            model_params[_name]["V"],
            axis=1,
        )
        _frame["active"] = model_params[_name]["active_mask"]
        _frame = _frame[_frame["active"]]

        for _family in _families:
            _subset = _frame[_frame["family"] == _family]
            if _subset.empty:
                continue
            _abs_w = _subset["abs_w"].to_numpy(dtype=np.float64)
            _v_norm = _subset["v_norm"].to_numpy(dtype=np.float64)
            _rows.append({
                "model": _name,
                "family": _family,
                "n_features": len(_subset),
                "mean_abs_w": float(_abs_w.mean()),
                "median_abs_w": float(np.median(_abs_w)),
                "p95_abs_w": float(np.quantile(_abs_w, 0.95)),
                "mean_v_norm": float(_v_norm.mean()),
                "median_v_norm": float(np.median(_v_norm)),
                "p95_v_norm": float(np.quantile(_v_norm, 0.95)),
            })

    param_summary = pd.DataFrame(_rows)
    mo.ui.table(param_summary, pagination=True)
    return (param_summary,)


@app.cell
def _(mo, model_names):
    _default = "C4_full" if "C4_full" in model_names else model_names[0]
    selected_model = mo.ui.dropdown(
        options=model_names,
        value=_default,
        label="Modèle sélectionné",
    )
    selected_model
    return (selected_model,)


@app.cell
def _(
    mo,
    model_feature_mapping,
    model_params,
    model_specs,
    np,
    plt,
    selected_model,
):
    _name = selected_model.value
    _spec = model_specs[_name]

    if _spec["feature_set"] == "none":
        mo.md("Ce modèle global n'a aucun paramètre de feature.")
    else:
        _mapping = model_feature_mapping[_name]
        _frame = _mapping.copy()
        _frame["abs_w"] = np.abs(model_params[_name]["w"])
        _frame["v_norm"] = np.linalg.norm(
            model_params[_name]["V"],
            axis=1,
        )
        _frame["active"] = model_params[_name]["active_mask"]
        _frame = _frame[_frame["active"]]

        _families = [
            _family
            for _family in ["user", "film", "genre", "year", "studio"]
            if (_frame["family"] == _family).any()
        ]

        _figure, _axes = plt.subplots(
            2,
            len(_families),
            figsize=(max(6, 4 * len(_families)), 8),
            squeeze=False,
        )

        for _column_index, _family in enumerate(_families):
            _abs_w = _frame.loc[
                _frame["family"] == _family, "abs_w"
            ].to_numpy(dtype=np.float64)
            _v_norm = _frame.loc[
                _frame["family"] == _family, "v_norm"
            ].to_numpy(dtype=np.float64)

            for _row_index, (_values, _label) in enumerate([
                (_abs_w, "|w|"),
                (_v_norm, "||v||"),
            ]):
                _axis = _axes[_row_index, _column_index]
                _positive = _values[_values > 0]
                if len(_positive):
                    _axis.hist(
                        np.log10(_positive),
                        bins=40,
                        alpha=0.75,
                    )
                    _axis.set_xlabel(f"log10({_label})")
                else:
                    _axis.text(
                        0.5,
                        0.5,
                        "aucune valeur > 0",
                        ha="center",
                        va="center",
                    )
                _axis.set_title(f"{_family} — {_label}")
                _axis.grid(True, alpha=0.3)

        _figure.tight_layout()
        _figure


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Relation entre paramètres et erreurs
    """)
    return


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
            alpha=0.15,
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

    def add_parameter_metrics(
        name,
        frame,
        model_params,
        model_specs,
        n_users,
    ):
        _spec = model_specs[name]
        _frame = frame.copy()
        _rows = _frame["row"].to_numpy(dtype=np.int64)
        _cols = _frame["col"].to_numpy(dtype=np.int64)

        if _spec["feature_set"] == "none":
            _frame["user_bias_abs"] = 0.0
            _frame["film_bias_abs"] = 0.0
            _frame["user_v_norm"] = 0.0
            _frame["film_v_norm"] = 0.0
            return _frame

        _w = model_params[name]["w"]
        _V = model_params[name]["V"]
        _frame["user_bias_abs"] = np.abs(_w[_rows])
        _frame["film_bias_abs"] = np.abs(_w[n_users + _cols])
        _frame["user_v_norm"] = np.linalg.norm(
            _V[_rows],
            axis=1,
        )
        _frame["film_v_norm"] = np.linalg.norm(
            _V[n_users + _cols],
            axis=1,
        )
        return _frame

    def compute_model_contributions(
        name,
        model_params,
        model_specs,
        test_blocks,
        model_feature_mapping,
        n_users,
        n_movies,
    ):
        _spec = model_specs[name]
        _w0 = float(model_params[name]["w0"])
        _w = model_params[name]["w"]
        _V = model_params[name]["V"]
        _n_test = test_blocks["user"].shape[0]
        _zeros = np.zeros(_n_test, dtype=np.float64)

        if _spec["feature_set"] == "none":
            return {
                "global": np.full(_n_test, _w0),
                "linear_user": _zeros,
                "linear_film": _zeros,
                "user_film_dot": _zeros,
                "linear_total": _zeros,
                "latent_total": _zeros,
                "linear_by_family": {},
                "latent_by_family": {},
                "reconstructed": np.full(_n_test, _w0),
            }

        if _spec["feature_set"] == "part1":
            _X_user = test_blocks["user"]
            _X_film = test_blocks["film"]
            _linear_user = _X_user @ _w[:n_users]
            _linear_film = _X_film @ _w[
                n_users:n_users + n_movies
            ]
            _S_user = _X_user @ _V[:n_users]
            _S_film = _X_film @ _V[
                n_users:n_users + n_movies
            ]
            _user_film_dot = np.einsum(
                "ij,ij->i",
                _S_user,
                _S_film,
            )
            _linear_total = _linear_user + _linear_film
            return {
                "global": np.full(_n_test, _w0),
                "linear_user": _linear_user,
                "linear_film": _linear_film,
                "user_film_dot": _user_film_dot,
                "linear_total": _linear_total,
                "latent_total": _user_film_dot,
                "linear_by_family": {
                    "user": _linear_user,
                    "film": _linear_film,
                },
                "latent_by_family": {},
                "reconstructed": (
                    _w0 + _linear_total + _user_film_dot
                ),
            }

        if _spec["feature_set"] == "context":
            _mapping = model_feature_mapping[name]
            _groups = _mapping["family"].to_numpy()
            _families = ["user", "film", "genre", "year", "studio"]
            _linear = {}
            _S = {}
            _latent_within = {}

            for _family in _families:
                _columns = np.where(_groups == _family)[0]
                _X = test_blocks[_family]
                _w_family = _w[_columns]
                _V_family = _V[_columns]
                _linear[_family] = _X @ _w_family
                _S[_family] = _X @ _V_family
                _latent_within[_family] = 0.5 * (
                    np.einsum(
                        "ij,ij->i",
                        _S[_family],
                        _S[_family],
                    )
                    - _X.multiply(_X)
                    @ np.einsum(
                        "ij,ij->i",
                        _V_family,
                        _V_family,
                    )
                )

            _cross = {}
            _latent_by_family = {
                _family: _latent_within[_family].copy()
                for _family in _families
            }
            for _first_index, _first in enumerate(_families):
                for _second in _families[_first_index + 1:]:
                    _value = np.einsum(
                        "ij,ij->i",
                        _S[_first],
                        _S[_second],
                    )
                    _cross[(_first, _second)] = _value
                    _latent_by_family[_first] = (
                        _latent_by_family[_first] + 0.5 * _value
                    )
                    _latent_by_family[_second] = (
                        _latent_by_family[_second] + 0.5 * _value
                    )

            _linear_total = sum(_linear.values())
            _latent_total = (
                sum(_latent_within.values())
                + sum(_cross.values())
            )
            return {
                "global": np.full(_n_test, _w0),
                "linear_user": _linear["user"],
                "linear_film": _linear["film"],
                "user_film_dot": _cross[("user", "film")],
                "linear_total": _linear_total,
                "latent_total": _latent_total,
                "linear_by_family": _linear,
                "latent_by_family": _latent_by_family,
                "reconstructed": (
                    _w0 + _linear_total + _latent_total
                ),
            }

        raise ValueError(
            f"Unknown feature_set for model {name}: "
            f"{_spec['feature_set']}"
        )

    return (
        add_parameter_metrics,
        compute_bin_curve,
        compute_model_contributions,
        scatter_with_curve,
        subsample_frame,
    )


@app.cell
def _(
    add_parameter_metrics,
    compute_model_contributions,
    model_feature_mapping,
    model_params,
    model_specs,
    n_movies,
    n_users,
    np,
    selected_model,
    test_blocks,
    test_predictions,
):
    _name = selected_model.value
    _prediction_frame = test_predictions[
        test_predictions["model"] == _name
    ].reset_index(drop=True)

    selected_contributions = compute_model_contributions(
        _name,
        model_params,
        model_specs,
        test_blocks,
        model_feature_mapping,
        n_users,
        n_movies,
    )
    selected_frame = add_parameter_metrics(
        _name,
        _prediction_frame,
        model_params,
        model_specs,
        n_users,
    )

    selected_frame["linear_user_abs"] = np.abs(
        selected_contributions["linear_user"]
    )
    selected_frame["linear_film_abs"] = np.abs(
        selected_contributions["linear_film"]
    )
    selected_frame["user_film_dot_abs"] = np.abs(
        selected_contributions["user_film_dot"]
    )
    selected_frame["linear_total_abs"] = np.abs(
        selected_contributions["linear_total"]
    )
    selected_frame["latent_total_abs"] = np.abs(
        selected_contributions["latent_total"]
    )
    selected_frame["reconstruction_error"] = np.abs(
        selected_contributions["reconstructed"]
        - _prediction_frame["rating_est"].to_numpy()
    )

    for _family, _values in selected_contributions[
        "linear_by_family"
    ].items():
        selected_frame[f"linear_{_family}_abs"] = np.abs(_values)
    for _family, _values in selected_contributions[
        "latent_by_family"
    ].items():
        selected_frame[f"latent_{_family}_abs"] = np.abs(_values)

    return selected_contributions, selected_frame


@app.cell
def _(
    mo,
    model_specs,
    plt,
    scatter_with_curve,
    selected_frame,
    selected_model,
):
    _spec = model_specs[selected_model.value]
    _figure, _axes = plt.subplots(2, 2, figsize=(16, 10))
    _variables = [
        ("user_bias_abs", "|b_u|"),
        ("film_bias_abs", "|b_i|"),
        ("user_v_norm", "||v_u||_2"),
        ("film_v_norm", "||v_i||_2"),
    ]

    for _axis, (_column, _label) in zip(_axes.ravel(), _variables):
        scatter_with_curve(
            _axis,
            selected_frame,
            _column,
            "abs_error",
            f"Erreur absolue vs {_label}",
            _label,
            "Erreur absolue",
        )

    _figure.tight_layout()

    if _spec["feature_set"] == "context":
        _families = ["user", "film", "genre", "year", "studio"]
        _context_figure, _context_axes = plt.subplots(
            2,
            5,
            figsize=(24, 8),
            squeeze=False,
        )
        for _column_index, _family in enumerate(_families):
            for _row_index, _prefix in enumerate(["linear", "latent"]):
                _axis = _context_axes[_row_index, _column_index]
                _column = f"{_prefix}_{_family}_abs"
                if _column in selected_frame:
                    scatter_with_curve(
                        _axis,
                        selected_frame,
                        _column,
                        "abs_error",
                        f"Erreur vs {_prefix} {_family}",
                        _column,
                        "Erreur absolue",
                        max_points=3000,
                    )
                else:
                    _axis.axis("off")
        _context_figure.tight_layout()
        mo.vstack([
            _figure,
            mo.md("### Erreur vs contributions par famille"),
            _context_figure,
        ])
    else:
        _figure


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Contributions effectives aux prédictions
    """)
    return


@app.cell
def _(
    EXPLORING_BIAS_DIR,
    compute_model_contributions,
    model_feature_mapping,
    model_names,
    model_params,
    model_specs,
    n_movies,
    n_users,
    np,
    pd,
    test_blocks,
    test_predictions,
):
    _rows = []
    for _name in model_names:
        _prediction_frame = test_predictions[
            test_predictions["model"] == _name
        ].reset_index(drop=True)
        _contributions = compute_model_contributions(
            _name,
            model_params,
            model_specs,
            test_blocks,
            model_feature_mapping,
            n_users,
            n_movies,
        )
        _y_true = _prediction_frame["rating_true"].to_numpy(
            dtype=np.float64
        )
        _y_pred = _prediction_frame["rating_est"].to_numpy(
            dtype=np.float64
        )
        _std_true = float(np.std(_y_true, ddof=0))
        _std_pred = float(np.std(_y_pred, ddof=0))
        _abs_user_bias = np.abs(_contributions["linear_user"])
        _abs_film_bias = np.abs(_contributions["linear_film"])
        _abs_user_film_dot = np.abs(
            _contributions["user_film_dot"]
        )

        _rows.append({
            "model": _name,
            "family": model_specs[_name]["family"],
            "feature_set": model_specs[_name]["feature_set"],
            "n_features": model_specs[_name]["n_features"],
            "n_factors": model_specs[_name]["n_factors"],
            "std_pred": _std_pred,
            "std_true": _std_true,
            "std_ratio": (
                _std_pred / _std_true
                if _std_true > 0
                else np.nan
            ),
            "mean_abs_user_bias": float(
                _abs_user_bias.mean()
            ),
            "mean_abs_film_bias": float(
                _abs_film_bias.mean()
            ),
            "mean_abs_user_film_dot": float(
                _abs_user_film_dot.mean()
            ),
            "mean_abs_linear_total": float(
                np.abs(_contributions["linear_total"]).mean()
            ),
            "mean_abs_latent_total": float(
                np.abs(_contributions["latent_total"]).mean()
            ),
            "p95_abs_user_bias": float(
                np.quantile(_abs_user_bias, 0.95)
            ),
            "p95_abs_film_bias": float(
                np.quantile(_abs_film_bias, 0.95)
            ),
            "p95_abs_user_film_dot": float(
                np.quantile(_abs_user_film_dot, 0.95)
            ),
            "max_reconstruction_error": float(
                np.max(
                    np.abs(
                        _contributions["reconstructed"] - _y_pred
                    )
                )
            ),
        })

    bias_summary = pd.DataFrame(_rows)
    EXPLORING_BIAS_DIR.mkdir(parents=True, exist_ok=True)
    bias_summary.to_csv(
        EXPLORING_BIAS_DIR / "bias_summary.csv",
        index=False,
    )
    return (bias_summary,)


@app.cell
def _(bias_summary, np, plt):
    _figure, _axis = plt.subplots(figsize=(14, 6))
    _x = np.arange(len(bias_summary))
    _width = 0.38
    _axis.bar(
        _x - _width / 2,
        bias_summary["std_pred"],
        width=_width,
        label="std(prédictions)",
    )
    _axis.bar(
        _x + _width / 2,
        bias_summary["std_true"],
        width=_width,
        label="std(notes réelles)",
    )
    _axis.set_xticks(_x)
    _axis.set_xticklabels(
        bias_summary["model"],
        rotation=35,
        ha="right",
    )
    _axis.set_ylabel("Écart-type")
    _axis.set_title(
        "Écart-type des prédictions vs notes réelles"
    )
    _axis.grid(True, axis="y", alpha=0.3)
    _axis.legend()
    _figure.tight_layout()
    _figure


@app.cell
def _(
    model_specs,
    mo,
    np,
    pd,
    plt,
    selected_contributions,
    selected_model,
):
    _spec = model_specs[selected_model.value]
    _component_map = {
        "global": selected_contributions["global"],
        "linear_user": selected_contributions["linear_user"],
        "linear_film": selected_contributions["linear_film"],
        "user_film_dot": selected_contributions["user_film_dot"],
        "linear_total": selected_contributions["linear_total"],
        "latent_total": selected_contributions["latent_total"],
    }

    if _spec["feature_set"] == "context":
        for _family, _values in selected_contributions[
            "linear_by_family"
        ].items():
            _component_map[f"linear:{_family}"] = _values
        for _family, _values in selected_contributions[
            "latent_by_family"
        ].items():
            _component_map[f"latent:{_family}"] = _values

    _rows = []
    for _component, _values in _component_map.items():
        _values = np.asarray(_values, dtype=np.float64)
        _rows.append({
            "component": _component,
            "mean_signed": float(_values.mean()),
            "mean_absolute": float(np.abs(_values).mean()),
            "std": float(_values.std(ddof=0)),
            "median_absolute": float(
                np.median(np.abs(_values))
            ),
            "p95_absolute": float(
                np.quantile(np.abs(_values), 0.95)
            ),
        })

    _component_table = pd.DataFrame(_rows)

    if _spec["feature_set"] == "context":
        _families = ["user", "film", "genre", "year", "studio"]
        _figure, _axes = plt.subplots(
            2,
            5,
            figsize=(24, 8),
            squeeze=False,
        )
        for _column_index, _family in enumerate(_families):
            for _row_index, _prefix in enumerate(["linear", "latent"]):
                _axis = _axes[_row_index, _column_index]
                _values = (
                    selected_contributions["linear_by_family"].get(
                        _family
                    )
                    if _prefix == "linear"
                    else selected_contributions[
                        "latent_by_family"
                    ].get(_family)
                )
                if _values is None:
                    _axis.axis("off")
                    continue
                _axis.hist(_values, bins=60, alpha=0.75)
                _axis.set_title(f"{_prefix} {_family}")
                _axis.set_xlabel("Contribution")
                _axis.set_ylabel("Observations")
                _axis.grid(True, alpha=0.3)
        _figure.tight_layout()
        mo.vstack([
            mo.ui.table(_component_table, pagination=True),
            _figure,
        ])
    elif _spec["feature_set"] == "part1":
        _figure, _axes = plt.subplots(1, 3, figsize=(18, 5))
        for _axis, _component in zip(
            _axes,
            ["linear_user", "linear_film", "user_film_dot"],
        ):
            _axis.hist(
                selected_contributions[_component],
                bins=60,
                alpha=0.75,
            )
            _axis.set_title(_component)
            _axis.set_xlabel("Contribution")
            _axis.set_ylabel("Observations")
            _axis.grid(True, alpha=0.3)
        _figure.tight_layout()
        mo.vstack([
            mo.ui.table(_component_table, pagination=True),
            _figure,
        ])
    else:
        mo.ui.table(_component_table, pagination=True)


@app.cell
def _(bias_summary, mo):
    mo.vstack([
        mo.md(
            "### Tableau récapitulatif final\n\n"
            "Le tableau est aussi sauvegardé dans "
            "`modeling/exploring_bias/bias_summary.csv`. Les valeurs sont "
            "descriptives. L'erreur de reconstruction permet de vérifier "
            "que la décomposition reproduit bien les prédictions "
            "sauvegardées."
        ),
        mo.ui.table(bias_summary, pagination=False),
    ])
    return


if __name__ == "__main__":
    app.run()
