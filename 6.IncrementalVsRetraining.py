#!/usr/bin/env python3

import argparse
import ast
import json
import time
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack, load_npz


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

        rng = np.random.default_rng(seed)
        self.w0 = 0.0
        self.w = np.zeros(self.n_features, dtype=np.float64)
        self.V = rng.normal(
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
        n_rows = X.shape[0]
        prediction = np.full(n_rows, self.w0, dtype=np.float64)
        if self.n_features == 0:
            return prediction

        if self.use_linear:
            prediction += X @ self.w

        if self.use_latent:
            XV = X @ self.V
            X2 = X.multiply(X)
            V2_sum = np.einsum("ij,ij->i", self.V, self.V)
            prediction += 0.5 * (
                np.einsum("ij,ij->i", XV, XV)
                - X2 @ V2_sum
            )
        return prediction

    def step(self, X, y):
        batch_size = X.shape[0]
        prediction = self.predict(X)
        error = prediction - y

        grad_w0 = float(error.mean())
        grad_w = np.zeros_like(self.w)
        grad_V = np.zeros_like(self.V)

        if self.use_linear and self.n_features:
            grad_w = (X.T @ error) / batch_size
            grad_w += self.reg_bias * self.w
            grad_w *= self.active_mask

        if self.use_latent and self.n_features:
            XV = X @ self.V
            X2 = X.multiply(X)
            G1 = X.T @ (error[:, None] * XV)
            G2 = X2.T @ error
            grad_V = (G1 - G2[:, None] * self.V) / batch_size
            grad_V += self.reg_factors * self.V
            grad_V *= self.active_mask[:, None]

        self.step_number += 1
        t = self.step_number
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8

        self.m_w0 = beta1 * self.m_w0 + (1 - beta1) * grad_w0
        self.v_w0 = beta2 * self.v_w0 + (1 - beta2) * grad_w0 ** 2
        m_hat_w0 = self.m_w0 / (1 - beta1 ** t)
        v_hat_w0 = self.v_w0 / (1 - beta2 ** t)
        self.w0 -= self.learning_rate * m_hat_w0 / (
            np.sqrt(v_hat_w0) + epsilon
        )

        if self.use_linear and self.n_features:
            self.m_w = beta1 * self.m_w + (1 - beta1) * grad_w
            self.v_w = (
                beta2 * self.v_w + (1 - beta2) * grad_w ** 2
            )
            m_hat_w = self.m_w / (1 - beta1 ** t)
            v_hat_w = self.v_w / (1 - beta2 ** t)
            self.w -= self.learning_rate * m_hat_w / (
                np.sqrt(v_hat_w) + epsilon
            )
            self.w[~self.active_mask] = 0.0

        if self.use_latent and self.n_features:
            self.m_V = beta1 * self.m_V + (1 - beta1) * grad_V
            self.v_V = (
                beta2 * self.v_V + (1 - beta2) * grad_V ** 2
            )
            m_hat_V = self.m_V / (1 - beta1 ** t)
            v_hat_V = self.v_V / (1 - beta2 ** t)
            self.V -= self.learning_rate * m_hat_V / (
                np.sqrt(v_hat_V) + epsilon
            )
            self.V[~self.active_mask] = 0.0

    def clone_parameters(self):
        return (float(self.w0), self.w.copy(), self.V.copy())

    def load_parameters(self, parameters):
        self.w0 = float(parameters[0])
        self.w = parameters[1].copy()
        self.V = parameters[2].copy()


def compute_metrics(y_true, y_pred):
    error = y_pred - y_true
    absolute = np.abs(error)
    return {
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "bias": float(error.mean()),
        "std_pred": float(np.std(y_pred, ddof=0)),
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
    engine = FMEngine(
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
    rng = np.random.default_rng(seed + 1)
    n_train = len(y_train)
    history = []
    best_rmse = np.inf
    best_epoch = 0
    best_parameters = None
    patience_counter = 0
    start = time.perf_counter()

    for epoch in range(1, int(n_epochs) + 1):
        if batch_size and int(batch_size) < n_train:
            order = rng.permutation(n_train)
            for start_index in range(0, n_train, int(batch_size)):
                batch = order[start_index:start_index + int(batch_size)]
                engine.step(X_train[batch], y_train[batch])
        else:
            engine.step(X_train, y_train)

        train_pred = engine.predict(X_train)
        val_pred = engine.predict(X_val)
        train_metrics = compute_metrics(y_train, train_pred)
        val_metrics = compute_metrics(y_val, val_pred)
        history.append({
            "epoch": epoch,
            "train_loss": float(
                0.5 * np.mean((train_pred - y_train) ** 2)
            ),
            "train_mae": train_metrics["mae"],
            "val_mae": val_metrics["mae"],
            "val_rmse": val_metrics["rmse"],
        })

        if val_metrics["rmse"] < best_rmse - 1e-10:
            best_rmse = val_metrics["rmse"]
            best_epoch = epoch
            best_parameters = engine.clone_parameters()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_parameters is not None:
        engine.load_parameters(best_parameters)

    return (
        engine,
        pd.DataFrame(history),
        best_epoch,
        int(history[-1]["epoch"]),
        time.perf_counter() - start,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Local adaptation vs global retraining."
    )
    parser.add_argument("--cleaning-dir", default=None)
    parser.add_argument("--model-output-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--n-factors", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--reg-bias", type=float, default=0.0)
    parser.add_argument("--reg-factors", type=float, default=1e-6)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--cold-eval-fraction", type=float, default=0.2)
    parser.add_argument("--max-users", type=int, default=None)
    parser.add_argument("--max-interactions-per-user", type=int, default=None)
    parser.add_argument("--levels", default=None)
    return parser.parse_args()


def load_model_config(model_output_dir):
    config_path = model_output_dir / "model_config.json"
    manifest_path = model_output_dir / "checkpoints" / "manifest.json"
    if config_path.exists():
        with open(config_path) as handle:
            config = json.load(handle)
        return {
            "hyperparameters": config["hyperparameters"],
            "split": config.get("split", {}),
            "feature_config": config.get("feature_config", {}),
        }
    if manifest_path.exists():
        with open(manifest_path) as handle:
            manifest = json.load(handle)
        run_context = manifest["run_context"]
        return {
            "hyperparameters": run_context["hyperparameters"],
            "split": run_context.get("split", {}),
            "feature_config": run_context.get("features", {}),
        }
    raise FileNotFoundError(
        "No model_config.json or checkpoints/manifest.json found."
    )


def load_inputs(cleaning_dir):
    matrix = load_npz(cleaning_dir / "ratings_matrix.npz").tocsr()
    user_ids = np.load(cleaning_dir / "user_ids.npy", allow_pickle=True)
    movie_ids = np.load(cleaning_dir / "movie_ids.npy", allow_pickle=True)

    if matrix.shape[0] != len(user_ids) or matrix.shape[1] != len(movie_ids):
        raise ValueError("Matrix shape and ID arrays are inconsistent.")

    user_mapping = pd.DataFrame({
        "user_id": user_ids,
        "row_index": np.arange(len(user_ids)),
    })
    movie_mapping = pd.DataFrame({
        "canonical_movie_id": movie_ids,
        "col_index": np.arange(len(movie_ids)),
    })

    movie_context = pd.read_json(cleaning_dir / "movie_context.json")
    movie_context = movie_context.rename(columns={
        "movie_id": "canonical_movie_id",
        "date": "year",
        "studio": "studios",
    })
    movie_context["col_index"] = np.arange(len(movie_context))

    if len(movie_context) != matrix.shape[1]:
        raise ValueError("movie_context does not match matrix columns.")
    if (
        movie_context["canonical_movie_id"].astype(str).tolist()
        != movie_mapping["canonical_movie_id"].astype(str).tolist()
    ):
        raise ValueError("movie_context is not aligned with movie_ids.npy.")

    matrix_coo = matrix.tocoo()
    observations = pd.DataFrame({
        "row": matrix_coo.row.astype(np.int64),
        "col": matrix_coo.col.astype(np.int64),
        "user_id": user_mapping["user_id"].to_numpy()[matrix_coo.row],
        "film_id": movie_mapping[
            "canonical_movie_id"
        ].to_numpy()[matrix_coo.col],
        "rating": matrix_coo.data.astype(np.float64),
    })
    if observations.duplicated(subset=["row", "col"]).any():
        raise ValueError("Duplicated user-film coordinates in the matrix.")

    return {
        "observations": observations,
        "user_mapping": user_mapping,
        "movie_mapping": movie_mapping,
        "movie_context": movie_context.sort_values(
            "col_index"
        ).reset_index(drop=True),
        "n_users": matrix.shape[0],
        "n_movies": matrix.shape[1],
    }


def cap_observations(
    observations,
    max_users,
    max_interactions_per_user,
    seed,
):
    obs = observations.copy()
    rng = np.random.default_rng(seed)

    if max_users is not None:
        counts = obs["row"].value_counts()
        candidates = counts[counts >= 2].index.to_numpy()
        if len(candidates) > max_users:
            selected = rng.choice(
                candidates,
                size=int(max_users),
                replace=False,
            )
            obs = obs[obs["row"].isin(selected)]

    if max_interactions_per_user is not None:
        kept = []
        for _, group in obs.groupby("row", sort=False):
            indices = group.index.to_numpy()
            if len(indices) > int(max_interactions_per_user):
                choice = rng.choice(
                    indices,
                    size=int(max_interactions_per_user),
                    replace=False,
                )
                kept.extend(choice.tolist())
            else:
                kept.extend(indices.tolist())
        obs = obs.loc[np.sort(np.asarray(kept))]

    return obs.reset_index(drop=True)


def split_users(observations, seed, cold_eval_fraction):
    rng = np.random.default_rng(seed + 17)
    counts = observations["row"].value_counts()
    candidate_users = counts[counts >= 2].index.to_numpy()
    if len(candidate_users) < 3:
        raise ValueError("Not enough users with at least two interactions.")

    n_test_users = max(1, int(round(0.1 * len(candidate_users))))
    n_test_users = min(n_test_users, len(candidate_users) - 2)
    test_users = rng.choice(
        candidate_users,
        size=n_test_users,
        replace=False,
    )
    test_mask = observations["row"].isin(test_users).to_numpy()
    test_indices = np.where(test_mask)[0]
    known_indices = np.where(~test_mask)[0]

    if len(known_indices) < 2:
        raise ValueError("Not enough known interactions.")

    n_val = max(1, int(round(0.1 * len(observations))))
    n_val = min(n_val, len(known_indices) - 1)
    val_indices = rng.choice(known_indices, size=n_val, replace=False)
    train_indices = np.setdiff1d(
        known_indices,
        val_indices,
        assume_unique=False,
    )

    train_pairs = observations.iloc[train_indices][["row", "col"]]
    val_pairs = observations.iloc[val_indices][["row", "col"]]
    test_pairs = observations.iloc[test_indices][["row", "col"]]
    all_pairs = pd.concat(
        [train_pairs, val_pairs, test_pairs],
        ignore_index=True,
    )
    if all_pairs.duplicated().any():
        raise ValueError("Duplicated user-film pairs across splits.")

    if set(observations.iloc[test_indices]["row"]) & set(
        observations.iloc[train_indices]["row"]
    ):
        raise ValueError("Test users overlap with train users.")
    if set(observations.iloc[test_indices]["row"]) & set(
        observations.iloc[val_indices]["row"]
    ):
        raise ValueError("Test users overlap with validation users.")

    test_films = set(observations.iloc[test_indices]["col"])
    train_films = set(observations.iloc[train_indices]["col"])
    test_film_coverage = (
        len(test_films & train_films) / len(test_films)
        if test_films
        else 0.0
    )

    cold_eval_indices = []
    pool_order_by_user = {}
    eval_rows_by_user = {}
    eval_positions_by_user = {}
    kept_test_users = []

    for user in np.sort(test_users):
        user_indices = np.where(
            observations["row"].to_numpy() == user
        )[0]
        if len(user_indices) < 2:
            continue
        user_rng = np.random.default_rng(seed + 100000 + int(user))
        order = user_rng.permutation(len(user_indices))
        ordered_indices = user_indices[order]
        n_eval = max(
            1,
            int(round(cold_eval_fraction * len(ordered_indices))),
        )
        n_eval = min(n_eval, len(ordered_indices) - 1)
        eval_indices_user = ordered_indices[:n_eval]
        pool_indices = ordered_indices[n_eval:]
        eval_rows_by_user[int(user)] = eval_indices_user
        pool_order_by_user[int(user)] = pool_indices
        cold_eval_indices.extend(eval_indices_user.tolist())
        kept_test_users.append(int(user))

    if not kept_test_users:
        raise ValueError("No cold-start users with enough interactions.")

    cold_eval_indices = np.asarray(
        sorted(set(cold_eval_indices)),
        dtype=np.int64,
    )
    eval_position = {
        int(index): position
        for position, index in enumerate(cold_eval_indices)
    }
    for user, indices in eval_rows_by_user.items():
        eval_positions_by_user[user] = np.asarray(
            [eval_position[int(index)] for index in indices],
            dtype=np.int64,
        )

    split_info = {
        "n_interactions": int(len(observations)),
        "n_train": int(len(train_indices)),
        "n_validation": int(len(val_indices)),
        "n_test": int(len(test_indices)),
        "train_fraction": float(len(train_indices) / len(observations)),
        "validation_fraction": float(len(val_indices) / len(observations)),
        "test_fraction": float(len(test_indices) / len(observations)),
        "n_test_users": int(len(kept_test_users)),
        "test_film_coverage_in_train": float(test_film_coverage),
    }

    return {
        "train_indices": train_indices,
        "val_indices": val_indices,
        "test_indices": test_indices,
        "test_users": np.asarray(sorted(kept_test_users), dtype=np.int64),
        "eval_indices": cold_eval_indices,
        "eval_rows_by_user": eval_rows_by_user,
        "eval_positions_by_user": eval_positions_by_user,
        "pool_order_by_user": pool_order_by_user,
        "split_info": split_info,
    }


def as_list(value):
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") and text.endswith("]"):
            try:
                value = ast.literal_eval(text)
            except (SyntaxError, ValueError):
                value = text
        elif "," in text:
            value = [
                item.strip()
                for item in text.split(",")
                if item.strip()
            ]
        else:
            value = text
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


def build_features(
    observations,
    movie_context,
    user_mapping,
    movie_mapping,
    n_users,
    n_movies,
    train_indices,
    max_studios,
):
    context_by_col = movie_context.set_index("col_index").sort_index()
    genres_by_film = [
        as_list(value)
        for value in context_by_col.get(
            "genres",
            pd.Series([None] * n_movies, index=context_by_col.index),
        )
    ]
    studios_by_film = [
        as_list(value)
        for value in context_by_col.get(
            "studios",
            pd.Series([None] * n_movies, index=context_by_col.index),
        )
    ]

    train_films = np.sort(
        observations.iloc[train_indices]["col"].unique()
    )
    genre_counter = Counter()
    for film in train_films:
        genres = genres_by_film[film]
        if genres:
            genre_counter.update(set(genres))

    studio_counter = Counter()
    for studios in studios_by_film:
        if studios:
            studio_counter.update(set(studios))

    studio_ranked = sorted(
        studio_counter.items(),
        key=lambda item: (-item[1], item[0]),
    )
    genre_vocab = sorted(genre_counter)
    studio_vocab = sorted(
        studio for studio, _ in studio_ranked[:max_studios]
    )
    retained_studios = set(studio_vocab)
    studios_by_film = [
        (
            None
            if studios is None
            else ([s for s in studios if s in retained_studios] or None)
        )
        for studios in studios_by_film
    ]

    genre_feature_names = (
        ["genre:<UNKNOWN>", "genre:<MISSING>"]
        + [f"genre:{value}" for value in genre_vocab]
    )
    studio_feature_names = (
        ["studio:<UNKNOWN>", "studio:<MISSING>"]
        + [f"studio:{value}" for value in studio_vocab]
    )
    genre_index = {
        value: index + 2 for index, value in enumerate(genre_vocab)
    }
    studio_index = {
        value: index + 2 for index, value in enumerate(studio_vocab)
    }

    genre_rows = []
    genre_cols = []
    studio_rows = []
    studio_cols = []
    for film in range(n_movies):
        genres = genres_by_film[film]
        if genres is None:
            genre_rows.append(film)
            genre_cols.append(1)
        elif len(genres) > 0:
            for value in set(genres):
                genre_rows.append(film)
                genre_cols.append(genre_index.get(value, 0))

        studios = studios_by_film[film]
        if studios is None:
            studio_rows.append(film)
            studio_cols.append(1)
        elif len(studios) > 0:
            for value in set(studios):
                studio_rows.append(film)
                studio_cols.append(studio_index.get(value, 0))

    item_genres = csr_matrix(
        (
            np.ones(len(genre_rows), dtype=np.float64),
            (genre_rows, genre_cols),
        ),
        shape=(n_movies, len(genre_feature_names)),
    )
    item_studios = csr_matrix(
        (
            np.ones(len(studio_rows), dtype=np.float64),
            (studio_rows, studio_cols),
        ),
        shape=(n_movies, len(studio_feature_names)),
    )

    film_years = pd.to_numeric(
        context_by_col.get(
            "year",
            pd.Series([np.nan] * n_movies, index=context_by_col.index),
        ),
        errors="coerce",
    ).to_numpy(dtype=np.float64)
    train_years = film_years[train_films]
    valid_years = train_years[np.isfinite(train_years)]
    if len(valid_years):
        year_mean = float(valid_years.mean())
        year_std = float(valid_years.std(ddof=0))
    else:
        year_mean = 0.0
        year_std = 1.0
    if year_std <= 0:
        year_std = 1.0

    n_obs = len(observations)
    rows = observations["row"].to_numpy(dtype=np.int64)
    cols = observations["col"].to_numpy(dtype=np.int64)
    index = np.arange(n_obs)

    user_block = csr_matrix(
        (np.ones(n_obs), (index, rows)),
        shape=(n_obs, n_users),
    )
    film_block = csr_matrix(
        (np.ones(n_obs), (index, cols)),
        shape=(n_obs, n_movies),
    )
    genre_block = item_genres[cols]
    studio_block = item_studios[cols]

    years = film_years[cols]
    missing = ~np.isfinite(years)
    year_z = np.where(
        missing,
        0.0,
        (years - year_mean) / year_std,
    )
    present_rows = index[~missing]
    missing_rows = index[missing]
    year_rows = np.concatenate([present_rows, missing_rows])
    year_cols = np.concatenate([
        np.zeros(len(present_rows), dtype=np.int64),
        np.ones(len(missing_rows), dtype=np.int64),
    ])
    year_data = np.concatenate([
        year_z[~missing],
        np.ones(len(missing_rows), dtype=np.float64),
    ])
    year_block = csr_matrix(
        (year_data, (year_rows, year_cols)),
        shape=(n_obs, 2),
    )

    part1_all = hstack([user_block, film_block], format="csr")
    context_all = hstack(
        [user_block, film_block, genre_block, year_block, studio_block],
        format="csr",
    )

    user_names = [f"user:{value}" for value in user_mapping["user_id"]]
    film_names = [
        f"film:{value}"
        for value in movie_mapping["canonical_movie_id"]
    ]
    context_groups = np.array(
        ["user"] * n_users
        + ["film"] * n_movies
        + ["genre"] * len(genre_feature_names)
        + ["year"] * 2
        + ["studio"] * len(studio_feature_names),
        dtype=object,
    )
    context_names = np.array(
        user_names
        + film_names
        + genre_feature_names
        + ["year:standardized", "year:missing"]
        + studio_feature_names,
        dtype=object,
    )
    context_mapping = pd.DataFrame({
        "feature_index": np.arange(len(context_names)),
        "feature_name": context_names,
        "family": context_groups,
    })
    context_family_indices = {
        family: np.where(context_groups == family)[0]
        for family in ["user", "film", "genre", "year", "studio"]
    }
    part1_mapping = pd.DataFrame({
        "feature_index": np.arange(n_users + n_movies),
        "feature_name": user_names + film_names,
        "family": ["user"] * n_users + ["film"] * n_movies,
    })

    return {
        "part1_all": part1_all,
        "context_all": context_all,
        "context_mapping": context_mapping,
        "part1_mapping": part1_mapping,
        "context_family_indices": context_family_indices,
        "year_mean": year_mean,
        "year_std": year_std,
    }


def make_model_specs(n_users, n_movies, n_context_features):
    part1_features = n_users + n_movies
    return [
        {
            "name": "M1",
            "feature_set": "none",
            "n_features": 0,
            "use_linear": False,
            "use_latent": False,
            "active_mask": np.zeros(0, dtype=bool),
        },
        {
            "name": "M2",
            "feature_set": "part1",
            "n_features": part1_features,
            "use_linear": True,
            "use_latent": False,
            "active_mask": np.concatenate([
                np.zeros(n_users, dtype=bool),
                np.ones(n_movies, dtype=bool),
            ]),
        },
        {
            "name": "M3",
            "feature_set": "part1",
            "n_features": part1_features,
            "use_linear": True,
            "use_latent": False,
            "active_mask": np.ones(part1_features, dtype=bool),
        },
        {
            "name": "M4",
            "feature_set": "part1",
            "n_features": part1_features,
            "use_linear": True,
            "use_latent": True,
            "active_mask": np.ones(part1_features, dtype=bool),
        },
        {
            "name": "M5",
            "feature_set": "context",
            "n_features": n_context_features,
            "use_linear": True,
            "use_latent": True,
            "active_mask": np.ones(n_context_features, dtype=bool),
        },
    ]


def save_model(path, spec, engine):
    np.savez_compressed(
        path,
        w0=np.array([engine.w0]),
        w=engine.w,
        V=engine.V,
        active_mask=engine.active_mask,
        n_features=np.array([engine.n_features]),
        n_factors=np.array([engine.n_factors]),
        use_linear=np.array([engine.use_linear]),
        use_latent=np.array([engine.use_latent]),
        feature_set=np.array([spec["feature_set"]]),
        model_name=np.array([spec["name"]]),
    )


def load_model(path):
    with np.load(path) as data:
        spec = {
            "name": str(data["model_name"][0]),
            "feature_set": str(data["feature_set"][0]),
            "n_features": int(data["n_features"][0]),
            "use_linear": bool(data["use_linear"][0]),
            "use_latent": bool(data["use_latent"][0]),
            "active_mask": data["active_mask"].astype(bool),
        }
        engine = FMEngine(
            n_features=spec["n_features"],
            n_factors=int(data["n_factors"][0]),
            use_linear=spec["use_linear"],
            use_latent=spec["use_latent"],
            active_mask=spec["active_mask"],
            learning_rate=0.0,
            reg_bias=0.0,
            reg_factors=0.0,
            seed=0,
        )
        engine.w0 = float(data["w0"][0])
        engine.w = data["w"].copy()
        engine.V = data["V"].copy()
    return spec, engine


def user_feature_index(spec, user_row, context_family_indices):
    if spec["feature_set"] == "part1":
        return int(user_row)
    if spec["feature_set"] == "context":
        return int(context_family_indices["user"][int(user_row)])
    return None


def predict_with_zeroed_users(
    spec,
    engine,
    X_all,
    eval_indices,
    users_to_zero,
    context_family_indices,
):
    if spec["feature_set"] == "none":
        return engine.predict(X_all[eval_indices])

    user_indices = [
        user_feature_index(spec, user, context_family_indices)
        for user in users_to_zero
    ]
    saved_w = {}
    saved_V = {}
    for index in user_indices:
        saved_w[index] = engine.w[index].copy()
        saved_V[index] = engine.V[index].copy()
        engine.w[index] = 0.0
        engine.V[index] = 0.0
    prediction = engine.predict(X_all[eval_indices])
    for index in user_indices:
        engine.w[index] = saved_w[index]
        engine.V[index] = saved_V[index]
    return prediction


def run_local_adaptation(
    spec,
    engine,
    X_all,
    y_all,
    eval_indices,
    eval_rows_by_user,
    eval_positions_by_user,
    revealed_by_user,
    context_family_indices,
    lambda_bias,
    lambda_factors,
):
    start = time.perf_counter()
    predictions = np.zeros(len(eval_indices), dtype=np.float64)
    adapted = {}

    if spec["feature_set"] == "none" or spec["name"] == "M2":
        predictions[:] = engine.predict(X_all[eval_indices])
        for user in eval_rows_by_user:
            adapted[int(user)] = (
                0.0,
                np.zeros(engine.n_factors, dtype=np.float64),
            )
        return predictions, adapted, time.perf_counter() - start

    n_factors = engine.n_factors
    lambda_b = max(float(lambda_bias), 1e-8)
    lambda_f = max(float(lambda_factors), 1e-8)

    for user, eval_rows_user in eval_rows_by_user.items():
        feature_index = user_feature_index(
            spec,
            int(user),
            context_family_indices,
        )
        original_w = engine.w[feature_index].copy()
        original_V = engine.V[feature_index].copy()
        engine.w[feature_index] = 0.0
        engine.V[feature_index] = 0.0

        revealed_rows = revealed_by_user.get(
            int(user),
            np.array([], dtype=np.int64),
        )
        if len(revealed_rows) == 0:
            b_u = 0.0
            p_u = np.zeros(n_factors, dtype=np.float64)
        else:
            pred_no_user = engine.predict(X_all[revealed_rows])
            target = y_all[revealed_rows] - pred_no_user

            if spec["name"] == "M3":
                denominator = len(revealed_rows) + lambda_b
                b_u = float(target.sum() / denominator)
                p_u = np.zeros(n_factors, dtype=np.float64)
            else:
                X_non_user = X_all[revealed_rows].copy()
                X_non_user[:, feature_index] = 0.0
                S = X_non_user @ engine.V
                A = np.column_stack([
                    np.ones(len(revealed_rows), dtype=np.float64),
                    S,
                ])
                regularization = np.diag(
                    [lambda_b] + [lambda_f] * n_factors
                )
                try:
                    theta = np.linalg.solve(
                        A.T @ A + regularization,
                        A.T @ target,
                    )
                except np.linalg.LinAlgError:
                    theta = np.linalg.lstsq(
                        A,
                        target,
                        rcond=None,
                    )[0]
                b_u = float(theta[0])
                p_u = theta[1:].astype(np.float64)

        engine.w[feature_index] = b_u
        engine.V[feature_index] = p_u
        eval_prediction = engine.predict(X_all[eval_rows_user])
        positions = eval_positions_by_user[int(user)]
        predictions[positions] = eval_prediction
        adapted[int(user)] = (b_u, p_u)

        engine.w[feature_index] = original_w
        engine.V[feature_index] = original_V

    return predictions, adapted, time.perf_counter() - start


def save_local_parameters(path, user_ids, adapted, n_factors):
    users = np.asarray(sorted(adapted.keys()), dtype=np.int64)
    b_u = np.asarray(
        [adapted[int(user)][0] for user in users],
        dtype=np.float64,
    )
    if len(users):
        p_u = np.vstack([
            adapted[int(user)][1].reshape(1, -1)
            for user in users
        ])
    else:
        p_u = np.zeros((0, n_factors), dtype=np.float64)
    np.savez_compressed(
        path,
        user_rows=users,
        user_ids=user_ids[users],
        b_u=b_u,
        p_u=p_u,
    )


def plot_learning_curves(histories, output_path):
    figure, axes = plt.subplots(2, 2, figsize=(16, 10))
    specs = [
        ("train_loss", "Loss train"),
        ("train_mae", "MAE train"),
        ("val_mae", "MAE validation"),
        ("val_rmse", "RMSE validation"),
    ]
    for axis, (column, title) in zip(axes.ravel(), specs):
        for model, history in histories.groupby("model"):
            axis.plot(
                history["epoch"],
                history[column],
                label=model,
                linewidth=1.2,
            )
        axis.set_title(title)
        axis.set_xlabel("Époque")
        axis.set_ylabel(column)
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def plot_strategy_curves(results, output_path, metric, ylabel, log_y=False):
    figure, axes = plt.subplots(5, 1, figsize=(14, 20), sharex=True)
    for axis, model in zip(axes, ["M1", "M2", "M3", "M4", "M5"]):
        subset = results[results["model"] == model]
        for strategy, color in [
            ("local", "tab:blue"),
            ("retrain", "tab:red"),
        ]:
            strategy_subset = subset[
                subset["strategy"] == strategy
            ].sort_values("n_revealed")
            values = strategy_subset[metric].to_numpy(dtype=np.float64)
            if log_y:
                values = np.maximum(values, 1e-6)
            axis.plot(
                strategy_subset["n_revealed"],
                values,
                marker="o",
                color=color,
                label=strategy,
            )
        axis.set_title(f"{model} — {ylabel}")
        axis.set_ylabel(ylabel)
        if log_y:
            axis.set_yscale("log")
        axis.grid(True, alpha=0.3)
        axis.legend()
    axes[-1].set_xlabel("Nombre de notes révélées")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def compute_delta_table(results):
    rows = []
    for model in ["M1", "M2", "M3", "M4", "M5"]:
        model_results = results[results["model"] == model]
        for level in sorted(model_results["level"].unique()):
            level_results = model_results[
                np.isclose(model_results["level"], level)
            ]
            local = level_results[level_results["strategy"] == "local"]
            retrain = level_results[
                level_results["strategy"] == "retrain"
            ]
            if local.empty or retrain.empty:
                continue
            rows.append({
                "model": model,
                "level": float(level),
                "n_revealed": int(retrain.iloc[0]["n_revealed"]),
                "mae_local": float(local.iloc[0]["mae"]),
                "mae_retrain": float(retrain.iloc[0]["mae"]),
                "delta_mae": float(
                    local.iloc[0]["mae"] - retrain.iloc[0]["mae"]
                ),
            })
    delta = pd.DataFrame(rows)
    if delta.empty:
        return delta
    delta["gain_stable"] = False
    for model in ["M1", "M2", "M3", "M4", "M5"]:
        subset = delta[delta["model"] == model].sort_values("level")
        if subset.empty:
            continue
        positive = subset["delta_mae"] > 0
        stable_level = None
        for position in range(len(subset)):
            if positive.iloc[position:].all():
                stable_level = float(subset.iloc[position]["level"])
                break
        if stable_level is not None:
            mask = (
                (delta["model"] == model)
                & (delta["level"] >= stable_level)
            )
            delta.loc[mask, "gain_stable"] = True
    return delta


def plot_best_diagnostics(eval_frame, output_path, output_csv):
    figure, axes = plt.subplots(2, 2, figsize=(16, 10))

    rating_stats = (
        eval_frame.groupby("rating_true")
        .agg(
            bias=("error", "mean"),
            mae=("abs_error", "mean"),
            n=("error", "size"),
        )
        .reset_index()
    )
    axes[0, 0].plot(
        rating_stats["rating_true"],
        rating_stats["bias"],
        marker="o",
        label="Biais",
    )
    axes[0, 0].plot(
        rating_stats["rating_true"],
        rating_stats["mae"],
        marker="o",
        label="MAE",
    )
    axes[0, 0].axhline(0, color="black", linestyle="--", linewidth=0.8)
    axes[0, 0].set_title("Erreur selon la note réelle")
    axes[0, 0].set_xlabel("Note réelle")
    axes[0, 0].set_ylabel("Erreur")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    film_bins = pd.qcut(
        eval_frame["film_popularity"].rank(method="first"),
        q=min(10, eval_frame["film_popularity"].nunique()),
        duplicates="drop",
    )
    film_stats = (
        eval_frame.assign(bin=film_bins)
        .groupby("bin", observed=True)
        .agg(
            mae=("abs_error", "mean"),
            n=("abs_error", "size"),
            center=("film_popularity", "median"),
        )
        .reset_index(drop=True)
    )
    axes[0, 1].plot(
        film_stats["center"],
        film_stats["mae"],
        marker="o",
    )
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_title("MAE selon la popularité film")
    axes[0, 1].set_xlabel("Notes film (log)")
    axes[0, 1].set_ylabel("MAE")
    axes[0, 1].grid(True, alpha=0.3)

    user_bins = pd.qcut(
        eval_frame["user_activity"].rank(method="first"),
        q=min(10, eval_frame["user_activity"].nunique()),
        duplicates="drop",
    )
    user_stats = (
        eval_frame.assign(bin=user_bins)
        .groupby("bin", observed=True)
        .agg(
            mae=("abs_error", "mean"),
            n=("abs_error", "size"),
            center=("user_activity", "median"),
        )
        .reset_index(drop=True)
    )
    axes[1, 0].plot(
        user_stats["center"],
        user_stats["mae"],
        marker="o",
    )
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_title("MAE selon l'activité utilisateur")
    axes[1, 0].set_xlabel("Notes révélées utilisateur (log)")
    axes[1, 0].set_ylabel("MAE")
    axes[1, 0].grid(True, alpha=0.3)

    heat_frame = eval_frame.copy()
    heat_frame["user_bin"] = pd.qcut(
        heat_frame["user_activity"].rank(method="first"),
        q=min(4, heat_frame["user_activity"].nunique()),
        duplicates="drop",
    )
    heat_frame["film_bin"] = pd.qcut(
        heat_frame["film_popularity"].rank(method="first"),
        q=min(4, heat_frame["film_popularity"].nunique()),
        duplicates="drop",
    )
    heat = (
        heat_frame.groupby(
            ["user_bin", "film_bin"],
            observed=True,
        )
        .agg(
            mae=("abs_error", "mean"),
            n=("abs_error", "size"),
        )
        .reset_index()
    )
    pivot = heat.pivot(
        index="user_bin",
        columns="film_bin",
        values="mae",
    )
    image = axes[1, 1].imshow(
        pivot.to_numpy(),
        aspect="auto",
        cmap="viridis",
    )
    axes[1, 1].set_title("MAE activité utilisateur × popularité film")
    axes[1, 1].set_xlabel("Popularité film")
    axes[1, 1].set_ylabel("Activité utilisateur")
    figure.colorbar(image, ax=axes[1, 1], label="MAE")

    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)

    summary = pd.concat(
        [
            rating_stats.assign(group="rating_true"),
            film_stats.assign(group="film_popularity"),
            user_stats.assign(group="user_activity"),
        ],
        ignore_index=True,
    )
    summary.to_csv(output_csv, index=False)


def matrix_for_spec(features, spec, n_obs):
    if spec["feature_set"] == "none":
        return csr_matrix((n_obs, 0))
    if spec["feature_set"] == "part1":
        return features["part1_all"]
    return features["context_all"]


def set_local_user_parameters(
    spec,
    engine,
    local_path,
    context_family_indices,
):
    with np.load(local_path) as data:
        user_rows = data["user_rows"].astype(np.int64)
        b_u = data["b_u"]
        p_u = data["p_u"]
    saved = {}
    for position, user_row in enumerate(user_rows):
        feature_index = user_feature_index(
            spec,
            int(user_row),
            context_family_indices,
        )
        saved[feature_index] = (
            engine.w[feature_index].copy(),
            engine.V[feature_index].copy(),
        )
        engine.w[feature_index] = b_u[position]
        engine.V[feature_index] = p_u[position]
    return saved


def restore_parameters(engine, saved):
    for feature_index, (w_value, V_value) in saved.items():
        engine.w[feature_index] = w_value
        engine.V[feature_index] = V_value


def compute_parameter_summary(
    spec,
    engine,
    features,
    n_users,
    n_movies,
    local_path=None,
    context_family_indices=None,
    test_users=None,
):
    context_family_indices = context_family_indices or features[
        "context_family_indices"
    ]
    saved = {}
    if local_path is not None and spec["feature_set"] != "none":
        saved = set_local_user_parameters(
            spec,
            engine,
            local_path,
            context_family_indices,
        )

    if spec["feature_set"] == "part1":
        user_indices = np.arange(n_users)
        film_indices = np.arange(n_users, n_users + n_movies)
        families = {
            "user": user_indices,
            "film": film_indices,
        }
        linear_families = {}
        latent_families = families
    elif spec["feature_set"] == "context":
        linear_families = {
            "genre": context_family_indices["genre"],
            "year": context_family_indices["year"],
            "studio": context_family_indices["studio"],
        }
        families = {
            "user": context_family_indices["user"],
            "film": context_family_indices["film"],
        }
        families.update(linear_families)
        latent_families = families
    else:
        families = {}
        linear_families = {}
        latent_families = {}

    rows = []
    for family, indices in families.items():
        w_values = engine.w[indices]
        v_norms = np.linalg.norm(engine.V[indices], axis=1)
        rows.append({
            "family": family,
            "mean_signed_bias": float(w_values.mean()),
            "mean_abs_bias": float(np.abs(w_values).mean()),
            "median_bias": float(np.median(w_values)),
            "p95_abs_bias": float(np.quantile(np.abs(w_values), 0.95)),
            "mean_latent_norm": float(v_norms.mean()),
            "n_features": int(len(indices)),
        })

    if saved:
        restore_parameters(engine, saved)
    return pd.DataFrame(
        rows,
        columns=[
            "family",
            "mean_signed_bias",
            "mean_abs_bias",
            "median_bias",
            "p95_abs_bias",
            "mean_latent_norm",
            "n_features",
        ],
    )


def compute_latent_contribution(spec, engine, X_eval, local_path=None, context_family_indices=None):
    saved = {}
    if local_path is not None and spec["feature_set"] != "none":
        saved = set_local_user_parameters(
            spec,
            engine,
            local_path,
            context_family_indices,
        )
    if not engine.use_latent or engine.n_features == 0:
        contribution = np.zeros(X_eval.shape[0], dtype=np.float64)
    else:
        XV = X_eval @ engine.V
        X2 = X_eval.multiply(X_eval)
        V2_sum = np.einsum("ij,ij->i", engine.V, engine.V)
        contribution = 0.5 * (
            np.einsum("ij,ij->i", XV, XV) - X2 @ V2_sum
        )
    if saved:
        restore_parameters(engine, saved)
    return contribution


def main():
    args = parse_args()
    cleaning_dir = Path(
        args.cleaning_dir if args.cleaning_dir is not None else Path.cwd()
    )
    model_output_dir = Path(
        args.model_output_dir
        if args.model_output_dir is not None
        else cleaning_dir / "modeling"
    )
    output_dir = Path(
        args.output_dir
        if args.output_dir is not None
        else model_output_dir / "incremental_vs_retraining"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = output_dir / "models"
    (models_dir / "warm_start").mkdir(parents=True, exist_ok=True)
    (models_dir / "retrain").mkdir(parents=True, exist_ok=True)
    (models_dir / "local").mkdir(parents=True, exist_ok=True)

    model_config = load_model_config(model_output_dir)
    hyperparameters = model_config["hyperparameters"]
    split_context = model_config["split"]
    feature_config = model_config["feature_config"]

    seed = int(
        args.seed
        if args.seed is not None
        else split_context.get("seed", hyperparameters.get("seed", 42))
    )
    n_factors = int(
        args.n_factors
        if args.n_factors is not None
        else (4 if args.fast else 8)
    )
    learning_rate = float(
        args.learning_rate
        if args.learning_rate is not None
        else hyperparameters.get("learning_rate", 0.01)
    )
    batch_size = int(
        args.batch_size
        if args.batch_size is not None
        else hyperparameters.get("batch_size", 65536)
    )
    n_epochs = int(
        args.epochs
        if args.epochs is not None
        else (5 if args.fast else hyperparameters.get("n_epochs", 20))
    )
    max_studios = int(feature_config.get("max_studios", 4000))

    if args.levels is not None:
        levels = [
            float(value)
            for value in args.levels.split(",")
            if value.strip()
        ]
    elif args.fast:
        levels = [0.0, 0.05, 0.2, 1.0]
    else:
        levels = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    levels = sorted(set(levels))
    if 0.05 not in levels:
        levels.append(0.05)
        levels = sorted(set(levels))

    print("Loading inputs...")
    data = load_inputs(cleaning_dir)
    max_users = args.max_users if args.max_users is not None else (
        200 if args.fast else None
    )
    max_interactions = (
        args.max_interactions_per_user
        if args.max_interactions_per_user is not None
        else (50 if args.fast else None)
    )
    observations = cap_observations(
        data["observations"],
        max_users,
        max_interactions,
        seed,
    )
    if len(observations) < 10:
        raise ValueError("Not enough observations after filtering.")

    print("Splitting users...")
    split = split_users(observations, seed, args.cold_eval_fraction)
    y_all = observations["rating"].to_numpy(dtype=np.float64)
    eval_indices = split["eval_indices"]
    y_eval = y_all[eval_indices]
    test_users = split["test_users"]
    all_user_ids = observations["user_id"].to_numpy()

    print("Building features...")
    features = build_features(
        observations,
        data["movie_context"],
        data["user_mapping"],
        data["movie_mapping"],
        data["n_users"],
        data["n_movies"],
        split["train_indices"],
        max_studios,
    )
    model_specs = make_model_specs(
        data["n_users"],
        data["n_movies"],
        len(features["context_mapping"]),
    )

    config_payload = {
        "cleaning_dir": str(cleaning_dir),
        "model_output_dir": str(model_output_dir),
        "output_dir": str(output_dir),
        "seed": seed,
        "n_factors": n_factors,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "reg_bias": float(args.reg_bias),
        "reg_factors": float(args.reg_factors),
        "cold_eval_fraction": float(args.cold_eval_fraction),
        "levels": levels,
        "fast": bool(args.fast),
        "split": split["split_info"],
    }
    with open(output_dir / "config.json", "w") as handle:
        json.dump(config_payload, handle, indent=2, default=str)
    pd.DataFrame([split["split_info"]]).to_csv(
        output_dir / "split_summary.csv",
        index=False,
    )

    print("Training warm-start models...")
    warm_engines = {}
    warm_metrics = []
    warm_histories = []
    for spec_index, spec in enumerate(model_specs):
        X_all = matrix_for_spec(
            features,
            spec,
            len(observations),
        )
        X_train = X_all[split["train_indices"]]
        X_val = X_all[split["val_indices"]]
        y_train = y_all[split["train_indices"]]
        y_val = y_all[split["val_indices"]]
        model_seed = seed + spec_index * 1000
        (
            engine,
            history,
            best_epoch,
            epochs_run,
            training_time,
        ) = train_fm_model(
            X_train,
            y_train,
            X_val,
            y_val,
            n_features=spec["n_features"],
            n_factors=n_factors,
            use_linear=spec["use_linear"],
            use_latent=spec["use_latent"],
            active_mask=spec["active_mask"],
            learning_rate=learning_rate,
            reg_bias=float(args.reg_bias),
            reg_factors=float(args.reg_factors),
            n_epochs=n_epochs,
            batch_size=batch_size,
            seed=model_seed,
            patience=int(args.patience),
        )
        warm_engines[spec["name"]] = engine
        save_model(
            models_dir / "warm_start" / f"{spec['name']}.npz",
            spec,
            engine,
        )
        history = history.assign(model=spec["name"])
        warm_histories.append(history)
        train_metrics = compute_metrics(y_train, engine.predict(X_train))
        val_metrics = compute_metrics(y_val, engine.predict(X_val))
        warm_metrics.append({
            "model": spec["name"],
            "train_mae": train_metrics["mae"],
            "val_mae": val_metrics["mae"],
            "val_rmse": val_metrics["rmse"],
            "best_epoch": int(best_epoch),
            "epochs_run": int(epochs_run),
            "training_time_seconds": float(training_time),
        })
        print(
            f"  {spec['name']} val_mae={val_metrics['mae']:.6f} "
            f"val_rmse={val_metrics['rmse']:.6f}"
        )

    warm_metrics = pd.DataFrame(warm_metrics)
    warm_histories = pd.concat(warm_histories, ignore_index=True)
    warm_metrics.to_csv(output_dir / "warm_start_metrics.csv", index=False)
    warm_histories.to_csv(
        output_dir / "warm_start_histories.csv",
        index=False,
    )
    plot_learning_curves(
        warm_histories,
        output_dir / "learning_curves.png",
    )

    results_rows = []
    results_path = output_dir / "results_all_levels.csv"

    for level in levels:
        revealed_by_user = {}
        all_revealed = []
        for user in split["test_users"]:
            pool = split["pool_order_by_user"][int(user)]
            k = int(np.floor(float(level) * len(pool)))
            revealed = pool[:k]
            revealed_by_user[int(user)] = revealed
            all_revealed.extend(revealed.tolist())
        all_revealed = np.asarray(
            sorted(set(all_revealed)),
            dtype=np.int64,
        )
        n_revealed = int(len(all_revealed))
        augmented_train = np.concatenate([
            split["train_indices"],
            all_revealed,
        ]) if n_revealed else split["train_indices"]
        augmented_train = np.sort(np.unique(augmented_train))

        for spec_index, spec in enumerate(model_specs):
            model_name = spec["name"]
            X_all = matrix_for_spec(
                features,
                spec,
                len(observations),
            )
            warm_engine = warm_engines[model_name]
            model_seed = seed + spec_index * 1000
            level_tag = f"p{int(round(float(level) * 100)):03d}"

            local_predictions, adapted, local_time = run_local_adaptation(
                spec,
                warm_engine,
                X_all,
                y_all,
                eval_indices,
                split["eval_rows_by_user"],
                split["eval_positions_by_user"],
                revealed_by_user,
                features["context_family_indices"],
                float(args.reg_bias),
                float(args.reg_factors),
            )
            local_metrics = compute_metrics(y_eval, local_predictions)
            save_local_parameters(
                models_dir / "local" / f"{model_name}_{level_tag}.npz",
                all_user_ids,
                adapted,
                n_factors,
            )
            results_rows.append({
                "model": model_name,
                "strategy": "local",
                "level": float(level),
                "n_revealed": n_revealed,
                "n_users": int(len(test_users)),
                "mae": local_metrics["mae"],
                "rmse": local_metrics["rmse"],
                "bias": local_metrics["bias"],
                "std_pred": local_metrics["std_pred"],
                "time_seconds": float(local_time),
            })

            revealed_counts = {
                int(user): len(revealed_by_user[int(user)])
                for user in split["test_users"]
            }
            if level == 0.0:
                retrain_engine = warm_engine
                retrain_time = 0.0
                users_to_zero = test_users.tolist()
            else:
                X_augmented = X_all[augmented_train]
                y_augmented = y_all[augmented_train]
                (
                    retrain_engine,
                    retrain_history,
                    retrain_best_epoch,
                    retrain_epochs,
                    retrain_time,
                ) = train_fm_model(
                    X_augmented,
                    y_augmented,
                    X_all[split["val_indices"]],
                    y_all[split["val_indices"]],
                    n_features=spec["n_features"],
                    n_factors=n_factors,
                    use_linear=spec["use_linear"],
                    use_latent=spec["use_latent"],
                    active_mask=spec["active_mask"],
                    learning_rate=learning_rate,
                    reg_bias=float(args.reg_bias),
                    reg_factors=float(args.reg_factors),
                    n_epochs=n_epochs,
                    batch_size=batch_size,
                    seed=model_seed,
                    patience=int(args.patience),
                )
                save_model(
                    models_dir / "retrain" / f"{model_name}_{level_tag}.npz",
                    spec,
                    retrain_engine,
                )
                users_to_zero = [
                    int(user)
                    for user in split["test_users"]
                    if revealed_counts[int(user)] == 0
                ]

            retrain_predictions = predict_with_zeroed_users(
                spec,
                retrain_engine,
                X_all,
                eval_indices,
                users_to_zero,
                features["context_family_indices"],
            )
            retrain_metrics = compute_metrics(y_eval, retrain_predictions)
            if level == 0.0:
                p0_difference = float(
                    np.max(np.abs(local_predictions - retrain_predictions))
                )
                if p0_difference > 1e-8:
                    raise ValueError(
                        f"p=0 local/retrain mismatch for {model_name}: "
                        f"{p0_difference}"
                    )
            results_rows.append({
                "model": model_name,
                "strategy": "retrain",
                "level": float(level),
                "n_revealed": n_revealed,
                "n_users": int(len(test_users)),
                "mae": retrain_metrics["mae"],
                "rmse": retrain_metrics["rmse"],
                "bias": retrain_metrics["bias"],
                "std_pred": retrain_metrics["std_pred"],
                "time_seconds": float(retrain_time),
            })
            pd.DataFrame(results_rows).to_csv(
                results_path,
                index=False,
            )

    results = pd.DataFrame(results_rows)
    results.to_csv(results_path, index=False)
    results_5 = results[np.isclose(results["level"], 0.05)]
    results_5.to_csv(output_dir / "comparison_5pct.csv", index=False)
    delta = compute_delta_table(results)
    delta.to_csv(output_dir / "delta_mae.csv", index=False)

    plot_strategy_curves(
        results,
        output_dir / "mae_vs_revealed.png",
        "mae",
        "MAE",
    )
    plot_strategy_curves(
        results,
        output_dir / "rmse_vs_revealed.png",
        "rmse",
        "RMSE",
    )
    plot_strategy_curves(
        results,
        output_dir / "time_vs_revealed.png",
        "time_seconds",
        "Temps (s)",
        log_y=True,
    )

    print("Best-model diagnostics...")
    best_row = results.sort_values("mae").iloc[0]
    best_model = str(best_row["model"])
    best_strategy = str(best_row["strategy"])
    best_level = float(best_row["level"])
    best_revealed_by_user = {}
    for user in split["test_users"]:
        pool = split["pool_order_by_user"][int(user)]
        k = int(np.floor(best_level * len(pool)))
        best_revealed_by_user[int(user)] = pool[:k]
    best_spec = next(
        spec for spec in model_specs if spec["name"] == best_model
    )
    X_best = matrix_for_spec(
        features,
        best_spec,
        len(observations),
    )
    level_tag = f"p{int(round(best_level * 100)):03d}"

    if best_strategy == "retrain":
        if best_level == 0.0:
            best_spec_loaded, best_engine = load_model(
                models_dir / "warm_start" / f"{best_model}.npz"
            )
        else:
            best_spec_loaded, best_engine = load_model(
                models_dir / "retrain" / f"{best_model}_{level_tag}.npz"
            )
        users_to_zero = [
            int(user)
            for user in split["test_users"]
            if len(best_revealed_by_user[int(user)]) == 0
        ] if best_level > 0.0 else test_users.tolist()
        best_predictions = predict_with_zeroed_users(
            best_spec_loaded,
            best_engine,
            X_best,
            eval_indices,
            users_to_zero,
            features["context_family_indices"],
        )
        local_path = None
    else:
        best_spec_loaded, best_engine = load_model(
            models_dir / "warm_start" / f"{best_model}.npz"
        )
        local_path = (
            models_dir / "local" / f"{best_model}_{level_tag}.npz"
        )
        best_predictions, _, _ = run_local_adaptation(
            best_spec_loaded,
            best_engine,
            X_best,
            y_all,
            eval_indices,
            split["eval_rows_by_user"],
            split["eval_positions_by_user"],
            best_revealed_by_user,
            features["context_family_indices"],
            float(args.reg_bias),
            float(args.reg_factors),
        )

    n_best_revealed_by_user = {
        int(user): len(best_revealed_by_user[int(user)])
        for user in split["test_users"]
    }
    aug_indices = split["train_indices"]
    if any(len(v) for v in best_revealed_by_user.values()):
        aug_indices = np.sort(np.unique(np.concatenate([
            split["train_indices"],
            np.concatenate(list(best_revealed_by_user.values())),
        ])))
    film_counts = np.bincount(
        observations.iloc[aug_indices]["col"].to_numpy(dtype=np.int64),
        minlength=data["n_movies"],
    )
    eval_frame = pd.DataFrame({
        "row": observations.iloc[eval_indices]["row"].to_numpy(dtype=np.int64),
        "col": observations.iloc[eval_indices]["col"].to_numpy(dtype=np.int64),
        "rating_true": y_eval,
        "rating_est": best_predictions,
    })
    eval_frame["error"] = eval_frame["rating_est"] - eval_frame["rating_true"]
    eval_frame["abs_error"] = eval_frame["error"].abs()
    eval_frame["user_activity"] = eval_frame["row"].map(n_best_revealed_by_user)
    eval_frame["film_popularity"] = film_counts[eval_frame["col"].to_numpy()]

    plot_best_diagnostics(
        eval_frame,
        output_dir / "best_model_diagnostics.png",
        output_dir / "best_model_diagnostics.csv",
    )

    print("Parameter diagnostics...")
    if best_strategy == "local":
        parameter_summary = compute_parameter_summary(
            best_spec_loaded,
            best_engine,
            features,
            data["n_users"],
            data["n_movies"],
            local_path=local_path,
            context_family_indices=features["context_family_indices"],
            test_users=test_users,
        )
    else:
        parameter_summary = compute_parameter_summary(
            best_spec_loaded,
            best_engine,
            features,
            data["n_users"],
            data["n_movies"],
            local_path=None,
            context_family_indices=features["context_family_indices"],
        )
    parameter_summary.to_csv(
        output_dir / "parameter_summary.csv",
        index=False,
    )

    latent_contribution = compute_latent_contribution(
        best_spec_loaded,
        best_engine,
        X_best[eval_indices],
        local_path=local_path,
        context_family_indices=features["context_family_indices"],
    )
    latent_summary = pd.DataFrame([{
        "model": best_model,
        "strategy": best_strategy,
        "level": best_level,
        "mean_abs_latent_contribution": float(
            np.abs(latent_contribution).mean()
        ),
        "p95_abs_latent_contribution": float(
            np.quantile(np.abs(latent_contribution), 0.95)
        ),
    }])
    latent_summary.to_csv(
        output_dir / "latent_contribution_summary.csv",
        index=False,
    )

    figure, axes = plt.subplots(1, 3, figsize=(18, 5))
    for axis, column, title in [
        (axes[0], "mean_abs_bias", "Biais absolu moyen"),
        (axes[1], "mean_latent_norm", "Norme latente moyenne"),
        (axes[2], "p95_abs_bias", "P95 biais absolu"),
    ]:
        axis.bar(
            parameter_summary["family"],
            parameter_summary[column],
        )
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=30)
        axis.grid(True, axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(
        output_dir / "parameter_summary.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(figure)

    print("Done.")
    print(f"Outputs written to {output_dir}")


if __name__ == "__main__":
    main()




