#!/usr/bin/env python3

import argparse
import importlib.util
import json
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_NAME = "M5"
DEFAULT_LEVELS = (0, 1, 2, 5, 10, 20, 50, 100)
DEFAULT_BIAS_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0)
DEFAULT_FACTOR_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0)
FALLBACK_BIAS = 1e-2
FALLBACK_FACTORS = 1e-1


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the contextual FM M5 (warm and cold start)."
    )
    parser.add_argument("--cleaning-dir", default=None)
    parser.add_argument("--model-output-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--previous-script",
        default=str(
            Path(__file__).with_name("6.IncrementalVsRetraining.py")
        ),
    )
    parser.add_argument("--context-fm-config", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--model-seed", type=int, default=None)
    parser.add_argument("--n-factors", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--reg-bias", type=float, default=None)
    parser.add_argument("--reg-factors", type=float, default=None)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--cold-eval-fraction", type=float, default=0.2)
    parser.add_argument("--reveal-levels", default=None)
    parser.add_argument("--reveal-seeds", type=int, default=5)
    parser.add_argument("--min-fixed-eval-notes", type=int, default=2)
    parser.add_argument("--calibration-fraction", type=float, default=0.1)
    parser.add_argument("--calibration-max-users", type=int, default=400)
    parser.add_argument("--calibration-seeds", type=int, default=None)
    parser.add_argument("--adapt-reg-bias", type=float, default=None)
    parser.add_argument("--adapt-reg-factors", type=float, default=None)
    parser.add_argument("--adapt-reg-bias-grid", default=None)
    parser.add_argument("--adapt-reg-factors-grid", default=None)
    parser.add_argument("--skip-calibration", action="store_true")
    parser.add_argument("--load-model", default=None)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--max-users", type=int, default=None)
    parser.add_argument("--max-interactions-per-user", type=int, default=None)
    return parser.parse_args()


def parse_float_list(value, fallback):
    if value is None:
        return list(fallback)
    items = [
        float(item)
        for item in str(value).split(",")
        if str(item).strip()
    ]
    if not items:
        return list(fallback)
    return sorted(set(items))


def parse_int_list(value, fallback):
    if value is None:
        return sorted(set(int(item) for item in fallback))
    items = [
        int(float(item))
        for item in str(value).split(",")
        if str(item).strip()
    ]
    if not items:
        return sorted(set(int(item) for item in fallback))
    return sorted(set(items))


def load_previous_module(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Previous script not found: {path}")
    spec = importlib.util.spec_from_file_location(
        "filmfinder_incremental",
        path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load previous script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def try_load_model_config(prev, model_output_dir):
    try:
        return prev.load_model_config(model_output_dir)
    except (FileNotFoundError, KeyError, ValueError):
        return {"hyperparameters": {}, "split": {}, "feature_config": {}}


def find_context_fm_config(explicit, model_output_dir):
    candidates = []
    if explicit is not None:
        candidates.append(Path(explicit))
    candidates.append(model_output_dir / "context_fm" / "experiment_config.json")
    candidates.append(
        Path.cwd() / "results" / "context_fm" / "experiment_config.json"
    )
    for path in candidates:
        if path.exists():
            with open(path) as handle:
                return json.load(handle), path
    return {}, None


def pick(value, best, fallback_hyper, key, default):
    if value is not None:
        return value
    if key in best:
        return best[key]
    if key in fallback_hyper:
        return fallback_hyper[key]
    return default


def resolve_settings(args, model_config, context_config):
    hyper = model_config.get("hyperparameters", {}) or {}
    split_context = model_config.get("split", {}) or {}
    feature_config = model_config.get("feature_config", {}) or {}
    best_models = context_config.get("best_models", {}) or {}
    best = best_models.get("C4", {}) or best_models.get("M5", {}) or {}

    split_seed = int(
        args.seed
        if args.seed is not None
        else split_context.get("seed", hyper.get("seed", 42))
    )
    model_seed = int(
        args.model_seed
        if args.model_seed is not None
        else context_config.get("model_seed", split_seed + 4000)
    )
    requested_factors = args.n_factors
    requested_epochs = args.epochs
    if args.fast and requested_factors is None:
        requested_factors = 4
    if args.fast and requested_epochs is None:
        requested_epochs = 5
    n_factors = int(
        pick(
            requested_factors,
            best,
            {},
            "n_factors",
            8,
        )
    )
    learning_rate = float(
        pick(args.learning_rate, best, {}, "learning_rate", 0.01)
    )
    n_epochs = int(
        pick(
            requested_epochs,
            best,
            {},
            "n_epochs",
            20,
        )
    )
    batch_size = int(
        pick(args.batch_size, best, {}, "batch_size", 65536)
    )
    reg_bias = float(
        pick(args.reg_bias, best, {}, "reg_bias", 1e-7)
    )
    reg_factors = float(
        pick(args.reg_factors, best, {}, "reg_factors", 1e-7)
    )
    max_studios = int(
        feature_config.get(
            "max_studios",
            context_config.get("max_studios", 4000),
        )
    )
    return {
        "split_seed": split_seed,
        "model_seed": model_seed,
        "n_factors": n_factors,
        "learning_rate": learning_rate,
        "n_epochs": n_epochs,
        "batch_size": batch_size,
        "reg_bias": reg_bias,
        "reg_factors": reg_factors,
        "max_studios": max_studios,
    }


def user_index_map(row_values):
    frame = pd.DataFrame({"row": np.asarray(row_values, dtype=np.int64)})
    return {
        int(user): np.asarray(indices, dtype=np.int64)
        for user, indices in frame.groupby("row").indices.items()
    }


def select_calibration_users(
    observations,
    known_users,
    fraction,
    max_users,
    seed,
):
    if fraction <= 0 or known_users.size == 0:
        return np.zeros(0, dtype=np.int64)
    rows = observations["row"].to_numpy(dtype=np.int64)
    counts = np.bincount(rows, minlength=int(rows.max()) + 1)
    candidates = known_users[counts[known_users] >= 2]
    if candidates.size < 3:
        return np.zeros(0, dtype=np.int64)
    n_selected = int(round(float(fraction) * candidates.size))
    n_selected = max(1, min(n_selected, int(max_users), candidates.size - 1))
    rng = np.random.default_rng(seed + 777)
    selected = rng.choice(candidates, size=n_selected, replace=False)
    return np.sort(selected.astype(np.int64))


def remove_user_indices(indices, user_rows, user_ids):
    if user_ids.size == 0:
        return np.asarray(indices, dtype=np.int64)
    mask = ~np.isin(user_rows[indices], user_ids)
    return np.asarray(indices, dtype=np.int64)[mask]


def split_warm_validation(val_indices, user_rows, seed):
    early_stop = []
    warm_eval = []
    if val_indices.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    groups = user_index_map(user_rows[val_indices])
    for user, local_indices in groups.items():
        global_indices = np.asarray(
            val_indices[np.asarray(local_indices, dtype=np.int64)],
            dtype=np.int64,
        )
        rng = np.random.default_rng(seed + 600000 + int(user))
        ordered = global_indices[rng.permutation(global_indices.size)]
        n_warm = ordered.size // 2
        if n_warm == 0:
            if int(user) % 2 == 0:
                warm_eval.extend(ordered.tolist())
            else:
                early_stop.extend(ordered.tolist())
        else:
            warm_eval.extend(ordered[:n_warm].tolist())
            early_stop.extend(ordered[n_warm:].tolist())
    return (
        np.asarray(sorted(early_stop), dtype=np.int64),
        np.asarray(sorted(warm_eval), dtype=np.int64),
    )


def build_cold_structures(
    observations,
    users,
    cold_eval_fraction,
    seed,
    offset,
):
    rows = observations["row"].to_numpy(dtype=np.int64)
    groups = user_index_map(rows)
    eval_rows_by_user = {}
    pool_by_user = {}
    for user in np.asarray(users, dtype=np.int64):
        user = int(user)
        indices = groups.get(user)
        if indices is None or indices.size < 2:
            continue
        rng = np.random.default_rng(seed + int(offset) + user)
        ordered = indices[rng.permutation(indices.size)]
        n_eval = max(
            1,
            int(round(float(cold_eval_fraction) * ordered.size)),
        )
        n_eval = min(n_eval, ordered.size - 1)
        eval_rows_by_user[user] = np.sort(ordered[:n_eval])
        pool_by_user[user] = ordered[n_eval:]
    return eval_rows_by_user, pool_by_user


def reveal_permutation(n_items, seed_index, user, base_seed):
    if n_items <= 1 or int(seed_index) == 0:
        return np.arange(n_items, dtype=np.int64)
    rng = np.random.default_rng(
        int(base_seed) + 1000003 * int(seed_index) + int(user)
    )
    return rng.permutation(n_items)


def prepare_rows(engine, X_all, rows, feature_index):
    rows = np.asarray(rows, dtype=np.int64)
    if rows.size == 0:
        return np.zeros(0), np.zeros((0, engine.n_factors))
    X_rows = X_all[rows]
    user_index = int(feature_index)
    user_vector = engine.V[user_index]
    S = np.asarray(X_rows @ engine.V, dtype=np.float64) - user_vector
    prediction = engine.predict(X_rows)
    prediction = prediction - engine.w[user_index] - S @ user_vector
    return prediction, S


def build_ridge_system(base, S, y):
    target = y - base
    A = np.column_stack([np.ones(target.size), S])
    return A, target


def solve_ridge(A, target, lambda_bias, lambda_factors):
    regularizer = np.full(A.shape[1], float(lambda_factors))
    regularizer[0] = float(lambda_bias)
    gram = A.T @ A
    gram[np.diag_indices_from(gram)] += regularizer
    try:
        return np.linalg.solve(gram, A.T @ target)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(A, target, rcond=None)[0]


def solve_ridge_system(gram_base, rhs, A, target, lambda_bias, lambda_factors):
    regularizer = np.full(gram_base.shape[0], float(lambda_factors))
    regularizer[0] = float(lambda_bias)
    gram = gram_base.copy()
    gram[np.diag_indices_from(gram)] += regularizer
    try:
        return np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(A, target, rcond=None)[0]


def apply_profile(base, S, theta):
    return base + theta[0] + S @ theta[1:]


def safe_nanmean(values):
    values = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(values)
    if not valid.any():
        return float("nan")
    return float(values[valid].mean())


def safe_nanstd(values):
    values = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(values)
    if valid.sum() < 2:
        return 0.0
    return float(values[valid].std(ddof=1))


def log_bin_edges(values, n_bins=12):
    values = np.asarray(values, dtype=np.float64)
    vmax = int(np.max(values))
    if vmax <= 1:
        return np.array([0, 2], dtype=np.int64)
    raw = np.geomspace(1.0, float(vmax + 1), max(3, int(n_bins)))
    edges = np.unique(np.rint(raw).astype(np.int64))
    edges = edges[edges >= 1]
    edges = np.concatenate([[0], edges])
    if edges.size < 2:
        edges = np.array([0, vmax + 1], dtype=np.int64)
    return edges


def warm_start_bins(per_user, note_counts):
    if per_user.empty:
        return pd.DataFrame(
            columns=[
                "bin",
                "center",
                "n_users",
                "n_observations",
                "mean_user_mae",
                "pooled_mae",
            ]
        )
    edges = log_bin_edges(per_user["n_train"].to_numpy())
    interior = edges[1:-1]
    per_user = per_user.copy()
    per_user["bin"] = np.digitize(
        per_user["n_train"].to_numpy(),
        interior,
        right=False,
    )
    rows = []
    for bin_index, group in per_user.groupby("bin"):
        user_rows = group.index.to_numpy(dtype=np.int64)
        n_obs = int(sum(int(note_counts[user]) for user in user_rows))
        rows.append({
            "bin": int(bin_index),
            "center": float(
                np.exp(np.mean(np.log(group["n_train"].to_numpy())))
            ),
            "n_users": int(group.shape[0]),
            "n_observations": n_obs,
            "mean_user_mae": float(group["mae"].mean()),
            "pooled_mae": float(
                np.average(
                    group["mae"].to_numpy(),
                    weights=group["n_eval"].to_numpy(),
                )
            ),
        })
    return pd.DataFrame(rows).sort_values("center").reset_index(drop=True)


def evaluate_warm_start(
    prev,
    engine,
    X_all,
    y_all,
    observations,
    train_indices,
    warm_eval_indices,
    n_users,
):
    if warm_eval_indices.size == 0:
        raise ValueError("No warm-start evaluation notes available.")
    rows = observations["row"].to_numpy(dtype=np.int64)
    train_counts = np.bincount(
        rows[train_indices],
        minlength=int(n_users),
    )
    eligible = train_counts[rows[warm_eval_indices]] >= 1
    warm_eval_indices = warm_eval_indices[eligible]
    if warm_eval_indices.size == 0:
        raise ValueError("No warm-start users with train interactions.")
    predictions = engine.predict(X_all[warm_eval_indices])
    y_true = y_all[warm_eval_indices]
    metrics = prev.compute_metrics(y_true, predictions)
    frame = pd.DataFrame({
        "user_row": rows[warm_eval_indices],
        "abs_error": np.abs(predictions - y_true),
    })
    per_user = frame.groupby("user_row").agg(
        mae=("abs_error", "mean"),
        n_eval=("abs_error", "size"),
    )
    per_user["n_train"] = train_counts[
        per_user.index.to_numpy(dtype=np.int64)
    ]
    if per_user.empty:
        raise ValueError("No warm-start users with train interactions.")
    note_counts = per_user["n_eval"].astype(int).to_dict()
    bins = warm_start_bins(per_user, note_counts)
    summary = {
        "mae": float(metrics["mae"]),
        "rmse": float(metrics["rmse"]),
        "bias": float(metrics["bias"]),
        "n_users": int(per_user.shape[0]),
        "n_observations": int(per_user["n_eval"].sum()),
        "median_train_notes": float(per_user["n_train"].median()),
        "mean_train_notes": float(per_user["n_train"].mean()),
    }
    return per_user.reset_index(), bins, summary


def prepare_cold_users(
    prev,
    engine,
    X_all,
    y_all,
    spec,
    eval_rows_by_user,
    pool_by_user,
    context_family_indices,
):
    prepared = {}
    for user in sorted(pool_by_user):
        user = int(user)
        pool_rows = np.asarray(pool_by_user[user], dtype=np.int64)
        eval_rows = np.asarray(
            eval_rows_by_user.get(user, []),
            dtype=np.int64,
        )
        if pool_rows.size == 0 or eval_rows.size == 0:
            continue
        feature_index = prev.user_feature_index(
            spec,
            user,
            context_family_indices,
        )
        base_pool, S_pool = prepare_rows(
            engine,
            X_all,
            pool_rows,
            feature_index,
        )
        base_eval, S_eval = prepare_rows(
            engine,
            X_all,
            eval_rows,
            feature_index,
        )
        prepared[user] = {
            "pool_rows": pool_rows,
            "eval_rows": eval_rows,
            "y_pool": y_all[pool_rows],
            "y_eval": y_all[eval_rows],
            "base_pool": base_pool,
            "S_pool": S_pool,
            "base_eval": base_eval,
            "S_eval": S_eval,
        }
    return prepared


def calibrate_ridge(
    prev,
    engine,
    X_all,
    y_all,
    spec,
    eval_rows_by_user,
    pool_by_user,
    context_family_indices,
    levels,
    seeds,
    bias_grid,
    factor_grid,
    seed_base,
):
    prepared = prepare_cold_users(
        prev,
        engine,
        X_all,
        y_all,
        spec,
        eval_rows_by_user,
        pool_by_user,
        context_family_indices,
    )
    positive_levels = [int(level) for level in levels if int(level) > 0]
    if not prepared or not positive_levels:
        return (
            float(FALLBACK_BIAS),
            float(FALLBACK_FACTORS),
            pd.DataFrame(),
            {"users": 0, "reason": "insufficient calibration data"},
        )
    n_bias = len(bias_grid)
    n_factors_grid = len(factor_grid)
    n_levels = len(positive_levels)
    absolute_sums = np.zeros((n_bias, n_factors_grid, n_levels, seeds))
    counts = np.zeros_like(absolute_sums)
    for user, data in prepared.items():
        n_pool = data["pool_rows"].size
        for seed_index in range(int(seeds)):
            permutation = reveal_permutation(
                n_pool,
                seed_index,
                user,
                seed_base,
            )
            for level_index, level in enumerate(positive_levels):
                if level > n_pool:
                    continue
                chosen = permutation[:level]
                A, target = build_ridge_system(
                    data["base_pool"][chosen],
                    data["S_pool"][chosen],
                    data["y_pool"][chosen],
                )
                gram = A.T @ A
                rhs = A.T @ target
                for bias_index, lambda_bias in enumerate(bias_grid):
                    for factor_index, lambda_factors in enumerate(
                        factor_grid
                    ):
                        theta = solve_ridge_system(
                            gram,
                            rhs,
                            A,
                            target,
                            lambda_bias,
                            lambda_factors,
                        )
                        predictions = apply_profile(
                            data["base_eval"],
                            data["S_eval"],
                            theta,
                        )
                        residual = predictions - data["y_eval"]
                        absolute_sums[
                            bias_index,
                            factor_index,
                            level_index,
                            seed_index,
                        ] += float(np.abs(residual).sum())
                        counts[
                            bias_index,
                            factor_index,
                            level_index,
                            seed_index,
                        ] += int(residual.size)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_mae = np.where(
            counts > 0,
            absolute_sums / counts,
            np.nan,
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        per_level = np.nanmean(mean_mae, axis=3)
        objective = np.nanmean(per_level, axis=2)
    if np.all(np.isnan(objective)):
        return (
            float(FALLBACK_BIAS),
            float(FALLBACK_FACTORS),
            pd.DataFrame(),
            {"users": len(prepared), "reason": "no valid calibration level"},
        )
    flat_index = int(np.nanargmin(objective))
    bias_index, factor_index = np.unravel_index(
        flat_index,
        objective.shape,
    )
    selected_bias = float(bias_grid[bias_index])
    selected_factors = float(factor_grid[factor_index])
    rows = []
    for b_index, lambda_bias in enumerate(bias_grid):
        for f_index, lambda_factors in enumerate(factor_grid):
            for level_index, level in enumerate(positive_levels):
                mae_by_seed = mean_mae[b_index, f_index, level_index, :]
                rows.append({
                    "reg_bias": float(lambda_bias),
                    "reg_factors": float(lambda_factors),
                    "n_revealed": int(level),
                    "mae_mean": safe_nanmean(mae_by_seed),
                    "mae_std": safe_nanstd(mae_by_seed),
                    "n_observations": int(
                        safe_nanmean(
                            counts[b_index, f_index, level_index, :]
                        )
                    ),
                    "objective": float(objective[b_index, f_index]),
                    "selected": bool(
                        b_index == bias_index
                        and f_index == factor_index
                    ),
                })
    summary = {
        "users": int(len(prepared)),
        "seeds": int(seeds),
        "levels": positive_levels,
        "reg_bias": selected_bias,
        "reg_factors": selected_factors,
        "objective": float(objective[bias_index, factor_index]),
    }
    return selected_bias, selected_factors, pd.DataFrame(rows), summary


def evaluate_cold_start(
    prev,
    engine,
    X_all,
    y_all,
    spec,
    eval_rows_by_user,
    pool_by_user,
    context_family_indices,
    levels,
    seeds,
    lambda_bias,
    lambda_factors,
    seed_base,
    min_fixed_eval_notes,
):
    prepared = prepare_cold_users(
        prev,
        engine,
        X_all,
        y_all,
        spec,
        eval_rows_by_user,
        pool_by_user,
        context_family_indices,
    )
    if not prepared:
        raise ValueError("No cold-start users available.")
    max_level = int(max(levels))
    fixed_users = {
        user
        for user, data in prepared.items()
        if data["pool_rows"].size >= max_level
        and data["eval_rows"].size >= int(min_fixed_eval_notes)
    }
    stats = {}
    for user, data in prepared.items():
        n_pool = data["pool_rows"].size
        n_factors = data["S_pool"].shape[1]
        for seed_index in range(int(seeds)):
            permutation = reveal_permutation(
                n_pool,
                seed_index,
                user,
                seed_base,
            )
            for level_index, level in enumerate(levels):
                n_revealed = int(level)
                if n_revealed > n_pool:
                    continue
                start = time.perf_counter()
                theta = np.zeros(n_factors + 1, dtype=np.float64)
                if n_revealed > 0:
                    chosen = permutation[:n_revealed]
                    A, target = build_ridge_system(
                        data["base_pool"][chosen],
                        data["S_pool"][chosen],
                        data["y_pool"][chosen],
                    )
                    theta = solve_ridge(
                        A,
                        target,
                        lambda_bias,
                        lambda_factors,
                    )
                predictions = apply_profile(
                    data["base_eval"],
                    data["S_eval"],
                    theta,
                )
                elapsed = time.perf_counter() - start
                residual = predictions - data["y_eval"]
                stats[(seed_index, level_index, user)] = (
                    float(np.abs(residual).sum()),
                    float(np.sum(residual ** 2)),
                    int(residual.size),
                    float(elapsed),
                )
    long_rows = []
    for cohort in ("variable", "fixed"):
        if cohort == "fixed" and not fixed_users:
            continue
        for seed_index in range(int(seeds)):
            for level_index, level in enumerate(levels):
                n_revealed = int(level)
                users = []
                for user, data in prepared.items():
                    if data["pool_rows"].size < n_revealed:
                        continue
                    if cohort == "fixed":
                        if user not in fixed_users:
                            continue
                    elif data["eval_rows"].size < 1:
                        continue
                    users.append(user)
                if not users:
                    continue
                sum_abs = 0.0
                sum_sq = 0.0
                n_observations = 0
                elapsed = 0.0
                for user in users:
                    entry = stats.get((seed_index, level_index, user))
                    if entry is None:
                        continue
                    sum_abs += entry[0]
                    sum_sq += entry[1]
                    n_observations += entry[2]
                    elapsed += entry[3]
                if n_observations == 0:
                    continue
                long_rows.append({
                    "cohort": cohort,
                    "seed_index": int(seed_index),
                    "n_revealed": n_revealed,
                    "n_users": int(len(users)),
                    "n_revealed_notes": int(n_revealed * len(users)),
                    "n_observations": int(n_observations),
                    "mae": float(sum_abs / n_observations),
                    "rmse": float(np.sqrt(sum_sq / n_observations)),
                    "adaptation_time_seconds": float(elapsed),
                    "adaptation_time_per_user": float(
                        elapsed / max(len(users), 1)
                    ),
                })
    long_frame = pd.DataFrame(long_rows)
    if long_frame.empty:
        raise ValueError(
            "No cold-start evaluation observations for these levels."
        )
    summary_rows = []
    for (cohort, n_revealed), group in long_frame.groupby(
        ["cohort", "n_revealed"]
    ):
        summary_rows.append({
            "cohort": cohort,
            "n_revealed": int(n_revealed),
            "n_users": int(group["n_users"].max()),
            "n_revealed_notes": int(group["n_revealed_notes"].max()),
            "n_observations": int(group["n_observations"].max()),
            "mae_mean": float(group["mae"].mean()),
            "mae_std": float(group["mae"].std(ddof=1))
            if group.shape[0] > 1
            else 0.0,
            "rmse_mean": float(group["rmse"].mean()),
            "rmse_std": float(group["rmse"].std(ddof=1))
            if group.shape[0] > 1
            else 0.0,
            "adaptation_time_mean": float(
                group["adaptation_time_seconds"].mean()
            ),
            "adaptation_time_per_user": float(
                group["adaptation_time_per_user"].mean()
            ),
            "n_seeds": int(group["seed_index"].nunique()),
        })
    return long_frame, pd.DataFrame(summary_rows)


def plot_learning_curves(history, output_path):
    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(
        history["epoch"],
        history["train_mae"],
        label="MAE train",
        linewidth=1.4,
    )
    axes[0].plot(
        history["epoch"],
        history["val_mae"],
        label="MAE validation (early-stop)",
        linewidth=1.4,
    )
    axes[0].set_title("MAE par époque")
    axes[0].set_xlabel("Époque")
    axes[0].set_ylabel("MAE")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        history["epoch"],
        history["train_loss"],
        label="Loss train",
        linewidth=1.4,
    )
    axes[1].plot(
        history["epoch"],
        history["val_rmse"],
        label="RMSE validation",
        linewidth=1.4,
    )
    axes[1].set_title("Loss et RMSE validation")
    axes[1].set_xlabel("Époque")
    axes[1].set_ylabel("Valeur")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def plot_warm_curve(per_user, bins, output_path):
    figure, axis = plt.subplots(figsize=(11, 6))
    axis.scatter(
        per_user["n_train"],
        per_user["mae"],
        s=8,
        alpha=0.15,
        color="tab:gray",
        label="Utilisateurs",
    )
    axis.plot(
        bins["center"],
        bins["mean_user_mae"],
        marker="o",
        color="tab:blue",
        linewidth=1.6,
        label="MAE moyenne par bin",
    )
    for _, row in bins.iterrows():
        axis.annotate(
            f"{int(row['n_users'])}u / {int(row['n_observations'])}o",
            (row["center"], row["mean_user_mae"]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=7,
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.75,
            },
        )
    axis.set_xscale("log")
    axis.set_title("Warm-start — MAE selon l'activité utilisateur")
    axis.set_xlabel("Nombre de notes de l'utilisateur dans le train")
    axis.set_ylabel("MAE sur les notes cachées")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def plot_cold_curve(cold_summary, output_path):
    figure, axis = plt.subplots(figsize=(11, 6))
    variable = cold_summary[cold_summary["cohort"] == "variable"]
    fixed = cold_summary[cold_summary["cohort"] == "fixed"]
    axis.errorbar(
        variable["n_revealed"],
        variable["mae_mean"],
        yerr=variable["mae_std"],
        marker="o",
        capsize=4,
        color="tab:red",
        label="Cohorte variable",
    )
    if not fixed.empty:
        axis.errorbar(
            fixed["n_revealed"],
            fixed["mae_mean"],
            yerr=fixed["mae_std"],
            marker="s",
            linestyle="--",
            capsize=4,
            color="tab:orange",
            label="Cohorte fixe",
        )
    for _, row in variable.iterrows():
        axis.annotate(
            f"{int(row['n_users'])}u / {int(row['n_revealed_notes'])} rév.",
            (row["n_revealed"], row["mae_mean"]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=7,
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.75,
            },
        )
    axis.set_xscale("symlog", linthresh=1)
    axis.set_title("Cold-start — MAE selon le nombre de notes révélées")
    axis.set_xlabel("Nombre de notes révélées")
    axis.set_ylabel("MAE sur les notes d'évaluation")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def plot_comparison(warm_bins, cold_summary, output_path, note):
    figure, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    axes[0].plot(
        warm_bins["center"],
        warm_bins["mean_user_mae"],
        marker="o",
        color="tab:blue",
        label="Warm-start (utilisateurs connus)",
    )
    axes[0].set_xscale("log")
    axes[0].set_title("Warm-start — utilisateurs connus")
    axes[0].set_xlabel("Notes de l'utilisateur dans le train")
    axes[0].set_ylabel("MAE")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    variable = cold_summary[cold_summary["cohort"] == "variable"]
    fixed = cold_summary[cold_summary["cohort"] == "fixed"]
    axes[1].errorbar(
        variable["n_revealed"],
        variable["mae_mean"],
        yerr=variable["mae_std"],
        marker="o",
        capsize=4,
        color="tab:red",
        label="Cold-start — cohorte variable",
    )
    if not fixed.empty:
        axes[1].errorbar(
            fixed["n_revealed"],
            fixed["mae_mean"],
            yerr=fixed["mae_std"],
            marker="s",
            linestyle="--",
            capsize=4,
            color="tab:orange",
            label="Cold-start — cohorte fixe",
        )
    axes[1].set_xscale("symlog", linthresh=1)
    axes[1].set_title("Cold-start — nouveaux utilisateurs")
    axes[1].set_xlabel("Notes révélées")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    figure.suptitle("MAE warm-start vs cold-start pour M5")
    figure.text(0.5, 0.01, note, ha="center", fontsize=9)
    figure.tight_layout(rect=(0, 0.04, 1, 0.96))
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def model_metadata_path(model_path):
    return model_path.with_name(model_path.stem + "_metadata.json")


def read_training_metadata(model_path):
    path = model_metadata_path(Path(model_path))
    if not path.exists():
        return {}
    with open(path) as handle:
        return json.load(handle)


def validate_loaded_model(spec, engine, expected_spec, context_mapping):
    expected_features = len(context_mapping)
    if int(spec["n_features"]) != expected_features:
        raise ValueError(
            "Loaded model feature count does not match the current split: "
            f"{spec['n_features']} != {expected_features}."
        )
    if int(engine.n_factors) <= 0:
        raise ValueError("Loaded model has no latent factors.")
    if not bool(spec["use_linear"]) or not bool(spec["use_latent"]):
        raise ValueError(
            "Loaded model is not a full linear + latent contextual FM."
        )
    if spec["feature_set"] != "context":
        raise ValueError("Loaded model does not use context features.")
    if spec["active_mask"].shape != expected_spec["active_mask"].shape:
        raise ValueError("Loaded active_mask shape does not match.")
    if not np.array_equal(
        spec["active_mask"],
        expected_spec["active_mask"],
    ):
        raise ValueError("Loaded active_mask does not match the M5 spec.")


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
        else model_output_dir / "evaluate_m5"
    )
    models_dir = output_dir / "model"
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    print("Loading previous implementation...")
    prev = load_previous_module(args.previous_script)
    model_config = try_load_model_config(prev, model_output_dir)
    context_config, context_config_path = find_context_fm_config(
        args.context_fm_config,
        model_output_dir,
    )
    settings = resolve_settings(args, model_config, context_config)
    levels = parse_int_list(args.reveal_levels, DEFAULT_LEVELS)
    if not levels or any(level < 0 for level in levels):
        raise ValueError("Reveal levels must be non-negative integers.")
    if int(args.reveal_seeds) < 1:
        raise ValueError("--reveal-seeds must be at least 1.")
    if not 0 < float(args.cold_eval_fraction) < 1:
        raise ValueError("--cold-eval-fraction must be in (0, 1).")
    bias_grid = parse_float_list(
        args.adapt_reg_bias_grid,
        DEFAULT_BIAS_GRID,
    )
    factor_grid = parse_float_list(
        args.adapt_reg_factors_grid,
        DEFAULT_FACTOR_GRID,
    )
    calibration_seeds = int(
        args.calibration_seeds
        if args.calibration_seeds is not None
        else min(3, int(args.reveal_seeds))
    )
    if calibration_seeds < 1:
        raise ValueError("--calibration-seeds must be at least 1.")
    max_users = args.max_users if args.max_users is not None else (
        200 if args.fast else None
    )
    max_interactions = (
        args.max_interactions_per_user
        if args.max_interactions_per_user is not None
        else (50 if args.fast else None)
    )

    print("Loading inputs...")
    data = prev.load_inputs(cleaning_dir)
    observations = prev.cap_observations(
        data["observations"],
        max_users,
        max_interactions,
        settings["split_seed"],
    )
    if len(observations) < 10:
        raise ValueError("Not enough observations after filtering.")

    print("Building cold-start split...")
    split = prev.split_users(
        observations,
        settings["split_seed"],
        float(args.cold_eval_fraction),
    )
    user_rows = observations["row"].to_numpy(dtype=np.int64)
    y_all = observations["rating"].to_numpy(dtype=np.float64)
    known_mask = np.zeros(len(observations), dtype=bool)
    known_mask[split["train_indices"]] = True
    known_mask[split["val_indices"]] = True
    known_users = np.unique(user_rows[known_mask])

    calibration_users = select_calibration_users(
        observations,
        known_users,
        float(args.calibration_fraction),
        int(args.calibration_max_users),
        settings["split_seed"],
    )
    train_indices = remove_user_indices(
        split["train_indices"],
        user_rows,
        calibration_users,
    )
    val_indices = remove_user_indices(
        split["val_indices"],
        user_rows,
        calibration_users,
    )
    if train_indices.size < 10 or val_indices.size < 4:
        raise ValueError(
            "Not enough training or validation interactions after "
            "removing calibration users."
        )
    train_users = np.unique(user_rows[train_indices])
    test_users = np.asarray(split["test_users"], dtype=np.int64)
    if np.intersect1d(calibration_users, test_users).size:
        raise ValueError("Calibration and test users overlap.")
    if train_users.size and np.intersect1d(train_users, test_users).size:
        raise ValueError("Test users leaked into the training set.")
    if np.intersect1d(train_users, calibration_users).size:
        raise ValueError("Calibration users leaked into the training set.")
    early_stop_indices, warm_eval_indices = split_warm_validation(
        val_indices,
        user_rows,
        settings["split_seed"],
    )
    train_user_counts = np.bincount(
        user_rows[train_indices],
        minlength=int(data["n_users"]),
    )
    early_stop_mask = train_user_counts[user_rows[early_stop_indices]] >= 1
    early_stop_indices = early_stop_indices[early_stop_mask]
    warm_eval_mask = train_user_counts[user_rows[warm_eval_indices]] >= 1
    warm_eval_indices = warm_eval_indices[warm_eval_mask]
    if early_stop_indices.size == 0 or warm_eval_indices.size == 0:
        raise ValueError(
            "No warm-start users remain after restricting to users "
            "present in training."
        )

    print("Building contextual features...")
    features = prev.build_features(
        observations,
        data["movie_context"],
        data["user_mapping"],
        data["movie_mapping"],
        data["n_users"],
        data["n_movies"],
        train_indices,
        settings["max_studios"],
    )
    specs = prev.make_model_specs(
        data["n_users"],
        data["n_movies"],
        len(features["context_mapping"]),
    )
    spec = next(item for item in specs if item["name"] == MODEL_NAME)
    X_all = prev.matrix_for_spec(features, spec, len(observations))

    metadata = {}
    training_time = np.nan
    epochs_run = 0
    best_epoch = 0
    load_time = 0.0
    if args.load_model is not None:
        print(f"Loading model from {args.load_model}...")
        load_start = time.perf_counter()
        loaded_spec, engine = prev.load_model(Path(args.load_model))
        load_time = time.perf_counter() - load_start
        validate_loaded_model(
            loaded_spec,
            engine,
            spec,
            features["context_mapping"],
        )
        spec = loaded_spec
        settings["n_factors"] = int(engine.n_factors)
        metadata = read_training_metadata(Path(args.load_model))
        if not metadata:
            print(
                "  Warning: no metadata sidecar. The checkpoint must share "
                "the current feature layout and year standardization."
            )
        if "training_time_seconds" in metadata:
            training_time = float(metadata["training_time_seconds"])
        if "best_epoch" in metadata:
            best_epoch = int(metadata["best_epoch"])
        if "epochs_run" in metadata:
            epochs_run = int(metadata["epochs_run"])
        print(f"  Loaded in {load_time:.3f}s.")
        if np.isfinite(training_time):
            print(
                "  Recorded training time: "
                f"{training_time:.3f}s ({epochs_run} epochs)."
            )
        else:
            print("  No recorded training time for this checkpoint.")
        history = pd.DataFrame()
    else:
        print("Training M5 once...")
        (
            engine,
            history,
            best_epoch,
            epochs_run,
            training_time,
        ) = prev.train_fm_model(
            X_all[train_indices],
            y_all[train_indices],
            X_all[early_stop_indices],
            y_all[early_stop_indices],
            n_features=spec["n_features"],
            n_factors=settings["n_factors"],
            use_linear=spec["use_linear"],
            use_latent=spec["use_latent"],
            active_mask=spec["active_mask"],
            learning_rate=settings["learning_rate"],
            reg_bias=settings["reg_bias"],
            reg_factors=settings["reg_factors"],
            n_epochs=settings["n_epochs"],
            batch_size=settings["batch_size"],
            seed=settings["model_seed"],
            patience=int(args.patience),
        )
        model_path = models_dir / f"{MODEL_NAME}.npz"
        prev.save_model(model_path, spec, engine)
        metadata = {
            "model_name": MODEL_NAME,
            "training_time_seconds": float(training_time),
            "mean_epoch_time_seconds": float(
                training_time / max(int(epochs_run), 1)
            ),
            "best_epoch": int(best_epoch),
            "epochs_run": int(epochs_run),
            "seed": int(settings["model_seed"]),
            "n_factors": int(settings["n_factors"]),
            "learning_rate": float(settings["learning_rate"]),
            "reg_bias": float(settings["reg_bias"]),
            "reg_factors": float(settings["reg_factors"]),
            "batch_size": int(settings["batch_size"]),
            "n_epochs": int(settings["n_epochs"]),
            "n_features": int(spec["n_features"]),
        }
        with open(model_metadata_path(model_path), "w") as handle:
            json.dump(metadata, handle, indent=2)
        history.to_csv(output_dir / "training_history.csv", index=False)
        plot_learning_curves(
            history,
            output_dir / "learning_curves.png",
        )
        train_metrics = prev.compute_metrics(
            y_all[train_indices],
            engine.predict(X_all[train_indices]),
        )
        val_metrics = prev.compute_metrics(
            y_all[early_stop_indices],
            engine.predict(X_all[early_stop_indices]),
        )
        print(
            f"  train_mae={train_metrics['mae']:.6f} "
            f"val_mae={val_metrics['mae']:.6f} "
            f"time={training_time:.2f}s "
            f"best_epoch={best_epoch} epochs={epochs_run}"
        )

    split_summary = {
        "n_observations": int(len(observations)),
        "n_train": int(train_indices.size),
        "n_early_stop": int(early_stop_indices.size),
        "n_warm_eval": int(warm_eval_indices.size),
        "n_known_users": int(known_users.size),
        "n_calibration_users": int(calibration_users.size),
        "n_test_users": int(len(split["test_users"])),
        "n_test_eval_notes": int(split["eval_indices"].size),
        "cold_eval_fraction": float(args.cold_eval_fraction),
        "reveal_levels": levels,
        "reveal_seeds": int(args.reveal_seeds),
        "split_seed": int(settings["split_seed"]),
        "model_seed": int(settings["model_seed"]),
    }
    pd.DataFrame([split_summary]).to_csv(
        output_dir / "split_summary.csv",
        index=False,
    )

    print("Evaluating warm-start...")
    warm_per_user, warm_bins, warm_summary = evaluate_warm_start(
        prev,
        engine,
        X_all,
        y_all,
        observations,
        train_indices,
        warm_eval_indices,
        data["n_users"],
    )
    warm_per_user.to_csv(
        output_dir / "warm_start_per_user.csv",
        index=False,
    )
    warm_bins.to_csv(
        output_dir / "warm_start_bins.csv",
        index=False,
    )
    with open(output_dir / "warm_start_summary.json", "w") as handle:
        json.dump(warm_summary, handle, indent=2)
    plot_warm_curve(
        warm_per_user,
        warm_bins,
        output_dir / "warm_start_mae_vs_activity.png",
    )
    print(
        f"  MAE={warm_summary['mae']:.6f} "
        f"RMSE={warm_summary['rmse']:.6f} "
        f"users={warm_summary['n_users']}"
    )

    lambda_bias = args.adapt_reg_bias
    lambda_factors = args.adapt_reg_factors
    calibration_summary = {}
    calibration_grid = pd.DataFrame()
    if (
        args.skip_calibration
        or lambda_bias is not None
        or lambda_factors is not None
        or calibration_users.size == 0
    ):
        lambda_bias = float(
            FALLBACK_BIAS if lambda_bias is None else lambda_bias
        )
        lambda_factors = float(
            FALLBACK_FACTORS if lambda_factors is None else lambda_factors
        )
        calibration_summary = {
            "users": int(calibration_users.size),
            "reg_bias": lambda_bias,
            "reg_factors": lambda_factors,
            "reason": "explicit_or_disabled",
        }
        print(
            "Ridge adaptation fixed: "
            f"reg_bias={lambda_bias} reg_factors={lambda_factors}"
        )
    else:
        print("Calibrating ridge adaptation on known users...")
        calibration_eval, calibration_pool = build_cold_structures(
            observations,
            calibration_users,
            float(args.cold_eval_fraction),
            settings["split_seed"],
            200000,
        )
        (
            lambda_bias,
            lambda_factors,
            calibration_grid,
            calibration_summary,
        ) = calibrate_ridge(
            prev,
            engine,
            X_all,
            y_all,
            spec,
            calibration_eval,
            calibration_pool,
            features["context_family_indices"],
            levels,
            calibration_seeds,
            bias_grid,
            factor_grid,
            settings["split_seed"] + 900000,
        )
        calibration_grid.to_csv(
            output_dir / "calibration_grid.csv",
            index=False,
        )
        with open(
            output_dir / "calibration_summary.json",
            "w",
        ) as handle:
            json.dump(calibration_summary, handle, indent=2)
        print(
            "  Selected reg_bias="
            f"{lambda_bias} reg_factors={lambda_factors} "
            f"on {calibration_summary.get('users', 0)} users"
        )

    print("Evaluating cold-start...")
    cold_long, cold_summary = evaluate_cold_start(
        prev,
        engine,
        X_all,
        y_all,
        spec,
        split["eval_rows_by_user"],
        split["pool_order_by_user"],
        features["context_family_indices"],
        levels,
        int(args.reveal_seeds),
        float(lambda_bias),
        float(lambda_factors),
        settings["split_seed"] + 500000,
        int(args.min_fixed_eval_notes),
    )
    cold_long.to_csv(
        output_dir / "cold_start_long.csv",
        index=False,
    )
    cold_summary.to_csv(
        output_dir / "cold_start_summary.csv",
        index=False,
    )
    plot_cold_curve(
        cold_summary,
        output_dir / "cold_start_mae_vs_revealed.png",
    )

    comparison_note = (
        "Les cohortes warm-start et cold-start sont disjointes : "
        "la comparaison n'est pas causale."
    )
    plot_comparison(
        warm_bins,
        cold_summary,
        output_dir / "warm_vs_cold_mae.png",
        comparison_note,
    )

    training_summary = {
        "loaded_model": args.load_model,
        "load_time_seconds": float(load_time),
        "training_time_seconds": float(training_time)
        if np.isfinite(training_time)
        else None,
        "mean_epoch_time_seconds": float(
            training_time / max(int(epochs_run), 1)
        )
        if np.isfinite(training_time) and int(epochs_run) > 0
        else None,
        "best_epoch": int(best_epoch),
        "epochs_run": int(epochs_run),
        "n_features": int(spec["n_features"]),
    }
    with open(output_dir / "training_summary.json", "w") as handle:
        json.dump(training_summary, handle, indent=2)

    variable = cold_summary[cold_summary["cohort"] == "variable"]
    comparison_rows = [{
        "scenario": "warm-start",
        "n_known": f"variable (median={warm_summary['median_train_notes']:.0f})",
        "mae": warm_summary["mae"],
        "mae_std": 0.0,
        "rmse": warm_summary["rmse"],
        "n_users": warm_summary["n_users"],
        "time_seconds": float(training_time)
        if np.isfinite(training_time)
        else np.nan,
        "time_kind": "training_global",
    }]
    for _, row in variable.iterrows():
        comparison_rows.append({
            "scenario": "cold-start",
            "n_known": int(row["n_revealed"]),
            "mae": float(row["mae_mean"]),
            "mae_std": float(row["mae_std"]),
            "rmse": float(row["rmse_mean"]),
            "n_users": int(row["n_users"]),
            "time_seconds": float(row["adaptation_time_per_user"]),
            "time_kind": "adaptation_per_user",
        })
    comparison = pd.DataFrame(comparison_rows)[
        [
            "scenario",
            "n_known",
            "mae",
            "mae_std",
            "rmse",
            "n_users",
            "time_seconds",
            "time_kind",
        ]
    ]
    comparison.to_csv(output_dir / "final_comparison.csv", index=False)

    config_payload = {
        "cleaning_dir": str(cleaning_dir),
        "model_output_dir": str(model_output_dir),
        "output_dir": str(output_dir),
        "previous_script": str(Path(args.previous_script)),
        "context_config": str(context_config_path)
        if context_config_path is not None
        else None,
        "settings": settings,
        "split": split_summary,
        "training": training_summary,
        "warm_start": warm_summary,
        "calibration": calibration_summary,
        "comparison_note": comparison_note,
        "loaded_model": args.load_model,
    }
    with open(output_dir / "config.json", "w") as handle:
        json.dump(config_payload, handle, indent=2, default=str)

    print()
    print(comparison.to_string(index=False))
    print()
    print(f"Outputs written to {output_dir}")


if __name__ == "__main__":
    main()
