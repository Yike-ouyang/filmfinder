#!/usr/bin/env python3

import argparse
import importlib.util
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCRIPTS_DIR = REPO_ROOT / "scripts"
DEFAULT_BASELINE_DIR = (
    REPO_ROOT / "recommandation" / "recommendation_system"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "recommandation" / "results" / "knn_m5_optimization"
)

DEFAULT_LEVELS = (0, 1, 2, 5, 10, 20, 50, 100)
DEFAULT_CUTOFFS = (5, 10, 20, 30, 50, 100, 200, 500)
FAST_CUTOFFS = (5, 10, 20, 50)
DEFAULT_NEIGHBORS = (5, 10, 25, 50, 100, 200)
DEFAULT_TOPK = (20, 50, 100, 200, 500)
DEFAULT_POOLS = (100, 200, 500, 1000, 2000)
FAST_NEIGHBORS = (10, 25, 50)
FAST_TOPK = (20, 50, 100)
FAST_POOLS = (200, 500)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Optimize the cold-start KNN + M5 recommender over multiple "
            "cutoffs, using the M5 model and splits produced by "
            "8.RecommendationSystem.py."
        )
    )
    parser.add_argument("--baseline-dir", default=None)
    parser.add_argument("--cleaning-dir", default=None)
    parser.add_argument("--model-output-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--scripts-dir", default=None)
    parser.add_argument("--m5-path", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--levels", default=None)
    parser.add_argument("--cutoffs", default=None)
    parser.add_argument("--neighbors-grid", default=None)
    parser.add_argument("--topk-grid", default=None)
    parser.add_argument("--pool-sizes", default=None)
    parser.add_argument(
        "--caps-only",
        dest="include_no_cap",
        action="store_false",
    )
    parser.set_defaults(include_no_cap=True)
    parser.add_argument("--similarity-methods", default="centered,latent")
    parser.add_argument("--similarity-shrink", type=float, default=None)
    parser.add_argument("--fixed-pool", type=int, default=100)
    parser.add_argument("--min-eval-notes", type=int, default=1)
    parser.add_argument("--recall-tolerance", type=float, default=0.02)
    parser.add_argument("--min-pool-coverage", type=float, default=0.9)
    parser.add_argument("--bootstrap-samples", type=int, default=None)
    parser.add_argument("--no-baseline-check", action="store_true")
    parser.add_argument("--skip-variable", action="store_true")
    parser.add_argument("--fast", action="store_true")
    return parser.parse_args()


def parse_int_list(value, fallback, label):
    if value is None:
        items = [int(item) for item in fallback]
    else:
        items = [
            int(float(item))
            for item in str(value).split(",")
            if str(item).strip()
        ]
    items = sorted(set(items))
    if not items or any(item < 0 for item in items):
        raise ValueError(f"{label} must contain non-negative integers.")
    return items


def parse_positive_int_list(value, fallback, label):
    items = parse_int_list(value, fallback, label)
    if any(item <= 0 for item in items):
        raise ValueError(f"{label} must contain positive integers.")
    return items


def parse_methods(value):
    methods = [
        item.strip().casefold()
        for item in str(value).split(",")
        if item.strip()
    ]
    allowed = {"centered", "latent"}
    unknown = set(methods) - allowed
    if not methods or unknown:
        raise ValueError(
            "Unsupported similarity methods: "
            f"{sorted(unknown)}. Use centered,latent."
        )
    return sorted(set(methods))


def load_module(path, name):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Module not found: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path, payload):
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, default=str)


def config_details(config):
    method, n_neighbors, top_k, pool_size = config
    return {
        "method": str(method),
        "n_neighbors": int(n_neighbors),
        "top_k": int(top_k),
        "pool_size": int(pool_size)
        if pool_size is not None
        else None,
    }


def config_id(config):
    method, n_neighbors, top_k, pool_size = config
    pool_text = "none" if pool_size is None else str(int(pool_size))
    return (
        f"{method}|{int(n_neighbors)}|{int(top_k)}|{pool_text}"
    )


def parse_baseline_configs(selected, levels):
    cold = selected.get("cold", {})
    configs = {}
    details = {}
    for level in levels:
        key = f"{int(level)}|knn_m5"
        item = cold.get(key)
        if item is None:
            continue
        details[int(level)] = dict(item)
        if int(level) == 0:
            configs[int(level)] = ("fallback", 0, 0, None)
        else:
            configs[int(level)] = (
                str(item["similarity"]),
                int(item["n_neighbors"]),
                int(item["top_k"]),
                None,
            )
    return configs, details


def make_configs_by_level(
    levels,
    methods,
    neighbors,
    topks,
    pools,
    include_no_cap,
):
    pool_grid = sorted(int(value) for value in pools)
    if include_no_cap:
        pool_grid.append(None)
    configs = {}
    for level in levels:
        level = int(level)
        if level == 0:
            configs[level] = [
                ("fallback", 0, 0, pool) for pool in pool_grid
            ]
        else:
            configs[level] = [
                (method, int(n_neighbors), int(top_k), pool)
                for method in methods
                for n_neighbors in neighbors
                for top_k in topks
                for pool in pool_grid
            ]
    return configs


def build_fallback_candidates(fallback_order, visible_cols):
    candidates = fallback_order
    if visible_cols.size:
        candidates = candidates[
            ~np.isin(candidates, visible_cols)
        ]
    return candidates.astype(np.int64)


def build_knn_union(
    neighbor_values,
    n_neighbors,
    top_k,
    topk_cache,
    catalogue_mask,
    visible_cols,
    popularity_counts,
):
    if neighbor_values is None or neighbor_values.size == 0:
        return None, None, None
    neighbors = neighbor_values[:int(n_neighbors)]
    arrays = [
        topk_cache[int(top_k)][int(neighbor)]
        for neighbor in neighbors
        if topk_cache[int(top_k)][int(neighbor)].size > 0
    ]
    if not arrays:
        return None, None, None
    candidates, support = np.unique(
        np.concatenate(arrays),
        return_counts=True,
    )
    keep = catalogue_mask[candidates]
    if visible_cols.size:
        keep = keep & ~np.isin(candidates, visible_cols)
    candidates = candidates[keep].astype(np.int64)
    support = support[keep].astype(np.float64)
    if candidates.size == 0:
        return None, None, None
    order = np.lexsort(
        (
            candidates,
            -popularity_counts[candidates],
            -support,
        )
    )
    return candidates, support, order


def apply_pool_cap(
    candidates,
    order,
    pool_size,
    fallback_size,
    source,
):
    if candidates.size == 0:
        return candidates
    if source == "fallback":
        limit = (
            int(fallback_size)
            if pool_size is None
            else int(pool_size)
        )
        return candidates[:limit]
    if pool_size is None:
        return candidates
    if candidates.size <= int(pool_size):
        return candidates
    return candidates[order[:int(pool_size)]]


def pool_coverage_mask(pool_sizes, cutoff):
    return np.asarray(pool_sizes) >= int(cutoff)


def load_baseline(eval8, fm, args):
    baseline_dir = Path(
        args.baseline_dir
        if args.baseline_dir is not None
        else DEFAULT_BASELINE_DIR
    )
    config_path = baseline_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Baseline config not found: {config_path}"
        )
    with open(config_path) as handle:
        config = json.load(handle)
    paths = config.get("paths", {})
    baseline_args = config.get("args", {})
    cleaning_dir = Path(
        args.cleaning_dir
        if args.cleaning_dir is not None
        else paths.get(
            "cleaning_dir",
            baseline_args.get("cleaning_dir", Path.cwd()),
        )
    )
    model_output_dir = Path(
        args.model_output_dir
        if args.model_output_dir is not None
        else paths.get(
            "model_output_dir",
            baseline_args.get(
                "model_output_dir",
                cleaning_dir / "modeling",
            ),
        )
    )
    seed = int(
        args.seed
        if args.seed is not None
        else config.get("split", {}).get("seed", 42)
    )
    cold_eval_fraction = float(
        config.get("split", {}).get(
            "cold_eval_fraction",
            baseline_args.get("cold_eval_fraction", 0.2),
        )
    )
    max_users = baseline_args.get("max_users")
    max_interactions = baseline_args.get("max_interactions_per_user")
    max_studios = int(
        config.get("feature_config", {}).get("max_studios", 4000)
    )
    fallback_size = int(
        baseline_args.get("fallback_candidates", 500)
    )
    fallback_m = float(
        baseline_args.get("fallback_popularity_m", 100.0)
    )
    shrink = float(
        args.similarity_shrink
        if args.similarity_shrink is not None
        else baseline_args.get("similarity_shrink", 10.0)
    )
    cold_reg_bias = float(
        baseline_args.get("cold_adapt_reg_bias", 1e-2)
    )
    cold_reg_factors = float(
        baseline_args.get("cold_adapt_reg_factors", 1e-1)
    )
    rating_info = config.get("rating", {}) or {}

    print("Loading observations and reconstructing the cold split...")
    data = fm.load_inputs(cleaning_dir)
    observations = fm.cap_observations(
        data["observations"],
        max_users,
        max_interactions,
        seed,
    )
    split = fm.split_users(observations, seed, cold_eval_fraction)
    user_rows = observations["row"].to_numpy(dtype=np.int64)
    movie_cols = observations["col"].to_numpy(dtype=np.int64)
    y_all = observations["rating"].to_numpy(dtype=np.float64)
    n_users = int(data["n_users"])
    n_movies = int(data["n_movies"])

    if not rating_info.get("factor") or "threshold" not in rating_info:
        detected = eval8.detect_rating_scale(y_all, None)
        rating_info = {
            "min": detected["min"],
            "max": detected["max"],
            "factor": detected["factor"],
            "threshold": detected["threshold"],
        }
    if "factor" in rating_info:
        rating_info["factor"] = float(rating_info["factor"])
    if "threshold" in rating_info:
        rating_info["threshold"] = float(rating_info["threshold"])

    cold_cohorts_path = baseline_dir / "splits" / "cold_cohorts.csv"
    if not cold_cohorts_path.exists():
        raise FileNotFoundError(
            f"Cold cohort file not found: {cold_cohorts_path}"
        )
    cold_cohorts = pd.read_csv(cold_cohorts_path)
    for _, row in cold_cohorts.iterrows():
        user = int(row["user_row"])
        actual_eval = len(split["eval_rows_by_user"].get(user, []))
        actual_pool = len(split["pool_order_by_user"].get(user, []))
        if (
            actual_eval != int(row["n_eval"])
            or actual_pool != int(row["n_pool"])
        ):
            raise ValueError(
                "Reconstructed cold split does not match the baseline: "
                f"user {user}, expected eval/pool "
                f"{int(row['n_eval'])}/{int(row['n_pool'])}, got "
                f"{actual_eval}/{actual_pool}."
            )

    train_visible_path = (
        baseline_dir / "splits" / "train_visible_indices.npy"
    )
    if not train_visible_path.exists():
        raise FileNotFoundError(
            f"Train-visible indices not found: {train_visible_path}"
        )
    train_visible_indices = np.load(train_visible_path)
    selected_path = baseline_dir / "selected_configs.json"
    if not selected_path.exists():
        raise FileNotFoundError(
            f"Selected configs not found: {selected_path}"
        )
    with open(selected_path) as handle:
        selected = json.load(handle)
    m5_path = Path(
        args.m5_path
        if args.m5_path is not None
        else baseline_dir / "models" / "M5.npz"
    )

    print("Building features and loading the fixed M5 model...")
    features = fm.build_features(
        observations,
        data["movie_context"],
        data["user_mapping"],
        data["movie_mapping"],
        n_users,
        n_movies,
        train_visible_indices,
        max_studios,
    )
    spec, engine = fm.load_model(m5_path)
    if int(spec["n_features"]) != len(features["context_mapping"]):
        raise ValueError(
            "M5 feature count does not match the reconstructed layout: "
            f"{spec['n_features']} != "
            f"{len(features['context_mapping'])}."
        )
    if spec["feature_set"] != "context":
        raise ValueError("Loaded model is not the contextual M5.")
    if not spec["use_linear"] or not spec["use_latent"]:
        raise ValueError("M5 must be a full linear + latent FM.")
    if not np.array_equal(
        spec["active_mask"],
        np.ones(len(features["context_mapping"]), dtype=bool),
    ):
        raise ValueError("M5 active mask is incompatible.")

    return {
        "baseline_dir": baseline_dir,
        "config": config,
        "paths": paths,
        "baseline_args": baseline_args,
        "cleaning_dir": cleaning_dir,
        "model_output_dir": model_output_dir,
        "seed": seed,
        "cold_eval_fraction": cold_eval_fraction,
        "max_studios": max_studios,
        "fallback_size": fallback_size,
        "fallback_m": fallback_m,
        "shrink": shrink,
        "cold_reg_bias": cold_reg_bias,
        "cold_reg_factors": cold_reg_factors,
        "rating_info": rating_info,
        "data": data,
        "observations": observations,
        "split": split,
        "user_rows": user_rows,
        "movie_cols": movie_cols,
        "y_all": y_all,
        "n_users": n_users,
        "n_movies": n_movies,
        "train_visible_indices": train_visible_indices,
        "selected": selected,
        "m5_path": m5_path,
        "features": features,
        "spec": spec,
        "engine": engine,
        "cold_cohorts": cold_cohorts,
    }


def build_context(eval8, fm, baseline, args, topks):
    observations = baseline["observations"]
    y_all = baseline["y_all"]
    n_users = baseline["n_users"]
    n_movies = baseline["n_movies"]
    features = baseline["features"]
    engine = baseline["engine"]
    train_indices = baseline["train_visible_indices"]

    item_scores = eval8.build_item_scores(
        fm,
        engine,
        features,
        observations,
        n_movies,
    )
    item_check = eval8.verify_item_scores(
        engine,
        features,
        observations,
        item_scores,
        baseline["seed"],
    )
    popularity = eval8.build_popularity(
        observations,
        train_indices,
        y_all,
        n_movies,
        [baseline["fallback_m"]],
    )
    catalogue_mask = popularity["counts"] > 0
    catalogue_cols = np.where(catalogue_mask)[0].astype(np.int64)
    if catalogue_cols.size == 0:
        raise ValueError("Empty catalogue.")
    visible_matrix = eval8.build_visible_matrix(
        observations,
        train_indices,
        y_all,
        n_users,
        n_movies,
    )
    similarity_structures = eval8.build_similarity_structures(
        visible_matrix,
        n_users,
    )
    topk_cache = eval8.build_topk_cache(
        visible_matrix,
        sorted(set(int(value) for value in topks)),
        popularity["counts"],
    )
    user_feature_columns = features["context_family_indices"]["user"]
    latent_matrix = engine.V[user_feature_columns]
    latent_norms = np.linalg.norm(latent_matrix, axis=1)
    latent_norms = np.where(
        similarity_structures["counts"] > 0,
        latent_norms,
        0.0,
    )
    latent_denominator = np.where(latent_norms > 0, latent_norms, 1.0)
    latent_normalized = latent_matrix / latent_denominator[:, None]
    fallback_scores = popularity["scores"][baseline["fallback_m"]]
    fallback_order = catalogue_cols[
        np.argsort(-fallback_scores[catalogue_cols])
    ]
    return {
        "item_scores": item_scores,
        "item_check": float(item_check),
        "popularity": popularity,
        "catalogue_mask": catalogue_mask,
        "catalogue_cols": catalogue_cols,
        "visible_matrix": visible_matrix,
        "similarity_structures": similarity_structures,
        "topk_cache": topk_cache,
        "latent_matrix": latent_normalized,
        "latent_norms": latent_norms,
        "fallback_order": fallback_order,
        "fallback_size": baseline["fallback_size"],
        "fallback_m": baseline["fallback_m"],
        "shrink": baseline["shrink"],
        "cold_reg_bias": baseline["cold_reg_bias"],
        "cold_reg_factors": baseline["cold_reg_factors"],
        "factor": float(baseline["rating_info"]["factor"]),
        "threshold": float(baseline["rating_info"]["threshold"]),
    }


def prepare_profile(eval8, evalm5, fm, baseline, user):
    split = baseline["split"]
    observations = baseline["observations"]
    y_all = baseline["y_all"]
    features = baseline["features"]
    engine = baseline["engine"]
    user = int(user)
    pool = np.asarray(
        split["pool_order_by_user"][user],
        dtype=np.int64,
    )
    hidden = np.asarray(
        split["eval_rows_by_user"][user],
        dtype=np.int64,
    )
    feature_index = fm.user_feature_index(
        baseline["spec"],
        user,
        features["context_family_indices"],
    )
    base_pool, latent_pool = evalm5.prepare_rows(
        engine,
        features["context_all"],
        pool,
        feature_index,
    )
    return {
        "user": user,
        "pool": pool,
        "pool_ratings": y_all[pool],
        "pool_cols": observations["col"].to_numpy(
            dtype=np.int64
        )[pool],
        "base_pool": base_pool,
        "latent_pool": latent_pool,
        "hidden_movies": observations["col"].to_numpy(
            dtype=np.int64
        )[hidden],
        "hidden_ratings": y_all[hidden],
    }


def adapt_profile(eval8, evalm5, baseline, profile, level):
    level = int(level)
    n_revealed = min(level, profile["pool"].size)
    start = time.perf_counter()
    if n_revealed == 0:
        theta = np.zeros(
            baseline["engine"].n_factors + 1,
            dtype=np.float64,
        )
    else:
        system, target = evalm5.build_ridge_system(
            profile["base_pool"][:n_revealed],
            profile["latent_pool"][:n_revealed],
            profile["pool_ratings"][:n_revealed],
        )
        theta = evalm5.solve_ridge(
            system,
            target,
            float(baseline["cold_reg_bias"]),
            float(baseline["cold_reg_factors"]),
        )
    adaptation_time = time.perf_counter() - start
    item_scores = (
        baseline["item_scores_ref"]["base"]
        + theta[0]
        + baseline["item_scores_ref"]["latent"] @ theta[1:]
    )
    visible_cols = profile["pool_cols"][:n_revealed]
    centered_target, binary_target = eval8.build_target_csr(
        visible_cols,
        profile["pool_ratings"][:n_revealed],
        baseline["n_movies"],
    )
    latent_target = eval8.normalize_vector(theta[1:])
    return {
        "n_revealed": n_revealed,
        "item_scores": item_scores,
        "visible_cols": visible_cols,
        "centered_target": centered_target,
        "binary_target": binary_target,
        "latent_target": latent_target,
        "adaptation_time": float(adaptation_time),
    }


def add_to_bucket(bucket, cutoff, recall, ndcg, heldout, known, pool_ge):
    per_k = bucket["per_k"][int(cutoff)]
    if pool_ge:
        per_k["ge"] += 1
        if recall is not None:
            per_k["recall_sum"] += float(recall)
            per_k["recall_n"] += 1
        if ndcg is not None:
            per_k["ndcg_sum"] += float(ndcg)
            per_k["ndcg_n"] += 1
        if heldout is not None:
            per_k["held_sum"] += float(heldout)
            per_k["held_n"] += 1
        per_k["known_sum"] += float(known)
        per_k["known_n"] += 1
    else:
        per_k["lt"] += 1


def new_bucket(cutoffs):
    return {
        "users": 0,
        "cand_recall_sum": 0.0,
        "cand_recall_n": 0,
        "cand_size_sum": 0.0,
        "time_similarity": 0.0,
        "time_candidate": 0.0,
        "time_ranking": 0.0,
        "time_adaptation": 0.0,
        "per_k": {
            int(cutoff): {
                "ge": 0,
                "lt": 0,
                "recall_sum": 0.0,
                "recall_n": 0,
                "ndcg_sum": 0.0,
                "ndcg_n": 0,
                "held_sum": 0.0,
                "held_n": 0,
                "known_sum": 0.0,
                "known_n": 0,
            }
            for cutoff in cutoffs
        },
    }


def evaluate_target_grid(
    eval8,
    user,
    cohort,
    level,
    prepared,
    configs,
    context,
    accumulators,
    config_lookup,
):
    item_scores = prepared["item_scores"]
    hidden_movies = prepared["hidden_movies"]
    hidden_ratings = prepared["hidden_ratings"]
    visible_cols = prepared["visible_cols"]
    if hidden_movies.size == 0 or not configs:
        return
    cutoffs = context["cutoffs"]
    max_cutoff = int(max(cutoffs))
    factor = context["factor"]
    threshold = context["threshold"]
    truth_map = {
        int(movie): float(rating)
        for movie, rating in zip(hidden_movies, hidden_ratings)
    }
    relevant = hidden_movies[hidden_ratings > threshold]
    ideal_by_cutoff = {
        int(cutoff): eval8.ideal_dcg(
            hidden_ratings,
            factor,
            int(cutoff),
        )
        for cutoff in cutoffs
    }
    hidden_ranked = eval8.top_k_columns(
        hidden_movies,
        item_scores[hidden_movies],
        max_cutoff,
    )

    methods = sorted(
        set(
            config[0]
            for config in configs
            if config[0] in ("centered", "latent")
        )
    )
    max_neighbors = {}
    for method in methods:
        max_neighbors[method] = max(
            int(config[1])
            for config in configs
            if config[0] == method
        )
    similarity_arrays = {}
    neighbor_orders = {}
    similarity_times = {}
    for method in methods:
        start = time.perf_counter()
        if method == "centered":
            similarity = eval8.centered_similarity(
                prepared["centered_target"],
                prepared["binary_target"],
                context["similarity_structures"]["centered"],
                context["similarity_structures"]["binary"],
                context["similarity_structures"]["norms"],
                context["shrink"],
            )
        else:
            similarity = eval8.latent_similarity(
                prepared["latent_target"],
                context["latent_matrix"],
                context["latent_norms"],
            )
        neighbor_orders[method] = eval8.neighbor_order(
            similarity,
            int(user),
            max_neighbors[method],
        )
        similarity_arrays[method] = similarity
        similarity_times[method] = time.perf_counter() - start

    start = time.perf_counter()
    fallback_candidates = build_fallback_candidates(
        context["fallback_order"],
        visible_cols,
    )
    fallback_time = time.perf_counter() - start

    union_cache = {}
    for config in configs:
        method, n_neighbors, top_k, pool_size = config
        key = config_id(config)
        bucket_key = (int(level), key)
        if bucket_key not in accumulators[cohort]:
            accumulators[cohort][bucket_key] = new_bucket(cutoffs)
        bucket = accumulators[cohort][bucket_key]
        if method == "fallback" or level == 0:
            start = time.perf_counter()
            candidates = apply_pool_cap(
                fallback_candidates,
                np.arange(fallback_candidates.size),
                pool_size,
                context["fallback_size"],
                "fallback",
            )
            candidate_time = fallback_time + time.perf_counter() - start
        else:
            union_key = (method, int(n_neighbors), int(top_k))
            if union_key not in union_cache:
                union_start = time.perf_counter()
                candidates, support, order = build_knn_union(
                    neighbor_orders.get(method),
                    int(n_neighbors),
                    int(top_k),
                    context["topk_cache"],
                    context["catalogue_mask"],
                    visible_cols,
                    context["popularity"]["counts"],
                )
                union_time = time.perf_counter() - union_start
                union_cache[union_key] = {
                    "candidates": candidates,
                    "order": order,
                    "time": union_time,
                }
            cached = union_cache[union_key]
            start = time.perf_counter()
            if cached["candidates"] is None:
                candidates = apply_pool_cap(
                    fallback_candidates,
                    np.arange(fallback_candidates.size),
                    pool_size,
                    context["fallback_size"],
                    "fallback",
                )
                candidate_time = (
                    fallback_time
                    + cached["time"]
                    + time.perf_counter()
                    - start
                )
            else:
                candidates = apply_pool_cap(
                    cached["candidates"],
                    cached["order"],
                    pool_size,
                    context["fallback_size"],
                    "knn",
                )
                candidate_time = (
                    cached["time"]
                    + time.perf_counter()
                    - start
                )
        candidate_size = int(candidates.size)
        cand_recall = eval8.candidate_recall(candidates, relevant)
        start = time.perf_counter()
        top_max = eval8.top_k_columns(
            candidates,
            item_scores[candidates],
            max_cutoff,
        )
        ranking_time = time.perf_counter() - start
        similarity_time = float(similarity_times.get(method, 0.0))
        bucket["users"] += 1
        if cand_recall is not None:
            bucket["cand_recall_sum"] += float(cand_recall)
            bucket["cand_recall_n"] += 1
        bucket["cand_size_sum"] += float(candidate_size)
        bucket["time_similarity"] += similarity_time
        bucket["time_candidate"] += float(candidate_time)
        bucket["time_ranking"] += float(ranking_time)
        bucket["time_adaptation"] += prepared["adaptation_time"]
        for cutoff in cutoffs:
            cutoff = int(cutoff)
            pool_ge = candidate_size >= cutoff
            recall = eval8.recall_at_k(
                top_max,
                relevant,
                k=cutoff,
            )
            ndcg, known = eval8.ndcg_known(
                top_max,
                truth_map,
                ideal_by_cutoff[cutoff],
                factor,
                k=cutoff,
            )
            heldout = eval8.ndcg_from_ranked(
                hidden_ranked,
                truth_map,
                ideal_by_cutoff[cutoff],
                factor,
                k=cutoff,
            )
            add_to_bucket(
                bucket,
                cutoff,
                recall,
                ndcg,
                heldout,
                known,
                pool_ge,
            )


def finalize_grid(accumulators, config_lookup, cutoffs):
    rows = []
    for cohort, buckets in accumulators.items():
        for bucket_key, bucket in buckets.items():
            level, key = bucket_key
            config = config_lookup[key]
            details = config_details(config)
            users = max(int(bucket["users"]), 1)
            cand_recall = (
                bucket["cand_recall_sum"] / bucket["cand_recall_n"]
                if bucket["cand_recall_n"]
                else np.nan
            )
            candidate_size = bucket["cand_size_sum"] / users
            time_similarity = bucket["time_similarity"] / users
            time_candidate = bucket["time_candidate"] / users
            time_ranking = bucket["time_ranking"] / users
            time_adaptation = bucket["time_adaptation"] / users
            time_total = (
                time_similarity
                + time_candidate
                + time_ranking
                + time_adaptation
            )
            for cutoff in cutoffs:
                cutoff = int(cutoff)
                per_k = bucket["per_k"][cutoff]
                ge = int(per_k["ge"])
                lt = int(per_k["lt"])
                total = ge + lt
                pool_coverage = ge / total if total else np.nan
                recall = (
                    per_k["recall_sum"] / per_k["recall_n"]
                    if per_k["recall_n"]
                    else np.nan
                )
                ndcg = (
                    per_k["ndcg_sum"] / per_k["ndcg_n"]
                    if per_k["ndcg_n"]
                    else np.nan
                )
                heldout = (
                    per_k["held_sum"] / per_k["held_n"]
                    if per_k["held_n"]
                    else np.nan
                )
                known_count = (
                    per_k["known_sum"] / per_k["known_n"]
                    if per_k["known_n"]
                    else np.nan
                )
                rows.append({
                    "cohort": cohort,
                    "level": int(level),
                    "config_id": key,
                    **details,
                    "cutoff": cutoff,
                    "users": int(bucket["users"]),
                    "n_pool_ge": ge,
                    "n_pool_lt": lt,
                    "pool_coverage": pool_coverage,
                    "candidate_recall": cand_recall,
                    "candidate_recall_n": int(bucket["cand_recall_n"]),
                    "candidate_recall_excluded": int(
                        bucket["users"] - bucket["cand_recall_n"]
                    ),
                    "candidate_size": candidate_size,
                    "recall": recall,
                    "recall_n": int(per_k["recall_n"]),
                    "recall_excluded": int(ge - per_k["recall_n"]),
                    "ndcg_known": ndcg,
                    "ndcg_known_n": int(per_k["ndcg_n"]),
                    "ndcg_known_excluded": int(
                        ge - per_k["ndcg_n"]
                    ),
                    "ndcg_heldout": heldout,
                    "ndcg_heldout_n": int(per_k["held_n"]),
                    "known_count": known_count,
                    "time_similarity": time_similarity,
                    "time_candidate": time_candidate,
                    "time_ranking": time_ranking,
                    "time_adaptation": time_adaptation,
                    "time_total": time_total,
                })
    return pd.DataFrame(rows)


def evaluate_cohort_grid(
    eval8,
    evalm5,
    fm,
    baseline,
    context,
    cohort,
    user_sets,
    levels,
    configs_by_level,
    accumulators,
    config_lookup,
):
    all_users = sorted(
        set().union(*[set(user_sets[int(level)]) for level in levels])
    ) if levels else []
    for user in all_users:
        profile = prepare_profile(
            eval8,
            evalm5,
            fm,
            baseline,
            user,
        )
        for level in levels:
            level = int(level)
            if profile["pool"].size < level:
                continue
            if user not in user_sets[level]:
                continue
            prepared = adapt_profile(
                eval8,
                evalm5,
                baseline,
                profile,
                level,
            )
            evaluate_target_grid(
                eval8,
                user,
                cohort,
                level,
                {
                    "item_scores": prepared["item_scores"],
                    "hidden_movies": profile["hidden_movies"],
                    "hidden_ratings": profile["hidden_ratings"],
                    "visible_cols": prepared["visible_cols"],
                    "centered_target": prepared["centered_target"],
                    "binary_target": prepared["binary_target"],
                    "latent_target": prepared["latent_target"],
                    "adaptation_time": prepared["adaptation_time"],
                },
                configs_by_level[level],
                context,
                accumulators,
                config_lookup,
            )


def select_for_cutoff(
    grid,
    level,
    cutoff,
    tolerance,
    min_coverage,
    cohort="fixed",
):
    subset = grid[
        (grid["cohort"] == cohort)
        & (grid["level"] == int(level))
        & (grid["cutoff"] == int(cutoff))
    ].copy()
    subset = subset[
        (subset["pool_coverage"] >= float(min_coverage))
        & subset["recall"].notna()
    ]
    if subset.empty:
        subset = grid[
            (grid["cohort"] == cohort)
            & (grid["level"] == int(level))
            & (grid["cutoff"] == int(cutoff))
        ].copy()
        subset = subset[
            (subset["pool_coverage"] > 0)
            & subset["recall"].notna()
        ]
    if subset.empty:
        return None
    max_recall = float(subset["recall"].max())
    best_recall = subset.sort_values(
        ["recall", "ndcg_known", "candidate_size", "time_total"],
        ascending=[False, False, True, True],
    ).iloc[0]
    acceptable = subset[
        subset["recall"] >= max_recall - float(tolerance)
    ]
    best_ndcg = acceptable.sort_values(
        ["ndcg_known", "recall", "candidate_size", "time_total"],
        ascending=[False, False, True, True],
    ).iloc[0]
    return {
        "level": int(level),
        "cutoff": int(cutoff),
        "max_recall": max_recall,
        "tolerance": float(tolerance),
        "min_coverage": float(min_coverage),
        "best_recall": best_recall,
        "best_ndcg": best_ndcg,
        "n_acceptable": int(acceptable.shape[0]),
    }


def select_all(
    grid,
    levels,
    cutoffs,
    config_lookup,
    tolerance,
    min_coverage,
):
    best_rows = []
    stage2 = {}
    for level in levels:
        for cutoff in cutoffs:
            selection = select_for_cutoff(
                grid,
                level,
                cutoff,
                tolerance,
                min_coverage,
            )
            if selection is None:
                continue
            best_rows.append({
                "level": int(level),
                "cutoff": int(cutoff),
                "max_recall": selection["max_recall"],
                "n_acceptable": selection["n_acceptable"],
                "best_recall_config_id": selection[
                    "best_recall"
                ]["config_id"],
                "best_recall": float(
                    selection["best_recall"]["recall"]
                ),
                "best_recall_ndcg": float(
                    selection["best_recall"]["ndcg_known"]
                ),
                "best_recall_config": config_lookup[
                    selection["best_recall"]["config_id"]
                ],
                "best_ndcg_config_id": selection[
                    "best_ndcg"
                ]["config_id"],
                "best_ndcg": float(
                    selection["best_ndcg"]["ndcg_known"]
                ),
                "best_ndcg_recall": float(
                    selection["best_ndcg"]["recall"]
                ),
                "best_ndcg_config": config_lookup[
                    selection["best_ndcg"]["config_id"]
                ],
                "best_ndcg_candidate_size": float(
                    selection["best_ndcg"]["candidate_size"]
                ),
                "best_ndcg_coverage": float(
                    selection["best_ndcg"]["pool_coverage"]
                ),
                "best_ndcg_time": float(
                    selection["best_ndcg"]["time_total"]
                ),
            })
            if int(cutoff) == 20:
                stage2[int(level)] = config_lookup[
                    selection["best_ndcg"]["config_id"]
                ]
    best_table = pd.DataFrame(best_rows)

    composite_rows = []
    for level in levels:
        if int(level) == 0:
            continue
        subset = grid[
            (grid["cohort"] == "fixed")
            & (grid["level"] == int(level))
            & (grid["cutoff"].isin([20, 50, 100]))
            & (grid["pool_coverage"] >= float(min_coverage))
        ]
        if subset.empty:
            continue
        recall_pivot = subset.pivot_table(
            index="config_id",
            columns="cutoff",
            values="recall",
            aggfunc="mean",
        )
        ndcg20 = subset[subset["cutoff"] == 20].set_index(
            "config_id"
        )["ndcg_known"]
        if not {20, 50, 100}.issubset(set(recall_pivot.columns)):
            continue
        recall_pivot = recall_pivot.dropna(
            subset=[20, 50, 100]
        )
        if recall_pivot.empty:
            continue
        composite = recall_pivot[[20, 50, 100]].mean(axis=1)
        max_composite = float(composite.max())
        acceptable = composite[
            composite >= max_composite - float(tolerance)
        ]
        if acceptable.empty:
            continue
        ndcg_candidates = ndcg20.reindex(acceptable.index).dropna()
        if ndcg_candidates.empty:
            best_config_id = acceptable.index[0]
        else:
            best_config_id = ndcg_candidates.sort_values(
                ascending=False
            ).index[0]
        composite_rows.append({
            "level": int(level),
            "max_composite_recall": max_composite,
            "selected_composite_recall": float(
                composite.loc[best_config_id]
            ),
            "selected_ndcg20": float(
                ndcg20.get(best_config_id, np.nan)
            ),
            "config_id": best_config_id,
            "config": config_lookup[best_config_id],
        })
    composite_table = pd.DataFrame(composite_rows)

    global_config = None
    global_row = None
    positive_levels = [int(level) for level in levels if int(level) > 0]
    if positive_levels:
        subset = grid[
            (grid["cohort"] == "fixed")
            & (grid["cutoff"] == 20)
            & (grid["pool_coverage"] >= float(min_coverage))
        ]
        recall_pivot = subset.pivot_table(
            index="config_id",
            columns="level",
            values="recall",
            aggfunc="mean",
        )
        ndcg_pivot = subset.pivot_table(
            index="config_id",
            columns="level",
            values="ndcg_known",
            aggfunc="mean",
        )
        if set(positive_levels).issubset(set(recall_pivot.columns)):
            recall_pivot = recall_pivot.dropna(
                subset=positive_levels
            )
            ndcg_pivot = ndcg_pivot.reindex(recall_pivot.index)
            ndcg_pivot = ndcg_pivot.dropna(subset=positive_levels)
            common = recall_pivot.index.intersection(
                ndcg_pivot.index
            )
            if common.size:
                mean_recall = recall_pivot.loc[common][
                    positive_levels
                ].mean(axis=1)
                mean_ndcg = ndcg_pivot.loc[common][
                    positive_levels
                ].mean(axis=1)
                max_mean_recall = float(mean_recall.max())
                acceptable = mean_recall[
                    mean_recall >= max_mean_recall - float(tolerance)
                ]
                if not acceptable.empty:
                    ndcg_candidates = mean_ndcg.reindex(
                        acceptable.index
                    ).dropna()
                    if not ndcg_candidates.empty:
                        best_id = ndcg_candidates.sort_values(
                            ascending=False
                        ).index[0]
                    else:
                        best_id = acceptable.index[0]
                    global_config = config_lookup[best_id]
                    global_row = {
                        "config_id": best_id,
                        "mean_recall20": float(mean_recall.loc[best_id]),
                        "mean_ndcg20": float(mean_ndcg.loc[best_id]),
                        "max_mean_recall20": max_mean_recall,
                        "per_level_recall20": {
                            int(level): float(
                                recall_pivot.loc[best_id, level]
                            )
                            for level in positive_levels
                        },
                        "per_level_ndcg20": {
                            int(level): float(
                                ndcg_pivot.loc[best_id, level]
                            )
                            for level in positive_levels
                        },
                    }
    return {
        "best_table": best_table,
        "stage2": stage2,
        "composite_table": composite_table,
        "global_config": global_config,
        "global_row": global_row,
    }


def evaluate_selected(
    eval8,
    evalm5,
    fm,
    baseline,
    context,
    cohort,
    user_sets,
    levels,
    roles_by_level,
    cutoffs,
):
    cutoffs = [int(cutoff) for cutoff in cutoffs]
    max_cutoff = int(max(cutoffs))
    factor = context["factor"]
    threshold = context["threshold"]
    rows = []
    all_users = sorted(
        set().union(*[set(user_sets[int(level)]) for level in levels])
    ) if levels else []
    for user in all_users:
        profile = prepare_profile(
            eval8,
            evalm5,
            fm,
            baseline,
            user,
        )
        for level in levels:
            level = int(level)
            if profile["pool"].size < level:
                continue
            if user not in user_sets[level]:
                continue
            prepared = adapt_profile(
                eval8,
                evalm5,
                baseline,
                profile,
                level,
            )
            item_scores = prepared["item_scores"]
            hidden_movies = profile["hidden_movies"]
            hidden_ratings = profile["hidden_ratings"]
            if hidden_movies.size == 0:
                continue
            truth_map = {
                int(movie): float(rating)
                for movie, rating in zip(
                    hidden_movies,
                    hidden_ratings,
                )
            }
            relevant = hidden_movies[hidden_ratings > threshold]
            ideal_by_cutoff = {
                int(cutoff): eval8.ideal_dcg(
                    hidden_ratings,
                    factor,
                    int(cutoff),
                )
                for cutoff in cutoffs
            }
            hidden_ranked = eval8.top_k_columns(
                hidden_movies,
                item_scores[hidden_movies],
                max_cutoff,
            )
            configs = list(roles_by_level[level].values())
            methods = sorted(
                set(
                    config[0]
                    for config in configs
                    if config[0] in ("centered", "latent")
                )
            )
            max_neighbors = {}
            for method in methods:
                max_neighbors[method] = max(
                    int(config[1])
                    for config in configs
                    if config[0] == method
                )
            similarity_arrays = {}
            neighbor_orders = {}
            similarity_times = {}
            for method in methods:
                start = time.perf_counter()
                if method == "centered":
                    similarity = eval8.centered_similarity(
                        prepared["centered_target"],
                        prepared["binary_target"],
                        context["similarity_structures"]["centered"],
                        context["similarity_structures"]["binary"],
                        context["similarity_structures"]["norms"],
                        context["shrink"],
                    )
                else:
                    similarity = eval8.latent_similarity(
                        prepared["latent_target"],
                        context["latent_matrix"],
                        context["latent_norms"],
                    )
                neighbor_orders[method] = eval8.neighbor_order(
                    similarity,
                    int(user),
                    max_neighbors[method],
                )
                similarity_arrays[method] = similarity
                similarity_times[method] = time.perf_counter() - start
            start = time.perf_counter()
            fallback_candidates = build_fallback_candidates(
                context["fallback_order"],
                prepared["visible_cols"],
            )
            fallback_time = time.perf_counter() - start
            union_cache = {}
            for role, config in roles_by_level[level].items():
                method, n_neighbors, top_k, pool_size = config
                if method == "fallback" or level == 0:
                    start = time.perf_counter()
                    candidates = apply_pool_cap(
                        fallback_candidates,
                        np.arange(fallback_candidates.size),
                        pool_size,
                        context["fallback_size"],
                        "fallback",
                    )
                    candidate_time = (
                        fallback_time + time.perf_counter() - start
                    )
                else:
                    union_key = (
                        method,
                        int(n_neighbors),
                        int(top_k),
                    )
                    if union_key not in union_cache:
                        start = time.perf_counter()
                        candidates, support, order = build_knn_union(
                            neighbor_orders.get(method),
                            int(n_neighbors),
                            int(top_k),
                            context["topk_cache"],
                            context["catalogue_mask"],
                            prepared["visible_cols"],
                            context["popularity"]["counts"],
                        )
                        union_time = time.perf_counter() - start
                        union_cache[union_key] = {
                            "candidates": candidates,
                            "order": order,
                            "time": union_time,
                        }
                    cached = union_cache[union_key]
                    start = time.perf_counter()
                    if cached["candidates"] is None:
                        candidates = apply_pool_cap(
                            fallback_candidates,
                            np.arange(fallback_candidates.size),
                            pool_size,
                            context["fallback_size"],
                            "fallback",
                        )
                        candidate_time = (
                            fallback_time
                            + cached["time"]
                            + time.perf_counter()
                            - start
                        )
                    else:
                        candidates = apply_pool_cap(
                            cached["candidates"],
                            cached["order"],
                            pool_size,
                            context["fallback_size"],
                            "knn",
                        )
                        candidate_time = (
                            cached["time"]
                            + time.perf_counter()
                            - start
                        )
                candidate_size = int(candidates.size)
                cand_recall = eval8.candidate_recall(
                    candidates,
                    relevant,
                )
                start = time.perf_counter()
                top_max = eval8.top_k_columns(
                    candidates,
                    item_scores[candidates],
                    max_cutoff,
                )
                ranking_time = time.perf_counter() - start
                similarity_time = float(
                    similarity_times.get(method, 0.0)
                )
                for cutoff in cutoffs:
                    cutoff = int(cutoff)
                    pool_ge = candidate_size >= cutoff
                    recall = eval8.recall_at_k(
                        top_max,
                        relevant,
                        k=cutoff,
                    )
                    ndcg, known = eval8.ndcg_known(
                        top_max,
                        truth_map,
                        ideal_by_cutoff[cutoff],
                        factor,
                        k=cutoff,
                    )
                    heldout = eval8.ndcg_from_ranked(
                        hidden_ranked,
                        truth_map,
                        ideal_by_cutoff[cutoff],
                        factor,
                        k=cutoff,
                    )
                    rows.append({
                        "cohort": cohort,
                        "role": role,
                        "level": level,
                        "user": int(user),
                        "cutoff": cutoff,
                        "pool_ge": bool(pool_ge),
                        "candidate_recall": cand_recall,
                        "candidate_size": candidate_size,
                        "recall": recall,
                        "ndcg_known": ndcg,
                        "ndcg_heldout": heldout,
                        "known_count": int(known),
                        "time_similarity": similarity_time,
                        "time_candidate": float(candidate_time),
                        "time_ranking": float(ranking_time),
                        "time_adaptation": float(
                            prepared["adaptation_time"]
                        ),
                        "time_total": float(
                            similarity_time
                            + candidate_time
                            + ranking_time
                            + prepared["adaptation_time"]
                        ),
                    })
    return pd.DataFrame(rows)


def summarize_selected(
    frame,
    bootstrap_samples,
    seed,
    require_pool_ge=True,
):
    if frame.empty:
        return pd.DataFrame()
    rows = []
    group_columns = ["cohort", "role", "level", "cutoff"]
    for key, group in frame.groupby(group_columns, dropna=False):
        key_dict = dict(zip(group_columns, key))
        users = int(group["user"].nunique())
        ge = int(group["pool_ge"].sum())
        lt = int((~group["pool_ge"]).sum())
        candidate_recall = group["candidate_recall"].dropna()
        metric_group = (
            group[group["pool_ge"]]
            if require_pool_ge
            else group
        )
        recall = metric_group["recall"].dropna()
        ndcg = metric_group["ndcg_known"].dropna()
        heldout = metric_group["ndcg_heldout"].dropna()
        record = {
            **key_dict,
            "users": users,
            "n_pool_ge": ge,
            "n_pool_lt": lt,
            "pool_coverage": ge / (ge + lt) if ge + lt else np.nan,
            "candidate_recall": float(candidate_recall.mean())
            if not candidate_recall.empty
            else np.nan,
            "candidate_recall_n": int(candidate_recall.size),
            "candidate_size": float(group["candidate_size"].mean()),
            "recall": float(recall.mean())
            if not recall.empty
            else np.nan,
            "recall_n": int(recall.size),
            "ndcg_known": float(ndcg.mean())
            if not ndcg.empty
            else np.nan,
            "ndcg_known_n": int(ndcg.size),
            "ndcg_heldout": float(heldout.mean())
            if not heldout.empty
            else np.nan,
            "ndcg_heldout_n": int(heldout.size),
            "known_count": float(metric_group["known_count"].mean())
            if not metric_group.empty
            else np.nan,
            "time_similarity": float(group["time_similarity"].mean()),
            "time_candidate": float(group["time_candidate"].mean()),
            "time_ranking": float(group["time_ranking"].mean()),
            "time_adaptation": float(group["time_adaptation"].mean()),
            "time_total": float(group["time_total"].mean()),
        }
        for metric in ["recall", "ndcg_known", "ndcg_heldout"]:
            values = metric_group[metric].dropna().to_numpy()
            low, high = eval8_bootstrap(
                values,
                bootstrap_samples,
                seed + len(rows),
            )
            record[f"{metric}_ci_low"] = low
            record[f"{metric}_ci_high"] = high
        rows.append(record)
    return pd.DataFrame(rows)


def eval8_bootstrap(values, samples, seed):
    values = np.asarray(
        [value for value in values if pd.notna(value)],
        dtype=np.float64,
    )
    if values.size == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(int(seed))
    means = np.empty(int(samples), dtype=np.float64)
    for index in range(int(samples)):
        draw = rng.choice(values, size=values.size, replace=True)
        means[index] = draw.mean()
    return float(np.percentile(means, 2.5)), float(
        np.percentile(means, 97.5)
    )


def paired_bootstrap(
    frame,
    role_a,
    role_b,
    cohort,
    levels,
    cutoffs,
    samples,
    seed,
):
    rows = []
    for level in levels:
        for cutoff in cutoffs:
            for metric in ["recall", "ndcg_known"]:
                subset = frame[
                    (frame["cohort"] == cohort)
                    & (frame["level"] == int(level))
                    & (frame["cutoff"] == int(cutoff))
                    & (frame["role"].isin([role_a, role_b]))
                ]
                pivot = subset.pivot_table(
                    index="user",
                    columns="role",
                    values=metric,
                    aggfunc="first",
                )
                if role_a not in pivot.columns or role_b not in pivot.columns:
                    continue
                pool = subset.pivot_table(
                    index="user",
                    columns="role",
                    values="pool_ge",
                    aggfunc="first",
                )
                if role_a not in pool.columns or role_b not in pool.columns:
                    continue
                valid = (
                    pivot[role_a].notna()
                    & pivot[role_b].notna()
                    & pool[role_a].reindex(pivot.index).fillna(False)
                    & pool[role_b].reindex(pivot.index).fillna(False)
                )
                valid = valid.reindex(pivot.index).fillna(False).astype(bool)
                values = pivot.loc[valid]
                if values.shape[0] < 2:
                    continue
                differences = (
                    values[role_b].to_numpy(dtype=np.float64)
                    - values[role_a].to_numpy(dtype=np.float64)
                )
                rng = np.random.default_rng(
                    int(seed) + len(rows)
                )
                means = np.empty(int(samples), dtype=np.float64)
                for index in range(int(samples)):
                    draw = rng.integers(
                        0,
                        differences.size,
                        size=differences.size,
                    )
                    means[index] = float(
                        differences[draw].mean()
                    )
                mean_a = float(values[role_a].mean())
                mean_b = float(values[role_b].mean())
                absolute = mean_b - mean_a
                relative = absolute / mean_a if mean_a != 0 else np.nan
                rows.append({
                    "cohort": cohort,
                    "level": int(level),
                    "cutoff": int(cutoff),
                    "metric": metric,
                    "role_a": role_a,
                    "role_b": role_b,
                    "n_pairs": int(differences.size),
                    "mean_a": mean_a,
                    "mean_b": mean_b,
                    "absolute_improvement": absolute,
                    "relative_improvement": relative,
                    "ci_low": float(np.percentile(means, 2.5)),
                    "ci_high": float(np.percentile(means, 97.5)),
                })
    return pd.DataFrame(rows)


def selected_roles(
    baseline_configs,
    optimized_configs,
    global_config,
    levels,
):
    roles = {}
    for level in levels:
        level = int(level)
        level_roles = {}
        if level in baseline_configs:
            level_roles["baseline"] = baseline_configs[level]
        if level in optimized_configs:
            level_roles["optimized"] = optimized_configs[level]
        if level == 0:
            if level in optimized_configs:
                level_roles["global"] = optimized_configs[level]
        elif global_config is not None:
            level_roles["global"] = global_config
        roles[level] = level_roles
    return roles


def config_label(config):
    method, n_neighbors, top_k, pool_size = config
    pool_text = "none" if pool_size is None else str(int(pool_size))
    if method == "fallback":
        return f"fallback|pool={pool_text}"
    return (
        f"{method}|N={int(n_neighbors)}|k={int(top_k)}|"
        f"pool={pool_text}"
    )


def plot_role_metric_vs_level(
    summary,
    metric,
    cutoff,
    roles,
    path,
    title,
    ylabel=None,
):
    if summary.empty:
        return
    figure, axis = plt.subplots(figsize=(9, 5))
    for role in roles:
        subset = summary[
            (summary["cohort"] == "fixed")
            & (summary["role"] == role)
            & (summary["cutoff"] == int(cutoff))
        ].sort_values("level")
        if subset.empty:
            continue
        mean = subset[metric].to_numpy(dtype=np.float64)
        low = subset[f"{metric}_ci_low"].to_numpy(dtype=np.float64)
        high = subset[f"{metric}_ci_high"].to_numpy(dtype=np.float64)
        yerr = np.vstack([
            mean - low,
            high - mean,
        ])
        axis.errorbar(
            subset["level"],
            mean,
            yerr=yerr,
            marker="o",
            capsize=4,
            label=role,
        )
    axis.set_xlabel("Nombre de notes connues")
    axis.set_ylabel(ylabel if ylabel is not None else metric)
    axis.set_title(title)
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def plot_metric_vs_cutoff(
    summary,
    metric,
    role,
    path,
    title,
    ylabel=None,
):
    if summary.empty:
        return
    figure, axis = plt.subplots(figsize=(9, 5))
    levels = sorted(
        summary[
            (summary["cohort"] == "fixed")
            & (summary["role"] == role)
        ]["level"].unique()
    )
    cmap = plt.get_cmap("viridis")
    for index, level in enumerate(levels):
        subset = summary[
            (summary["cohort"] == "fixed")
            & (summary["role"] == role)
            & (summary["level"] == int(level))
        ].sort_values("cutoff")
        if subset.empty:
            continue
        mean = subset[metric].to_numpy(dtype=np.float64)
        low = subset[f"{metric}_ci_low"].to_numpy(dtype=np.float64)
        high = subset[f"{metric}_ci_high"].to_numpy(dtype=np.float64)
        yerr = np.vstack([mean - low, high - mean])
        axis.errorbar(
            subset["cutoff"],
            mean,
            yerr=yerr,
            marker="o",
            capsize=2,
            color=cmap(index / max(len(levels) - 1, 1)),
            label=f"n={int(level)}",
        )
    axis.set_xscale("log")
    axis.set_xlabel("Cutoff K")
    axis.set_ylabel(ylabel if ylabel is not None else metric)
    axis.set_title(title)
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def plot_metric_vs_level_for_cutoffs(
    summary,
    metric,
    role,
    cutoffs,
    path,
    title,
    ylabel=None,
):
    if summary.empty:
        return
    figure, axis = plt.subplots(figsize=(9, 5))
    cmap = plt.get_cmap("plasma")
    for index, cutoff in enumerate(cutoffs):
        subset = summary[
            (summary["cohort"] == "fixed")
            & (summary["role"] == role)
            & (summary["cutoff"] == int(cutoff))
        ].sort_values("level")
        if subset.empty:
            continue
        mean = subset[metric].to_numpy(dtype=np.float64)
        low = subset[f"{metric}_ci_low"].to_numpy(dtype=np.float64)
        high = subset[f"{metric}_ci_high"].to_numpy(dtype=np.float64)
        yerr = np.vstack([mean - low, high - mean])
        axis.errorbar(
            subset["level"],
            mean,
            yerr=yerr,
            marker="o",
            capsize=2,
            color=cmap(index / max(len(cutoffs) - 1, 1)),
            label=f"K={int(cutoff)}",
        )
    axis.set_xlabel("Nombre de notes connues")
    axis.set_ylabel(ylabel if ylabel is not None else metric)
    axis.set_title(title)
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def plot_tradeoff(
    grid,
    baseline_configs,
    optimized_configs,
    min_coverage,
    path,
):
    if grid.empty:
        return
    subset = grid[
        (grid["cohort"] == "fixed")
        & (grid["cutoff"] == 20)
        & (grid["pool_coverage"] >= float(min_coverage))
        & grid["recall"].notna()
        & grid["ndcg_known"].notna()
    ]
    if subset.empty:
        return
    figure, axis = plt.subplots(figsize=(9, 6))
    scatter = axis.scatter(
        subset["recall"],
        subset["ndcg_known"],
        c=subset["level"],
        cmap="viridis",
        s=14,
        alpha=0.35,
    )
    for role, configs, marker in [
        ("baseline", baseline_configs, "*"),
        ("optimized", optimized_configs, "X"),
    ]:
        points = subset[
            subset["config_id"].isin(
                [config_id(config) for config in configs.values()]
            )
        ]
        if not points.empty:
            axis.scatter(
                points["recall"],
                points["ndcg_known"],
                marker=marker,
                s=130,
                facecolor="none",
                edgecolor="red",
                linewidth=1.5,
                label=role,
            )
    figure.colorbar(scatter, ax=axis, label="Nombre de notes connues")
    axis.set_xlabel("Recall@20")
    axis.set_ylabel("Known NDCG@20")
    axis.set_title("Trade-off Recall@20 vs NDCG@20 (validation)")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def plot_pool_vs_recall(
    grid,
    baseline_configs,
    optimized_configs,
    min_coverage,
    cutoff,
    path,
):
    if grid.empty:
        return
    subset = grid[
        (grid["cohort"] == "fixed")
        & (grid["cutoff"] == int(cutoff))
        & (grid["pool_coverage"] >= float(min_coverage))
        & grid["recall"].notna()
    ]
    if subset.empty:
        return
    figure, axis = plt.subplots(figsize=(9, 6))
    scatter = axis.scatter(
        subset["candidate_size"],
        subset["recall"],
        c=subset["level"],
        cmap="viridis",
        s=14,
        alpha=0.35,
    )
    points = subset[
        subset["config_id"].isin(
            [config_id(config) for config in optimized_configs.values()]
        )
    ]
    if not points.empty:
        axis.scatter(
            points["candidate_size"],
            points["recall"],
            marker="X",
            s=130,
            facecolor="none",
            edgecolor="red",
            linewidth=1.5,
            label="optimized",
        )
    figure.colorbar(scatter, ax=axis, label="Nombre de notes connues")
    axis.set_xscale("log")
    axis.set_xlabel("Taille du pool de candidats")
    axis.set_ylabel(f"Recall@{int(cutoff)}")
    axis.set_title("Taille du pool vs Recall")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def plot_pool_vs_recall_by_cutoff(
    grid,
    cutoffs,
    min_coverage,
    path,
):
    if grid.empty:
        return
    selected = [int(cutoff) for cutoff in cutoffs][:4]
    if not selected:
        return
    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    for axis, cutoff in zip(axes.ravel(), selected):
        subset = grid[
            (grid["cohort"] == "fixed")
            & (grid["cutoff"] == cutoff)
            & (grid["pool_coverage"] >= float(min_coverage))
            & grid["recall"].notna()
        ]
        if subset.empty:
            axis.set_visible(False)
            continue
        scatter = axis.scatter(
            subset["candidate_size"],
            subset["recall"],
            c=subset["level"],
            cmap="viridis",
            s=10,
            alpha=0.35,
        )
        axis.set_xscale("log")
        axis.set_xlabel("Taille du pool")
        axis.set_ylabel(f"Recall@{cutoff}")
        axis.set_title(f"Recall@{cutoff}")
        axis.grid(True, alpha=0.3)
        figure.colorbar(scatter, ax=axis, label="n")
    for axis in axes.ravel()[len(selected):]:
        axis.set_visible(False)
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def write_summary(
    path,
    eval8,
    selection,
    test_summary,
    paired_base,
    paired_global,
    baseline_info,
    global_row,
    fixed_counts,
    variable_counts,
    args,
    levels,
    cutoffs,
    shrink,
    grid,
):
    lines = [
        "# KNN + M5 cold-start optimization",
        "",
        "## Setup",
        "",
        f"- M5 is loaded from `{baseline_info['m5_path']}` and is not retrained.",
        f"- Fixed cohort: pool >= {args.fixed_pool} "
        f"({fixed_counts['validation']} validation, "
        f"{fixed_counts['test']} test users).",
        f"- Variable cohort: {variable_counts['validation']} validation, "
        f"{variable_counts['test']} test users total.",
        f"- Profile levels: {levels}",
        f"- Cutoffs: {cutoffs}",
        f"- Recall tolerance: {args.recall_tolerance} absolute.",
        f"- Minimum pool coverage per cutoff: {args.min_pool_coverage}.",
        f"- Similarity shrink: {shrink}.",
        "",
        "## Selected KNN configurations by profile size",
        "",
    ]
    best_table = selection["best_table"]
    if not best_table.empty:
        stage2 = best_table[best_table["cutoff"] == 20].copy()
        display = []
        for _, row in stage2.sort_values("level").iterrows():
            display.append({
                "n": int(row["level"]),
                "max_Recall@20": row["max_recall"],
                "best_recall_config": config_label(
                    row["best_recall_config"]
                ),
                "best_recall": row["best_recall"],
                "best_ndcg_config": config_label(
                    row["best_ndcg_config"]
                ),
                "known_NDCG@20": row["best_ndcg"],
                "recall_at_best_ndcg": row["best_ndcg_recall"],
                "pool": row["best_ndcg_candidate_size"],
                "coverage": row["best_ndcg_coverage"],
                "time": row["best_ndcg_time"],
            })
        lines.append(eval8.df_to_markdown(pd.DataFrame(display)))
    else:
        lines.append("No validation selection available.")

    composite = selection["composite_table"]
    if not composite.empty:
        lines.extend([
            "",
            "### Multi-cutoff recall configuration",
            "",
            eval8.df_to_markdown(
                composite[
                    [
                        "level",
                        "max_composite_recall",
                        "selected_composite_recall",
                        "selected_ndcg20",
                        "config_id",
                    ]
                ]
            ),
        ])

    if global_row is not None and selection["global_config"] is not None:
        lines.extend([
            "",
            "### Single global configuration (n > 0)",
            "",
            f"- `{config_label(selection['global_config'])}`",
            f"- Mean Recall@20: {global_row['mean_recall20']:.4f}",
            f"- Mean Known NDCG@20: {global_row['mean_ndcg20']:.4f}",
        ])

    lines.extend([
        "",
        "## Final test comparison",
        "",
    ])
    if not test_summary.empty:
        comparison = test_summary[
            (test_summary["cohort"] == "fixed")
            & (test_summary["cutoff"] == 20)
        ][
            [
                "role",
                "level",
                "users",
                "n_pool_ge",
                "candidate_recall",
                "candidate_size",
                "recall",
                "ndcg_known",
                "ndcg_heldout",
            ]
        ].sort_values(["level", "role"])
        lines.append(eval8.df_to_markdown(comparison))
    else:
        lines.append("No test metrics available.")

    lines.extend([
        "",
        "## Paired bootstrap — baseline vs optimized",
        "",
    ])
    if not paired_base.empty:
        lines.append(
            eval8.df_to_markdown(
                paired_base[
                    [
                        "level",
                        "cutoff",
                        "metric",
                        "n_pairs",
                        "mean_a",
                        "mean_b",
                        "absolute_improvement",
                        "relative_improvement",
                        "ci_low",
                        "ci_high",
                    ]
                ]
            )
        )
    else:
        lines.append("No paired comparison available.")

    lines.extend([
        "",
        "## Paired bootstrap — optimized vs single global",
        "",
    ])
    if not paired_global.empty:
        lines.append(
            eval8.df_to_markdown(
                paired_global[
                    [
                        "level",
                        "cutoff",
                        "metric",
                        "n_pairs",
                        "mean_a",
                        "mean_b",
                        "absolute_improvement",
                        "relative_improvement",
                        "ci_low",
                        "ci_high",
                    ]
                ]
            )
        )
    else:
        lines.append("No global-config comparison available.")

    lines.extend([
        "",
        "## Interpretation",
        "",
        "- Candidate Recall is a property of the candidate pool; "
        "Recall@K and NDCG@K are computed after M5 ranking and are "
        "distinct metrics.",
        "- Known NDCG@K is conditional: unknown movies are removed and "
        "the original Top-K positions are kept. Results must be read "
        "with the number of known movies and the fraction of evaluable "
        "users.",
        "- Metrics at cutoff K use only users whose candidate pool has at "
        "least K items; pool coverage and excluded users are reported.",
        "- The baseline configurations were selected by 8 with Known "
        "NDCG@20 as the primary objective. Stage 1 here optimizes "
        "Recall@20 first, so some improvement may reflect the different "
        "selection objective.",
        "- The fixed cohort (pool >= "
        f"{args.fixed_pool}) removes population changes across profile "
        "sizes; the variable cohort is reported as a secondary "
        "comparison.",
        "- The explicit candidate-pool cap prioritizes movies supported "
        "by multiple neighbors, then training popularity.",
        "- `ndcg_known` is the conditional Known NDCG; `ndcg_heldout` is "
        "the standard NDCG over all hidden test movies ranked by M5.",
        "",
        "## Answers to the objectives",
        "",
    ])
    best_table = selection["best_table"]
    if not best_table.empty:
        distinct_configs = int(
            best_table["best_ndcg_config_id"].nunique()
        )
        mean_recall_loss = float(
            (
                best_table["best_recall"]
                - best_table["best_ndcg_recall"]
            ).mean()
        )
        mean_ndcg_gain = float(
            (
                best_table["best_ndcg"]
                - best_table["best_recall_ndcg"]
            ).mean()
        )
        lines.append(
            f"- The best NDCG configuration takes {distinct_configs} "
            f"distinct values across the {best_table.shape[0]} "
            "selected (profile, cutoff) pairs."
        )
        lines.append(
            "- Moving from the best-recall configuration to the best-NDCG "
            f"configuration within tolerance changes Recall@K by "
            f"{mean_recall_loss:+.4f} and Known NDCG@K by "
            f"{mean_ndcg_gain:+.4f} on average."
        )
    if not grid.empty:
        pool_subset = grid[
            (grid["cohort"] == "fixed")
            & (grid["cutoff"] == 20)
            & (grid["pool_coverage"] >= float(args.min_pool_coverage))
            & grid["recall"].notna()
        ]
        if not pool_subset.empty:
            pool_means = (
                pool_subset.groupby("pool_size", dropna=False)[
                    "recall"
                ]
                .mean()
                .reset_index()
            )
            pool_means["_pool_sort"] = pool_means["pool_size"].apply(
                lambda value: (
                    1,
                    0.0,
                )
                if pd.isna(value)
                else (0, float(value))
            )
            pool_means = pool_means.sort_values("_pool_sort").drop(
                columns=["_pool_sort"]
            )
            pool_means["pool_size"] = pool_means["pool_size"].apply(
                lambda value: "no cap" if pd.isna(value) else str(int(value))
            )
            pool_means = pool_means.rename(
                columns={
                    "pool_size": "pool",
                    "recall": "mean_Recall@20",
                }
            )
            lines.extend([
                "- Mean validation Recall@20 by candidate-pool size:",
                "",
                eval8.df_to_markdown(pool_means),
                "",
            ])
    if not test_summary.empty:
        optimized20 = test_summary[
            (test_summary["cohort"] == "fixed")
            & (test_summary["role"] == "optimized")
            & (test_summary["cutoff"] == 20)
        ].sort_values("level")
        if not optimized20.empty:
            first = optimized20.iloc[0]
            last = optimized20.iloc[-1]
            lines.append(
                "- With the optimized KNN, Known NDCG@20 moves from "
                f"{first['ndcg_known']:.4f} at n={int(first['level'])} to "
                f"{last['ndcg_known']:.4f} at n={int(last['level'])}; "
                f"Recall@20 moves from {first['recall']:.4f} to "
                f"{last['recall']:.4f}."
            )
    lines.append(
        "- The candidate-pool and cutoff plots indicate whether larger "
        "pools or cutoff-specific configurations are worth their cost."
    )
    lines.extend([
        "",
        "## Recommendation",
        "",
    ])
    if not test_summary.empty:
        optimized20 = test_summary[
            (test_summary["cohort"] == "fixed")
            & (test_summary["role"] == "optimized")
            & (test_summary["cutoff"] == 20)
        ]
        global20 = test_summary[
            (test_summary["cohort"] == "fixed")
            & (test_summary["role"] == "global")
            & (test_summary["cutoff"] == 20)
        ]
        if not optimized20.empty:
            lines.append(
                f"- Per-profile optimized mean NDCG@20: "
                f"{optimized20['ndcg_known'].mean():.4f}, "
                f"mean Recall@20: {optimized20['recall'].mean():.4f}."
            )
        if not global20.empty:
            lines.append(
                f"- Single global config mean NDCG@20 (n>0): "
                f"{global20[global20['level'] > 0]['ndcg_known'].mean():.4f}, "
                f"mean Recall@20: "
                f"{global20[global20['level'] > 0]['recall'].mean():.4f}."
            )
        if not paired_global.empty:
            ndcg_pairs = paired_global[
                (paired_global["metric"] == "ndcg_known")
                & (paired_global["cutoff"] == 20)
            ]
            if not ndcg_pairs.empty:
                crosses_zero = (
                    ndcg_pairs["ci_low"] <= 0
                ) & (ndcg_pairs["ci_high"] >= 0)
                if crosses_zero.all():
                    lines.append(
                        "- The paired bootstrap does not show a clear "
                        "NDCG@20 advantage of per-profile configurations "
                        "over one global configuration. Prefer the single "
                        "global configuration for deployment simplicity."
                    )
                else:
                    lines.append(
                        "- Per-profile configurations show a detectable "
                        "NDCG@20 advantage over the single global "
                        "configuration; use profile-size-specific configs "
                        "if the additional complexity is acceptable."
                    )
    lines.append(
        "- If the optimal configuration changes materially with K, "
        "consider a cutoff-aware configuration; otherwise keep one "
        "configuration per profile size (or one global configuration)."
    )
    path.write_text("\n".join(lines))


def compare_value(label, ours, theirs, tolerance=1e-6):
    ours_nan = pd.isna(ours)
    theirs_nan = pd.isna(theirs)
    if ours_nan and theirs_nan:
        return
    if ours_nan != theirs_nan:
        raise ValueError(
            f"Baseline mismatch for {label}: ours={ours}, saved={theirs}."
        )
    if abs(float(ours) - float(theirs)) > float(tolerance):
        raise ValueError(
            f"Baseline mismatch for {label}: ours={ours}, saved={theirs}."
        )


def check_baseline_grid(
    baseline,
    summary,
    baseline_details,
    levels,
    tolerance=1e-6,
):
    path = baseline["baseline_dir"] / "cold" / "grid.csv"
    if not path.exists():
        print("  Baseline grid not found; skipping K=20 grid check.")
        return
    grid = pd.read_csv(path)
    for level in levels:
        level = int(level)
        if level not in baseline_details:
            continue
        details = baseline_details[level]
        rows = grid[
            (grid["approach"] == "knn_m5")
            & (grid["level"] == level)
            & (grid["similarity"] == details["similarity"])
            & (grid["n_neighbors"] == int(details["n_neighbors"]))
            & (grid["top_k"] == int(details["top_k"]))
        ]
        if rows.empty:
            continue
        expected = rows.iloc[0]
        ours = summary[
            (summary["cohort"] == "variable")
            & (summary["role"] == "baseline")
            & (summary["level"] == level)
            & (summary["cutoff"] == 20)
        ]
        if ours.empty:
            continue
        ours = ours.iloc[0]
        compare_value(
            f"grid level {level} users",
            ours["users"],
            expected["users"],
            tolerance,
        )
        compare_value(
            f"grid level {level} candidate_recall",
            ours["candidate_recall"],
            expected["candidate_recall"],
            tolerance,
        )
        compare_value(
            f"grid level {level} recall20",
            ours["recall"],
            expected["recall20"],
            tolerance,
        )
        compare_value(
            f"grid level {level} ndcg20",
            ours["ndcg_known"],
            expected["ndcg_known"],
            tolerance,
        )
        compare_value(
            f"grid level {level} candidate_size",
            ours["candidate_size"],
            expected["candidate_size"],
            tolerance,
        )
    print("  Baseline K=20 validation check passed.")


def check_baseline_final(
    baseline,
    summary,
    levels,
    tolerance=1e-6,
):
    path = baseline["baseline_dir"] / "cold" / "final_metrics.csv"
    if not path.exists():
        print("  Baseline final metrics not found; skipping K=20 test check.")
        return
    final = pd.read_csv(path)
    for level in levels:
        level = int(level)
        rows = final[
            (final["approach"] == "knn_m5")
            & (final["level"] == level)
        ]
        if rows.empty:
            continue
        expected = rows.iloc[0]
        ours = summary[
            (summary["cohort"] == "variable")
            & (summary["role"] == "baseline")
            & (summary["level"] == level)
            & (summary["cutoff"] == 20)
        ]
        if ours.empty:
            continue
        ours = ours.iloc[0]
        compare_value(
            f"final level {level} users",
            ours["users"],
            expected["users"],
            tolerance,
        )
        compare_value(
            f"final level {level} candidate_recall",
            ours["candidate_recall"],
            expected["candidate_recall"],
            tolerance,
        )
        compare_value(
            f"final level {level} recall20",
            ours["recall"],
            expected["recall20"],
            tolerance,
        )
        compare_value(
            f"final level {level} ndcg20",
            ours["ndcg_known"],
            expected["ndcg_known"],
            tolerance,
        )
        compare_value(
            f"final level {level} candidate_size",
            ours["candidate_size"],
            expected["candidate_size"],
            tolerance,
        )
    print("  Baseline K=20 test check passed.")


def build_user_sets(
    cold_cohorts,
    cohort_name,
    levels,
    min_pool,
    min_eval,
):
    subset = cold_cohorts[
        (cold_cohorts["cohort"] == cohort_name)
        & (cold_cohorts["n_pool"] >= int(min_pool))
        & (cold_cohorts["n_eval"] >= int(min_eval))
    ]
    pool_map = dict(
        zip(
            subset["user_row"].astype(int),
            subset["n_pool"].astype(int),
        )
    )
    return {
        int(level): np.asarray(
            sorted(
                user
                for user, n_pool in pool_map.items()
                if n_pool >= int(level)
            ),
            dtype=np.int64,
        )
        for level in levels
    }


def main():
    args = parse_args()
    fast = bool(args.fast)
    scripts_dir = Path(
        args.scripts_dir
        if args.scripts_dir is not None
        else DEFAULT_SCRIPTS_DIR
    )
    output_dir = Path(
        args.output_dir
        if args.output_dir is not None
        else DEFAULT_OUTPUT_DIR
    )
    for subdir in [
        "cohorts",
        "validation",
        "selection",
        "test",
        "figures",
    ]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    print("Loading project modules...")
    eval8 = load_module(
        Path(__file__).with_name("8.RecommendationSystem.py"),
        "_filmfinder_rec8",
    )
    fm = load_module(
        scripts_dir / "6.IncrementalVsRetraining.py",
        "_filmfinder_fm",
    )
    evalm5 = load_module(
        scripts_dir / "7.EvaluateM5.py",
        "_filmfinder_evalm5",
    )

    print("Loading baseline artefacts and fixed M5...")
    baseline = load_baseline(eval8, fm, args)
    levels = parse_int_list(
        args.levels,
        DEFAULT_LEVELS if not fast else (0, 5, 20),
        "levels",
    )
    cutoffs = parse_positive_int_list(
        args.cutoffs,
        DEFAULT_CUTOFFS if not fast else FAST_CUTOFFS,
        "cutoffs",
    )
    if 20 not in cutoffs:
        cutoffs = sorted(set(cutoffs + [20]))
    neighbors = parse_positive_int_list(
        args.neighbors_grid,
        DEFAULT_NEIGHBORS if not fast else FAST_NEIGHBORS,
        "neighbors",
    )
    topks = parse_positive_int_list(
        args.topk_grid,
        DEFAULT_TOPK if not fast else FAST_TOPK,
        "top-k",
    )
    pools = parse_positive_int_list(
        args.pool_sizes,
        DEFAULT_POOLS if not fast else FAST_POOLS,
        "pool sizes",
    )
    methods = parse_methods(args.similarity_methods)
    shrink = float(
        args.similarity_shrink
        if args.similarity_shrink is not None
        else baseline["shrink"]
    )
    bootstrap_samples = int(
        args.bootstrap_samples
        if args.bootstrap_samples is not None
        else (200 if fast else 1000)
    )
    if args.fixed_pool <= 0:
        raise ValueError("--fixed-pool must be positive.")
    if args.min_eval_notes < 1:
        raise ValueError("--min-eval-notes must be at least 1.")
    if args.recall_tolerance < 0:
        raise ValueError("--recall-tolerance must be non-negative.")
    if not 0 < args.min_pool_coverage <= 1:
        raise ValueError("--min-pool-coverage must be in (0, 1].")
    if bootstrap_samples < 1:
        raise ValueError("--bootstrap-samples must be positive.")

    baseline["shrink"] = shrink
    topk_for_cache = sorted(
        set(topks)
        | {
            int(item["top_k"])
            for item in baseline["selected"].get(
                "cold",
                {},
            ).values()
            if item.get("approach") == "knn_m5"
            and "top_k" in item
        }
    )
    print("Building scoring context...")
    context = build_context(eval8, fm, baseline, args, topk_for_cache)
    baseline["item_scores_ref"] = context["item_scores"]
    context["cutoffs"] = cutoffs

    print("Building fixed and variable cohorts...")
    fixed_validation_sets = build_user_sets(
        baseline["cold_cohorts"],
        "cold_validation",
        levels,
        args.fixed_pool,
        args.min_eval_notes,
    )
    fixed_test_sets = build_user_sets(
        baseline["cold_cohorts"],
        "cold_test",
        levels,
        args.fixed_pool,
        args.min_eval_notes,
    )
    variable_validation_sets = build_user_sets(
        baseline["cold_cohorts"],
        "cold_validation",
        levels,
        0,
        args.min_eval_notes,
    )
    variable_test_sets = build_user_sets(
        baseline["cold_cohorts"],
        "cold_test",
        levels,
        0,
        args.min_eval_notes,
    )
    fixed_counts = {
        "validation": int(len(fixed_validation_sets[levels[0]])),
        "test": int(len(fixed_test_sets[levels[0]])),
    }
    variable_counts = {
        "validation": int(
            len(variable_validation_sets[levels[0]])
        ),
        "test": int(len(variable_test_sets[levels[0]])),
    }
    if fixed_counts["validation"] == 0 or fixed_counts["test"] == 0:
        raise ValueError(
            "Fixed cohort is empty. Lower --fixed-pool for this dataset."
        )
    pd.DataFrame([
        {
            "cohort": "fixed_validation",
            "users": fixed_counts["validation"],
            "fixed_pool": args.fixed_pool,
        },
        {
            "cohort": "fixed_test",
            "users": fixed_counts["test"],
            "fixed_pool": args.fixed_pool,
        },
        {
            "cohort": "variable_validation",
            "users": variable_counts["validation"],
            "fixed_pool": 0,
        },
        {
            "cohort": "variable_test",
            "users": variable_counts["test"],
            "fixed_pool": 0,
        },
    ]).to_csv(
        output_dir / "cohorts" / "cohort_counts.csv",
        index=False,
    )

    baseline_configs, baseline_details = parse_baseline_configs(
        baseline["selected"],
        levels,
    )
    for level in levels:
        level = int(level)
        if level not in baseline_configs:
            fallback = (
                ("fallback", 0, 0, None)
                if level == 0
                else ("centered", 10, 20, None)
            )
            baseline_configs[level] = fallback
            print(
                f"  Warning: no saved baseline for n={level}; "
                f"using {config_label(fallback)}."
            )

    configs_by_level = make_configs_by_level(
        levels,
        methods,
        neighbors,
        topks,
        pools,
        bool(args.include_no_cap),
    )
    config_lookup = {}
    for level, configs in configs_by_level.items():
        for config in configs:
            config_lookup[config_id(config)] = config
    for config in baseline_configs.values():
        config_lookup.setdefault(config_id(config), config)

    print("Evaluating the KNN grid on fixed validation users...")
    accumulators = {"fixed": {}}
    evaluate_cohort_grid(
        eval8,
        evalm5,
        fm,
        baseline,
        context,
        "fixed",
        fixed_validation_sets,
        levels,
        configs_by_level,
        accumulators,
        config_lookup,
    )
    grid = finalize_grid(accumulators, config_lookup, cutoffs)
    if grid.empty:
        raise ValueError("The validation grid produced no metrics.")
    grid.to_csv(
        output_dir / "validation" / "grid_summary.csv",
        index=False,
    )

    print("Selecting configurations...")
    selection = select_all(
        grid,
        levels,
        cutoffs,
        config_lookup,
        float(args.recall_tolerance),
        float(args.min_pool_coverage),
    )
    selection["best_table"].to_csv(
        output_dir / "selection" / "best_per_profile_and_cutoff.csv",
        index=False,
    )
    selection["composite_table"].to_csv(
        output_dir / "selection" / "multi_cutoff_configs.csv",
        index=False,
    )
    optimized_configs = dict(selection["stage2"])
    for level in levels:
        level = int(level)
        if level not in optimized_configs:
            optimized_configs[level] = baseline_configs[level]
    global_config = selection["global_config"]
    if global_config is None:
        global_config = optimized_configs.get(
            levels[-1],
            baseline_configs.get(levels[-1]),
        )
    write_json(
        output_dir / "selection" / "selected_configs.json",
        {
            "baseline": {
                str(level): config_details(config)
                for level, config in baseline_configs.items()
            },
            "optimized": {
                str(level): config_details(config)
                for level, config in optimized_configs.items()
            },
            "global": config_details(global_config)
            if global_config is not None
            else None,
            "global_summary": selection["global_row"],
            "recall_tolerance": float(args.recall_tolerance),
            "min_pool_coverage": float(args.min_pool_coverage),
        },
    )

    roles = selected_roles(
        baseline_configs,
        optimized_configs,
        global_config,
        levels,
    )

    print("Evaluating selected configurations on fixed test users...")
    fixed_test_rows = evaluate_selected(
        eval8,
        evalm5,
        fm,
        baseline,
        context,
        "fixed",
        fixed_test_sets,
        levels,
        roles,
        cutoffs,
    )
    fixed_test_summary = summarize_selected(
        fixed_test_rows,
        bootstrap_samples,
        baseline["seed"] + 610000,
        require_pool_ge=True,
    )
    fixed_test_rows.to_csv(
        output_dir / "test" / "fixed_per_user.csv",
        index=False,
    )
    fixed_test_summary.to_csv(
        output_dir / "test" / "fixed_metrics.csv",
        index=False,
    )

    variable_test_rows = pd.DataFrame()
    if not args.skip_variable:
        print(
            "Evaluating selected configurations on variable test users..."
        )
        variable_test_rows = evaluate_selected(
            eval8,
            evalm5,
            fm,
            baseline,
            context,
            "variable",
            variable_test_sets,
            levels,
            roles,
            cutoffs,
        )
        variable_test_summary = summarize_selected(
            variable_test_rows,
            bootstrap_samples,
            baseline["seed"] + 620000,
            require_pool_ge=True,
        )
        variable_test_rows.to_csv(
            output_dir / "test" / "variable_per_user.csv",
            index=False,
        )
        variable_test_summary.to_csv(
            output_dir / "test" / "variable_metrics.csv",
            index=False,
        )

    if not args.no_baseline_check and not args.skip_variable:
        print("Checking baseline reproduction against 8...")
        variable_validation_rows = evaluate_selected(
            eval8,
            evalm5,
            fm,
            baseline,
            context,
            "variable",
            variable_validation_sets,
            levels,
            {
                level: {"baseline": baseline_configs[level]}
                for level in levels
            },
            [20],
        )
        variable_validation_summary = summarize_selected(
            variable_validation_rows,
            max(bootstrap_samples, 1),
            baseline["seed"] + 630000,
            require_pool_ge=False,
        )
        variable_test_baseline_summary = summarize_selected(
            variable_test_rows[
                variable_test_rows["role"] == "baseline"
            ],
            max(bootstrap_samples, 1),
            baseline["seed"] + 640000,
            require_pool_ge=False,
        )
        check_baseline_grid(
            baseline,
            variable_validation_summary,
            baseline_details,
            levels,
        )
        check_baseline_final(
            baseline,
            variable_test_baseline_summary,
            levels,
        )

    print("Running paired bootstrap comparisons...")
    paired_base = paired_bootstrap(
        fixed_test_rows,
        "baseline",
        "optimized",
        "fixed",
        levels,
        cutoffs,
        bootstrap_samples,
        baseline["seed"] + 650000,
    )
    paired_global = paired_bootstrap(
        fixed_test_rows,
        "optimized",
        "global",
        "fixed",
        levels,
        cutoffs,
        bootstrap_samples,
        baseline["seed"] + 660000,
    )
    paired_base.to_csv(
        output_dir / "test" / "paired_baseline_vs_optimized.csv",
        index=False,
    )
    paired_global.to_csv(
        output_dir / "test" / "paired_optimized_vs_global.csv",
        index=False,
    )

    print("Generating figures...")
    plot_role_metric_vs_level(
        fixed_test_summary,
        "ndcg_known",
        20,
        ["baseline", "optimized", "global"],
        output_dir / "figures" / "plot1_ndcg20_vs_n.png",
        "Known NDCG@20 vs known ratings",
        "Known NDCG@20",
    )
    plot_role_metric_vs_level(
        fixed_test_summary,
        "recall",
        20,
        ["baseline", "optimized", "global"],
        output_dir / "figures" / "plot2_recall20_vs_n.png",
        "Recall@20 vs known ratings",
        "Recall@20",
    )
    plot_tradeoff(
        grid,
        baseline_configs,
        optimized_configs,
        float(args.min_pool_coverage),
        output_dir / "figures" / "plot3_recall20_vs_ndcg20.png",
    )
    plot_pool_vs_recall(
        grid,
        baseline_configs,
        optimized_configs,
        float(args.min_pool_coverage),
        20,
        output_dir / "figures" / "plot4_pool_vs_recall20.png",
    )
    plot_metric_vs_cutoff(
        fixed_test_summary,
        "recall",
        "optimized",
        output_dir / "figures" / "ext1_recallK_vs_K.png",
        "Optimized Recall@K vs cutoff",
        "Recall@K",
    )
    plot_metric_vs_cutoff(
        fixed_test_summary,
        "ndcg_known",
        "optimized",
        output_dir / "figures" / "ext2_ndcgK_vs_K.png",
        "Optimized Known NDCG@K vs cutoff",
        "Known NDCG@K",
    )
    plot_metric_vs_level_for_cutoffs(
        fixed_test_summary,
        "ndcg_known",
        "optimized",
        [10, 20, 50, 100, 200],
        output_dir / "figures" / "ext3_ndcgK_vs_n.png",
        "Known NDCG@K vs known ratings",
        "Known NDCG@K",
    )
    plot_metric_vs_level_for_cutoffs(
        fixed_test_summary,
        "recall",
        "optimized",
        [10, 20, 50, 100, 200],
        output_dir / "figures" / "ext4_recallK_vs_n.png",
        "Recall@K vs known ratings",
        "Recall@K",
    )
    plot_pool_vs_recall_by_cutoff(
        grid,
        [20, 50, 100, 200],
        float(args.min_pool_coverage),
        output_dir / "figures" / "ext5_pool_vs_recallK.png",
    )

    print("Writing summary and configuration...")
    write_summary(
        output_dir / "summary.md",
        eval8,
        selection,
        fixed_test_summary,
        paired_base,
        paired_global,
        baseline,
        selection["global_row"],
        fixed_counts,
        variable_counts,
        args,
        levels,
        cutoffs,
        shrink,
        grid,
    )
    write_json(
        output_dir / "config.json",
        {
            "paths": {
                "baseline_dir": str(baseline["baseline_dir"]),
                "cleaning_dir": str(baseline["cleaning_dir"]),
                "output_dir": str(output_dir),
                "m5_path": str(baseline["m5_path"]),
            },
            "args": vars(args),
            "levels": levels,
            "cutoffs": cutoffs,
            "neighbors": neighbors,
            "topks": topks,
            "pools": pools,
            "include_no_cap": bool(args.include_no_cap),
            "methods": methods,
            "similarity_shrink": shrink,
            "fixed_counts": fixed_counts,
            "variable_counts": variable_counts,
            "rating": baseline["rating_info"],
            "item_score_check": context["item_check"],
            "selection": {
                "optimized": {
                    str(level): config_details(config)
                    for level, config in optimized_configs.items()
                },
                "global": config_details(global_config)
                if global_config is not None
                else None,
                "global_summary": selection["global_row"],
            },
        },
    )
    write_json(
        output_dir / "baseline_config.json",
        {
            "baseline_dir": str(baseline["baseline_dir"]),
            "configs": {
                str(level): config_details(config)
                for level, config in baseline_configs.items()
            },
            "details": baseline_details,
        },
    )

    print()
    if not fixed_test_summary.empty:
        print("Fixed-cohort test metrics at K=20")
        print(
            fixed_test_summary[
                (fixed_test_summary["cutoff"] == 20)
            ][
                [
                    "role",
                    "level",
                    "users",
                    "n_pool_ge",
                    "candidate_recall",
                    "candidate_size",
                    "recall",
                    "ndcg_known",
                ]
            ].to_string(index=False)
        )
    print()
    print(f"Outputs written to {output_dir}")


if __name__ == "__main__":
    main()
