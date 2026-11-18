#!/usr/bin/env python3

import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCRIPTS_DIR = REPO_ROOT / "scripts"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "recommandation" / "results" / "recommendation_system"
)

DEFAULT_LEVELS = (0, 1, 2, 5, 10, 20, 50, 100)
FAST_LEVELS = (0, 5, 20)
DEFAULT_POP_M = (10, 50, 100, 500)
DEFAULT_KNN_N = (10, 25, 50, 100)
DEFAULT_KNN_K = (20, 50, 100)
FAST_POP_M = (10, 100)
FAST_KNN_N = (10, 25)
FAST_KNN_K = (20, 50)

M5_DEFAULTS = {
    "n_factors": 8,
    "learning_rate": 0.01,
    "n_epochs": 20,
    "batch_size": 65536,
    "reg_bias": 0.0,
    "reg_factors": 1e-6,
    "patience": 3,
    "seed": 4042,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Top-20 recommendation strategies with the "
            "contextual FM M5."
        )
    )
    parser.add_argument("--cleaning-dir", default=None)
    parser.add_argument("--model-output-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--scripts-dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--m5-seed", type=int, default=None)
    parser.add_argument("--n-factors", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--reg-bias", type=float, default=None)
    parser.add_argument("--reg-factors", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--max-studios", type=int, default=None)
    parser.add_argument("--cold-eval-fraction", type=float, default=0.2)
    parser.add_argument("--warm-validation-users", type=int, default=None)
    parser.add_argument("--warm-test-users", type=int, default=None)
    parser.add_argument("--warm-hidden-fraction", type=float, default=0.2)
    parser.add_argument("--min-visible-per-user", type=int, default=5)
    parser.add_argument("--min-hidden-per-user", type=int, default=2)
    parser.add_argument("--cold-validation-fraction", type=float, default=0.3)
    parser.add_argument("--reveal-levels", default=None)
    parser.add_argument("--popularity-m-grid", default=None)
    parser.add_argument("--knn-neighbors-grid", default=None)
    parser.add_argument("--knn-topk-grid", default=None)
    parser.add_argument(
        "--similarity-methods",
        default="centered,latent",
    )
    parser.add_argument("--similarity-shrink", type=float, default=10.0)
    parser.add_argument("--fallback-candidates", type=int, default=500)
    parser.add_argument("--fallback-popularity-m", type=float, default=100.0)
    parser.add_argument("--cold-adapt-reg-bias", type=float, default=1e-2)
    parser.add_argument("--cold-adapt-reg-factors", type=float, default=1e-1)
    parser.add_argument("--relevance-threshold", type=float, default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=None)
    parser.add_argument("--reuse-m5", dest="reuse_m5", action="store_true")
    parser.add_argument(
        "--no-reuse-m5",
        dest="reuse_m5",
        action="store_false",
    )
    parser.set_defaults(reuse_m5=True)
    parser.add_argument("--load-m5", default=None)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--max-users", type=int, default=None)
    parser.add_argument(
        "--max-interactions-per-user",
        type=int,
        default=None,
    )
    return parser.parse_args()


def parse_int_list(value, fallback, label):
    if value is None:
        items = list(fallback)
    else:
        items = [
            int(float(item))
            for item in str(value).split(",")
            if str(item).strip()
        ]
    items = sorted(set(int(item) for item in items))
    if not items or any(item < 0 for item in items):
        raise ValueError(f"{label} must contain non-negative integers.")
    return items


def parse_float_list(value, fallback, label):
    if value is None:
        items = list(fallback)
    else:
        items = [
            float(item)
            for item in str(value).split(",")
            if str(item).strip()
        ]
    items = sorted(set(float(item) for item in items))
    if not items or any(item <= 0 for item in items):
        raise ValueError(f"{label} must contain positive numbers.")
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


def resolve_max_studios(fm, model_output_dir, explicit):
    if explicit is not None:
        return int(explicit)
    try:
        config = fm.load_model_config(model_output_dir)
        return int(
            config.get("feature_config", {}).get("max_studios", 4000)
        )
    except (FileNotFoundError, KeyError, ValueError):
        return 4000


def resolve_hyperparameters(args):
    fast = bool(args.fast)
    return {
        "n_factors": int(
            args.n_factors
            if args.n_factors is not None
            else (4 if fast else M5_DEFAULTS["n_factors"])
        ),
        "learning_rate": float(
            args.learning_rate
            if args.learning_rate is not None
            else M5_DEFAULTS["learning_rate"]
        ),
        "n_epochs": int(
            args.epochs
            if args.epochs is not None
            else (3 if fast else M5_DEFAULTS["n_epochs"])
        ),
        "batch_size": int(
            args.batch_size
            if args.batch_size is not None
            else M5_DEFAULTS["batch_size"]
        ),
        "reg_bias": float(
            args.reg_bias
            if args.reg_bias is not None
            else M5_DEFAULTS["reg_bias"]
        ),
        "reg_factors": float(
            args.reg_factors
            if args.reg_factors is not None
            else M5_DEFAULTS["reg_factors"]
        ),
        "patience": int(
            args.patience
            if args.patience is not None
            else M5_DEFAULTS["patience"]
        ),
        "seed": int(
            args.m5_seed
            if args.m5_seed is not None
            else M5_DEFAULTS["seed"]
        ),
    }


def split_signature(indices, extra):
    digest = hashlib.sha1()
    digest.update(np.ascontiguousarray(indices, dtype=np.int64).tobytes())
    digest.update(json.dumps(extra, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


def write_json(path, payload):
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, default=str)


def split_warm_cohorts(
    observations,
    train_indices,
    seed,
    n_validation,
    n_test,
    hidden_fraction,
    min_visible,
    min_hidden,
):
    user_rows = observations["row"].to_numpy(dtype=np.int64)
    train_users = user_rows[train_indices]
    users, counts = np.unique(train_users, return_counts=True)
    eligible = users[counts >= min_visible + min_hidden]
    if eligible.size < 2:
        raise ValueError("Not enough users for warm-start cohorts.")
    rng = np.random.default_rng(seed + 110000)
    order = rng.permutation(eligible)
    requested_total = max(2, int(n_validation) + int(n_test))
    max_total = max(2, int(eligible.size * 0.5))
    if requested_total > max_total:
        scale = max_total / requested_total
        n_validation = max(1, int(round(int(n_validation) * scale)))
        n_test = max(1, int(round(int(n_test) * scale)))
    n_validation = min(int(n_validation), order.size - 1)
    n_test = min(int(n_test), order.size - n_validation)
    validation_users = np.sort(order[:n_validation]).astype(np.int64)
    test_users = np.sort(
        order[n_validation:n_validation + n_test]
    ).astype(np.int64)
    if validation_users.size == 0 or test_users.size == 0:
        raise ValueError("Warm validation or test cohort is empty.")

    groups = pd.DataFrame({"row": user_rows[train_indices]})
    positions = groups.groupby("row").indices
    hidden_by_user = {}
    visible_by_user = {}
    cohort_rows = []

    def assign(users, cohort):
        kept = []
        for user in np.asarray(users, dtype=np.int64):
            key = int(user)
            local = np.asarray(positions.get(key, []), dtype=np.int64)
            if local.size < min_visible + min_hidden:
                continue
            global_indices = train_indices[local]
            user_rng = np.random.default_rng(
                seed + 120000 + key
            )
            permutation = user_rng.permutation(global_indices.size)
            ordered = global_indices[permutation]
            n_hidden = max(
                min_hidden,
                int(round(hidden_fraction * ordered.size)),
            )
            n_hidden = min(n_hidden, ordered.size - min_visible)
            if n_hidden < min_hidden:
                continue
            hidden_by_user[key] = np.sort(ordered[:n_hidden])
            visible_by_user[key] = np.sort(ordered[n_hidden:])
            kept.append(key)
            cohort_rows.append({
                "user_row": key,
                "cohort": cohort,
                "n_train": int(ordered.size),
                "n_visible": int(ordered.size - n_hidden),
                "n_hidden": int(n_hidden),
            })
        return kept

    validation_kept = assign(validation_users, "warm_validation")
    test_kept = assign(test_users, "warm_test")
    if not validation_kept or not test_kept:
        raise ValueError(
            "Warm cohorts are empty after enforcing visible/hidden sizes."
        )
    return {
        "validation_users": np.asarray(
            validation_kept,
            dtype=np.int64,
        ),
        "test_users": np.asarray(test_kept, dtype=np.int64),
        "hidden_by_user": hidden_by_user,
        "visible_by_user": visible_by_user,
        "cohort_rows": cohort_rows,
    }


def split_cold_cohorts(test_users, seed, validation_fraction):
    users = np.asarray(test_users, dtype=np.int64)
    if users.size < 2:
        raise ValueError("Not enough cold-start users.")
    rng = np.random.default_rng(seed + 130000)
    order = rng.permutation(users.size)
    ordered = users[order]
    n_validation = int(round(validation_fraction * ordered.size))
    n_validation = max(1, min(n_validation, ordered.size - 1))
    return {
        "validation_users": np.sort(ordered[:n_validation]),
        "test_users": np.sort(ordered[n_validation:]),
    }


def build_visible_matrix(
    observations,
    indices,
    y_all,
    n_users,
    n_movies,
):
    rows = observations["row"].to_numpy(dtype=np.int64)[indices]
    cols = observations["col"].to_numpy(dtype=np.int64)[indices]
    values = y_all[indices].astype(np.float64)
    matrix = csr_matrix(
        (values, (rows, cols)),
        shape=(int(n_users), int(n_movies)),
        dtype=np.float64,
    )
    matrix.sum_duplicates()
    return matrix


def build_similarity_structures(visible_matrix, user_count):
    row_counts = np.diff(visible_matrix.indptr)
    row_sums = np.asarray(visible_matrix.sum(axis=1)).ravel()
    means = np.divide(
        row_sums,
        np.maximum(row_counts, 1),
        out=np.zeros_like(row_sums),
        where=row_counts > 0,
    )
    centered = visible_matrix.copy()
    centered.data -= np.repeat(means, row_counts)
    row_norms = np.sqrt(
        np.asarray(centered.multiply(centered).sum(axis=1)).ravel()
    )
    binary = visible_matrix.copy()
    binary.data[:] = 1.0
    return {
        "centered": centered,
        "binary": binary,
        "means": means,
        "norms": row_norms,
        "counts": row_counts,
        "user_count": int(user_count),
    }


def build_topk_cache(visible_matrix, ks, tie_break):
    n_users = visible_matrix.shape[0]
    max_k = max(ks)
    cache = {int(k): [None] * n_users for k in ks}
    for user in range(n_users):
        start = visible_matrix.indptr[user]
        end = visible_matrix.indptr[user + 1]
        cols = visible_matrix.indices[start:end]
        values = visible_matrix.data[start:end]
        if cols.size == 0:
            for k in ks:
                cache[int(k)][user] = np.zeros(0, dtype=np.int64)
            continue
        order = np.lexsort(
            (cols, -tie_break[cols], -values)
        )[:max_k]
        selected = cols[order]
        for k in ks:
            cache[int(k)][user] = selected[:int(k)]
    return cache


def build_target_csr(cols, ratings, n_movies):
    cols = np.asarray(cols, dtype=np.int64)
    if cols.size == 0:
        return None, None
    ratings = np.asarray(ratings, dtype=np.float64)
    mean = float(ratings.mean())
    centered = csr_matrix(
        (
            ratings - mean,
            (np.zeros(cols.size, dtype=np.int64), cols),
        ),
        shape=(1, int(n_movies)),
        dtype=np.float64,
    )
    binary = csr_matrix(
        (
            np.ones(cols.size, dtype=np.float64),
            (np.zeros(cols.size, dtype=np.int64), cols),
        ),
        shape=(1, int(n_movies)),
        dtype=np.float64,
    )
    return centered, binary


def centered_similarity(
    target_centered,
    target_binary,
    centered,
    binary,
    norms,
    shrink,
):
    if target_centered is None:
        return None
    target_norm = float(
        np.sqrt(target_centered.multiply(target_centered).sum())
    )
    if target_norm <= 0:
        return None
    target_values = np.asarray(target_centered.todense()).ravel()
    target_presence = np.asarray(target_binary.todense()).ravel()
    dots = np.asarray(centered.dot(target_values)).ravel()
    overlap = np.asarray(binary.dot(target_presence)).ravel()
    similarity = dots / (norms * target_norm + 1e-12)
    similarity *= overlap / (overlap + float(shrink))
    similarity[overlap <= 0] = -np.inf
    similarity[norms <= 0] = -np.inf
    return similarity


def latent_similarity(target_vector, latent_matrix, latent_norms):
    if target_vector is None:
        return None
    target = np.asarray(target_vector, dtype=np.float64)
    target_norm = float(np.linalg.norm(target))
    if target_norm <= 0:
        return None
    similarity = latent_matrix @ (target / target_norm)
    similarity[latent_norms <= 0] = -np.inf
    return similarity


def neighbor_order(similarity, user, max_neighbors):
    if similarity is None:
        return np.zeros(0, dtype=np.int64)
    similarity = np.asarray(similarity, dtype=np.float64).copy()
    if 0 <= int(user) < similarity.size:
        similarity[int(user)] = -np.inf
    valid = np.isfinite(similarity)
    if not valid.any():
        return np.zeros(0, dtype=np.int64)
    order = np.argsort(-similarity)
    order = order[valid[order]]
    return order[:int(max_neighbors)].astype(np.int64)


def generate_candidates(
    neighbor_order_values,
    visible_cols,
    n_neighbors,
    top_k,
    topk_cache,
    catalogue_mask,
    fallback_order,
    fallback_size,
):
    if (
        neighbor_order_values is not None
        and neighbor_order_values.size > 0
        and int(n_neighbors) > 0
    ):
        neighbors = neighbor_order_values[:int(n_neighbors)]
        arrays = [
            topk_cache[int(top_k)][int(neighbor)]
            for neighbor in neighbors
            if topk_cache[int(top_k)][int(neighbor)].size > 0
        ]
        if arrays:
            candidates = np.unique(np.concatenate(arrays))
            candidates = candidates[
                catalogue_mask[candidates]
            ]
            if visible_cols.size:
                candidates = candidates[
                    ~np.isin(candidates, visible_cols)
                ]
            if candidates.size:
                return candidates.astype(np.int64), "knn"
    candidates = fallback_order
    if visible_cols.size:
        candidates = candidates[
            ~np.isin(candidates, visible_cols)
        ]
    return (
        candidates[:int(fallback_size)].astype(np.int64),
        "fallback",
    )


def top_k_columns(cols, scores, k=20):
    cols = np.asarray(cols, dtype=np.int64)
    if cols.size == 0:
        return np.zeros(0, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if cols.size <= int(k):
        order = np.lexsort((cols, -scores))
        return cols[order]
    part = np.argpartition(-scores, int(k) - 1)[:int(k)]
    order = part[np.lexsort((cols[part], -scores[part]))]
    return cols[order]


def graded_relevance(ratings, factor):
    ratings = np.asarray(ratings, dtype=np.float64)
    scaled = np.rint(ratings * float(factor)).astype(np.int64)
    return np.maximum(0, scaled - 5)


def ideal_dcg(ratings, factor, k=20):
    relevance = np.sort(graded_relevance(ratings, factor))[::-1][:int(k)]
    if relevance.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, relevance.size + 2))
    gains = np.power(2.0, relevance.astype(np.float64)) - 1.0
    return float(np.sum(gains / discounts))


def ndcg_from_ranked(ranked_cols, truth_map, ideal, factor, k=20):
    if ideal <= 0:
        return None
    dcg = 0.0
    for position, movie in enumerate(ranked_cols[:int(k)], start=1):
        if int(movie) not in truth_map:
            continue
        relevance = float(
            graded_relevance(
                np.array([truth_map[int(movie)]]),
                factor,
            )[0]
        )
        dcg += (2.0 ** relevance - 1.0) / np.log2(position + 1)
    return float(dcg / ideal)


def ndcg_known(top20, truth_map, ideal, factor, k=20):
    if ideal <= 0:
        return None, 0
    dcg = 0.0
    known = 0
    for position, movie in enumerate(top20[:int(k)], start=1):
        if int(movie) not in truth_map:
            continue
        known += 1
        relevance = float(
            graded_relevance(
                np.array([truth_map[int(movie)]]),
                factor,
            )[0]
        )
        dcg += (2.0 ** relevance - 1.0) / np.log2(position + 1)
    return float(dcg / ideal), known


def candidate_recall(candidates, relevant):
    relevant = np.asarray(relevant, dtype=np.int64)
    if relevant.size == 0:
        return None
    return float(
        np.isin(relevant, candidates).sum() / relevant.size
    )


def recall_at_k(top20, relevant, k=20):
    relevant = np.asarray(relevant, dtype=np.int64)
    if relevant.size == 0:
        return None
    return float(
        np.isin(top20[:int(k)], relevant).sum() / relevant.size
    )


def mean_true_rating(top20, truth_map):
    values = [
        float(truth_map[int(movie)])
        for movie in top20
        if int(movie) in truth_map
    ]
    if not values:
        return None, 0
    return float(np.mean(values)), len(values)


def prediction_mae(top20, truth_map, scores):
    errors = [
        abs(float(scores[int(movie)]) - float(truth_map[int(movie)]))
        for movie in top20
        if int(movie) in truth_map
    ]
    if not errors:
        return None
    return float(np.mean(errors))


def config_details(config):
    approach = config[0]
    if approach == "pop":
        return {
            "approach": "popularity",
            "similarity": None,
            "n_neighbors": None,
            "top_k": None,
            "m": float(config[1]),
        }
    if approach == "m5":
        return {
            "approach": "m5_direct",
            "similarity": None,
            "n_neighbors": None,
            "top_k": None,
            "m": None,
        }
    if approach == "knn_m5":
        return {
            "approach": "knn_m5",
            "similarity": str(config[1]),
            "n_neighbors": int(config[2]),
            "top_k": int(config[3]),
            "m": None,
        }
    if approach == "knn_pop":
        return {
            "approach": "knn_popularity",
            "similarity": str(config[1]),
            "n_neighbors": int(config[2]),
            "top_k": int(config[3]),
            "m": float(config[4]),
        }
    raise ValueError(f"Unknown config: {config}")


def config_id(config):
    return "|".join(str(item) for item in config)


def make_configs(pop_ms, neighbors, topks, methods):
    configs = [("m5",)]
    configs.extend(("pop", float(m)) for m in pop_ms)
    for method in methods:
        for n_neighbors in neighbors:
            for top_k in topks:
                configs.append(
                    ("knn_m5", method, int(n_neighbors), int(top_k))
                )
                for m in pop_ms:
                    configs.append(
                        (
                            "knn_pop",
                            method,
                            int(n_neighbors),
                            int(top_k),
                            float(m),
                        )
                    )
    return configs


def build_item_scores(fm, engine, features, observations, n_movies):
    context = features["context_all"]
    cols = observations["col"].to_numpy(dtype=np.int64)
    rows = observations["row"].to_numpy(dtype=np.int64)
    item_obs = np.full(int(n_movies), -1, dtype=np.int64)
    item_obs[cols] = np.arange(cols.size, dtype=np.int64)
    valid_movies = np.where(item_obs >= 0)[0]
    selected_obs = item_obs[valid_movies]
    user_columns = features["context_family_indices"]["user"]
    user_feature = user_columns[rows[selected_obs]]
    item_matrix = context[selected_obs].tolil()
    item_rows = np.arange(valid_movies.size, dtype=np.int64)
    item_matrix[item_rows, user_feature] = 0.0
    item_matrix = item_matrix.tocsr()
    base = engine.predict(item_matrix)
    latent = np.asarray(item_matrix @ engine.V, dtype=np.float64)
    base_items = np.full(int(n_movies), np.nan, dtype=np.float64)
    latent_items = np.zeros(
        (int(n_movies), engine.n_factors),
        dtype=np.float64,
    )
    base_items[valid_movies] = base
    latent_items[valid_movies] = latent
    return {
        "item_obs": item_obs,
        "valid_movies": valid_movies,
        "base": base_items,
        "latent": latent_items,
    }


def build_popularity(
    observations,
    train_indices,
    y_all,
    n_movies,
    pop_ms,
):
    cols = observations["col"].to_numpy(dtype=np.int64)[train_indices]
    ratings = y_all[train_indices].astype(np.float64)
    counts = np.bincount(cols, minlength=int(n_movies)).astype(np.float64)
    sums = np.bincount(
        cols,
        weights=ratings,
        minlength=int(n_movies),
    ).astype(np.float64)
    global_mean = float(ratings.mean())
    means = np.divide(
        sums,
        counts,
        out=np.zeros_like(sums),
        where=counts > 0,
    )
    scores = {}
    for m in pop_ms:
        m = float(m)
        scores[m] = (
            counts / (counts + m) * means
            + m / (counts + m) * global_mean
        )
    return {
        "counts": counts,
        "means": means,
        "global_mean": global_mean,
        "scores": scores,
    }


def warm_item_scores(engine, item_scores, features, user):
    user_feature = features["context_family_indices"]["user"][int(user)]
    return (
        item_scores["base"]
        + engine.w[int(user_feature)]
        + item_scores["latent"] @ engine.V[int(user_feature)]
    )


def train_or_load_m5(
    fm,
    features,
    observations,
    y_all,
    train_indices,
    early_stop_indices,
    spec,
    hyperparameters,
    output_dir,
    load_path,
    reuse,
    signature_payload,
):
    model_path = output_dir / "models" / "M5.npz"
    metadata_path = output_dir / "models" / "M5_metadata.json"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    signature = split_signature(
        np.concatenate([train_indices, early_stop_indices]),
        signature_payload,
    )
    metadata = {}
    training_time = np.nan
    history = pd.DataFrame()
    load_time = 0.0
    best_epoch = 0
    epochs_run = 0
    loaded = False

    if load_path is not None:
        load_start = time.perf_counter()
        loaded_spec, engine = fm.load_model(Path(load_path))
        load_time = time.perf_counter() - load_start
        loaded = True
        sidecar = Path(load_path).with_name(
            Path(load_path).stem + "_metadata.json"
        )
        if sidecar.exists():
            with open(sidecar) as handle:
                metadata = json.load(handle)
        training_time = float(
            metadata.get("training_time_seconds", np.nan)
        )
        best_epoch = int(metadata.get("best_epoch", 0))
        epochs_run = int(metadata.get("epochs_run", 0))
        spec = loaded_spec
    elif reuse and model_path.exists() and metadata_path.exists():
        with open(metadata_path) as handle:
            metadata = json.load(handle)
        if metadata.get("split_signature") == signature:
            load_start = time.perf_counter()
            loaded_spec, engine = fm.load_model(model_path)
            load_time = time.perf_counter() - load_start
            loaded = True
            training_time = float(
                metadata.get("training_time_seconds", np.nan)
            )
            best_epoch = int(metadata.get("best_epoch", 0))
            epochs_run = int(metadata.get("epochs_run", 0))
            spec = loaded_spec

    if loaded:
        if int(spec["n_features"]) != len(features["context_mapping"]):
            raise ValueError(
                "Loaded M5 feature count does not match the current split."
            )
        if spec["feature_set"] != "context":
            raise ValueError("Loaded model is not the contextual M5.")
        if not spec["use_linear"] or not spec["use_latent"]:
            raise ValueError("Loaded M5 must be a full linear + latent FM.")
        if not np.array_equal(
            spec["active_mask"],
            np.ones(len(features["context_mapping"]), dtype=bool),
        ):
            raise ValueError("Loaded M5 active mask is incompatible.")
    else:
        engine, history, best_epoch, epochs_run, training_time = (
            fm.train_fm_model(
                features["context_all"][train_indices],
                y_all[train_indices],
                features["context_all"][early_stop_indices],
                y_all[early_stop_indices],
                n_features=spec["n_features"],
                n_factors=hyperparameters["n_factors"],
                use_linear=spec["use_linear"],
                use_latent=spec["use_latent"],
                active_mask=spec["active_mask"],
                learning_rate=hyperparameters["learning_rate"],
                reg_bias=hyperparameters["reg_bias"],
                reg_factors=hyperparameters["reg_factors"],
                n_epochs=hyperparameters["n_epochs"],
                batch_size=hyperparameters["batch_size"],
                seed=hyperparameters["seed"],
                patience=hyperparameters["patience"],
            )
        )
        fm.save_model(model_path, spec, engine)
        metadata = {
            "model": "M5",
            "training_time_seconds": float(training_time),
            "mean_epoch_time_seconds": float(
                training_time / max(int(epochs_run), 1)
            ),
            "best_epoch": int(best_epoch),
            "epochs_run": int(epochs_run),
            "n_features": int(spec["n_features"]),
            "split_signature": signature,
            **{key: float(value) if isinstance(value, float) else value
               for key, value in hyperparameters.items()},
        }
        write_json(metadata_path, metadata)

    training_summary = {
        "loaded": bool(loaded),
        "load_time_seconds": float(load_time),
        "training_time_seconds": float(training_time)
        if np.isfinite(training_time)
        else None,
        "best_epoch": int(best_epoch),
        "epochs_run": int(epochs_run),
        "n_factors": int(engine.n_factors),
        "n_features": int(engine.n_features),
        "split_signature": signature,
    }
    return engine, spec, history, metadata, training_summary


def criterion_sort(frame):
    ordered = frame.copy()
    for source, target in [
        ("ndcg_known", "_ndcg"),
        ("recall20", "_recall"),
        ("candidate_recall", "_candidate"),
    ]:
        values = pd.to_numeric(ordered[source], errors="coerce")
        ordered[target] = values.fillna(-np.inf)
    return ordered.sort_values(
        ["_ndcg", "_recall", "_candidate", "time_total_mean"],
        ascending=[False, False, False, True],
    )


def aggregate_rows(rows, coverage_masks, catalogue_size, config_lookup):
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame, {}
    group_columns = [
        "scenario",
        "level",
        "config_id",
        "approach",
        "similarity",
        "n_neighbors",
        "top_k",
        "m",
    ]
    aggregated = frame.groupby(group_columns, dropna=False).agg(
        users=("user", "nunique"),
        candidate_recall=("candidate_recall", "mean"),
        candidate_recall_n=("candidate_recall", "count"),
        recall20=("recall20", "mean"),
        recall20_n=("recall20", "count"),
        ndcg_known=("ndcg_known", "mean"),
        ndcg_known_n=("ndcg_known", "count"),
        ndcg_heldout=("ndcg_heldout", "mean"),
        ndcg_heldout_n=("ndcg_heldout", "count"),
        mean_true_rating=("mean_true_rating", "mean"),
        mean_true_rating_n=("mean_true_rating", "count"),
        mae=("mae", "mean"),
        mae_n=("mae", "count"),
        candidate_size=("candidate_size", "mean"),
        known_count=("known_count", "mean"),
        time_similarity=("time_similarity", "mean"),
        time_candidate=("time_candidate", "mean"),
        time_ranking=("time_ranking", "mean"),
        time_adaptation=("time_adaptation", "mean"),
        time_total_mean=("time_total", "mean"),
    ).reset_index()
    aggregated["pct_ndcg_known"] = (
        aggregated["ndcg_known_n"] / aggregated["users"]
    )
    aggregated["pct_recall20"] = (
        aggregated["recall20_n"] / aggregated["users"]
    )
    aggregated["pct_candidate_recall"] = (
        aggregated["candidate_recall_n"] / aggregated["users"]
    )
    aggregated["recall20_excluded"] = (
        aggregated["users"] - aggregated["recall20_n"]
    )
    aggregated["candidate_recall_excluded"] = (
        aggregated["users"] - aggregated["candidate_recall_n"]
    )
    aggregated["ndcg_known_excluded"] = (
        aggregated["users"] - aggregated["ndcg_known_n"]
    )
    coverage_values = []
    coverage_users = []
    for _, row in aggregated.iterrows():
        key = (
            row["scenario"],
            int(row["level"]),
            row["config_id"],
        )
        mask = coverage_masks.get(key)
        if mask is None:
            coverage_values.append(np.nan)
            coverage_users.append(0)
            continue
        coverage_values.append(
            float(mask.sum() / max(catalogue_size, 1))
        )
        coverage_users.append(int(mask.sum()))
    aggregated["catalogue_coverage"] = coverage_values
    aggregated["catalogue_movies_recommended"] = coverage_users
    aggregated["config"] = aggregated["config_id"].map(config_lookup)
    return aggregated, frame


def select_configs(aggregated, approaches, scenario, levels):
    selected = {}
    for level in levels:
        for approach in approaches:
            subset = aggregated[
                (aggregated["scenario"] == scenario)
                & (aggregated["level"] == int(level))
                & (aggregated["approach"] == approach)
            ]
            if subset.empty:
                continue
            ordered = criterion_sort(subset)
            selected[(int(level), approach)] = ordered.iloc[0]["config"]
    return selected


def bootstrap_ci(values, samples, seed):
    values = np.asarray(
        [value for value in values if pd.notna(value)],
        dtype=np.float64,
    )
    if values.size == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.empty(int(samples), dtype=np.float64)
    for index in range(int(samples)):
        draw = rng.choice(values, size=values.size, replace=True)
        means[index] = draw.mean()
    return float(np.percentile(means, 2.5)), float(
        np.percentile(means, 97.5)
    )


def final_metrics_frame(
    rows,
    coverage_masks,
    catalogue_size,
    bootstrap_samples,
    seed,
    scenario,
):
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(), pd.DataFrame()
    group_columns = [
        "scenario",
        "level",
        "config_id",
        "approach",
        "similarity",
        "n_neighbors",
        "top_k",
        "m",
    ]
    records = []
    ci_records = []
    for key, group in frame.groupby(group_columns, dropna=False):
        key_dict = dict(zip(group_columns, key))
        mask_key = (
            key_dict["scenario"],
            int(key_dict["level"]),
            key_dict["config_id"],
        )
        mask = coverage_masks.get(mask_key)
        coverage = (
            float(mask.sum() / max(catalogue_size, 1))
            if mask is not None
            else np.nan
        )
        record = dict(key_dict)
        record["users"] = int(group["user"].nunique())
        record["catalogue_coverage"] = coverage
        record["known_count"] = float(group["known_count"].mean())
        record["candidate_size"] = float(group["candidate_size"].mean())
        record["time_total"] = float(group["time_total"].mean())
        record["time_similarity"] = float(
            group["time_similarity"].mean()
        )
        record["time_candidate"] = float(
            group["time_candidate"].mean()
        )
        record["time_ranking"] = float(group["time_ranking"].mean())
        record["time_adaptation"] = float(
            group["time_adaptation"].mean()
        )
        for metric in [
            "candidate_recall",
            "recall20",
            "ndcg_known",
            "ndcg_heldout",
            "mean_true_rating",
            "mae",
        ]:
            values = group[metric].dropna()
            record[metric] = (
                float(values.mean()) if not values.empty else np.nan
            )
            record[f"{metric}_std"] = (
                float(values.std(ddof=1))
                if values.size > 1
                else 0.0
            )
            record[f"{metric}_n"] = int(values.size)
            low, high = bootstrap_ci(
                values.to_numpy(),
                bootstrap_samples,
                seed + 500000 + len(ci_records),
            )
            ci_records.append({
                **key_dict,
                "metric": metric,
                "mean": record[metric],
                "ci_low": low,
                "ci_high": high,
                "n_users": int(values.size),
            })
        records.append(record)
    wide = pd.DataFrame(records)
    wide["pct_ndcg_known"] = wide["ndcg_known_n"] / wide["users"]
    wide["pct_recall20"] = wide["recall20_n"] / wide["users"]
    wide["pct_candidate_recall"] = (
        wide["candidate_recall_n"] / wide["users"]
    )
    wide["recall20_excluded"] = wide["users"] - wide["recall20_n"]
    wide["candidate_recall_excluded"] = (
        wide["users"] - wide["candidate_recall_n"]
    )
    wide["ndcg_known_excluded"] = (
        wide["users"] - wide["ndcg_known_n"]
    )
    return wide, pd.DataFrame(ci_records)


def df_to_markdown(frame):
    if frame.empty:
        return ""
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                if np.isnan(value):
                    values.append("")
                else:
                    values.append(f"{value:.5f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def normalize_vector(vector):
    if vector is None:
        return None
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm <= 0:
        return None
    return vector / norm


def verify_item_scores(
    engine,
    features,
    observations,
    item_scores,
    seed,
    sample_size=2000,
):
    rng = np.random.default_rng(seed + 140000)
    size = min(int(sample_size), len(observations))
    indices = rng.choice(len(observations), size=size, replace=False)
    user_rows = observations["row"].to_numpy(dtype=np.int64)[indices]
    movie_cols = observations["col"].to_numpy(dtype=np.int64)[indices]
    user_features = features["context_family_indices"]["user"][user_rows]
    expected = engine.predict(features["context_all"][indices])
    predicted = (
        item_scores["base"][movie_cols]
        + engine.w[user_features]
        + np.einsum(
            "ij,ij->i",
            item_scores["latent"][movie_cols],
            engine.V[user_features],
        )
    )
    difference = float(np.max(np.abs(expected - predicted)))
    if difference > 1e-6:
        raise ValueError(
            "Item-score decomposition does not match FMEngine: "
            f"max difference {difference:.3e}."
        )
    return difference


def evaluate_target(
    user,
    scenario,
    level,
    visible_cols,
    hidden_movies,
    hidden_ratings,
    item_scores,
    centered_target,
    binary_target,
    latent_target,
    configs,
    context,
):
    hidden_movies = np.asarray(hidden_movies, dtype=np.int64)
    hidden_ratings = np.asarray(hidden_ratings, dtype=np.float64)
    if hidden_movies.size == 0 or not configs:
        return []
    factor = float(context["factor"])
    threshold = float(context["threshold"])
    truth_map = {
        int(movie): float(rating)
        for movie, rating in zip(hidden_movies, hidden_ratings)
    }
    ideal = ideal_dcg(hidden_ratings, factor)
    relevant = hidden_movies[hidden_ratings > threshold]

    knn_configs = [
        config
        for config in configs
        if config[0] in ("knn_m5", "knn_pop")
    ]
    methods = sorted(set(config[1] for config in knn_configs))
    max_neighbors = {}
    for method in methods:
        max_neighbors[method] = max(
            int(config[2])
            for config in knn_configs
            if config[1] == method
        )

    similarity_times = {}
    neighbor_orders = {}
    for method in methods:
        start = time.perf_counter()
        if method == "centered":
            similarity = centered_similarity(
                centered_target,
                binary_target,
                context["centered"],
                context["binary"],
                context["norms"],
                float(context["shrink"]),
            )
        else:
            similarity = latent_similarity(
                latent_target,
                context["latent_matrix"],
                context["latent_norms"],
            )
        neighbor_orders[method] = neighbor_order(
            similarity,
            int(user),
            max_neighbors[method],
        )
        similarity_times[method] = time.perf_counter() - start

    eligible = None
    eligible_time = 0.0
    eligible_recall = None
    if any(config[0] in ("pop", "m5") for config in configs):
        start = time.perf_counter()
        eligible = context["catalogue_cols"][
            ~np.isin(context["catalogue_cols"], visible_cols)
        ]
        eligible_time = time.perf_counter() - start
        eligible_recall = candidate_recall(eligible, relevant)

    candidate_cache = {}
    heldout_cache = {}
    rows = []
    for config in configs:
        details = config_details(config)
        approach = details["approach"]
        method = details["similarity"]
        if approach in ("popularity", "m5_direct"):
            candidates = eligible
            candidate_time = eligible_time
            cand_recall = eligible_recall
            candidate_source = "all"
        else:
            key = (
                method,
                int(details["n_neighbors"]),
                int(details["top_k"]),
            )
            if key not in candidate_cache:
                start = time.perf_counter()
                candidates, candidate_source = generate_candidates(
                    neighbor_orders.get(method),
                    visible_cols,
                    int(details["n_neighbors"]),
                    int(details["top_k"]),
                    context["topk_cache"],
                    context["catalogue_mask"],
                    context["fallback_order"],
                    int(context["fallback_size"]),
                )
                cand_recall = candidate_recall(candidates, relevant)
                candidate_cache[key] = (
                    candidates,
                    time.perf_counter() - start,
                    cand_recall,
                    candidate_source,
                )
            (
                candidates,
                candidate_time,
                cand_recall,
                candidate_source,
            ) = candidate_cache[key]

        if approach in ("m5_direct", "knn_m5"):
            score_array = item_scores
            score_key = ("m5",)
        else:
            score_array = context["pop_scores"][float(details["m"])]
            score_key = ("pop", float(details["m"]))

        start = time.perf_counter()
        top20 = top_k_columns(
            candidates,
            score_array[candidates],
            20,
        )
        ranking_time = time.perf_counter() - start

        if score_key not in heldout_cache:
            ranked_hidden = top_k_columns(
                hidden_movies,
                score_array[hidden_movies],
                20,
            )
            heldout_cache[score_key] = ndcg_from_ranked(
                ranked_hidden,
                truth_map,
                ideal,
                factor,
            )
        ndcg_heldout = heldout_cache[score_key]
        ndcg_known_value, known_count = ndcg_known(
            top20,
            truth_map,
            ideal,
            factor,
        )
        recall_value = recall_at_k(top20, relevant)
        mean_true, _ = mean_true_rating(top20, truth_map)
        mae_value = (
            prediction_mae(top20, truth_map, score_array)
            if approach in ("m5_direct", "knn_m5")
            else None
        )

        adaptation_time = float(context.get("adaptation_time", 0.0))
        similarity_time = float(similarity_times.get(method, 0.0))
        total_time = (
            similarity_time
            + candidate_time
            + ranking_time
            + adaptation_time
        )
        mask_key = (
            scenario,
            int(level),
            config_id(config),
        )
        mask = context["coverage_masks"].get(mask_key)
        if mask is None:
            mask = np.zeros(int(context["n_movies"]), dtype=bool)
            context["coverage_masks"][mask_key] = mask
        mask[top20] = True

        rows.append({
            "scenario": scenario,
            "level": int(level),
            "user": int(user),
            "config_id": config_id(config),
            **details,
            "candidate_recall": cand_recall,
            "recall20": recall_value,
            "ndcg_known": ndcg_known_value,
            "ndcg_heldout": ndcg_heldout,
            "mean_true_rating": mean_true,
            "mae": mae_value,
            "candidate_size": int(candidates.size),
            "candidate_source": candidate_source,
            "known_count": int(known_count),
            "time_similarity": similarity_time,
            "time_candidate": float(candidate_time),
            "time_ranking": float(ranking_time),
            "time_adaptation": adaptation_time,
            "time_total": float(total_time),
        })
    return rows


def evaluate_warm(
    users,
    hidden_by_user,
    visible_matrix,
    similarity_structures,
    latent_matrix,
    latent_norms,
    engine,
    features,
    item_scores,
    observations,
    y_all,
    configs,
    context,
):
    rows = []
    for user in np.asarray(users, dtype=np.int64):
        visible_row = visible_matrix.getrow(int(user))
        visible_cols = visible_row.indices.astype(np.int64)
        binary_target = visible_row.copy()
        binary_target.data[:] = 1.0
        centered_target = similarity_structures["centered"].getrow(
            int(user)
        )
        latent_target = normalize_vector(latent_matrix[int(user)])
        m5_scores = warm_item_scores(
            engine,
            item_scores,
            features,
            int(user),
        )
        hidden_indices = hidden_by_user[int(user)]
        hidden_movies = observations["col"].to_numpy(
            dtype=np.int64
        )[hidden_indices]
        hidden_ratings = y_all[hidden_indices]
        rows.extend(
            evaluate_target(
                int(user),
                "warm",
                0,
                visible_cols,
                hidden_movies,
                hidden_ratings,
                m5_scores,
                centered_target,
                binary_target,
                latent_target,
                configs,
                {**context, "adaptation_time": 0.0},
            )
        )
    return rows


def evaluate_cold(
    users,
    levels,
    configs_by_level,
    split,
    observations,
    y_all,
    spec,
    features,
    engine,
    fm,
    evalm5,
    item_scores,
    context,
):
    rows = []
    for user in np.asarray(users, dtype=np.int64):
        user = int(user)
        pool = np.asarray(
            split["pool_order_by_user"][user],
            dtype=np.int64,
        )
        hidden = np.asarray(
            split["eval_rows_by_user"][user],
            dtype=np.int64,
        )
        if hidden.size == 0:
            continue
        hidden_movies = observations["col"].to_numpy(
            dtype=np.int64
        )[hidden]
        hidden_ratings = y_all[hidden]
        feature_index = fm.user_feature_index(
            spec,
            user,
            features["context_family_indices"],
        )
        base_pool, latent_pool = evalm5.prepare_rows(
            engine,
            features["context_all"],
            pool,
            feature_index,
        )
        pool_ratings = y_all[pool]
        pool_cols = observations["col"].to_numpy(dtype=np.int64)[pool]
        for level in levels:
            n_revealed = int(level)
            if n_revealed > pool.size:
                continue
            start = time.perf_counter()
            if n_revealed == 0:
                theta = np.zeros(
                    engine.n_factors + 1,
                    dtype=np.float64,
                )
            else:
                system, target = evalm5.build_ridge_system(
                    base_pool[:n_revealed],
                    latent_pool[:n_revealed],
                    pool_ratings[:n_revealed],
                )
                theta = evalm5.solve_ridge(
                    system,
                    target,
                    float(context["cold_reg_bias"]),
                    float(context["cold_reg_factors"]),
                )
            adaptation_time = time.perf_counter() - start
            item_scores_n = (
                item_scores["base"]
                + theta[0]
                + item_scores["latent"] @ theta[1:]
            )
            visible_cols = pool_cols[:n_revealed]
            centered_target, binary_target = build_target_csr(
                visible_cols,
                pool_ratings[:n_revealed],
                context["n_movies"],
            )
            latent_target = normalize_vector(theta[1:])
            level_configs = configs_by_level.get(int(level), [])
            rows.extend(
                evaluate_target(
                    user,
                    "cold",
                    n_revealed,
                    visible_cols,
                    hidden_movies,
                    hidden_ratings,
                    item_scores_n,
                    centered_target,
                    binary_target,
                    latent_target,
                    level_configs,
                    {**context, "adaptation_time": adaptation_time},
                )
            )
    return rows


def choose_example_user(users, sizes_by_user):
    best_user = None
    best_size = -1
    for user in np.asarray(users, dtype=np.int64):
        size = int(sizes_by_user.get(int(user), 0))
        if size > best_size:
            best_user = int(user)
            best_size = size
    return best_user


def build_example(
    user,
    scenario,
    level,
    config,
    visible_cols,
    visible_ratings,
    hidden_movies,
    hidden_ratings,
    item_scores,
    centered_target,
    binary_target,
    latent_target,
    context,
    movie_titles,
    movie_ids,
    user_ids,
    factor,
    threshold,
):
    details = config_details(config)
    approach = details["approach"]
    method = details["similarity"]
    if approach in ("knn_m5", "knn_popularity"):
        if method == "centered":
            similarity = centered_similarity(
                centered_target,
                binary_target,
                context["centered"],
                context["binary"],
                context["norms"],
                float(context["shrink"]),
            )
        else:
            similarity = latent_similarity(
                latent_target,
                context["latent_matrix"],
                context["latent_norms"],
            )
        order = neighbor_order(
            similarity,
            int(user),
            int(details["n_neighbors"]),
        )
        candidates, source = generate_candidates(
            order,
            visible_cols,
            int(details["n_neighbors"]),
            int(details["top_k"]),
            context["topk_cache"],
            context["catalogue_mask"],
            context["fallback_order"],
            int(context["fallback_size"]),
        )
        neighbor_rows = []
        for neighbor in order[:5]:
            neighbor_rows.append({
                "neighbor_row": int(neighbor),
                "neighbor_user_id": user_ids[int(neighbor)],
                "similarity": float(similarity[int(neighbor)])
                if similarity is not None
                else np.nan,
            })
    else:
        candidates = context["catalogue_cols"][
            ~np.isin(context["catalogue_cols"], visible_cols)
        ]
        source = "all"
        neighbor_rows = []
    if approach in ("m5_direct", "knn_m5"):
        score_array = item_scores
        score_type = "m5_prediction"
    else:
        score_array = context["pop_scores"][float(details["m"])]
        score_type = "popularity_score"
    top20 = top_k_columns(candidates, score_array[candidates], 20)
    truth_map = {
        int(movie): float(rating)
        for movie, rating in zip(hidden_movies, hidden_ratings)
    }
    top_rows = []
    for rank, movie in enumerate(top20, start=1):
        true_rating = truth_map.get(int(movie))
        top_rows.append({
            "rank": rank,
            "movie_col": int(movie),
            "movie_id": movie_ids[int(movie)],
            "title": movie_titles[int(movie)],
            "score": float(score_array[int(movie)]),
            "score_type": score_type,
            "true_rating": true_rating,
            "relevant": (
                bool(true_rating > float(threshold))
                if true_rating is not None
                else None
            ),
            "known_in_test": true_rating is not None,
            "candidate_source": source,
            "candidate_size": int(candidates.size),
        })
    visible_rows = []
    for movie, rating in zip(visible_cols, visible_ratings):
        visible_rows.append({
            "movie_col": int(movie),
            "movie_id": movie_ids[int(movie)],
            "title": movie_titles[int(movie)],
            "rating": float(rating),
        })
    return (
        pd.DataFrame(top_rows),
        pd.DataFrame(neighbor_rows),
        pd.DataFrame(visible_rows),
    )


def plot_warm_comparison(frame, path):
    if frame.empty:
        return
    approaches = [
        "popularity",
        "m5_direct",
        "knn_m5",
        "knn_popularity",
    ]
    metrics = [
        ("candidate_recall", "Candidate Recall"),
        ("recall20", "Recall@20"),
        ("ndcg_known", "Known NDCG@20"),
        ("time_total", "Temps (s)"),
    ]
    figure, axes = plt.subplots(1, 4, figsize=(20, 5))
    colors = {
        "popularity": "tab:gray",
        "m5_direct": "tab:blue",
        "knn_m5": "tab:red",
        "knn_popularity": "tab:orange",
    }
    for axis, (metric, title) in zip(axes, metrics):
        values = []
        labels = []
        bar_colors = []
        for approach in approaches:
            subset = frame[frame["approach"] == approach]
            if subset.empty:
                continue
            values.append(float(subset.iloc[0][metric]))
            labels.append(approach)
            bar_colors.append(colors[approach])
        axis.bar(labels, values, color=bar_colors)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=25)
        axis.grid(True, axis="y", alpha=0.3)
    figure.suptitle("Warm-start — comparaison des approches")
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def plot_cold_curves(frame, path):
    if frame.empty:
        return
    approaches = [
        "popularity",
        "m5_direct",
        "knn_m5",
        "knn_popularity",
    ]
    colors = {
        "popularity": "tab:gray",
        "m5_direct": "tab:blue",
        "knn_m5": "tab:red",
        "knn_popularity": "tab:orange",
    }
    metrics = [
        ("ndcg_known", "Known NDCG@20"),
        ("recall20", "Recall@20"),
        ("candidate_recall", "Candidate Recall"),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(19, 5))
    for axis, (metric, title) in zip(axes, metrics):
        for approach in approaches:
            subset = frame[frame["approach"] == approach].sort_values(
                "level"
            )
            if subset.empty:
                continue
            standard_error = (
                subset[f"{metric}_std"]
                / np.sqrt(np.maximum(subset[f"{metric}_n"], 1))
            )
            axis.errorbar(
                subset["level"],
                subset[metric],
                yerr=standard_error,
                marker="o",
                capsize=3,
                color=colors[approach],
                label=approach,
            )
        axis.set_title(title)
        axis.set_xscale("symlog", linthresh=1)
        axis.set_xlabel("Notes révélées")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)
    figure.suptitle("Cold-start — qualité selon le profil révélé")
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def plot_knn_heatmaps(frame, path):
    if frame.empty:
        return
    figure, axes = plt.subplots(2, 2, figsize=(15, 11))

    def draw_heatmap(axis, subset, value, title, cmap, label):
        pivot = subset.pivot_table(
            index="n_neighbors",
            columns="top_k",
            values=value,
            aggfunc="mean",
        ).sort_index().sort_index(axis=1)
        image = axis.imshow(pivot.to_numpy(), aspect="auto", cmap=cmap)
        axis.set_xticks(range(pivot.shape[1]))
        axis.set_xticklabels([str(item) for item in pivot.columns])
        axis.set_yticks(range(pivot.shape[0]))
        axis.set_yticklabels([str(item) for item in pivot.index])
        axis.set_xlabel("k films par voisin")
        axis.set_ylabel("N voisins")
        axis.set_title(title)
        for row in range(pivot.shape[0]):
            for column in range(pivot.shape[1]):
                item = pivot.iloc[row, column]
                if pd.notna(item):
                    text = f"{item:.3f}" if value == "ndcg_known" else f"{item:.0f}"
                    axis.text(
                        column,
                        row,
                        text,
                        ha="center",
                        va="center",
                        color="white",
                        fontsize=8,
                    )
        figure.colorbar(image, ax=axis, label=label)

    best_methods = {}
    for approach in ("knn_m5", "knn_popularity"):
        subset = frame[frame["approach"] == approach]
        if subset.empty:
            continue
        best_method = (
            subset.groupby("similarity")["ndcg_known"]
            .mean()
            .sort_values(ascending=False)
            .index[0]
        )
        best_methods[approach] = best_method
        subset = subset[subset["similarity"] == best_method]
        title = approach
        if approach == "knn_popularity":
            best_m = (
                subset.groupby("m")["ndcg_known"]
                .mean()
                .sort_values(ascending=False)
                .index[0]
            )
            subset = subset[subset["m"] == best_m]
            title = f"{title} (m={best_m:g})"
        target_axis = axes[0, 0] if approach == "knn_m5" else axes[0, 1]
        draw_heatmap(
            target_axis,
            subset,
            "ndcg_known",
            f"{title} — {best_method}",
            "viridis",
            "Known NDCG@20",
        )

    subset_size = frame[frame["approach"] == "knn_m5"]
    if not subset_size.empty:
        method = best_methods.get("knn_m5")
        subset_size = subset_size[subset_size["similarity"] == method]
        draw_heatmap(
            axes[1, 0],
            subset_size,
            "candidate_size",
            f"Taille des candidats — {method}",
            "magma",
            "Candidats",
        )

    comparison = (
        frame[frame["approach"].isin(["knn_m5", "knn_popularity"])]
        .groupby(["approach", "similarity"])["ndcg_known"]
        .mean()
        .reset_index()
    )
    if not comparison.empty:
        methods = sorted(comparison["similarity"].dropna().unique())
        width = 0.35
        positions = np.arange(len(methods))
        for offset, approach in enumerate(
            ["knn_m5", "knn_popularity"]
        ):
            subset = comparison[comparison["approach"] == approach]
            values = [
                float(
                    subset[subset["similarity"] == method][
                        "ndcg_known"
                    ].mean()
                )
                for method in methods
            ]
            axes[1, 1].bar(
                positions + (offset - 0.5) * width,
                values,
                width=width,
                label=approach,
            )
        axes[1, 1].set_xticks(positions)
        axes[1, 1].set_xticklabels(methods)
        axes[1, 1].set_ylabel("Known NDCG@20")
        axes[1, 1].set_title("Influence de la similarité")
        axes[1, 1].legend()
        axes[1, 1].grid(True, axis="y", alpha=0.3)

    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def plot_popularity(frame, path):
    if frame.empty:
        return
    subset = frame[frame["approach"] == "popularity"].sort_values("m")
    if subset.empty:
        return
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(subset["m"], subset["ndcg_known"], marker="o", label="NDCG@20")
    axis.plot(subset["m"], subset["recall20"], marker="s", label="Recall@20")
    axis.set_xscale("log")
    axis.set_xlabel("m (shrinkage)")
    axis.set_ylabel("Métrique")
    axis.set_title("Popularité — influence de m")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def plot_timing(frame, path):
    if frame.empty:
        return
    figure, axis = plt.subplots(figsize=(8, 5))
    approaches = frame["approach"].to_list()
    values = frame["time_total"].to_list()
    axis.bar(approaches, values, color="tab:purple")
    axis.set_ylabel("Temps moyen par utilisateur (s)")
    axis.set_title("Coût de génération du Top-20")
    axis.tick_params(axis="x", rotation=25)
    axis.grid(True, axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def selected_payload(selected):
    payload = {}
    for (level, approach), config in selected.items():
        payload[f"{int(level)}|{approach}"] = config_details(config)
    return payload


def write_summary(
    path,
    warm_wide,
    cold_wide,
    warm_ci,
    cold_ci,
    selected_warm,
    selected_cold,
    args,
    hyperparameters,
    training_summary,
    rating_info,
    split_info,
):
    lines = [
        "# FilmFinder — Recommendation system evaluation",
        "",
        "## Run configuration",
        "",
        f"- M5 hyperparameters: `{hyperparameters}`",
        f"- Rating scale: {rating_info['min']:g}–"
        f"{rating_info['max']:g}, relevant threshold "
        f"{rating_info['threshold']:g}",
        f"- Seeds: split={args.seed}, M5={hyperparameters['seed']}",
        f"- Cold validation fraction: {args.cold_validation_fraction}",
        f"- Warm hidden fraction: {args.warm_hidden_fraction}",
        f"- M5 loaded: {training_summary['loaded']}",
        "",
        "## Dataset split",
        "",
        f"```json\n{json.dumps(split_info, indent=2, default=str)}\n```",
        "",
        "## Warm-start results",
        "",
    ]
    warm_columns = [
        "approach",
        "candidate_recall",
        "recall20",
        "ndcg_known",
        "ndcg_heldout",
        "mean_true_rating",
        "mae",
        "catalogue_coverage",
        "candidate_size",
        "time_total",
        "users",
        "ndcg_known_n",
        "pct_ndcg_known",
        "known_count",
        "recall20_excluded",
        "candidate_recall_excluded",
        "ndcg_known_excluded",
    ]
    if not warm_wide.empty:
        warm_display = warm_wide[warm_columns].copy()
        warm_display = warm_display.sort_values(
            "ndcg_known",
            ascending=False,
        )
        lines.append(df_to_markdown(warm_display))
        best_warm = warm_display.iloc[0]
        lines.extend([
            "",
            f"Meilleure approche warm : **{best_warm['approach']}** "
            f"(Known NDCG@20={best_warm['ndcg_known']:.4f}, "
            f"Recall@20={best_warm['recall20']:.4f}, "
            f"temps={best_warm['time_total']:.4f}s).",
        ])
    lines.extend([
        "",
        "## Cold-start results",
        "",
    ])
    cold_columns = [
        "approach",
        "level",
        "candidate_recall",
        "recall20",
        "ndcg_known",
        "ndcg_heldout",
        "mean_true_rating",
        "mae",
        "catalogue_coverage",
        "candidate_size",
        "time_total",
        "users",
        "ndcg_known_n",
        "pct_ndcg_known",
        "known_count",
        "recall20_excluded",
        "candidate_recall_excluded",
        "ndcg_known_excluded",
    ]
    if not cold_wide.empty:
        cold_display = cold_wide[cold_columns].copy()
        cold_display = cold_display.sort_values(
            ["approach", "level"],
        )
        lines.append(df_to_markdown(cold_display))
        cold_by_approach = (
            cold_wide.groupby("approach")["ndcg_known"]
            .mean()
            .sort_values(ascending=False)
        )
        if not cold_by_approach.empty:
            lines.extend([
                "",
                "Meilleure approche cold (moyenne sur les niveaux) : "
                f"**{cold_by_approach.index[0]}** "
                f"(Known NDCG@20={cold_by_approach.iloc[0]:.4f}).",
            ])
    warm_best_name = None
    warm_best_ndcg = None
    warm_best_time = None
    if not warm_wide.empty:
        warm_best_row = warm_wide.sort_values(
            "ndcg_known",
            ascending=False,
        ).iloc[0]
        warm_best_name = warm_best_row["approach"]
        warm_best_ndcg = float(warm_best_row["ndcg_known"])
        warm_best_time = float(warm_best_row["time_total"])
    cold_best_name = None
    cold_best_ndcg = None
    if not cold_wide.empty:
        cold_scores = (
            cold_wide.groupby("approach")["ndcg_known"]
            .mean()
            .sort_values(ascending=False)
        )
        if not cold_scores.empty:
            cold_best_name = cold_scores.index[0]
            cold_best_ndcg = float(cold_scores.iloc[0])
    lines.extend([
        "",
        "## Recommendation",
        "",
    ])
    if warm_best_name is not None:
        lines.append(
            f"- Warm-start: **{warm_best_name}** is best on known NDCG@20 "
            f"({warm_best_ndcg:.4f}) with {warm_best_time:.4f}s per user."
        )
    if cold_best_name is not None:
        lines.append(
            f"- Cold-start: **{cold_best_name}** has the best mean known "
            f"NDCG@20 across revealed-profile levels ({cold_best_ndcg:.4f})."
        )
    if warm_best_name is not None and cold_best_name is not None:
        if warm_best_name == cold_best_name:
            lines.append(
                f"- For FilmFinder, prefer **{warm_best_name}** if its "
                "latency and memory fit the web deployment budget; it is "
                "the best compromise observed in both scenarios."
            )
        else:
            lines.append(
                "- The warm and cold winners differ. Choose based on the "
                "dominant traffic: M5 direct is the simplest deployment "
                "candidate, while KNN should only be retained if its "
                "quality gain clearly offsets candidate-generation latency."
            )
    lines.extend([
        "",
        "## Selected configurations",
        "",
        "### Warm",
        "",
        f"```json\n"
        f"{json.dumps(selected_payload(selected_warm), indent=2, default=str)}"
        f"\n```",
        "",
        "### Cold",
        "",
        f"```json\n"
        f"{json.dumps(selected_payload(selected_cold), indent=2, default=str)}"
        f"\n```",
        "",
        "## Interpretation",
        "",
        "- The known-movie NDCG@20 is conditional: unknown movies are "
        "removed from the ranking, so it must be read together with the "
        "number of known recommendations and the fraction of evaluable "
        "users.",
        "- Known-movie NDCG keeps the original Top-20 positions of the "
        "retained movies; missing known movies are not re-indexed. The "
        "ideal ranking (IDCG@20) is the user's full hidden test set "
        "sorted by graded relevance descending, truncated to 20.",
        "- The held-out ranking NDCG@20 is a separate diagnostic: all "
        "hidden test movies are ranked by the approach score, without "
        "candidate generation.",
        "- Warm and cold cohorts are disjoint; their metrics are not a "
        "causal comparison.",
        "- M5 hyperparameters were fixed; only recommendation "
        "hyperparameters were tuned on validation cohorts.",
        "- KNN + M5 is not assumed to be superior. The best deployment "
        "choice should balance known NDCG@20, Recall@20, candidate recall, "
        "cold-start behavior, latency and implementation complexity.",
        "",
        "## Metrics with bootstrap confidence intervals",
        "",
    ])
    if not warm_ci.empty:
        lines.extend([
            "### Warm",
            "",
            df_to_markdown(warm_ci),
            "",
        ])
    if not cold_ci.empty:
        lines.extend([
            "### Cold",
            "",
            df_to_markdown(cold_ci),
            "",
        ])
    path.write_text("\n".join(lines))


def detect_rating_scale(ratings, explicit_threshold):
    values = np.asarray(ratings, dtype=np.float64)
    max_rating = float(np.nanmax(values))
    min_rating = float(np.nanmin(values))
    factor = 2.0 if max_rating <= 5.5 else 1.0
    default_threshold = 4.0 if max_rating <= 5.5 else 8.0
    threshold = (
        float(explicit_threshold)
        if explicit_threshold is not None
        else default_threshold
    )
    return {
        "min": min_rating,
        "max": max_rating,
        "factor": factor,
        "threshold": threshold,
    }


def main():
    args = parse_args()
    fast = bool(args.fast)
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
        else DEFAULT_OUTPUT_DIR
    )
    scripts_dir = Path(
        args.scripts_dir
        if args.scripts_dir is not None
        else DEFAULT_SCRIPTS_DIR
    )
    for subdir in [
        "models",
        "splits",
        "warm",
        "cold",
        "figures",
        "examples",
    ]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    hyperparameters = resolve_hyperparameters(args)
    levels = parse_int_list(
        args.reveal_levels,
        FAST_LEVELS if fast else DEFAULT_LEVELS,
        "reveal levels",
    )
    pop_ms = parse_float_list(
        args.popularity_m_grid,
        FAST_POP_M if fast else DEFAULT_POP_M,
        "popularity m grid",
    )
    neighbors = parse_int_list(
        args.knn_neighbors_grid,
        FAST_KNN_N if fast else DEFAULT_KNN_N,
        "KNN neighbors grid",
    )
    topks = parse_int_list(
        args.knn_topk_grid,
        FAST_KNN_K if fast else DEFAULT_KNN_K,
        "KNN top-k grid",
    )
    methods = parse_methods(args.similarity_methods)
    bootstrap_samples = int(
        args.bootstrap_samples
        if args.bootstrap_samples is not None
        else (100 if fast else 1000)
    )
    warm_validation_users = int(
        args.warm_validation_users
        if args.warm_validation_users is not None
        else (60 if fast else 300)
    )
    warm_test_users = int(
        args.warm_test_users
        if args.warm_test_users is not None
        else (60 if fast else 300)
    )
    max_users = (
        args.max_users
        if args.max_users is not None
        else (200 if fast else None)
    )
    max_interactions = (
        args.max_interactions_per_user
        if args.max_interactions_per_user is not None
        else (50 if fast else None)
    )
    if not 0 < args.cold_eval_fraction < 1:
        raise ValueError("--cold-eval-fraction must be in (0, 1).")
    if not 0 < args.warm_hidden_fraction < 1:
        raise ValueError("--warm-hidden-fraction must be in (0, 1).")
    if not 0 < args.cold_validation_fraction < 1:
        raise ValueError("--cold-validation-fraction must be in (0, 1).")
    if bootstrap_samples < 1:
        raise ValueError("--bootstrap-samples must be positive.")
    if warm_validation_users < 1 or warm_test_users < 1:
        raise ValueError("Warm cohort sizes must be positive.")
    if args.fallback_popularity_m <= 0:
        raise ValueError("--fallback-popularity-m must be positive.")
    if args.fallback_candidates < 0:
        raise ValueError("--fallback-candidates must be non-negative.")
    if args.similarity_shrink < 0:
        raise ValueError("--similarity-shrink must be non-negative.")
    if args.min_visible_per_user < 0 or args.min_hidden_per_user < 0:
        raise ValueError("Minimum visible/hidden notes must be non-negative.")

    print("Loading project modules...")
    fm = load_module(
        scripts_dir / "6.IncrementalVsRetraining.py",
        "_filmfinder_fm",
    )
    evalm5 = load_module(
        scripts_dir / "7.EvaluateM5.py",
        "_filmfinder_evalm5",
    )

    print("Loading observations and split...")
    data = fm.load_inputs(cleaning_dir)
    observations = fm.cap_observations(
        data["observations"],
        max_users,
        max_interactions,
        args.seed,
    )
    if len(observations) < 20:
        raise ValueError("Not enough observations after filtering.")
    split = fm.split_users(
        observations,
        args.seed,
        float(args.cold_eval_fraction),
    )
    user_rows = observations["row"].to_numpy(dtype=np.int64)
    movie_cols = observations["col"].to_numpy(dtype=np.int64)
    y_all = observations["rating"].to_numpy(dtype=np.float64)
    n_users = int(data["n_users"])
    n_movies = int(data["n_movies"])

    rating_info = detect_rating_scale(
        y_all,
        args.relevance_threshold,
    )
    print(
        "Rating scale: "
        f"{rating_info['min']:g}-{rating_info['max']:g}, "
        f"relevance threshold={rating_info['threshold']:g}"
    )

    print("Building warm and cold cohorts...")
    warm = split_warm_cohorts(
        observations,
        split["train_indices"],
        args.seed,
        warm_validation_users,
        warm_test_users,
        float(args.warm_hidden_fraction),
        int(args.min_visible_per_user),
        int(args.min_hidden_per_user),
    )
    cold = split_cold_cohorts(
        split["test_users"],
        args.seed,
        float(args.cold_validation_fraction),
    )
    warm_cohort_users = set(
        warm["validation_users"].tolist()
        + warm["test_users"].tolist()
    )
    warm_hidden_indices = np.concatenate(
        [
            warm["hidden_by_user"][int(user)]
            for user in sorted(warm["hidden_by_user"])
        ]
    )
    train_visible_indices = np.setdiff1d(
        split["train_indices"],
        warm_hidden_indices,
        assume_unique=False,
    )
    if train_visible_indices.size < 10:
        raise ValueError("Not enough training interactions after warm split.")
    val_indices = split["val_indices"]
    val_users = user_rows[val_indices]
    early_stop_indices = val_indices[
        ~np.isin(val_users, list(warm_cohort_users))
    ]
    if early_stop_indices.size == 0:
        raise ValueError("Early-stop validation set is empty.")
    cold_users = np.asarray(split["test_users"], dtype=np.int64)
    if np.intersect1d(
        np.unique(user_rows[train_visible_indices]),
        cold_users,
    ).size:
        raise ValueError("Cold users leaked into M5 training.")
    if np.intersect1d(
        np.unique(user_rows[early_stop_indices]),
        cold_users,
    ).size:
        raise ValueError("Cold users leaked into M5 validation.")
    if np.intersect1d(warm_hidden_indices, train_visible_indices).size:
        raise ValueError("Warm hidden notes leaked into M5 training.")
    if np.intersect1d(warm_hidden_indices, early_stop_indices).size:
        raise ValueError("Warm hidden notes leaked into early stopping.")

    pd.DataFrame(warm["cohort_rows"]).to_csv(
        output_dir / "splits" / "warm_cohorts.csv",
        index=False,
    )
    cold_rows = []
    for cohort, users in (
        ("cold_validation", cold["validation_users"]),
        ("cold_test", cold["test_users"]),
    ):
        for user in users:
            cold_rows.append({
                "user_row": int(user),
                "cohort": cohort,
                "n_eval": int(
                    len(split["eval_rows_by_user"][int(user)])
                ),
                "n_pool": int(
                    len(split["pool_order_by_user"][int(user)])
                ),
            })
    pd.DataFrame(cold_rows).to_csv(
        output_dir / "splits" / "cold_cohorts.csv",
        index=False,
    )
    split_info = {
        "n_observations": int(len(observations)),
        "n_users": n_users,
        "n_movies": n_movies,
        "n_train_original": int(split["train_indices"].size),
        "n_train_visible": int(train_visible_indices.size),
        "n_warm_hidden": int(warm_hidden_indices.size),
        "n_early_stop": int(early_stop_indices.size),
        "n_warm_validation_users": int(
            warm["validation_users"].size
        ),
        "n_warm_test_users": int(warm["test_users"].size),
        "n_cold_validation_users": int(
            cold["validation_users"].size
        ),
        "n_cold_test_users": int(cold["test_users"].size),
        "n_test_users_original": int(cold_users.size),
        "seed": int(args.seed),
        "cold_eval_fraction": float(args.cold_eval_fraction),
        "warm_hidden_fraction": float(args.warm_hidden_fraction),
        "cold_validation_fraction": float(
            args.cold_validation_fraction
        ),
    }
    write_json(output_dir / "splits" / "split_summary.json", split_info)
    np.save(
        output_dir / "splits" / "train_visible_indices.npy",
        train_visible_indices,
    )
    np.save(
        output_dir / "splits" / "warm_hidden_indices.npy",
        warm_hidden_indices,
    )
    np.save(
        output_dir / "splits" / "early_stop_indices.npy",
        early_stop_indices,
    )

    print("Building contextual features...")
    max_studios = resolve_max_studios(
        fm,
        model_output_dir,
        args.max_studios,
    )
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
    specs = fm.make_model_specs(
        n_users,
        n_movies,
        len(features["context_mapping"]),
    )
    spec = next(item for item in specs if item["name"] == "M5")

    print("Training or loading M5...")
    signature_payload = {
        "hyperparameters": hyperparameters,
        "max_studios": int(max_studios),
        "n_users": n_users,
        "n_movies": n_movies,
        "n_features": len(features["context_mapping"]),
        "seed": int(args.seed),
    }
    (
        engine,
        spec,
        history,
        metadata,
        training_summary,
    ) = train_or_load_m5(
        fm,
        features,
        observations,
        y_all,
        train_visible_indices,
        early_stop_indices,
        spec,
        hyperparameters,
        output_dir,
        args.load_m5,
        bool(args.reuse_m5),
        signature_payload,
    )
    if not history.empty:
        history.to_csv(
            output_dir / "models" / "training_history.csv",
            index=False,
        )
    write_json(
        output_dir / "models" / "training_summary.json",
        training_summary,
    )
    print(
        f"  features={engine.n_features}, factors={engine.n_factors}, "
        f"loaded={training_summary['loaded']}, "
        f"training_time={training_summary['training_time_seconds']}"
    )
    if training_summary["loaded"]:
        print(
            "  Warning: the loaded M5 must share the current feature "
            "layout and year standardization."
        )

    print("Precomputing item scores and statistics...")
    item_scores = build_item_scores(
        fm,
        engine,
        features,
        observations,
        n_movies,
    )
    score_difference = verify_item_scores(
        engine,
        features,
        observations,
        item_scores,
        args.seed,
    )
    popularity_ms = sorted(set(pop_ms + [float(args.fallback_popularity_m)]))
    popularity = build_popularity(
        observations,
        train_visible_indices,
        y_all,
        n_movies,
        popularity_ms,
    )
    catalogue_mask = popularity["counts"] > 0
    catalogue_cols = np.where(catalogue_mask)[0].astype(np.int64)
    if catalogue_cols.size == 0:
        raise ValueError("Empty catalogue after building popularity.")
    visible_matrix = build_visible_matrix(
        observations,
        train_visible_indices,
        y_all,
        n_users,
        n_movies,
    )
    similarity_structures = build_similarity_structures(
        visible_matrix,
        n_users,
    )
    topk_cache = build_topk_cache(
        visible_matrix,
        topks,
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
    fallback_scores = popularity["scores"][
        float(args.fallback_popularity_m)
    ]
    fallback_order = catalogue_cols[
        np.argsort(-fallback_scores[catalogue_cols])
    ]
    context = {
        "centered": similarity_structures["centered"],
        "binary": similarity_structures["binary"],
        "norms": similarity_structures["norms"],
        "latent_matrix": latent_normalized,
        "latent_norms": latent_norms,
        "topk_cache": topk_cache,
        "catalogue_mask": catalogue_mask,
        "catalogue_cols": catalogue_cols,
        "pop_scores": popularity["scores"],
        "fallback_order": fallback_order,
        "fallback_size": int(args.fallback_candidates),
        "factor": float(rating_info["factor"]),
        "threshold": float(rating_info["threshold"]),
        "shrink": float(args.similarity_shrink),
        "coverage_masks": {},
        "n_movies": int(n_movies),
        "cold_reg_bias": float(args.cold_adapt_reg_bias),
        "cold_reg_factors": float(args.cold_adapt_reg_factors),
    }

    approaches = [
        "popularity",
        "m5_direct",
        "knn_m5",
        "knn_popularity",
    ]
    all_configs = make_configs(pop_ms, neighbors, topks, methods)
    config_lookup = {
        config_id(config): config for config in all_configs
    }

    print("Evaluating warm validation cohort...")
    coverage_warm_validation = {}
    rows_warm_validation = evaluate_warm(
        warm["validation_users"],
        warm["hidden_by_user"],
        visible_matrix,
        similarity_structures,
        latent_normalized,
        latent_norms,
        engine,
        features,
        item_scores,
        observations,
        y_all,
        all_configs,
        {**context, "coverage_masks": coverage_warm_validation},
    )
    warm_validation, _ = aggregate_rows(
        rows_warm_validation,
        coverage_warm_validation,
        catalogue_cols.size,
        config_lookup,
    )
    warm_validation.to_csv(
        output_dir / "warm" / "grid.csv",
        index=False,
    )
    del rows_warm_validation
    if warm_validation.empty:
        raise ValueError("Warm validation produced no metrics.")
    selected_warm = select_configs(
        warm_validation,
        approaches,
        "warm",
        [0],
    )
    write_json(
        output_dir / "warm" / "selected_configs.json",
        selected_payload(selected_warm),
    )

    print("Evaluating warm test cohort...")
    final_configs_warm = [
        selected_warm[(0, approach)]
        for approach in approaches
        if (0, approach) in selected_warm
    ]
    coverage_warm_test = {}
    rows_warm_test = evaluate_warm(
        warm["test_users"],
        warm["hidden_by_user"],
        visible_matrix,
        similarity_structures,
        latent_normalized,
        latent_norms,
        engine,
        features,
        item_scores,
        observations,
        y_all,
        final_configs_warm,
        {**context, "coverage_masks": coverage_warm_test},
    )
    warm_wide, warm_ci = final_metrics_frame(
        rows_warm_test,
        coverage_warm_test,
        catalogue_cols.size,
        bootstrap_samples,
        args.seed + 210000,
        "warm",
    )
    warm_wide.to_csv(
        output_dir / "warm" / "final_metrics.csv",
        index=False,
    )
    warm_ci.to_csv(
        output_dir / "warm" / "final_metrics_ci.csv",
        index=False,
    )
    pd.DataFrame(rows_warm_test).to_csv(
        output_dir / "warm" / "final_per_user.csv",
        index=False,
    )

    print("Evaluating cold validation cohort...")
    cold_configs_by_level = {
        int(level): all_configs for level in levels
    }
    coverage_cold_validation = {}
    rows_cold_validation = evaluate_cold(
        cold["validation_users"],
        levels,
        cold_configs_by_level,
        split,
        observations,
        y_all,
        spec,
        features,
        engine,
        fm,
        evalm5,
        item_scores,
        {**context, "coverage_masks": coverage_cold_validation},
    )
    cold_validation, _ = aggregate_rows(
        rows_cold_validation,
        coverage_cold_validation,
        catalogue_cols.size,
        config_lookup,
    )
    cold_validation.to_csv(
        output_dir / "cold" / "grid.csv",
        index=False,
    )
    del rows_cold_validation
    if cold_validation.empty:
        raise ValueError("Cold validation produced no metrics.")
    selected_cold = select_configs(
        cold_validation,
        approaches,
        "cold",
        levels,
    )
    write_json(
        output_dir / "cold" / "selected_configs.json",
        selected_payload(selected_cold),
    )

    print("Evaluating cold test cohort...")
    cold_test_configs = {}
    for level in levels:
        configs = [
            selected_cold[(int(level), approach)]
            for approach in approaches
            if (int(level), approach) in selected_cold
        ]
        cold_test_configs[int(level)] = configs
    coverage_cold_test = {}
    rows_cold_test = evaluate_cold(
        cold["test_users"],
        levels,
        cold_test_configs,
        split,
        observations,
        y_all,
        spec,
        features,
        engine,
        fm,
        evalm5,
        item_scores,
        {**context, "coverage_masks": coverage_cold_test},
    )
    cold_wide, cold_ci = final_metrics_frame(
        rows_cold_test,
        coverage_cold_test,
        catalogue_cols.size,
        bootstrap_samples,
        args.seed + 220000,
        "cold",
    )
    cold_wide.to_csv(
        output_dir / "cold" / "final_metrics.csv",
        index=False,
    )
    cold_ci.to_csv(
        output_dir / "cold" / "final_metrics_ci.csv",
        index=False,
    )
    pd.DataFrame(rows_cold_test).to_csv(
        output_dir / "cold" / "final_per_user.csv",
        index=False,
    )

    write_json(
        output_dir / "selected_configs.json",
        {
            "warm": selected_payload(selected_warm),
            "cold": selected_payload(selected_cold),
        },
    )

    print("Generating figures...")
    plot_warm_comparison(
        warm_wide,
        output_dir / "figures" / "warm_comparison.png",
    )
    plot_cold_curves(
        cold_wide,
        output_dir / "figures" / "cold_curves.png",
    )
    plot_knn_heatmaps(
        warm_validation,
        output_dir / "figures" / "knn_heatmaps.png",
    )
    plot_popularity(
        warm_validation,
        output_dir / "figures" / "popularity_m.png",
    )
    plot_timing(
        warm_wide,
        output_dir / "figures" / "warm_timing.png",
    )

    movie_titles = (
        data["movie_context"]["name"].astype(str).to_numpy()
        if "name" in data["movie_context"].columns
        else data["movie_mapping"]["canonical_movie_id"].astype(str).to_numpy()
    )
    movie_ids = (
        data["movie_mapping"]["canonical_movie_id"]
        .astype(str)
        .to_numpy()
    )
    user_ids = (
        data["user_mapping"]["user_id"].astype(str).to_numpy()
    )

    print("Building recommendation examples...")
    warm_visible_sizes = {
        int(user): len(indices)
        for user, indices in warm["visible_by_user"].items()
    }
    warm_example_user = choose_example_user(
        warm["test_users"],
        warm_visible_sizes,
    )
    if warm_example_user is not None and not warm_wide.empty:
        warm_best = warm_wide.sort_values(
            "ndcg_known",
            ascending=False,
        ).iloc[0]
        warm_best_config = selected_warm.get(
            (0, warm_best["approach"])
        )
        if warm_best_config is None:
            warm_best_config = final_configs_warm[0]
        visible_row = visible_matrix.getrow(warm_example_user)
        visible_cols = visible_row.indices.astype(np.int64)
        visible_ratings = visible_row.data.astype(np.float64)
        binary_target = visible_row.copy()
        binary_target.data[:] = 1.0
        centered_target = similarity_structures["centered"].getrow(
            warm_example_user
        )
        latent_target = normalize_vector(
            latent_normalized[warm_example_user]
        )
        hidden_indices = warm["hidden_by_user"][warm_example_user]
        hidden_movies = movie_cols[hidden_indices]
        hidden_ratings = y_all[hidden_indices]
        m5_scores = warm_item_scores(
            engine,
            item_scores,
            features,
            warm_example_user,
        )
        top_df, neighbor_df, visible_df = build_example(
            warm_example_user,
            "warm",
            0,
            warm_best_config,
            visible_cols,
            visible_ratings,
            hidden_movies,
            hidden_ratings,
            m5_scores,
            centered_target,
            binary_target,
            latent_target,
            context,
            movie_titles,
            movie_ids,
            user_ids,
            rating_info["factor"],
            rating_info["threshold"],
        )
        top_df.to_csv(
            output_dir / "examples" / "warm_top20.csv",
            index=False,
        )
        neighbor_df.to_csv(
            output_dir / "examples" / "warm_neighbors.csv",
            index=False,
        )
        visible_df.to_csv(
            output_dir / "examples" / "warm_visible.csv",
            index=False,
        )
        print("  Warm example recommendation")
        print(
            top_df[
                [
                    "rank",
                    "title",
                    "score",
                    "true_rating",
                    "relevant",
                    "known_in_test",
                ]
            ].head(10).to_string(index=False)
        )
        if not neighbor_df.empty:
            print("  Nearest neighbors")
            print(neighbor_df.to_string(index=False))
        if not visible_df.empty:
            print(
                f"  Visible profile: {len(visible_df)} rated movies"
            )

    cold_pool_sizes = {
        int(user): len(split["pool_order_by_user"][int(user)])
        for user in cold["test_users"]
    }
    cold_example_user = choose_example_user(
        cold["test_users"],
        cold_pool_sizes,
    )
    if cold_example_user is not None and not cold_wide.empty:
        cold_example_pool = np.asarray(
            split["pool_order_by_user"][int(cold_example_user)],
            dtype=np.int64,
        )
        example_level = 20 if cold_example_pool.size >= 20 else int(
            min(levels)
        )
        available_levels = [
            level
            for level in levels
            if int(level) <= cold_example_pool.size
        ]
        if available_levels:
            example_level = int(max(available_levels))
        cold_best = (
            cold_wide.groupby("approach")["ndcg_known"]
            .mean()
            .sort_values(ascending=False)
        )
        cold_best_approach = (
            cold_best.index[0] if not cold_best.empty else "m5_direct"
        )
        cold_best_config = selected_cold.get(
            (example_level, cold_best_approach)
        )
        if cold_best_config is None:
            for level in sorted(levels, reverse=True):
                candidate = selected_cold.get(
                    (int(level), cold_best_approach)
                )
                if candidate is not None:
                    cold_best_config = candidate
                    example_level = int(level)
                    break
        if cold_best_config is not None:
            pool = cold_example_pool
            hidden = np.asarray(
                split["eval_rows_by_user"][int(cold_example_user)],
                dtype=np.int64,
            )
            hidden_movies = movie_cols[hidden]
            hidden_ratings = y_all[hidden]
            feature_index = fm.user_feature_index(
                spec,
                int(cold_example_user),
                features["context_family_indices"],
            )
            base_pool, latent_pool = evalm5.prepare_rows(
                engine,
                features["context_all"],
                pool,
                feature_index,
            )
            pool_ratings = y_all[pool]
            pool_cols = movie_cols[pool]
            system, target = evalm5.build_ridge_system(
                base_pool[:example_level],
                latent_pool[:example_level],
                pool_ratings[:example_level],
            )
            theta = evalm5.solve_ridge(
                system,
                target,
                float(args.cold_adapt_reg_bias),
                float(args.cold_adapt_reg_factors),
            )
            cold_item_scores = (
                item_scores["base"]
                + theta[0]
                + item_scores["latent"] @ theta[1:]
            )
            visible_cols = pool_cols[:example_level]
            visible_ratings = pool_ratings[:example_level]
            centered_target, binary_target = build_target_csr(
                visible_cols,
                visible_ratings,
                n_movies,
            )
            latent_target = normalize_vector(theta[1:])
            top_df, neighbor_df, visible_df = build_example(
                int(cold_example_user),
                "cold",
                example_level,
                cold_best_config,
                visible_cols,
                visible_ratings,
                hidden_movies,
                hidden_ratings,
                cold_item_scores,
                centered_target,
                binary_target,
                latent_target,
                context,
                movie_titles,
                movie_ids,
                user_ids,
                rating_info["factor"],
                rating_info["threshold"],
            )
            top_df.to_csv(
                output_dir / "examples" / "cold_top20.csv",
                index=False,
            )
            neighbor_df.to_csv(
                output_dir / "examples" / "cold_neighbors.csv",
                index=False,
            )
            visible_df.to_csv(
                output_dir / "examples" / "cold_visible.csv",
                index=False,
            )
            print("  Cold example recommendation")
            print(
                top_df[
                    [
                        "rank",
                        "title",
                        "score",
                        "true_rating",
                        "relevant",
                        "known_in_test",
                    ]
                ].head(10).to_string(index=False)
            )
            if not neighbor_df.empty:
                print("  Nearest neighbors")
                print(neighbor_df.to_string(index=False))
            if not visible_df.empty:
                print(
                    f"  Visible profile: {len(visible_df)} rated movies"
                )

    print("Writing summary...")
    write_summary(
        output_dir / "summary.md",
        warm_wide,
        cold_wide,
        warm_ci,
        cold_ci,
        selected_warm,
        selected_cold,
        args,
        hyperparameters,
        training_summary,
        rating_info,
        split_info,
    )
    write_json(
        output_dir / "config.json",
        {
            "paths": {
                "cleaning_dir": str(cleaning_dir),
                "model_output_dir": str(model_output_dir),
                "scripts_dir": str(scripts_dir),
                "output_dir": str(output_dir),
            },
            "args": vars(args),
            "hyperparameters": hyperparameters,
            "feature_config": {"max_studios": int(max_studios)},
            "training": training_summary,
            "rating": rating_info,
            "split": split_info,
            "levels": levels,
            "popularity_m_grid": pop_ms,
            "knn_neighbors_grid": neighbors,
            "knn_topk_grid": topks,
            "similarity_methods": methods,
            "similarity_shrink": float(args.similarity_shrink),
            "item_score_check": float(score_difference),
            "catalogue_size": int(catalogue_cols.size),
            "selected": {
                "warm": selected_payload(selected_warm),
                "cold": selected_payload(selected_cold),
            },
        },
    )

    print()
    if not warm_wide.empty:
        print("Warm-start final metrics")
        print(
            warm_wide[
                [
                    "approach",
                    "candidate_recall",
                    "recall20",
                    "ndcg_known",
                    "time_total",
                    "users",
                ]
            ].to_string(index=False)
        )
    print()
    if not cold_wide.empty:
        print("Cold-start final metrics")
        print(
            cold_wide[
                [
                    "approach",
                    "level",
                    "candidate_recall",
                    "recall20",
                    "ndcg_known",
                    "time_total",
                    "users",
                ]
            ].to_string(index=False)
        )
    print()
    print(f"Outputs written to {output_dir}")


if __name__ == "__main__":
    main()
