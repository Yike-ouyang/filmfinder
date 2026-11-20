from __future__ import annotations

import argparse
import importlib.util
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz, load_npz


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_M5 = {
    "n_factors": 8,
    "learning_rate": 0.01,
    "n_epochs": 20,
    "batch_size": 65536,
    "reg_bias": 0.0,
    "reg_factors": 1e-6,
    "patience": 3,
    "seed": 4042,
}


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_json(path):
    with open(path) as handle:
        return json.load(handle)


def native(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and np.isnan(value):
        return None
    if pd.isna(value):
        return None
    return value


def as_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") and text.endswith("]"):
            try:
                import ast

                parsed = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                return [text] if text else []
            if isinstance(parsed, (list, tuple, set)):
                return [str(item) for item in parsed if str(item).strip()]
            return [str(parsed)] if str(parsed).strip() else []
        return [item.strip() for item in text.split(",") if item.strip()]
    if isinstance(value, (list, tuple, set, np.ndarray, pd.Series)):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)] if str(value).strip() else []


def resolve_paths(args):
    cleaning = Path(
        args.cleaning_dir
        or os.environ.get("FILMFINDER_CLEANING_DIR")
        or Path.cwd()
    ).expanduser()
    baseline = Path(
        args.baseline_dir
        or os.environ.get("FILMFINDER_RECOMMENDATION_DIR")
        or REPO_ROOT / "research" / "results" / "recommendation_system"
    ).expanduser()
    if not baseline.exists():
        baseline = REPO_ROOT / "research" / "results" / "recommendation_system"
    knn = Path(
        args.knn_dir
        or os.environ.get("FILMFINDER_KNN_M5_DIR")
        or REPO_ROOT / "research" / "results" / "knn_m5_optimization"
    ).expanduser()
    if not knn.exists():
        knn = REPO_ROOT / "research" / "results" / "knn_m5_optimization"
    output = Path(
        args.output_dir or REPO_ROOT / "packages" / "recommendation-engine"
    ).expanduser()
    for name, path in (
        ("cleaning", cleaning),
        ("baseline", baseline),
        ("knn", knn),
        ("output", output),
    ):
        if not path.exists():
            raise FileNotFoundError(f"Missing {name} directory: {path}")
    return cleaning, baseline, knn, output


def build_final_item_features(context_all, observations, n_movies):
    cols = observations["col"].to_numpy(dtype=np.int64)
    rows = observations["row"].to_numpy(dtype=np.int64)
    item_obs = np.full(int(n_movies), -1, dtype=np.int64)
    item_obs[cols] = np.arange(cols.size, dtype=np.int64)
    valid_movies = np.where(item_obs >= 0)[0]
    selected_obs = item_obs[valid_movies]
    item_matrix = context_all[selected_obs].tolil()
    item_rows = np.arange(valid_movies.size, dtype=np.int64)
    return item_matrix, item_rows, valid_movies, rows, selected_obs


def build_item_csr(context_all, observations, n_movies, user_feature):
    cols = observations["col"].to_numpy(dtype=np.int64)
    rows = observations["row"].to_numpy(dtype=np.int64)
    item_obs = np.full(int(n_movies), -1, dtype=np.int64)
    item_obs[cols] = np.arange(cols.size, dtype=np.int64)
    valid_movies = np.where(item_obs >= 0)[0]
    selected_obs = item_obs[valid_movies]
    sub = context_all[selected_obs].tolil()
    sub[np.arange(valid_movies.size), user_feature[rows[selected_obs]]] = 0.0
    sub = sub.tocsr()
    row_ids = np.repeat(valid_movies, np.diff(sub.indptr))
    return csr_matrix(
        (sub.data, (row_ids, sub.indices)),
        shape=(int(n_movies), context_all.shape[1]),
        dtype=np.float64,
    )


def train_final_m5(fm, features, y, hyperparameters, seed):
    spec = next(
        item for item in fm.make_model_specs(
            features["_n_users"],
            features["_n_movies"],
            len(features["context_mapping"]),
        )
        if item["name"] == "M5"
    )
    engine = fm.FMEngine(
        n_features=spec["n_features"],
        n_factors=int(hyperparameters["n_factors"]),
        use_linear=spec["use_linear"],
        use_latent=spec["use_latent"],
        active_mask=spec["active_mask"],
        learning_rate=float(hyperparameters["learning_rate"]),
        reg_bias=float(hyperparameters["reg_bias"]),
        reg_factors=float(hyperparameters["reg_factors"]),
        seed=int(seed),
    )
    X = features["context_all"]
    n_rows = X.shape[0]
    batch_size = int(hyperparameters["batch_size"])
    rng = np.random.default_rng(int(seed) + 1)
    started = time.perf_counter()
    history = []
    for epoch in range(1, int(hyperparameters["n_epochs"]) + 1):
        order = rng.permutation(n_rows)
        for start in range(0, n_rows, batch_size):
            batch = order[start:start + batch_size]
            engine.step(X[batch], y[batch])
        prediction = np.empty(n_rows, dtype=np.float64)
        for start in range(0, n_rows, batch_size):
            end = min(start + batch_size, n_rows)
            prediction[start:end] = engine.predict(X[start:end])
        error = prediction - y
        history.append({
            "epoch": epoch,
            "train_mae": float(np.mean(np.abs(error))),
            "train_rmse": float(np.sqrt(np.mean(error ** 2))),
        })
    elapsed = time.perf_counter() - started
    return spec, engine, elapsed, history


def build_metadata(data):
    context = data["movie_context"].reset_index(drop=True)
    movie_ids = data["movie_mapping"]["canonical_movie_id"].to_numpy()
    movies = []
    source_id_columns = [
        column for column in context.columns
        if column in ("_id", "1_id", "movie_id", "tmdb_id", "letterboxd_id")
    ]
    for col, movie_id in enumerate(movie_ids):
        row = context.iloc[col]
        title = row.get("name")
        try:
            title_missing = bool(pd.isna(title))
        except (TypeError, ValueError):
            title_missing = False
        if title is None or title_missing:
            title = str(native(movie_id))
        genres = as_list(row.get("genres"))
        year = row.get("year")
        year = native(year)
        source_ids = {}
        for column in source_id_columns:
            value = native(row.get(column))
            if value is not None:
                source_ids[column] = value
        movies.append({
            "internal_id": int(col),
            "movie_id": native(movie_id),
            "title": str(title),
            "year": year,
            "genres": genres,
            "source_ids": source_ids,
        })
    return {"n_movies": len(movies), "movies": movies}


def write_report(path, output, artifacts, load_time, recommend_time, new_time, extra):
    rows = []
    for name, file_path in sorted(artifacts.items()):
        rows.append((name, file_path.stat().st_size))
    total = sum(size for _, size in rows)
    lines = [
        "# FilmFinder deployment export",
        "",
        "## Artifact sizes",
        "",
        "| Artifact | Size |",
        "|---|---:|",
    ]
    for name, size in rows:
        lines.append(f"| `{name}` | {size / (1024 ** 2):.2f} MiB |")
    lines.extend([
        f"| **Total artifacts** | **{total / (1024 ** 2):.2f} MiB** |",
        "",
        "## Runtime",
        "",
        f"- Engine load time: {load_time:.3f} s",
        f"- Existing-user recommendation: {recommend_time:.4f} s",
        f"- New-user recommendation: {new_time:.4f} s",
        f"- Peak RSS: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2:.2f} GiB",
        "",
        "## Model and data",
        "",
        f"- Users: {extra['n_users']}",
        f"- Movies: {extra['n_movies']}",
        f"- Ratings: {extra['n_ratings']}",
        f"- Features: {extra['n_features']}",
        f"- Training time: {extra['training_time']:.1f} s",
        f"- Final training epoch: {extra['final_epoch']}",
        f"- Training MAE: {extra['train_mae']:.6f}",
        f"- Training RMSE: {extra['train_rmse']:.6f}",
        "",
        "## Verification",
        "",
        f"- M5 max prediction difference: {extra['m5_diff']:.3e}",
        f"- Existing-user Top-20 matches: {extra['existing_matches']}",
        f"- New-user Top-20 matches: {extra['new_matches']}",
        f"- Largest artifacts: {extra['largest']}",
        "",
        "The final M5 model was retrained on all valid ratings. "
        "Experimental validation/test metrics do not apply to this model.",
        "",
    ])
    path.write_text("\n".join(lines))


def verify_export(output_dir, reference, sample_users):
    sys.path.insert(0, str(Path(output_dir) / "src"))
    sys.path.insert(0, str(output_dir))
    from filmfinder import FilmFinderEngine

    start = time.perf_counter()
    engine = FilmFinderEngine.load(Path(output_dir) / "artifacts")
    load_time = time.perf_counter() - start
    existing_matches = 0
    new_matches = 0
    start = time.perf_counter()
    existing_ids = []
    for user_row in sample_users:
        user_id = reference["user_ids"][user_row]
        result = engine.recommend(user_id, top_n=20)
        existing_ids.append([row["movie_id"] for row in result])
    recommend_time = (time.perf_counter() - start) / max(len(sample_users), 1)
    cold_movies = reference["cold_movies"]
    cold_ratings = np.array([8.0, 9.0, 6.0], dtype=np.float64)
    start = time.perf_counter()
    new_result = engine.recommend_new_user(
        {
            reference["movie_ids"][movie_col]: float(rating)
            for movie_col, rating in zip(cold_movies, cold_ratings)
        },
        top_n=20,
    )
    new_time = time.perf_counter() - start
    for index, user_row in enumerate(sample_users):
        expected = reference["existing"][index]
        if existing_ids[index] == expected:
            existing_matches += 1
    new_ids = [row["movie_id"] for row in new_result]
    if new_ids == reference["new"]:
        new_matches = 1
    return {
        "load_time": load_time,
        "recommend_time": recommend_time,
        "new_time": new_time,
        "existing_matches": f"{existing_matches}/{len(sample_users)}",
        "new_matches": f"{new_matches}/1",
        "engine": engine,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cleaning-dir", default=None)
    parser.add_argument("--baseline-dir", default=None)
    parser.add_argument("--knn-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    cleaning_dir, baseline_dir, knn_dir, output_dir = resolve_paths(args)
    artifacts_dir = output_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    scripts_dir = REPO_ROOT / "research" / "modeling"
    rec_dir = Path(__file__).resolve().parent
    fm = load_module(scripts_dir / "6.IncrementalVsRetraining.py", "_export_fm")
    evalm5 = load_module(scripts_dir / "7.EvaluateM5.py", "_export_evalm5")
    eval8 = load_module(rec_dir / "8.RecommendationSystem.py", "_export_eval8")
    knn9 = load_module(rec_dir / "9.KNNM5Optimization.py", "_export_knn9")

    baseline_config = read_json(baseline_dir / "config.json")
    selected = read_json(knn_dir / "selection" / "selected_configs.json")
    global_config = selected.get("global")
    if not global_config:
        raise ValueError("No global recommendation configuration found.")
    if global_config.get("method") not in ("centered", "latent"):
        raise ValueError("Unsupported global similarity method.")
    if global_config.get("method") != "centered":
        raise ValueError(
            "The deployment recommender currently supports centered similarity."
        )
    hyperparameters = dict(DEFAULT_M5)
    hyperparameters.update(baseline_config.get("hyperparameters", {}))
    if "n_epochs" not in hyperparameters and "epochs" in baseline_config.get("args", {}):
        hyperparameters["n_epochs"] = baseline_config["args"]["epochs"]
    baseline_args = baseline_config.get("args", {})
    max_studios = int(
        baseline_config.get("feature_config", {}).get("max_studios", 4000)
    )
    fallback_m = float(baseline_args.get("fallback_popularity_m", 100.0))
    fallback_candidates = int(baseline_args.get("fallback_candidates", 500))
    cold_bias = float(baseline_args.get("cold_adapt_reg_bias", 0.01))
    cold_factors = float(baseline_args.get("cold_adapt_reg_factors", 0.1))
    shrink = float(baseline_args.get("similarity_shrink", 10.0))

    print("Loading full dataset...")
    data = fm.load_inputs(cleaning_dir)
    observations = data["observations"]
    n_users = int(data["n_users"])
    n_movies = int(data["n_movies"])
    y = observations["rating"].to_numpy(dtype=np.float64)
    all_indices = np.arange(len(observations), dtype=np.int64)
    print("Building final features on all ratings...")
    features = fm.build_features(
        observations,
        data["movie_context"],
        data["user_mapping"],
        data["movie_mapping"],
        n_users,
        n_movies,
        all_indices,
        max_studios,
    )
    features["_n_users"] = n_users
    features["_n_movies"] = n_movies
    print("Training final M5 on all ratings...")
    spec, engine, training_time, history = train_final_m5(
        fm,
        features,
        y,
        hyperparameters,
        int(hyperparameters["seed"]),
    )
    final_metrics = history[-1]
    user_feature = features["context_family_indices"]["user"]
    print("Building deployment item features...")
    movie_features = build_item_csr(
        features["context_all"],
        observations,
        n_movies,
        user_feature,
    )
    ratings_matrix = load_npz(cleaning_dir / "ratings_matrix.npz").tocsr()

    artifacts = {}
    weights_path = artifacts_dir / "m5_weights.npz"
    np.savez_compressed(
        weights_path,
        w0=np.array([engine.w0], dtype=np.float64),
        w=np.asarray(engine.w, dtype=np.float64),
        V=np.asarray(engine.V, dtype=np.float64),
        active_mask=np.asarray(engine.active_mask, dtype=bool),
        n_features=np.array([engine.n_features]),
        n_factors=np.array([engine.n_factors]),
        use_linear=np.array([engine.use_linear]),
        use_latent=np.array([engine.use_latent]),
    )
    artifacts["m5_weights.npz"] = weights_path
    features_path = artifacts_dir / "movie_features.npz"
    save_npz(features_path, movie_features)
    artifacts["movie_features.npz"] = features_path
    ids_paths = {}
    for name, values in (
        ("user_ids", data["user_mapping"]["user_id"].to_numpy()),
        ("movie_ids", data["movie_mapping"]["canonical_movie_id"].to_numpy()),
    ):
        path = artifacts_dir / f"{name}.npy"
        np.save(path, values, allow_pickle=True)
        artifacts[f"{name}.npy"] = path
        ids_paths[name] = path
    matrix_path = artifacts_dir / "ratings_matrix.npz"
    save_npz(matrix_path, ratings_matrix)
    artifacts["ratings_matrix.npz"] = matrix_path

    movie_metadata = build_metadata(data)
    metadata_path = artifacts_dir / "movie_metadata.json"
    metadata_path.write_text(json.dumps(movie_metadata, ensure_ascii=False))
    artifacts["movie_metadata.json"] = metadata_path
    mapping_payload = {
        "n_features": int(engine.n_features),
        "feature_names": features["context_mapping"]["feature_name"].tolist(),
        "feature_families": features["context_mapping"]["family"].tolist(),
        "family_indices": {
            key: np.asarray(value, dtype=np.int64).tolist()
            for key, value in features["context_family_indices"].items()
        },
        "user_feature_indices": np.asarray(
            features["context_family_indices"]["user"], dtype=np.int64
        ).tolist(),
        "n_users": n_users,
        "n_movies": n_movies,
        "max_studios": max_studios,
        "year_mean": float(features["year_mean"]),
        "year_std": float(features["year_std"]),
    }
    mapping_path = artifacts_dir / "feature_mappings.json"
    mapping_path.write_text(json.dumps(mapping_payload, ensure_ascii=False))
    artifacts["feature_mappings.json"] = mapping_path
    m5_config = {
        "model": "M5",
        "model_version": "filmfinder-m5-all-ratings-v1",
        "trained_on": "all valid ratings",
        "n_users": n_users,
        "n_movies": n_movies,
        "n_ratings": int(len(observations)),
        "n_features": int(engine.n_features),
        "n_factors": int(engine.n_factors),
        "use_linear": bool(engine.use_linear),
        "use_latent": bool(engine.use_latent),
        "feature_config": {
            "max_studios": max_studios,
            "year_mean": float(features["year_mean"]),
            "year_std": float(features["year_std"]),
        },
        "hyperparameters": hyperparameters,
        "training": {
            "optimizer": "adam",
            "epochs": int(hyperparameters["n_epochs"]),
            "batch_size": int(hyperparameters["batch_size"]),
            "early_stopping": False,
            "seed": int(hyperparameters["seed"]),
            "init_scale": 0.01,
        },
        "training_diagnostics": {
            "train_mae": float(final_metrics["train_mae"]),
            "train_rmse": float(final_metrics["train_rmse"]),
        },
    }
    m5_path = artifacts_dir / "m5_config.json"
    m5_path.write_text(json.dumps(m5_config, indent=2))
    artifacts["m5_config.json"] = m5_path
    recommendation_config = {
        "algorithm": "knn_m5",
        "similarity": "centered_cosine",
        "n_neighbors": int(global_config["n_neighbors"]),
        "top_k_per_neighbor": int(global_config["top_k"]),
        "candidate_pool_size": int(global_config["pool_size"]),
        "similarity_shrink": shrink,
        "fallback_popularity_m": fallback_m,
        "fallback_candidates": fallback_candidates,
        "ranking": "m5",
        "candidate_order": ["support", "popularity", "movie_id"],
        "tie_breaking": "lexsort_support_popularity_movie_id",
        "cold_adapt_reg_bias": cold_bias,
        "cold_adapt_reg_factors": cold_factors,
        "default_top_n": 20,
        "source": str(knn_dir / "selection" / "selected_configs.json") + "#global",
    }
    rec_path = artifacts_dir / "recommendation_config.json"
    rec_path.write_text(json.dumps(recommendation_config, indent=2))
    artifacts["recommendation_config.json"] = rec_path

    print("Building reference predictions...")
    item_scores = eval8.build_item_scores(
        fm, engine, features, observations, n_movies
    )
    visible_matrix = eval8.build_visible_matrix(
        observations, all_indices, y, n_users, n_movies
    )
    similarity_structures = eval8.build_similarity_structures(
        visible_matrix, n_users
    )
    popularity = eval8.build_popularity(
        observations, all_indices, y, n_movies, [fallback_m]
    )
    catalogue_mask = popularity["counts"] > 0
    catalogue_cols = np.flatnonzero(catalogue_mask).astype(np.int64)
    topk_cache = eval8.build_topk_cache(
        visible_matrix, [int(global_config["top_k"])], popularity["counts"]
    )
    fallback_order = catalogue_cols[
        np.argsort(-popularity["scores"][fallback_m][catalogue_cols])
    ]
    n_user_features = len(features["context_family_indices"]["user"])
    user_feature_index = features["context_family_indices"]["user"]

    existing_users = []
    for user in range(n_users):
        if visible_matrix.getrow(user).nnz > 0:
            existing_users.append(user)
        if len(existing_users) >= 3:
            break
    reference = {
        "user_ids": data["user_mapping"]["user_id"].to_numpy(),
        "movie_ids": data["movie_mapping"]["canonical_movie_id"].to_numpy(),
        "existing": [],
    }
    for user in existing_users:
        row = visible_matrix.getrow(user)
        centered_target = similarity_structures["centered"].getrow(user)
        binary_target = row.copy()
        binary_target.data[:] = 1.0
        similarity = eval8.centered_similarity(
            centered_target,
            binary_target,
            similarity_structures["centered"],
            similarity_structures["binary"],
            similarity_structures["norms"],
            shrink,
        )
        order = eval8.neighbor_order(similarity, user, int(global_config["n_neighbors"]))
        candidates, support, candidate_order = knn9.build_knn_union(
            order,
            int(global_config["n_neighbors"]),
            int(global_config["top_k"]),
            topk_cache,
            catalogue_mask,
            row.indices.astype(np.int64),
            popularity["counts"],
        )
        if candidates is None:
            candidates = knn9.build_fallback_candidates(
                fallback_order, row.indices.astype(np.int64)
            )
            candidates = knn9.apply_pool_cap(
                candidates,
                np.arange(candidates.size),
                int(global_config["pool_size"]),
                fallback_candidates,
                "fallback",
            )
        else:
            candidates = knn9.apply_pool_cap(
                candidates,
                candidate_order,
                int(global_config["pool_size"]),
                fallback_candidates,
                "knn",
            )
        user_index = int(user_feature_index[user])
        scores = (
            item_scores["base"][candidates]
            + engine.w[user_index]
            + item_scores["latent"][candidates] @ engine.V[user_index]
        )
        top = eval8.top_k_columns(candidates, scores, 20)
        reference["existing"].append(
            [native(data["movie_mapping"]["canonical_movie_id"].to_numpy()[int(movie)]) for movie in top]
        )
    cold_movies = np.asarray(
        [int(movie) for movie in np.flatnonzero(catalogue_mask)[:3]],
        dtype=np.int64,
    )
    reference["cold_movies"] = cold_movies.tolist()
    cold_ratings = np.array([8.0, 9.0, 6.0], dtype=np.float64)
    centered_target, binary_target = eval8.build_target_csr(
        cold_movies, cold_ratings, n_movies
    )
    similarity = eval8.centered_similarity(
        centered_target,
        binary_target,
        similarity_structures["centered"],
        similarity_structures["binary"],
        similarity_structures["norms"],
        shrink,
    )
    order = eval8.neighbor_order(similarity, -1, int(global_config["n_neighbors"]))
    candidates, support, candidate_order = knn9.build_knn_union(
        order,
        int(global_config["n_neighbors"]),
        int(global_config["top_k"]),
        topk_cache,
        catalogue_mask,
        cold_movies,
        popularity["counts"],
    )
    if candidates is None:
        candidates = knn9.build_fallback_candidates(
            fallback_order, cold_movies
        )
        candidates = knn9.apply_pool_cap(
            candidates,
            np.arange(candidates.size),
            int(global_config["pool_size"]),
            fallback_candidates,
            "fallback",
        )
    else:
        candidates = knn9.apply_pool_cap(
            candidates,
            candidate_order,
            int(global_config["pool_size"]),
            fallback_candidates,
            "knn",
        )
    base = item_scores["base"][cold_movies]
    latent = item_scores["latent"][cold_movies]
    ridge_matrix, ridge_target = evalm5.build_ridge_system(
        base, latent, cold_ratings
    )
    theta = evalm5.solve_ridge(
        ridge_matrix,
        ridge_target,
        cold_bias,
        cold_factors,
    )
    scores = (
        item_scores["base"][candidates]
        + theta[0]
        + item_scores["latent"][candidates] @ theta[1:]
    )
    top = eval8.top_k_columns(candidates, scores, 20)
    reference["new"] = [
        native(data["movie_mapping"]["canonical_movie_id"].to_numpy()[int(movie)])
        for movie in top
    ]

    print("Verifying exported package...")
    verification = verify_export(output_dir, reference, existing_users)
    engine_deployed = verification["engine"]
    sample_rows = np.arange(min(200, len(observations)), dtype=np.int64)
    user_rows = observations["row"].to_numpy(dtype=np.int64)[sample_rows]
    movie_rows = observations["col"].to_numpy(dtype=np.int64)[sample_rows]
    expected = engine.predict(features["context_all"][sample_rows])
    deployed = engine_deployed.model.score_user_rows(user_rows, movie_rows)
    m5_diff = float(np.max(np.abs(expected - deployed)))
    if m5_diff > 1e-8:
        raise RuntimeError(
            f"M5 exported prediction mismatch: {m5_diff:.3e}"
        )
    if verification["existing_matches"] != f"{len(existing_users)}/{len(existing_users)}":
        raise RuntimeError(
            "Existing-user Top-20 mismatch between deployment and reference."
        )
    if verification["new_matches"] != "1/1":
        raise RuntimeError(
            "New-user Top-20 mismatch between deployment and reference."
        )
    largest = sorted(
        ((name, path.stat().st_size) for name, path in artifacts.items()),
        key=lambda item: item[1],
        reverse=True,
    )[:3]
    extra = {
        "n_users": n_users,
        "n_movies": n_movies,
        "n_ratings": int(len(observations)),
        "n_features": int(engine.n_features),
        "training_time": training_time,
        "final_epoch": int(history[-1]["epoch"]),
        "train_mae": float(final_metrics["train_mae"]),
        "train_rmse": float(final_metrics["train_rmse"]),
        "m5_diff": m5_diff,
        "existing_matches": verification["existing_matches"],
        "new_matches": verification["new_matches"],
        "largest": ", ".join(
            f"{name} ({size / (1024 ** 2):.2f} MiB)"
            for name, size in largest
        ),
    }
    report_path = output_dir / "EXPORT_REPORT.md"
    write_report(
        report_path,
        output_dir,
        artifacts,
        verification["load_time"],
        verification["recommend_time"],
        verification["new_time"],
        extra,
    )
    print(f"Export complete: {output_dir}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
