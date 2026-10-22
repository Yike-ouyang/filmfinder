"""
Grid Search for the Filmfinder SVD recommender.

This standalone script loads the sparse user-film matrix produced by
``1.Preprocessing.py``, evaluates every combination of latent dimension,
regularization and number of epochs with a fixed ShuffleSplit protocol, and
saves the raw results, aggregated summary, best hyperparameters and figures.

The sparse matrix is never converted to a dense representation.

Usage
-----
    python grid_search.py --n-jobs 1
    python grid_search.py --n-jobs 2 --no-resume
    python grid_search.py --analysis-only

Dependencies
------------
    pandas numpy scipy scikit-surprise matplotlib joblib
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy.sparse import load_npz
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import ShuffleSplit


PARAM_GRID = {
    "n_factors": [20, 50, 100],
    "reg_all": [0.02, 0.05, 0.1, 0.2],
    "n_epochs": [5, 10, 15, 20],
}

LR_ALL = 0.005
BIASED = True
RANDOM_STATE = 42
N_SPLITS = 2
TEST_SIZE = 0.2

RESULTS_COLUMNS = [
    "n_factors",
    "reg_all",
    "n_epochs",
    "fold",
    "train_rmse",
    "val_rmse",
    "train_mae",
    "val_mae",
    "train_time",
    "warm_start_ratio",
    "warm_val_rmse",
    "warm_val_mae",
]

SUMMARY_COLUMNS = [
    "n_factors",
    "reg_all",
    "n_epochs",
    "mean_train_rmse",
    "std_train_rmse",
    "mean_val_rmse",
    "std_val_rmse",
    "mean_train_mae",
    "std_train_mae",
    "mean_val_mae",
    "std_val_mae",
    "generalization_gap",
    "mean_warm_start_ratio",
    "mean_warm_val_rmse",
    "mean_warm_val_mae",
    "total_train_time",
]


def load_ratings(data_dir):
    """Load the sparse matrix and build a Surprise Dataset from COO triplets."""
    sparse_path = data_dir / "ratings_sparse_final.npz"
    users_path = data_dir / "user_ids_final.npy"
    movies_path = data_dir / "movie_ids_final.npy"

    for path in (sparse_path, users_path, movies_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing input file: {path}")

    ratings_sparse = load_npz(sparse_path).tocsr()
    user_ids = np.load(users_path, allow_pickle=True)
    movie_ids = np.load(movies_path, allow_pickle=True)

    assert ratings_sparse.shape == (
        len(user_ids),
        len(movie_ids),
    ), "Mismatch between the sparse matrix and the identifier arrays."

    # Convert only observed entries to explicit triplets (never densify).
    ratings_coo = ratings_sparse.tocoo(copy=False)
    ratings_df = pd.DataFrame(
        {
            "user_id": user_ids[ratings_coo.row],
            "movie_id": movie_ids[ratings_coo.col],
            "rating": ratings_coo.data.astype(np.float64, copy=False),
        },
        copy=False,
    )

    min_rating = float(ratings_df["rating"].min())
    max_rating = float(ratings_df["rating"].max())
    if not min_rating < max_rating:
        raise ValueError("The rating scale must contain at least two distinct values.")

    n_users, n_movies = ratings_sparse.shape
    n_ratings = ratings_sparse.nnz
    density = n_ratings / (n_users * n_movies)

    print(f"Users: {n_users:,}")
    print(f"Films: {n_movies:,}")
    print(f"Observed ratings: {n_ratings:,}")
    print(f"Density: {density:.6%}")
    print(f"Sparsity: {1 - density:.6%}")
    print(f"Rating scale: [{min_rating}, {max_rating}]")

    # Release the sparse structures before Surprise builds its own index.
    del ratings_sparse, ratings_coo, user_ids, movie_ids

    reader = Reader(rating_scale=(min_rating, max_rating))
    dataset = Dataset.load_from_df(ratings_df, reader)

    return dataset


def build_folds(dataset):
    """Build the fixed cross-validation folds shared by all configurations."""
    splitter = ShuffleSplit(
        n_splits=N_SPLITS,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    folds = []
    for fold_idx, (trainset, testset) in enumerate(splitter.split(dataset)):
        folds.append((fold_idx, trainset, testset))
        print(
            f"Fold {fold_idx}: {trainset.n_ratings:,} training ratings, "
            f"{len(testset):,} validation ratings"
        )

    return folds


def _rmse_mae(predictions):
    """Return RMSE and MAE, or NaN when there is no prediction."""
    if not predictions:
        return float("nan"), float("nan")
    return (
        float(accuracy.rmse(predictions, verbose=False)),
        float(accuracy.mae(predictions, verbose=False)),
    )


def _is_warm_start(trainset, user_id, movie_id):
    """A validation pair is warm when both its user and film were seen in training."""
    try:
        trainset.to_inner_uid(user_id)
        trainset.to_inner_iid(movie_id)
        return True
    except ValueError:
        return False


def evaluate_configuration(n_factors, reg_all, n_epochs, fold_idx, trainset, testset):
    """Train one SVD configuration on one fold and return all its metrics."""
    algo = SVD(
        n_factors=int(n_factors),
        n_epochs=int(n_epochs),
        lr_all=LR_ALL,
        reg_all=float(reg_all),
        biased=BIASED,
        random_state=RANDOM_STATE,
    )

    start = time.perf_counter()
    algo.fit(trainset)
    train_time = time.perf_counter() - start

    # In-sample predictions only: the validation interactions are never fitted.
    train_predictions = algo.test(trainset.build_testset())
    validation_predictions = algo.test(testset)

    train_rmse, train_mae = _rmse_mae(train_predictions)
    val_rmse, val_mae = _rmse_mae(validation_predictions)

    warm_predictions = [
        prediction
        for prediction in validation_predictions
        if _is_warm_start(trainset, prediction.uid, prediction.iid)
    ]
    warm_val_rmse, warm_val_mae = _rmse_mae(warm_predictions)
    warm_start_ratio = (
        len(warm_predictions) / len(validation_predictions)
        if validation_predictions
        else float("nan")
    )

    return {
        "n_factors": int(n_factors),
        "reg_all": float(reg_all),
        "n_epochs": int(n_epochs),
        "fold": int(fold_idx),
        "train_rmse": train_rmse,
        "val_rmse": val_rmse,
        "train_mae": train_mae,
        "val_mae": val_mae,
        "train_time": train_time,
        "warm_start_ratio": warm_start_ratio,
        "warm_val_rmse": warm_val_rmse,
        "warm_val_mae": warm_val_mae,
    }


def build_configurations(param_grid=None):
    """Expand the parameter grid into a deterministic list of configurations."""
    grid = PARAM_GRID if param_grid is None else param_grid
    return [
        {
            "n_factors": int(n_factors),
            "reg_all": float(reg_all),
            "n_epochs": int(n_epochs),
        }
        for n_factors in grid["n_factors"]
        for reg_all in grid["reg_all"]
        for n_epochs in grid["n_epochs"]
    ]


def _task_key(n_factors, reg_all, n_epochs, fold):
    """Stable key used to resume a partially completed grid search."""
    return (
        int(n_factors),
        round(float(reg_all), 10),
        int(n_epochs),
        int(fold),
    )


def _atomic_write_csv(dataframe, path):
    """Write a CSV atomically so an interrupted run keeps the previous file."""
    temporary_path = path.with_name(path.name + ".tmp")
    dataframe.to_csv(temporary_path, index=False)
    temporary_path.replace(path)


def _parallel_results(n_jobs, task_args):
    """Yield results as workers finish; fall back for old joblib versions."""
    def job_generator():
        for args in task_args:
            yield delayed(evaluate_configuration)(*args)

    try:
        return Parallel(
            n_jobs=n_jobs,
            verbose=10,
            return_as="generator_unordered",
        )(job_generator())
    except TypeError:
        # Older joblib releases do not support return_as="generator_unordered".
        return iter(Parallel(n_jobs=n_jobs, verbose=10)(job_generator()))


def run_grid_search(folds, results_path, n_jobs=1, resume=True, param_grid=None):
    """Run or resume the grid search, saving each result as soon as it is ready."""
    configurations = build_configurations(param_grid)
    expected_trainings = len(configurations) * len(folds)

    if resume and results_path.exists():
        existing_results = pd.read_csv(results_path)
        existing_results = existing_results.drop_duplicates(
            subset=["n_factors", "reg_all", "n_epochs", "fold"],
            keep="last",
        )
    else:
        existing_results = pd.DataFrame(columns=RESULTS_COLUMNS)

    completed_keys = {
        _task_key(row.n_factors, row.reg_all, row.n_epochs, row.fold)
        for row in existing_results.itertuples()
    }

    tasks = []
    for fold_idx, trainset, testset in folds:
        for configuration in configurations:
            key = _task_key(
                configuration["n_factors"],
                configuration["reg_all"],
                configuration["n_epochs"],
                fold_idx,
            )
            if key not in completed_keys:
                tasks.append((fold_idx, trainset, testset, configuration))

    print(f"Grid: {len(configurations)} configurations x {len(folds)} folds = {expected_trainings} trainings")
    print(f"Already completed: {len(completed_keys)} | Remaining: {len(tasks)} | n_jobs={n_jobs}")

    all_results = existing_results.copy()

    if not tasks:
        print("Nothing to run: all configurations are already saved.")
        return all_results

    def save_row(row):
        nonlocal all_results
        all_results = pd.concat(
            [all_results, pd.DataFrame([row], columns=RESULTS_COLUMNS)],
            ignore_index=True,
        )
        _atomic_write_csv(all_results, results_path)

    loop_start = time.perf_counter()

    if n_jobs == 1:
        for completed, (fold_idx, trainset, testset, configuration) in enumerate(tasks, start=1):
            row = evaluate_configuration(
                configuration["n_factors"],
                configuration["reg_all"],
                configuration["n_epochs"],
                fold_idx,
                trainset,
                testset,
            )
            save_row(row)
            elapsed = time.perf_counter() - loop_start
            eta = elapsed / completed * (len(tasks) - completed)
            print(
                f"[{completed}/{len(tasks)}] "
                f"n_factors={row['n_factors']} reg_all={row['reg_all']:g} "
                f"n_epochs={row['n_epochs']} fold={row['fold']} "
                f"val_rmse={row['val_rmse']:.4f} | "
                f"{elapsed:.1f}s elapsed, ETA {eta:.1f}s"
            )
    else:
        task_args = [
            (
                configuration["n_factors"],
                configuration["reg_all"],
                configuration["n_epochs"],
                fold_idx,
                trainset,
                testset,
            )
            for fold_idx, trainset, testset, configuration in tasks
        ]

        for completed, row in enumerate(_parallel_results(n_jobs, task_args), start=1):
            save_row(row)
            elapsed = time.perf_counter() - loop_start
            eta = elapsed / completed * (len(tasks) - completed)
            print(
                f"[{completed}/{len(tasks)}] "
                f"n_factors={row['n_factors']} reg_all={row['reg_all']:g} "
                f"n_epochs={row['n_epochs']} fold={row['fold']} "
                f"val_rmse={row['val_rmse']:.4f} | "
                f"{elapsed:.1f}s elapsed, ETA {eta:.1f}s"
            )

    return all_results


def summarize_results(results):
    """Aggregate folds into mean/std metrics and the generalization gap."""
    if results.empty:
        raise ValueError("No grid-search results are available to summarize.")

    summary = (
        results.groupby(["n_factors", "reg_all", "n_epochs"], as_index=False)
        .agg(
            mean_train_rmse=("train_rmse", "mean"),
            std_train_rmse=("train_rmse", "std"),
            mean_val_rmse=("val_rmse", "mean"),
            std_val_rmse=("val_rmse", "std"),
            mean_train_mae=("train_mae", "mean"),
            std_train_mae=("train_mae", "std"),
            mean_val_mae=("val_mae", "mean"),
            std_val_mae=("val_mae", "std"),
            mean_warm_start_ratio=("warm_start_ratio", "mean"),
            mean_warm_val_rmse=("warm_val_rmse", "mean"),
            mean_warm_val_mae=("warm_val_mae", "mean"),
            total_train_time=("train_time", "sum"),
        )
    )

    summary["generalization_gap"] = summary["mean_val_rmse"] - summary["mean_train_rmse"]

    # Model selection is primarily based on validation RMSE.
    summary = summary.sort_values(
        ["mean_val_rmse", "generalization_gap"],
        ascending=[True, True],
    ).reset_index(drop=True)

    return summary[SUMMARY_COLUMNS]


def _json_number(value):
    """Convert a scalar to JSON-compatible float or None."""
    if value is None or pd.isna(value):
        return None
    return float(value)


def save_best_configuration(path, best_row):
    """Save the best hyperparameters and their validation metrics as JSON."""
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "n_splits": N_SPLITS,
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "lr_all": LR_ALL,
            "biased": BIASED,
        },
        "best_params": {
            "n_factors": int(best_row["n_factors"]),
            "reg_all": float(best_row["reg_all"]),
            "n_epochs": int(best_row["n_epochs"]),
            "lr_all": LR_ALL,
            "biased": BIASED,
            "random_state": RANDOM_STATE,
        },
        "validation_metrics": {
            "mean_train_rmse": _json_number(best_row["mean_train_rmse"]),
            "std_train_rmse": _json_number(best_row["std_train_rmse"]),
            "mean_val_rmse": _json_number(best_row["mean_val_rmse"]),
            "std_val_rmse": _json_number(best_row["std_val_rmse"]),
            "mean_train_mae": _json_number(best_row["mean_train_mae"]),
            "std_train_mae": _json_number(best_row["std_train_mae"]),
            "mean_val_mae": _json_number(best_row["mean_val_mae"]),
            "std_val_mae": _json_number(best_row["std_val_mae"]),
            "generalization_gap": _json_number(best_row["generalization_gap"]),
            "mean_warm_start_ratio": _json_number(best_row["mean_warm_start_ratio"]),
            "mean_warm_val_rmse": _json_number(best_row["mean_warm_val_rmse"]),
            "mean_warm_val_mae": _json_number(best_row["mean_warm_val_mae"]),
        },
    }

    temporary_path = path.with_name(path.name + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
    temporary_path.replace(path)


def _regression_colors(reg_values):
    """Assign a distinct color to each regularization value."""
    cmap = plt.get_cmap("viridis")
    positions = np.linspace(0.05, 0.9, len(reg_values))
    return {reg_all: cmap(position) for reg_all, position in zip(reg_values, positions)}


def plot_rmse_vs_epochs(summary, path):
    """Plot train and validation RMSE against epochs, split by latent dimension."""
    n_factors_values = sorted(summary["n_factors"].unique())
    reg_values = sorted(summary["reg_all"].unique())
    reg_colors = _regression_colors(reg_values)

    fig, axes = plt.subplots(
        1,
        len(n_factors_values),
        figsize=(5.5 * len(n_factors_values), 4.8),
        sharey=True,
        squeeze=False,
    )

    for ax, n_factors in zip(axes[0], n_factors_values):
        subset = summary[summary["n_factors"] == n_factors]
        for reg_all in reg_values:
            curve = subset[subset["reg_all"] == reg_all].sort_values("n_epochs")
            if curve.empty:
                continue
            color = reg_colors[reg_all]
            ax.plot(
                curve["n_epochs"],
                curve["mean_train_rmse"],
                color=color,
                linestyle="-",
                marker="o",
                markersize=4,
            )
            ax.plot(
                curve["n_epochs"],
                curve["mean_val_rmse"],
                color=color,
                linestyle="--",
                marker="s",
                markersize=4,
            )

        ax.set_title(f"n_factors = {int(n_factors)}")
        ax.set_xlabel("n_epochs")
        ax.grid(True, alpha=0.3)

    axes[0][0].set_ylabel("RMSE")

    legend_handles = [
        Line2D([0], [0], color=reg_colors[reg_all], lw=2, label=f"reg_all = {reg_all:g}")
        for reg_all in reg_values
    ]
    legend_handles.extend(
        [
            Line2D([0], [0], color="black", lw=2, linestyle="-", label="Train"),
            Line2D([0], [0], color="black", lw=2, linestyle="--", label="Validation"),
        ]
    )

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=min(len(legend_handles), 6),
        frameon=False,
        fontsize=9,
    )
    fig.suptitle("RMSE vs number of epochs")
    fig.tight_layout(rect=(0, 0.12, 1, 0.95))
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_factor_influence(summary, path):
    """Plot validation RMSE against the latent dimension."""
    n_factors_values = sorted(summary["n_factors"].unique())
    reg_values = sorted(summary["reg_all"].unique())
    reg_colors = _regression_colors(reg_values)

    fig, ax = plt.subplots(figsize=(8, 5))
    for reg_all in reg_values:
        curve = (
            summary[summary["reg_all"] == reg_all]
            .groupby("n_factors", as_index=False)["mean_val_rmse"]
            .mean()
            .sort_values("n_factors")
        )
        ax.plot(
            curve["n_factors"],
            curve["mean_val_rmse"],
            color=reg_colors[reg_all],
            marker="o",
            label=f"reg_all = {reg_all:g}",
        )

    ax.set_xticks(n_factors_values)
    ax.set_xlabel("n_factors")
    ax.set_ylabel("Mean validation RMSE")
    ax.set_title("Influence of latent dimension on validation RMSE")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_regularization_influence(summary, path):
    """Plot validation RMSE against the regularization strength."""
    n_factors_values = sorted(summary["n_factors"].unique())
    reg_values = sorted(summary["reg_all"].unique())

    cmap = plt.get_cmap("viridis")
    positions = np.linspace(0.05, 0.9, len(n_factors_values))
    factor_colors = {
        n_factors: cmap(position)
        for n_factors, position in zip(n_factors_values, positions)
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    for n_factors in n_factors_values:
        curve = (
            summary[summary["n_factors"] == n_factors]
            .groupby("reg_all", as_index=False)["mean_val_rmse"]
            .mean()
            .sort_values("reg_all")
        )
        ax.plot(
            curve["reg_all"],
            curve["mean_val_rmse"],
            color=factor_colors[n_factors],
            marker="o",
            label=f"n_factors = {int(n_factors)}",
        )

    ax.set_xticks(reg_values)
    ax.set_xticklabels([f"{reg_all:g}" for reg_all in reg_values])
    ax.set_xlabel("reg_all")
    ax.set_ylabel("Mean validation RMSE")
    ax.set_title("Influence of regularization on validation RMSE")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_performance_heatmap(summary, path):
    """Heatmap of the best validation RMSE for each factor/regularization pair."""
    pivot = (
        summary.groupby(["reg_all", "n_factors"], as_index=False)["mean_val_rmse"]
        .min()
        .pivot(index="reg_all", columns="n_factors", values="mean_val_rmse")
        .sort_index()
    )
    pivot = pivot.reindex(columns=sorted(pivot.columns))

    fig, ax = plt.subplots(figsize=(8, 5))
    image = ax.imshow(
        pivot.to_numpy(dtype=float),
        aspect="auto",
        cmap="viridis_r",
        origin="lower",
    )

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{int(value)}" for value in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{value:g}" for value in pivot.index])
    ax.set_xlabel("n_factors")
    ax.set_ylabel("reg_all")
    ax.set_title("Best mean validation RMSE by latent dimension and regularization")

    normalization = image.norm
    for row_idx in range(pivot.shape[0]):
        for col_idx in range(pivot.shape[1]):
            value = pivot.iat[row_idx, col_idx]
            if pd.isna(value):
                continue
            text_color = "black" if normalization(value) < 0.55 else "white"
            ax.text(
                col_idx,
                row_idx,
                f"{value:.4f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=9,
            )

    best_row = summary.loc[summary["mean_val_rmse"].idxmin()]
    best_y = list(pivot.index).index(best_row["reg_all"])
    best_x = list(pivot.columns).index(best_row["n_factors"])
    ax.add_patch(
        Rectangle(
            (best_x - 0.5, best_y - 0.5),
            1,
            1,
            fill=False,
            edgecolor="red",
            linewidth=2,
        )
    )

    fig.colorbar(image, ax=ax, label="Mean validation RMSE")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def print_top_configurations(summary, count=10):
    """Print the best configurations sorted by mean validation RMSE."""
    display_columns = [
        "n_factors",
        "reg_all",
        "n_epochs",
        "mean_train_rmse",
        "mean_val_rmse",
        "std_val_rmse",
        "mean_train_mae",
        "mean_val_mae",
        "generalization_gap",
    ]

    top = summary[display_columns].head(count).copy()
    for column in top.columns:
        if top[column].dtype.kind == "f":
            top[column] = top[column].round(4)

    print(f"\nTop {min(count, len(top))} configurations:")
    print(top.to_string(index=False))


def print_best_configuration(best_row):
    """Print the final best configuration banner."""
    print("\n" + "=" * 36)
    print("BEST SVD CONFIGURATION")
    print("=" * 36)
    print()
    print(f"n_factors : {int(best_row['n_factors'])}")
    print(f"reg_all   : {best_row['reg_all']:g}")
    print(f"n_epochs  : {int(best_row['n_epochs'])}")
    print()
    print(f"Mean Train RMSE : {best_row['mean_train_rmse']:.4f}")
    print(f"Mean Val RMSE   : {best_row['mean_val_rmse']:.4f}")
    print(f"Std Val RMSE    : {best_row['std_val_rmse']:.4f}")
    print()
    print(f"Mean Train MAE  : {best_row['mean_train_mae']:.4f}")
    print(f"Mean Val MAE    : {best_row['mean_val_mae']:.4f}")
    print()
    print(f"Generalization Gap : {best_row['generalization_gap']:.4f}")
    print("=" * 36)


def parse_args():
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description="Run the Filmfinder SVD grid search.",
    )
    parser.add_argument(
        "--data-dir",
        default=".",
        help="Directory containing ratings_sparse_final.npz and the .npy files.",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory where results and figures are written.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel workers (start with 1 to limit RAM usage).",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Ignore existing results and retrain every configuration.",
    )
    parser.add_argument(
        "--analysis-only",
        action="store_true",
        help="Only rebuild the summary, JSON and figures from saved CSV results.",
    )
    parser.set_defaults(resume=True)
    return parser.parse_args()


def main():
    """Run the grid search and its analysis."""
    args = parse_args()

    if args.n_jobs < 1:
        raise ValueError("--n-jobs must be at least 1.")

    data_dir = Path(args.data_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    results_path = results_dir / "svd_grid_results.csv"
    summary_path = results_dir / "svd_grid_summary.csv"
    best_params_path = results_dir / "svd_best_params.json"

    if not args.analysis_only:
        dataset = load_ratings(data_dir)
        folds = build_folds(dataset)

        try:
            run_grid_search(
                folds=folds,
                results_path=results_path,
                n_jobs=args.n_jobs,
                resume=args.resume,
            )
        except KeyboardInterrupt:
            print("\nInterrupted. Partial results remain saved on disk.")

    if not results_path.exists():
        raise FileNotFoundError(
            f"No saved results found at {results_path}. Run the grid search first."
        )

    df_grid_results = pd.read_csv(results_path)
    df_grid_summary = summarize_results(df_grid_results)
    _atomic_write_csv(df_grid_summary, summary_path)

    print_top_configurations(df_grid_summary)

    best_row = df_grid_summary.iloc[0]
    save_best_configuration(best_params_path, best_row)

    plot_rmse_vs_epochs(df_grid_summary, results_dir / "svd_grid_rmse.png")
    plot_factor_influence(df_grid_summary, results_dir / "svd_grid_factors.png")
    plot_regularization_influence(
        df_grid_summary,
        results_dir / "svd_grid_regularization.png",
    )
    plot_performance_heatmap(df_grid_summary, results_dir / "svd_grid_heatmap.png")

    print(f"\nResults saved to: {results_dir.resolve()}")
    print_best_configuration(best_row)


if __name__ == "__main__":
    main()
