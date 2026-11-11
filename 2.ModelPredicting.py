import marimo

__generated_with = "0.17.6"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # Filmfinder — Model Predicting

    This notebook loads the sparse user–film matrix produced by
    `1.Preprocessing.py` and prepares it for the recommendation model.

    **Pipeline**
    1. Load the CSR sparse matrix and its user/film identifiers.
    2. Check that the matrix and identifiers are consistent.
    3. Report basic statistics (density, sparsity, memory).
    4. Convert the matrix to COO format for model training.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from scipy.sparse import load_npz
    import pandas as pd
    import matplotlib.pyplot as plt
    from surprise import Dataset, Reader, SVD
    from surprise.model_selection import ShuffleSplit
    from surprise import accuracy
    return Dataset, Reader, SVD, ShuffleSplit, accuracy, load_npz, mo, np, pd, plt


@app.cell
def _(mo):
    mo.md(r"""
    ## Loading the sparse matrix

    The matrix and the identifier arrays must be loaded together so that row
    and column indices can be mapped back to actual users and films.
    """)
    return


@app.cell
def _(load_npz, np):
    # 1. Load the CSR sparse matrix
    ratings_sparse_final = load_npz("ratings_sparse_final.npz").tocsr()

    # 2. Load the identifiers
    user_ids_final = np.load(
        "user_ids_final.npy",
        allow_pickle=True
    )

    movie_ids_final = np.load(
        "movie_ids_final.npy",
        allow_pickle=True
    )

    # 3. Check that the dimensions are consistent
    assert ratings_sparse_final.shape == (
        len(user_ids_final),
        len(movie_ids_final)
    ), "Mismatch between the matrix and the identifiers."
    return (ratings_sparse_final,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Matrix statistics

    Basic indicators to verify that the loaded matrix matches the one saved
    during preprocessing.
    """)
    return


@app.cell
def _(ratings_sparse_final):
    # 4. Display the information
    n_users, n_movies = ratings_sparse_final.shape
    n_ratings = ratings_sparse_final.nnz

    density = n_ratings / (n_users * n_movies)

    print(f"Users: {n_users:,}")
    print(f"Films: {n_movies:,}")
    print(f"Observed ratings: {n_ratings:,}")
    print(f"Density: {density:.6%}")
    print(f"Sparsity: {1 - density:.6%}")
    print(f"CSR memory: {ratings_sparse_final.data.nbytes / 1024**2:.2f} MB (values only)")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Converting to COO format

    The COO representation exposes explicit `(user, film, rating)` triplets,
    which is the format expected by most recommendation algorithms.
    """)
    return


@app.cell
def _(ratings_sparse_final):
    # CSR -> COO conversion
    ratings_coo = ratings_sparse_final.tocoo()

    # User and film indices
    user_idx = ratings_coo.row
    movie_idx = ratings_coo.col

    # Observed ratings
    ratings = ratings_coo.data

    print("Number of ratings:", len(ratings))
    print("Number of users:", ratings_sparse_final.shape[0])
    print("Number of films:", ratings_sparse_final.shape[1])
    return movie_idx, ratings, user_idx


@app.cell
def _(Dataset, Reader, movie_idx, pd, ratings, user_idx):
    df_ratings = pd.DataFrame({
        "user_id": user_idx,
        "movie_id": movie_idx,
        "rating": ratings
    })
    rating_min = float(df_ratings["rating"].min())
    rating_max = float(df_ratings["rating"].max())

    reader = Reader(
        rating_scale=(rating_min, rating_max)
    )

    data = Dataset.load_from_df(
        df_ratings[["user_id", "movie_id", "rating"]],
        reader
    )
    return data, df_ratings


@app.cell
def _(mo):
    mo.md(r"""
    ## Training the model
    """)
    return


@app.cell
def _(SVD, ShuffleSplit, accuracy, data, pd):


    # ==========================================
    # CONFIGURATION
    # ==========================================

    N_FACTORS = 100
    LR_ALL = 0.005
    REG_ALL = 0.05
    ANALYSIS_EPOCH = 20

    epochs = [1, 2, 5, 10, 15, 20, 30, 40, 50]

    cv = ShuffleSplit(
        n_splits=2,
        test_size=0.2,
        random_state=42
    )

    results = []
    val_records = []
    cold_start_rows = []

    # ==========================================
    # CROSS VALIDATION
    # ==========================================

    for fold, (trainset, testset) in enumerate(cv.split(data), start=1):

        print(f"\n===== Fold {fold}/2 =====")

        train_users = {trainset.to_raw_uid(u) for u in trainset.ur}
        train_items = {trainset.to_raw_iid(i) for i in trainset.ir}

        test_users = {uid for uid, _, _ in testset}
        test_items = {iid for _, iid, _ in testset}

        unseen_users = test_users - train_users
        unseen_items = test_items - train_items

        n_test = len(testset)
        n_unseen_user_ratings = sum(
            uid not in train_users for uid, _, _ in testset
        )
        n_unseen_item_ratings = sum(
            iid not in train_items for _, iid, _ in testset
        )
        n_unseen_either_ratings = sum(
            uid not in train_users or iid not in train_items
            for uid, iid, _ in testset
        )

        print(
            f"Cold start | users jamais vus: "
            f"{len(unseen_users):,}/{len(test_users):,} | "
            f"films jamais vus: "
            f"{len(unseen_items):,}/{len(test_items):,} | "
            f"notes avec user inconnu: "
            f"{n_unseen_user_ratings:,} ({n_unseen_user_ratings / n_test:.2%}) | "
            f"notes avec film inconnu: "
            f"{n_unseen_item_ratings:,} ({n_unseen_item_ratings / n_test:.2%}) | "
            f"notes concernées: "
            f"{n_unseen_either_ratings:,} ({n_unseen_either_ratings / n_test:.2%})"
        )

        cold_start_rows.append({
            "fold": fold,
            "test_users": len(test_users),
            "unseen_users": len(unseen_users),
            "test_items": len(test_items),
            "unseen_items": len(unseen_items),
            "test_ratings": n_test,
            "ratings_unseen_user": n_unseen_user_ratings,
            "ratings_unseen_item": n_unseen_item_ratings,
            "ratings_unseen_either": n_unseen_either_ratings
        })

        # Notes d'entraînement pour évaluation
        train_testset = trainset.build_testset()

        for n_epoch in epochs:

            model = SVD(
                n_factors=N_FACTORS,
                n_epochs=n_epoch,
                lr_all=LR_ALL,
                reg_all=REG_ALL,
                biased=True,
                random_state=42,
                verbose=False
            )

            model.fit(trainset)

            # Prédictions sur train
            train_predictions = model.test(train_testset)

            # Prédictions sur validation
            val_predictions = model.test(testset)

            # Métriques
            train_rmse = accuracy.rmse(
                train_predictions,
                verbose=False
            )

            val_rmse = accuracy.rmse(
                val_predictions,
                verbose=False
            )

            train_mae = accuracy.mae(
                train_predictions,
                verbose=False
            )

            val_mae = accuracy.mae(
                val_predictions,
                verbose=False
            )

            if n_epoch == ANALYSIS_EPOCH:
                val_records.extend({
                    "fold": fold,
                    "user_id": pred.uid,
                    "movie_id": pred.iid,
                    "rating_true": pred.r_ui,
                    "rating_est": pred.est,
                    "error": pred.est - pred.r_ui,
                    "abs_error": abs(pred.est - pred.r_ui)
                } for pred in val_predictions)

            results.append({
                "fold": fold,
                "epoch": n_epoch,
                "train_rmse": train_rmse,
                "val_rmse": val_rmse,
                "train_mae": train_mae,
                "val_mae": val_mae
            })

            print(
                f"Epoch {n_epoch:3d} | "
                f"Train RMSE: {train_rmse:.4f} | "
                f"Val RMSE: {val_rmse:.4f}"
            )

    df_learning = pd.DataFrame(results)

    df_cold_start = pd.DataFrame(
        cold_start_rows,
        columns=[
            "fold",
            "test_users",
            "unseen_users",
            "test_items",
            "unseen_items",
            "test_ratings",
            "ratings_unseen_user",
            "ratings_unseen_item",
            "ratings_unseen_either"
        ]
    )

    df_val_predictions = pd.DataFrame(
        val_records,
        columns=[
            "fold",
            "user_id",
            "movie_id",
            "rating_true",
            "rating_est",
            "error",
            "abs_error"
        ]
    )

    return df_cold_start, df_val_predictions, results


@app.cell
def _(pd, results):
    df_results = pd.DataFrame(results)

    df_results
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Cold start in the validation sets

    Counts are reported per fold for users and films that appear in the test
    set but were never observed during training.
    """)
    return


@app.cell
def _(df_cold_start):
    df_cold_start
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Error analysis

    The validation predictions of the grid-search configuration
    (`n_factors=100`, `reg_all=0.05`, `n_epochs=20`, `lr_all=0.005`) are
    aggregated per user and per film, then merged with the total number of
    ratings and the mean rating observed in the full matrix.
    """)
    return


@app.cell
def _(df_ratings, df_val_predictions, pd, plt):
    user_stats = df_ratings.groupby("user_id").agg(
        n_ratings=("rating", "size"),
        mean_rating=("rating", "mean")
    )

    movie_stats = df_ratings.groupby("movie_id").agg(
        n_ratings=("rating", "size"),
        mean_rating=("rating", "mean")
    )

    df_user = (
        df_val_predictions
        .groupby("user_id")
        .agg(
            mae=("abs_error", "mean"),
            bias=("error", "mean"),
            n_test=("abs_error", "size")
        )
        .join(user_stats, how="inner")
        .reset_index()
    )

    df_movie = (
        df_val_predictions
        .groupby("movie_id")
        .agg(
            mae=("abs_error", "mean"),
            bias=("error", "mean"),
            n_test=("abs_error", "size")
        )
        .join(movie_stats, how="inner")
        .reset_index()
    )

    def add_trend(ax, frame, xcol, ycol):
        trend_data = frame[[xcol, ycol]].dropna()
        n_bins = min(20, trend_data[xcol].nunique())
        if n_bins < 2:
            return
        trend_data = trend_data.assign(
            bin=pd.qcut(
                trend_data[xcol],
                q=n_bins,
                duplicates="drop"
            )
        )
        trend = trend_data.groupby("bin", observed=True).agg(
            x=(xcol, "median"),
            y=(ycol, "median")
        )
        ax.plot(
            trend["x"],
            trend["y"],
            color="crimson",
            marker="o",
            linewidth=1.5,
            markersize=4,
            label="moyenne par bin"
        )
        ax.legend()

    def make_dashboard(metric, ylabel, title_prefix, zero_line=False):
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        panels = [
            (
                axes[0, 0],
                df_user,
                "n_ratings",
                "Nombre de notes de l'utilisateur (log)",
                f"{title_prefix} par utilisateur vs activité",
                True
            ),
            (
                axes[0, 1],
                df_movie,
                "n_ratings",
                "Nombre de notes du film (log)",
                f"{title_prefix} par film vs activité",
                True
            ),
            (
                axes[1, 0],
                df_movie,
                "mean_rating",
                "Note moyenne du film",
                f"{title_prefix} par film vs note moyenne",
                False
            ),
            (
                axes[1, 1],
                df_user,
                "mean_rating",
                "Note moyenne de l'utilisateur",
                f"{title_prefix} par utilisateur vs note moyenne",
                False
            )
        ]

        for ax, frame, xcol, xlabel, title, log_x in panels:
            ax.scatter(
                frame[xcol],
                frame[metric],
                s=8,
                alpha=0.2,
                edgecolors="none"
            )
            add_trend(ax, frame, xcol, metric)

            if log_x:
                ax.set_xscale("log")
            if zero_line:
                ax.axhline(
                    0,
                    color="black",
                    linestyle="--",
                    linewidth=0.8
                )

            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(True, which="both", alpha=0.3)

        fig.tight_layout()
        return fig

    return df_movie, df_user, make_dashboard


@app.cell
def _(make_dashboard):
    fig_mae = make_dashboard(
        "mae",
        "Erreur absolue moyenne (MAE)",
        "MAE"
    )

    fig_mae
    return


@app.cell
def _(make_dashboard):
    fig_bias = make_dashboard(
        "bias",
        "Erreur signée moyenne (estimation - réel)",
        "Biais",
        zero_line=True
    )

    fig_bias
    return


if __name__ == "__main__":
    app.run()
