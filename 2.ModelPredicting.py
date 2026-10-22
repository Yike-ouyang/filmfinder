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
    from surprise import Dataset, Reader, SVD
    from surprise.model_selection import ShuffleSplit
    from surprise import accuracy
    return Dataset, Reader, SVD, ShuffleSplit, accuracy, load_npz, mo, np, pd


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
    return (data,)


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

    epochs = [1, 2, 5, 10, 15, 20, 30, 40, 50]

    cv = ShuffleSplit(
        n_splits=2,
        test_size=0.2,
        random_state=42
    )

    results = []

    # ==========================================
    # CROSS VALIDATION
    # ==========================================

    for fold, (trainset, testset) in enumerate(cv.split(data), start=1):

        print(f"\n===== Fold {fold}/2 =====")

        # Notes d'entraînement pour évaluation
        train_testset = trainset.build_testset()

        for n_epoch in epochs:

            model = SVD(
                n_factors=100,
                n_epochs=n_epoch,
                lr_all=0.005,
                reg_all=0.02,
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
    return (results,)


@app.cell
def _(pd, results):
    df_results = pd.DataFrame(results)

    df_results
    return


if __name__ == "__main__":
    app.run()
