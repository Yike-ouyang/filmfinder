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

    return load_npz, mo, np


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

    return movie_ids_final, ratings_sparse_final, user_ids_final


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
    return


if __name__ == "__main__":
    app.run()
