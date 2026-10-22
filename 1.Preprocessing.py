import marimo

__generated_with = "0.17.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import time
    time.sleep(120)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Filmfinder — Data Preprocessing

    This notebook builds the user–film ratings matrix used by the
    recommendation model.

    **Pipeline**
    1. Load the Letterboxd ratings and the filtered film catalogue.
    2. Keep ratings only for films present in the catalogue.
    3. Pivot the data into a user × film matrix.
    4. Inspect the matrix density and rating distributions.
    5. Filter out films and users with too few ratings.
    6. Save the final sparse matrix and its identifiers to disk.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path
    from scipy.sparse import csr_matrix, save_npz
    return Path, csr_matrix, mo, np, pd, plt, save_npz


@app.cell
def _(mo):
    mo.md(r"""
    ## Loading the data

    `movies_final.csv` is the film catalogue, while the raw Letterboxd
    exports (`ratings_export.csv` and `users_export.csv`) come from the cached
    Kaggle dataset.
    """)
    return


@app.cell
def _(pd):
    df = pd.read_csv("movies_final.csv")
    return (df,)


@app.cell
def _(Path, pd):
    data_path = Path("/users/eleves-b/2023/yike.ouyang/.cache/kagglehub/datasets/samlearner/letterboxd-movie-ratings-data/versions/6")

    ratings_export_user = pd.read_csv(data_path / "ratings_export.csv", engine="python")

    users_export_user = pd.read_csv(data_path / "users_export.csv", engine="python")
    return (ratings_export_user,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Creating the user–film matrix

    Ratings are restricted to films that belong to the catalogue, then pivoted
    so that rows are users and columns are films.
    """)
    return


@app.cell
def _(df, ratings_export_user):
    ratings_export_user_filtered = ratings_export_user[
        ratings_export_user["movie_id"].isin(df["movie_id"].explode())
    ]
    return (ratings_export_user_filtered,)


@app.cell
def _(ratings_export_user_filtered):
    ratings_matrix = ratings_export_user_filtered.pivot_table(
        index="user_id",
        columns="movie_id",
        values="rating_val",
        aggfunc="mean"
    )

    ratings_matrix = ratings_matrix.astype("Float64")
    ratings_matrix.shape
    return (ratings_matrix,)


@app.cell
def _(ratings_matrix):
    # Drop films that were never rated
    ratings_matrix_cleaned = ratings_matrix.dropna(axis=1, how="all")
    return (ratings_matrix_cleaned,)


@app.cell
def _(ratings_matrix_cleaned):
    ratings_matrix_cleaned.shape
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Matrix density heatmap

    Rows and columns are sorted by the number of observed ratings so that the
    densest users and films appear at the top-left. Missing ratings are shown
    in white.
    """)
    return


@app.cell
def _(np, plt, ratings_matrix):
    # 1. Sort rows and columns by the number of non-NA values
    rows_sorted = ratings_matrix.notna().sum(axis=1).sort_values(ascending=False).index
    cols_sorted = ratings_matrix.notna().sum(axis=0).sort_values(ascending=False).index

    ratings_sorted = ratings_matrix.loc[rows_sorted, cols_sorted]

    # 2. Convert to a NumPy array
    data = ratings_sorted.to_numpy(dtype=np.float32, na_value=np.nan)
    data = np.ma.masked_invalid(data)

    # 3. Define the colormap (NA in white)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white")

    # 4. Create an 8000 x 8000 pixel figure
    fig, ax = plt.subplots(figsize=(20, 20), dpi=400)

    im = ax.imshow(
        data,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=0,
        vmax=5
    )

    ax.set_title("Heatmap of ratings sorted by density")
    ax.set_xlabel("Films (number of ratings, descending)")
    ax.set_ylabel("Users (number of ratings, descending)")

    # Hide ticks for readability
    ax.set_xticks([])
    ax.set_yticks([])

    plt.colorbar(im, ax=ax, label="Rating")

    # 5. Save as an 8000 x 8000 pixel image
    plt.savefig(
        "ratings_heatmap_8000.png",
        dpi=400,
        bbox_inches=None
    )

    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Distribution and minimum filtering

    Films rated fewer than 10 times are removed, then the number of ratings per
    film and per user is inspected on log–log scales.
    """)
    return


@app.cell
def _(ratings_matrix, ratings_matrix_cleaned):
    # Keep only films with at least 10 ratings
    ratings_matrix_10 = ratings_matrix_cleaned.loc[
        :, ratings_matrix.notna().sum(axis=0) >= 10
    ]

    # Check the new dimensions
    print(ratings_matrix_10.shape)
    return (ratings_matrix_10,)


@app.cell
def _(np, plt, ratings_matrix_10):
    # Number of ratings per film
    ratings_per_movie = ratings_matrix_10.notna().sum(axis=0)

    # Histogram with logarithmic bins
    values = ratings_per_movie[ratings_per_movie > 0]

    bins = np.logspace(
        np.log10(values.min()),
        np.log10(values.max()),
        100
    )

    plt.figure(figsize=(12, 6))

    plt.hist(values, bins=bins, edgecolor="black")

    # Logarithmic scales on both axes
    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel("Number of ratings per film (log)")
    plt.ylabel("Number of films (log)")
    plt.title("Distribution of ratings per film (log-log)")

    plt.grid(True, which="both", alpha=0.3)
    plt.show()
    return


@app.cell
def _(np, plt, ratings_matrix_10):
    # Count the number of ratings per user (row)
    ratings_per_user_10 = ratings_matrix_10.notna().sum(axis=1)

    # Exclude users without any rating
    user_counts_log_10 = ratings_per_user_10[ratings_per_user_10 > 0]

    # Logarithmic bins
    bins_user_log_10 = np.logspace(
        np.log10(user_counts_log_10.min()),
        np.log10(user_counts_log_10.max()),
        100
    )

    fig_user_log_10, ax_user_log_10 = plt.subplots(figsize=(10, 6))

    ax_user_log_10.hist(
        user_counts_log_10,
        bins=bins_user_log_10,
        edgecolor="black"
    )

    ax_user_log_10.set_xscale("log")
    ax_user_log_10.set_yscale("log")

    ax_user_log_10.set_xlabel("Number of ratings per user (log)")
    ax_user_log_10.set_ylabel("Number of users (log)")
    ax_user_log_10.set_title("Distribution of ratings per user (log-log)")

    ax_user_log_10.grid(True, which="both", alpha=0.3)

    fig_user_log_10
    return


@app.cell
def _(ratings_matrix_10):
    # Keep only users with at least 15 ratings
    ratings_matrix_final = ratings_matrix_10.loc[
        ratings_matrix_10.notna().sum(axis=1) >= 15, :
    ]

    # Check the new dimensions
    print(ratings_matrix_final.shape)
    return (ratings_matrix_final,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Saving the sparse matrix

    The final matrix is stored as a CSR sparse matrix, together with the user
    and film identifiers, so that the modelling notebook can load it without
    rerunning the whole preprocessing.
    """)
    return


@app.cell
def _(csr_matrix, np, ratings_matrix_final, save_npz):
    # 1. Save the identifiers
    user_ids_final = ratings_matrix_final.index.to_numpy()
    movie_ids_final = ratings_matrix_final.columns.to_numpy()

    # 2. Convert to a sparse CSR matrix
    ratings_sparse_final = csr_matrix(
        ratings_matrix_final.fillna(0).to_numpy(dtype=np.float32)
    )

    # 3. Save the sparse matrix
    save_npz("ratings_sparse_final.npz", ratings_sparse_final)

    # 4. Save the identifiers
    np.save("user_ids_final.npy", user_ids_final)
    np.save("movie_ids_final.npy", movie_ids_final)

    print("Sparse matrix saved!")
    print("Shape:", ratings_sparse_final.shape)
    print("Number of ratings:", ratings_sparse_final.nnz)
    return


if __name__ == "__main__":
    app.run()
