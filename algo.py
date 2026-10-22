import marimo

__generated_with = "0.17.6"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # Filmfinder
    An algorithm for recommending films to users based on similar profil users, built on Letterbox Database
    """)
    return


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path
    import seaborn as sbn
    from scipy.sparse import csr_matrix, save_npz,load_npz
    from surprise import Dataset, Reader, SVD
    from surprise.model_selection import ShuffleSplit
    from surprise import accuracy
    return Path, csr_matrix, load_npz, mo, np, pd, plt, save_npz


@app.cell
def _(mo):
    mo.md(r"""
    # Data Preprocessing
    """)
    return


@app.cell
def _(pd):
    df=pd.read_csv("movies_final.csv")
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
    ## Creating User-Film Matrix
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
    ratings_matrix_cleaned = ratings_matrix.dropna(axis=1, how="all")
    return (ratings_matrix_cleaned,)


@app.cell
def _(ratings_matrix_cleaned):
    ratings_matrix_cleaned.shape
    return


@app.cell
def _(mo):
    mo.md(r"""
    Heatmap(4000 x 4000) of the matrix
    """)
    return


@app.cell
def _(np, plt, ratings_matrix):


    # 1. Trier les lignes et colonnes par nombre de valeurs non-NA
    rows_sorted = ratings_matrix.notna().sum(axis=1).sort_values(ascending=False).index
    cols_sorted = ratings_matrix.notna().sum(axis=0).sort_values(ascending=False).index

    ratings_sorted = ratings_matrix.loc[rows_sorted, cols_sorted]

    # 2. Convertir en tableau NumPy
    data = ratings_sorted.to_numpy(dtype=np.float32, na_value=np.nan)
    data = np.ma.masked_invalid(data)

    # 3. Définir la colormap (NA en blanc)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white")

    # 4. Créer une figure de 8000 x 8000 pixels
    fig, ax = plt.subplots(figsize=(20, 20), dpi=400)

    im = ax.imshow(
        data,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=0,
        vmax=5
    )

    ax.set_title("Heatmap des notes triées par densité")
    ax.set_xlabel("Films (nombre de notes décroissant)")
    ax.set_ylabel("Utilisateurs (nombre de notes décroissant)")

    # Masquer les ticks pour plus de lisibilité
    ax.set_xticks([])
    ax.set_yticks([])

    plt.colorbar(im, ax=ax, label="Note")

    # 5. Sauvegarder en 8000 x 8000 pixels
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
    ## Distribution and remove min
    """)
    return


@app.cell
def _(ratings_matrix, ratings_matrix_cleaned):
    # Conserver uniquement les films ayant au moins 10 notes
    ratings_matrix_10 = ratings_matrix_cleaned.loc[
        :, ratings_matrix.notna().sum(axis=0) >= 10
    ]

    # Vérifier les nouvelles dimensions
    print(ratings_matrix_10.shape)
    return (ratings_matrix_10,)


@app.cell
def _(np, plt, ratings_matrix_10):


    # Nombre de notes par film
    ratings_per_movie = ratings_matrix_10.notna().sum(axis=0)

    # Histogramme avec des bins logarithmiques
    values = ratings_per_movie[ratings_per_movie > 0]

    bins = np.logspace(
        np.log10(values.min()),
        np.log10(values.max()),
        100
    )

    plt.figure(figsize=(12, 6))

    plt.hist(values, bins=bins, edgecolor="black")

    # Échelles logarithmiques sur les deux axes
    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel("Nombre de notes par film (log)")
    plt.ylabel("Nombre de films (log)")
    plt.title("Distribution du nombre de notes par film (log-log)")

    plt.grid(True, which="both", alpha=0.3)
    plt.show()
    return


@app.cell
def _(np, plt, ratings_matrix_10):


    # Compter le nombre de notes par utilisateur (ligne)
    ratings_per_user_10 = ratings_matrix_10.notna().sum(axis=1)

    # Exclure les utilisateurs sans note
    user_counts_log_10 = ratings_per_user_10[ratings_per_user_10 > 0]

    # Intervalles logarithmiques
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

    ax_user_log_10.set_xlabel("Nombre de notes par utilisateur (log)")
    ax_user_log_10.set_ylabel("Nombre d'utilisateurs (log)")
    ax_user_log_10.set_title("Distribution du nombre de notes par utilisateur (log-log)")

    ax_user_log_10.grid(True, which="both", alpha=0.3)

    fig_user_log_10
    return


@app.cell
def _(ratings_matrix_10):
    # Conserver uniquement les utilisateurs ayant au moins 15 notes
    ratings_matrix_final = ratings_matrix_10.loc[
        ratings_matrix_10.notna().sum(axis=1) >= 15, :
    ]

    # Vérifier les nouvelles dimensions
    print(ratings_matrix_final.shape)
    return (ratings_matrix_final,)


@app.cell
def _(csr_matrix, np, ratings_matrix_final, save_npz):


    # 1. Sauvegarder les identifiants
    user_ids_final = ratings_matrix_final.index.to_numpy()
    movie_ids_final = ratings_matrix_final.columns.to_numpy()

    # 2. Convertir en matrice sparse CSR
    ratings_sparse_final = csr_matrix(
        ratings_matrix_final.fillna(0).to_numpy(dtype=np.float32)
    )

    # 3. Sauvegarder la matrice sparse
    save_npz("ratings_sparse_final.npz", ratings_sparse_final)

    # 4. Sauvegarder les identifiants
    np.save("user_ids_final.npy", user_ids_final)
    np.save("movie_ids_final.npy", movie_ids_final)

    print("Matrice sparse sauvegardée !")
    print("Dimensions :", ratings_sparse_final.shape)
    print("Nombre de notes :", ratings_sparse_final.nnz)
    return (ratings_sparse_final,)


@app.cell
def _(load_npz, np):
    # 1. Charger la matrice sparse CSR
    ratings_sparse_final = load_npz("ratings_sparse_final.npz").tocsr()

    # 2. Charger les identifiants
    user_ids_final = np.load(
        "user_ids_final.npy",
        allow_pickle=True
    )

    movie_ids_final = np.load(
        "movie_ids_final.npy",
        allow_pickle=True
    )

    # 3. Vérifier la cohérence des dimensions
    assert ratings_sparse_final.shape == (
        len(user_ids_final),
        len(movie_ids_final)
    ), "Incohérence entre la matrice et les identifiants."

    # 4. Afficher les informations
    n_users, n_movies = ratings_sparse_final.shape
    n_ratings = ratings_sparse_final.nnz

    density = n_ratings / (n_users * n_movies)

    print(f"Utilisateurs : {n_users:,}")
    print(f"Films : {n_movies:,}")
    print(f"Notes observées : {n_ratings:,}")
    print(f"Densité : {density:.6%}")
    print(f"Sparsité : {1 - density:.6%}")
    print(f"Mémoire CSR : {ratings_sparse_final.data.nbytes / 1024**2:.2f} MB (valeurs uniquement)")
    return (ratings_sparse_final,)


@app.cell
def _(ratings_sparse_final):




    # Conversion CSR -> COO
    ratings_coo = ratings_sparse_final.tocoo()

    # Indices utilisateurs et films
    user_idx = ratings_coo.row
    movie_idx = ratings_coo.col

    # Notes observées
    ratings = ratings_coo.data

    print("Nombre de notes :", len(ratings))
    print("Nombre d'utilisateurs :", ratings_sparse_final.shape[0])
    print("Nombre de films :", ratings_sparse_final.shape[1])
    return


if __name__ == "__main__":
    app.run()
